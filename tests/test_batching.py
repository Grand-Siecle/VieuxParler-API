"""Batching engine tests with a fake model (no fairseq, no checkpoint).

The fake model implements the slice of the fairseq hub interface the engine
relies on: ``translate(list, beam=)``, ``encode(str)`` and
``cfg.dataset.max_tokens``. Token counts are words + 1 (EOS), so a line of
``k`` words is ``k + 1`` tokens.
"""

import threading
import time
from types import SimpleNamespace

import pytest

from app.translator import (
    BatchTranslator,
    InferenceSettings,
    TokenBudget,
    is_oom_error,
)

GiB = 1024 ** 3
MiB = 1024 ** 2


class FakeModel:
    def __init__(self, oom_above=None, gate=None):
        self.cfg = SimpleNamespace(dataset=SimpleNamespace(max_tokens=200))
        self.calls = []  # texts passed to each translate() call (as lists)
        self.single_calls = []  # str arguments, i.e. reference-path calls
        self.budgets = []  # cfg.dataset.max_tokens seen by each call
        self.oom_above = oom_above  # raise OOM while budget > this
        self.gate = gate  # threading.Event the first call blocks on
        self._gated = False

    def encode(self, text):
        return text.split() + ["</s>"]

    def translate(self, sentences, beam=5, **kwargs):
        if isinstance(sentences, str):
            self.single_calls.append(sentences)
            return self.translate([sentences], beam=beam)[0]
        if self.gate is not None and not self._gated:
            self._gated = True
            self.gate.wait(5)
        self.calls.append(list(sentences))
        self.budgets.append(self.cfg.dataset.max_tokens)
        if self.oom_above is not None and self.cfg.dataset.max_tokens > self.oom_above:
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")
        for s in sentences:
            if "BOOM" in s:
                raise ValueError(f"bad line: {s}")
        return [s.upper() for s in sentences]


def make_engine(model=None, coalesce=False, device="cpu", mem_info=None, **overrides):
    settings = InferenceSettings(coalesce=coalesce, **overrides)
    model = model or FakeModel()
    return BatchTranslator(model, device, settings, mem_info=mem_info), model


def words(n, sep=" "):
    return sep.join(f"w{i}" for i in range(n))


# ── order, empties, duplicates, reference path ──────────────────────


def test_batched_path_keeps_order_and_length():
    engine, model = make_engine()
    texts = ["un deux", "trois", "quatre cinq six", "sept"]
    out = engine.translate_batch(texts, batch_size=64)
    assert out == [t.upper() for t in texts]
    assert len(model.calls) == 1  # one model call for the whole request


def test_empty_lines_short_circuit():
    engine, model = make_engine()
    out = engine.translate_batch(["", "   ", "\t\n", "bonjour"], batch_size=64)
    assert out == ["", "", "", "BONJOUR"]
    assert model.calls == [["bonjour"]]


def test_duplicates_translated_once():
    engine, model = make_engine()
    texts = ["titre courant", "ligne a", "titre courant", "ligne b", "titre  courant"]
    out = engine.translate_batch(texts, batch_size=64)
    assert out == ["TITRE COURANT", "LIGNE A", "TITRE COURANT", "LIGNE B", "TITRE COURANT"]
    assert model.calls == [["titre courant", "ligne a", "ligne b"]]


def test_batch_size_one_is_line_by_line_reference_path():
    engine, model = make_engine()
    texts = ["un", "", "un", words(250)]  # duplicates and long lines untouched
    out = engine.translate_batch(texts, batch_size=1)
    assert out == ["UN", "", "UN", words(250).upper()]
    assert model.single_calls == ["un", "un", words(250)]  # one str call per line
    assert model.calls == [["un"], ["un"], [words(250)]]
    assert model.budgets[-1] >= 251  # budget raised to the line length


def test_whitespace_normalisation_matches_reference():
    engine, _ = make_engine()
    assert engine.translate_batch(["  a   b  "], batch_size=64) == ["A B"]
    assert engine.translate_batch(["  a   b  "], batch_size=1) == ["A B"]


# ── long lines ──────────────────────────────────────────────────────


def test_split_threshold_is_exclusive():
    engine, model = make_engine(split_over_tokens=200)
    exactly = words(199)  # 200 tokens with EOS → not split
    over = words(200)  # 201 tokens → split
    engine.translate_batch([exactly], batch_size=64)
    assert model.calls[-1] == [exactly]
    engine.translate_batch([over], batch_size=64)
    assert len(model.calls[-1]) > 1


def test_split_at_punctuation_and_rejoin():
    engine, model = make_engine(split_over_tokens=200, split_segment_tokens=40)
    # 25 clauses of 10 words, comma separated: 250 words
    clauses = [" ".join(f"c{i}w{j}" for j in range(10)) + "," for i in range(25)]
    line = " ".join(clauses)
    out = engine.translate_batch([line], batch_size=64)
    segments = model.calls[-1]
    assert len(segments) > 1
    assert all(engine.count_tokens(s) <= 40 for s in segments)
    # segments end on the punctuation, nothing lost, nothing duplicated
    assert all(s.endswith(",") for s in segments)
    assert " ".join(segments) == line
    assert out == [" ".join(s.upper() for s in segments)]


def test_split_without_punctuation_falls_back_to_words():
    engine, model = make_engine(split_over_tokens=200, split_segment_tokens=40)
    line = words(250)
    out = engine.translate_batch([line], batch_size=64)
    segments = model.calls[-1]
    assert len(segments) == 7  # 39 words per segment (+EOS = 40 tokens)
    assert all(engine.count_tokens(s) <= 40 for s in segments)
    assert " ".join(segments) == line
    assert out == [line.upper()]


def test_split_single_giant_word_by_characters():
    engine, model = make_engine(split_over_tokens=5, split_segment_tokens=4)
    line = "x" * 20  # one word, no spaces: 2 tokens for the fake, but force it
    # make the fake count characters for this test
    model.encode = lambda text: list(text) + ["</s>"]
    out = engine.translate_batch([line], batch_size=64)
    segments = model.calls[-1]
    assert all(len(s) <= 3 for s in segments)
    assert sorted(segments) == ["xx", "xxx"]  # identical slices de-duplicated
    assert out[0].replace(" ", "") == line.upper()


def test_split_disabled_with_zero_threshold():
    engine, model = make_engine(split_over_tokens=0)
    line = words(500)
    engine.translate_batch([line], batch_size=64)
    assert model.calls[-1] == [line]


# ── token budget ────────────────────────────────────────────────────


def test_fixed_budget_never_below_longest_line():
    budget = TokenBudget(InferenceSettings(max_tokens=200), "cuda")
    assert budget.compute(longest_line=50) == 200
    assert budget.compute(longest_line=300) == 300


def test_cpu_auto_budget_is_fixed():
    budget = TokenBudget(InferenceSettings(max_tokens=None, cpu_max_tokens=3000), "cpu")
    assert budget.compute(longest_line=10) == 3000
    assert budget.mode == "auto"


def test_budget_floor_when_memory_is_tight():
    budget = TokenBudget(
        InferenceSettings(floor=1024, ceil=16000, bytes_per_token=MiB, memory_fraction=0.6),
        "cuda",
        mem_info=lambda: (100 * MiB, 8 * GiB),
    )
    assert budget.compute(longest_line=10) == 1024


@pytest.mark.parametrize(
    "free_bytes, expected",
    [
        (2 * GiB, 1228),  # 2 GiB × 0.6 / 1 MiB
        (8 * GiB, 4915),
        (16 * GiB, 9830),
        (64 * GiB, 16000),  # ceiling
    ],
)
def test_budget_scales_with_free_memory(free_bytes, expected):
    budget = TokenBudget(
        InferenceSettings(
            floor=1024, ceil=16000, bytes_per_token=MiB, memory_fraction=0.6, beam=5
        ),
        "cuda",
        mem_info=lambda: (free_bytes, 80 * GiB),
    )
    assert budget.compute(longest_line=10) == expected


def test_budget_scales_with_beam_and_fp16():
    mem = lambda: (8 * GiB, 8 * GiB)  # noqa: E731
    b5 = TokenBudget(InferenceSettings(beam=5, bytes_per_token=MiB), "cuda", mem).compute()
    b1 = TokenBudget(InferenceSettings(beam=1, bytes_per_token=MiB), "cuda", mem).compute()
    b5_fp16 = TokenBudget(
        InferenceSettings(beam=5, fp16=True, bytes_per_token=MiB), "cuda", mem
    ).compute()
    assert b1 > b5
    assert b1 == 3 * b5  # (1+5)/(1+1)
    assert b5_fp16 == 2 * b5


def test_budget_reports_mode_and_last():
    budget = TokenBudget(InferenceSettings(max_tokens=None), "cpu")
    assert budget.last is None
    budget.compute(longest_line=1)
    assert budget.last == budget.settings.cpu_max_tokens


# ── OOM fallback ────────────────────────────────────────────────────


def test_oom_halves_budget_and_remembers_cap():
    model = FakeModel(oom_above=1000)
    engine, _ = make_engine(model=model, max_tokens=4000)
    out = engine.translate_batch(["a b", "c"], batch_size=64)
    assert out == ["A B", "C"]
    assert model.budgets == [4000, 2000, 1000]
    assert engine.budget.oom_cap == 1000
    # next call starts from the remembered cap
    engine.translate_batch(["d"], batch_size=64)
    assert model.budgets[-1] == 1000


def test_oom_never_goes_below_longest_line():
    model = FakeModel(oom_above=10)  # always OOM at any usable budget
    engine, _ = make_engine(model=model, max_tokens=100)
    with pytest.raises(RuntimeError, match="out of memory"):
        engine.translate_batch([words(60)], batch_size=64)  # 61 tokens
    assert model.budgets[-1] == 61  # tried down to the line length, then gave up


def test_non_oom_error_is_not_swallowed():
    model = FakeModel()
    engine, _ = make_engine(model=model, max_tokens=4000)
    with pytest.raises(ValueError, match="bad line"):
        engine.translate_batch(["ok", "BOOM"], batch_size=64)
    assert len(model.calls) == 1  # no retry
    assert engine.budget.oom_cap is None


def test_is_oom_error_recognises_torch_exception():
    import torch

    exc_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if exc_type is None:
        pytest.skip("torch build without OutOfMemoryError")
    assert is_oom_error(exc_type("CUDA out of memory"))
    assert is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate"))
    assert not is_oom_error(RuntimeError("shape mismatch"))


# ── request coalescing ──────────────────────────────────────────────


def run_concurrently(engine, requests, batch_size=64):
    results = {}
    errors = {}

    def worker(name, texts):
        try:
            results[name] = engine.translate_batch(texts, batch_size=batch_size)
        except Exception as exc:  # noqa: BLE001
            errors[name] = exc

    threads = [threading.Thread(target=worker, args=(n, t)) for n, t in requests]
    return threads, results, errors


def test_concurrent_requests_are_merged_into_one_model_call():
    gate = threading.Event()
    model = FakeModel(gate=gate)
    engine, _ = make_engine(model=model, coalesce=True)

    threads, results, errors = run_concurrently(
        engine, [("A", ["a1", "a2"]), ("B", ["b1", "b2"]), ("C", ["c1"])]
    )
    threads[0].start()
    time.sleep(0.2)  # A is inside the (blocked) model call
    threads[1].start()
    threads[2].start()
    time.sleep(0.2)  # B and C are queued
    gate.set()
    for t in threads:
        t.join(5)

    assert not errors
    assert results == {"A": ["A1", "A2"], "B": ["B1", "B2"], "C": ["C1"]}
    assert model.calls[0] == ["a1", "a2"]
    assert sorted(model.calls[1]) == ["b1", "b2", "c1"]  # one merged call
    assert len(model.calls) == 2


def test_faulty_request_is_isolated_from_the_others():
    gate = threading.Event()
    model = FakeModel(gate=gate)
    engine, _ = make_engine(model=model, coalesce=True)

    threads, results, errors = run_concurrently(
        engine, [("A", ["a1"]), ("B", ["b1", "BOOM"]), ("C", ["c1", "c2"])]
    )
    threads[0].start()
    time.sleep(0.2)
    threads[1].start()
    threads[2].start()
    time.sleep(0.2)
    gate.set()
    for t in threads:
        t.join(5)

    assert results["A"] == ["A1"]
    assert results["C"] == ["C1", "C2"]
    assert isinstance(errors["B"], ValueError)
    # merged call failed, then each request replayed on its own
    assert len(model.calls) == 4
    assert sorted(model.calls[1]) == ["BOOM", "b1", "c1", "c2"]
    assert sorted(map(tuple, model.calls[2:])) == [("b1", "BOOM"), ("c1", "c2")]


def test_coalescing_disabled_calls_model_directly():
    engine, model = make_engine(coalesce=False)
    assert engine.translate_batch(["x"], batch_size=64) == ["X"]
    assert engine._coalescer._thread is None


def test_coalesced_error_on_single_request_propagates():
    engine, model = make_engine(coalesce=True)
    with pytest.raises(ValueError, match="bad line"):
        engine.translate_batch(["BOOM"], batch_size=64)
    assert engine.translate_batch(["fine"], batch_size=64) == ["FINE"]  # worker alive


# ── info ────────────────────────────────────────────────────────────


def test_info_exposes_budget_and_options():
    engine, _ = make_engine(max_tokens=None, cpu_max_tokens=1234, fp16=False, beam=5)
    engine.translate_batch(["a"], batch_size=64)
    info = engine.info()
    assert info["token_budget"] == 1234
    assert info["token_budget_mode"] == "auto"
    assert info["fp16"] is False
    assert info["beam_size"] == 5
