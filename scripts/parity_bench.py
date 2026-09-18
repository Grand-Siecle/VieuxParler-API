#!/usr/bin/env python3
"""Parity and throughput benchmark: reference path vs batched path.

Loads the model exactly like the API does, samples the FreEMnorm test set
shipped in the model zip (or reads ``--file``), translates the sample once
line by line (``batch_size=1``, the original API behaviour) and once through
the batched path, then reports:

* how many outputs are identical, with examples of the differences (and,
  with ``--explain-diffs``, whether they come from beam-score ties);
* throughput of both paths (lines/s, tokens/s) and the speed-up;
* the token budget used, GPU peak memory and the measured bytes per token
  (with a probe batch pinned at the budget) to calibrate ``BYTES_PER_TOKEN``;
* chrF per length class against the reference translations.

Run inside the image::

    python scripts/parity_bench.py --n 400
    FP16=true python scripts/parity_bench.py --n 400
    BEAM_SIZE=1 python scripts/parity_bench.py --n 400
"""

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from app import config  # noqa: E402
from app.translator import translator  # noqa: E402

DEFAULT_CLASSES = "10,30,60,200"


# ── data ────────────────────────────────────────────────────────────


def load_test_set(model):
    """Decode the binarised test set: list of (source, reference)."""
    from fairseq.data import Dictionary, data_utils

    data = Path(config.MODEL_DIR) / config.DATA_DIR
    src_dict = Dictionary.load(str(data / f"dict.{config.SOURCE_LANG}.txt"))
    tgt_dict = Dictionary.load(str(data / f"dict.{config.TARGET_LANG}.txt"))
    prefix = str(data / f"test.{config.SOURCE_LANG}-{config.TARGET_LANG}")
    src_ds = data_utils.load_indexed_dataset(f"{prefix}.{config.SOURCE_LANG}", src_dict)
    tgt_ds = data_utils.load_indexed_dataset(f"{prefix}.{config.TARGET_LANG}", tgt_dict)
    if src_ds is None or tgt_ds is None:
        raise SystemExit(f"test set not found under {data}")
    pairs = []
    for i in range(len(src_ds)):
        src = model.bpe.decode(src_dict.string(src_ds[i]))
        tgt = model.bpe.decode(tgt_dict.string(tgt_ds[i]))
        pairs.append((src, tgt))
    return pairs


def length_class(n_tokens, bounds):
    for b in bounds:
        if n_tokens <= b:
            return f"<={b}"
    return f">{bounds[-1]}"


def sample_by_class(pairs, engine, n, bounds, seed):
    rng = random.Random(seed)
    by_class = {}
    for src, tgt in pairs:
        cls = length_class(engine.count_tokens(src), bounds)
        by_class.setdefault(cls, []).append((src, tgt))
    per_class = max(1, n // len(by_class))
    sample = []
    for cls in sorted(by_class, key=lambda c: (c.startswith(">"), int(c.lstrip("<=>")))):
        items = by_class[cls]
        rng.shuffle(items)
        sample.extend(items[:per_class])
    return sample, {c: len(v) for c, v in by_class.items()}


# ── measurement ─────────────────────────────────────────────────────


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    cuda_sync()
    t0 = time.perf_counter()
    out = fn()
    cuda_sync()
    return out, time.perf_counter() - t0


def peak_tracker():
    if not torch.cuda.is_available():
        return lambda: None
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    return lambda: torch.cuda.max_memory_allocated() - base


def probe_bytes_per_token(engine, pairs, budget, target_len):
    """Fill one batch of near-equal-length lines up to ``budget`` tokens.

    fairseq counts a batch as ``n_sentences × longest_line`` (padded tokens),
    so lines of nearly the same length give a batch whose token count is
    known: peak memory / that count = bytes per (padded) token.
    """
    if not torch.cuda.is_available():
        return None
    candidates = [s for s, _ in pairs if abs(engine.count_tokens(s) - target_len) <= 2]
    if not candidates:
        return None
    longest = max(engine.count_tokens(s) for s in candidates)
    n = max(1, budget // longest)
    lines = list(dict.fromkeys(candidates))[:n]
    if len(lines) < n:
        # not enough distinct lines: repeat with a marker so de-duplication keeps them
        lines = [f"{lines[i % len(lines)]} {i}" for i in range(n)]
        longest = max(engine.count_tokens(s) for s in lines)
    tokens = len(lines) * longest
    saved = engine.settings.max_tokens
    engine.settings.max_tokens = tokens  # pin the budget so it is one batch
    try:
        torch.cuda.empty_cache()
        peak = peak_tracker()
        engine.translate_batch(lines, batch_size=64)
        used = peak()
    finally:
        engine.settings.max_tokens = saved
    return {
        "lines": len(lines),
        "line_tokens": longest,
        "batch_tokens": tokens,
        "peak_bytes": used,
        "bytes_per_token": used / tokens,
    }


def explain_diff(engine, src, ref_out, batch_out, beam):
    """Score gap between the two candidates in the reference n-best list."""
    model = engine.model
    with engine._lock:
        engine._set_budget(engine.budget.compute(engine.count_tokens(src)))
        hypos = model.generate([model.encode(src)], beam=beam)[0]
    scored = {model.decode(h["tokens"]): float(h["score"]) for h in hypos}
    s_ref = scored.get(ref_out)
    s_bat = scored.get(batch_out)
    top2 = sorted(scored.values(), reverse=True)[:2]
    return {
        "ref_score": s_ref,
        "batched_score": s_bat,
        "batched_in_ref_nbest": s_bat is not None,
        "top2_gap": (top2[0] - top2[1]) if len(top2) == 2 else None,
    }


def chrf(hyps, refs):
    try:
        from sacrebleu.metrics import CHRF
    except ImportError:
        return None
    if not hyps:
        return None
    return CHRF().corpus_score(hyps, [refs]).score


# ── main ────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=400, help="sample size (spread over length classes)")
    ap.add_argument("--file", help="one source line per file instead of the test set (no chrF)")
    ap.add_argument("--classes", default=DEFAULT_CLASSES, help="length class upper bounds in tokens")
    ap.add_argument("--batch-size", type=int, default=64, help="batch_size sent on the batched path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=10, help="differences to print")
    ap.add_argument("--explain-diffs", action="store_true", help="score differing outputs against the n-best list")
    ap.add_argument("--probe-len", type=int, default=0, help="line length (tokens) for the bytes/token probe; 0 = median")
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--json", help="write the full report to this file")
    args = ap.parse_args()

    bounds = [int(b) for b in args.classes.split(",")]

    t0 = time.perf_counter()
    translator.load()
    engine = translator.engine
    print(f"model loaded in {time.perf_counter() - t0:.1f}s on {translator.device}"
          f" | beam={engine.settings.beam} fp16={engine.settings.fp16}"
          f" | MAX_TOKENS={'auto' if engine.settings.max_tokens is None else engine.settings.max_tokens}"
          f" | split_over={engine.settings.split_over_tokens} coalesce={engine.settings.coalesce}")

    if args.file:
        lines = [l.rstrip("\n") for l in open(args.file, encoding="utf-8") if l.strip()]
        pairs = [(l, None) for l in lines]
        sample = pairs
        population = {}
    else:
        pairs = load_test_set(engine.model)
        sample, population = sample_by_class(pairs, engine, args.n, bounds, args.seed)
        print(f"test set: {len(pairs)} sentences; population per class: {population}")

    sources = [s for s, _ in sample]
    refs = [t for _, t in sample]
    lengths = [engine.count_tokens(s) for s in sources]
    classes = [length_class(n, bounds) for n in lengths]
    total_tokens = sum(lengths)
    print(f"sample: {len(sources)} lines, {total_tokens} tokens, "
          f"per class: { {c: classes.count(c) for c in dict.fromkeys(classes)} }")

    # reference path
    peak = peak_tracker()
    ref_out, ref_time = timed(lambda: engine.translate_batch(sources, batch_size=1))
    ref_peak = peak()

    # batched path
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    peak = peak_tracker()
    bat_out, bat_time = timed(lambda: engine.translate_batch(sources, batch_size=args.batch_size))
    bat_peak = peak()
    budget = engine.budget.last

    identical = sum(1 for a, b in zip(ref_out, bat_out) if a == b)
    diffs = [i for i, (a, b) in enumerate(zip(ref_out, bat_out)) if a != b]

    threshold = engine.settings.split_over_tokens
    split_idx = {i for i, n in enumerate(lengths) if threshold > 0 and n > threshold}
    unsplit = [i for i in range(len(sources)) if i not in split_idx]
    same_unsplit = sum(1 for i in unsplit if ref_out[i] == bat_out[i])
    print("\n== Parity (reference vs batched) ==")
    print(f"identical: {identical}/{len(sources)}  different: {len(diffs)}")
    print(f"  lines <= {threshold} tokens (not split): {same_unsplit}/{len(unsplit)} identical")
    print(f"  lines >  {threshold} tokens (split, expected to differ from the truncated reference): "
          f"{len(split_idx)}, of which {sum(1 for i in split_idx if ref_out[i] != bat_out[i])} differ")
    diff_records = []
    for i in diffs[: args.show]:
        rec = {"class": classes[i], "tokens": lengths[i], "source": sources[i],
               "reference_path": ref_out[i], "batched_path": bat_out[i], "gold": refs[i]}
        if args.explain_diffs:
            rec["explain"] = explain_diff(engine, sources[i], ref_out[i], bat_out[i], engine.settings.beam)
        diff_records.append(rec)
        print(f"- [{classes[i]} / {lengths[i]} tok] {sources[i]}")
        print(f"    ref : {ref_out[i]}")
        print(f"    bat : {bat_out[i]}")
        if refs[i] is not None:
            print(f"    gold: {refs[i]}")
        if "explain" in rec:
            print(f"    scores: {rec['explain']}")

    print("\n== Throughput ==")
    print(f"reference: {ref_time:8.1f}s  {len(sources)/ref_time:7.1f} lines/s  {total_tokens/ref_time:8.1f} tok/s")
    print(f"batched  : {bat_time:8.1f}s  {len(sources)/bat_time:7.1f} lines/s  {total_tokens/bat_time:8.1f} tok/s")
    print(f"speed-up : x{ref_time / bat_time:.1f}")

    print("\n== Budget / memory ==")
    print(f"token budget used: {budget} ({engine.budget.mode}), oom cap: {engine.budget.oom_cap}")
    probe = None
    if torch.cuda.is_available():
        print(f"GPU peak above baseline: reference {ref_peak/2**20:.0f} MiB, batched {bat_peak/2**20:.0f} MiB")
        largest_batch = min(budget, total_tokens)
        print(f"rough bytes/token (batched peak / min(budget, tokens)): {bat_peak/largest_batch/1024:.0f} KiB")
        if not args.no_probe and not args.file:
            target = args.probe_len or int(statistics.median(lengths))
            probe = probe_bytes_per_token(engine, pairs, budget, target)
            if probe:
                print(f"probe: {probe['lines']} lines x {probe['line_tokens']} tok = {probe['batch_tokens']} padded tokens,"
                      f" peak {probe['peak_bytes']/2**20:.0f} MiB -> {probe['bytes_per_token']/1024:.0f} KiB/token")
                print(f"suggested BYTES_PER_TOKEN (x2 margin): {int(probe['bytes_per_token'] * 2)}"
                      f"  (current default {config.BYTES_PER_TOKEN})")
    else:
        print("CPU: no memory measurement")

    per_class = {}
    if not args.file:
        print("\n== chrF per length class (vs gold) ==")
        print(f"{'class':>8} {'n':>5} {'reference':>10} {'batched':>10} {'identical':>10}")
        for cls in dict.fromkeys(classes):
            idx = [i for i, c in enumerate(classes) if c == cls]
            r = chrf([ref_out[i] for i in idx], [refs[i] for i in idx])
            b = chrf([bat_out[i] for i in idx], [refs[i] for i in idx])
            same = sum(1 for i in idx if ref_out[i] == bat_out[i])
            per_class[cls] = {"n": len(idx), "chrf_reference": r, "chrf_batched": b, "identical": same}
            fmt = lambda v: f"{v:10.2f}" if v is not None else f"{'n/a':>10}"  # noqa: E731
            print(f"{cls:>8} {len(idx):>5} {fmt(r)} {fmt(b)} {same:>7}/{len(idx)}")

    if args.json:
        report = {
            "device": translator.device,
            "settings": engine.info(),
            "n": len(sources),
            "tokens": total_tokens,
            "identical": identical,
            "different": len(diffs),
            "identical_unsplit": same_unsplit,
            "unsplit": len(unsplit),
            "split": len(split_idx),
            "differences": diff_records,
            "reference_seconds": ref_time,
            "batched_seconds": bat_time,
            "speedup": ref_time / bat_time,
            "token_budget": budget,
            "peak_bytes_reference": ref_peak,
            "peak_bytes_batched": bat_peak,
            "probe": probe,
            "per_class": per_class,
        }
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\nreport written to {args.json}")


if __name__ == "__main__":
    main()
