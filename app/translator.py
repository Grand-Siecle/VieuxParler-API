"""Inference engine around the fairseq LSTM hub interface.

Three layers, from the outside in:

* :class:`FairseqTranslator` — the process-wide singleton the routes use. It
  loads the model and delegates to an engine.
* :class:`BatchTranslator` — the engine: normalisation, empty lines, long-line
  splitting, de-duplication, request coalescing. It only needs an object with
  the fairseq hub interface (``translate(list, beam=)``, ``encode(str)``,
  ``cfg.dataset.max_tokens``), so tests inject a fake model.
* :class:`TokenBudget` — how many source tokens one model batch may hold.

Vocabulary used below:

* *token* = one SentencePiece piece (plus the end-of-sentence marker fairseq
  appends), i.e. what the model actually sees.
* *token budget* = fairseq ``cfg.dataset.max_tokens``: the total number of
  source tokens packed into one batch. It is not a maximum line length, but
  fairseq asserts that no single line exceeds it.
* *reference path* = one ``model.translate(line)`` call per line, the
  behaviour of the original API; selected with ``batch_size=1``.
"""

import queue
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from . import config
from .metrics import (
    COALESCED_BATCH_REQUESTS,
    LINES_SPLIT,
    MODEL_LOAD_TIME,
    OOM_FALLBACKS,
    REQUESTS_COALESCED,
    TOKEN_BUDGET,
    logger,
)

_WS_RE = re.compile(r"\s+")
_PUNCT_SPLIT_RE = re.compile(r"(?<=[;:,.!?])\s+")


# ── Settings ────────────────────────────────────────────────────────


@dataclass
class InferenceSettings:
    beam: int = 5
    fp16: bool = False
    max_tokens: Optional[int] = None  # None = auto
    cpu_max_tokens: int = 4000
    floor: int = 1024
    ceil: int = 16000
    memory_fraction: float = 0.6
    bytes_per_token: int = 1024 * 1024
    reference_beam: int = 5
    split_over_tokens: int = 200
    split_segment_tokens: int = 40
    coalesce: bool = True

    @classmethod
    def from_config(cls) -> "InferenceSettings":
        return cls(
            beam=config.BEAM_SIZE,
            fp16=config.FP16,
            max_tokens=config.MAX_TOKENS,
            cpu_max_tokens=config.CPU_MAX_TOKENS,
            floor=config.MAX_TOKENS_FLOOR,
            ceil=config.MAX_TOKENS_CEIL,
            memory_fraction=config.GPU_MEMORY_FRACTION,
            bytes_per_token=config.BYTES_PER_TOKEN,
            reference_beam=config.REFERENCE_BEAM,
            split_over_tokens=config.SPLIT_OVER_TOKENS,
            split_segment_tokens=config.SPLIT_SEGMENT_TOKENS,
            coalesce=config.COALESCE_REQUESTS,
        )


# ── Token budget ────────────────────────────────────────────────────


def _cuda_mem_info() -> Tuple[int, int]:
    """(free bytes usable by a new batch, total bytes) on the current GPU.

    ``mem_get_info`` reports what the driver has left; PyTorch's caching
    allocator may additionally hold reserved-but-unallocated blocks that a
    new batch can reuse without asking the driver, so they are counted as
    free too.
    """
    free, total = torch.cuda.mem_get_info()
    cached = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    return free + max(cached, 0), total


class TokenBudget:
    """Computes the fairseq token budget for the next model call.

    * Fixed value when ``settings.max_tokens`` is an int (``MAX_TOKENS=200``).
    * On CPU with ``auto``: ``settings.cpu_max_tokens``.
    * On GPU with ``auto``: ``free × fraction / bytes_per_token``, where
      ``bytes_per_token`` is scaled by the beam width (encoder cost is
      beam-independent, decoder cost grows with the beam, hence the
      ``(1 + beam) / (1 + reference_beam)`` factor) and halved for fp16;
      clamped to ``[floor, ceil]``.
    * After an out-of-memory error the halved budget is remembered as a cap
      that stays for the life of the process.
    * Never below the longest line of the call, or fairseq asserts.
    """

    def __init__(
        self,
        settings: InferenceSettings,
        device: str,
        mem_info: Optional[Callable[[], Tuple[int, int]]] = None,
    ):
        self.settings = settings
        self.device = device
        self._mem_info = mem_info or _cuda_mem_info
        self._oom_cap: Optional[int] = None
        self.last: Optional[int] = None

    @property
    def mode(self) -> str:
        return "fixed" if self.settings.max_tokens is not None else "auto"

    @property
    def oom_cap(self) -> Optional[int]:
        return self._oom_cap

    def scale(self) -> float:
        s = self.settings
        beam_scale = (1 + max(s.beam, 1)) / (1 + max(s.reference_beam, 1))
        return beam_scale * (0.5 if s.fp16 else 1.0)

    def base(self) -> int:
        """Budget before the OOM cap and the longest-line floor."""
        s = self.settings
        if s.max_tokens is not None:
            return s.max_tokens
        if self.device != "cuda":
            return s.cpu_max_tokens
        free, _total = self._mem_info()
        per_token = max(s.bytes_per_token * self.scale(), 1.0)
        budget = int(free * s.memory_fraction / per_token)
        return max(s.floor, min(s.ceil, budget))

    def compute(self, longest_line: int = 0) -> int:
        budget = self.base()
        if self._oom_cap is not None:
            budget = min(budget, self._oom_cap)
        budget = max(budget, longest_line)
        self.last = budget
        TOKEN_BUDGET.set(budget)
        return budget

    def halve(self, current: int, longest_line: int) -> Optional[int]:
        """Budget to retry with after an OOM, or None if it cannot shrink."""
        new = max(current // 2, longest_line)
        if new >= current:
            return None
        self._oom_cap = new if self._oom_cap is None else min(self._oom_cap, new)
        return new


def is_oom_error(exc: BaseException) -> bool:
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cublas_status_alloc_failed" in msg


# ── Request coalescing ──────────────────────────────────────────────


class _Job:
    __slots__ = ("texts", "event", "result", "error")

    def __init__(self, texts: List[str]):
        self.texts = texts
        self.event = threading.Event()
        self.result: Optional[List[str]] = None
        self.error: Optional[BaseException] = None


class RequestCoalescer:
    """Single worker thread that drains whatever is queued into one model call.

    Requests arriving while the model is busy pile up in the queue; when the
    worker frees up it takes them all, runs ``infer`` once on the
    concatenated lines and hands each request its slice back. If the merged
    call fails, every request is replayed on its own so that one faulty
    request cannot take the others down with it.
    """

    def __init__(self, infer: Callable[[List[str]], List[str]]):
        self._infer = infer
        self._queue: "queue.Queue[_Job]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._start_lock = threading.Lock()

    def _ensure_worker(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        with self._start_lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run, name="vieuxparler-coalescer", daemon=True
                )
                self._thread.start()

    def submit(self, texts: List[str]) -> List[str]:
        job = _Job(texts)
        self._ensure_worker()
        self._queue.put(job)
        job.event.wait()
        if job.error is not None:
            raise job.error
        return job.result  # type: ignore[return-value]

    def _drain(self) -> List[_Job]:
        jobs = [self._queue.get()]
        while True:
            try:
                jobs.append(self._queue.get_nowait())
            except queue.Empty:
                return jobs

    def _run(self) -> None:
        while True:
            jobs = self._drain()
            try:
                self._process(jobs)
            except BaseException:  # keep the worker alive whatever happens
                logger.exception("coalescer worker error")
                for job in jobs:
                    if not job.event.is_set():
                        job.error = RuntimeError("coalescer worker error")
                        job.event.set()

    def _process(self, jobs: List[_Job]) -> None:
        if len(jobs) > 1:
            REQUESTS_COALESCED.inc(len(jobs))
            COALESCED_BATCH_REQUESTS.set(len(jobs))
        merged: List[str] = []
        for job in jobs:
            merged.extend(job.texts)
        try:
            outputs = self._infer(merged)
            if len(outputs) != len(merged):
                raise RuntimeError(
                    f"model returned {len(outputs)} outputs for {len(merged)} inputs"
                )
        except BaseException as exc:
            if len(jobs) == 1:
                jobs[0].error = exc
                jobs[0].event.set()
                return
            logger.warning(
                "coalesced call failed, replaying requests separately",
                extra={"requests": len(jobs), "error": str(exc)},
            )
            for job in jobs:
                try:
                    job.result = self._infer(job.texts)
                except BaseException as job_exc:
                    job.error = job_exc
                job.event.set()
            return
        pos = 0
        for job in jobs:
            job.result = outputs[pos : pos + len(job.texts)]
            pos += len(job.texts)
            job.event.set()


# ── Engine ──────────────────────────────────────────────────────────


class BatchTranslator:
    """Batched inference on top of a fairseq hub interface."""

    def __init__(
        self,
        model,
        device: str,
        settings: Optional[InferenceSettings] = None,
        mem_info: Optional[Callable[[], Tuple[int, int]]] = None,
        lock: Optional[threading.RLock] = None,
    ):
        self.model = model
        self.device = device
        self.settings = settings or InferenceSettings.from_config()
        self.budget = TokenBudget(self.settings, device, mem_info)
        self._lock = lock or threading.RLock()
        self._coalescer = RequestCoalescer(self._infer_unique)

    # ── helpers ─────────────────────────────────────────────────

    @staticmethod
    def normalize(text: str) -> str:
        return _WS_RE.sub(" ", text.strip())

    def count_tokens(self, text: str) -> int:
        """Number of source tokens fairseq will see (pieces + EOS)."""
        return len(self.model.encode(text))

    def _set_budget(self, budget: int) -> None:
        self.model.cfg.dataset.max_tokens = budget

    def _oom_cleanup(self) -> None:
        if self.device == "cuda":
            torch.cuda.empty_cache()

    # ── long lines ──────────────────────────────────────────────

    def split_long(self, text: str) -> List[str]:
        """Cut a long line into segments of about ``split_segment_tokens``.

        Split points are taken after punctuation (``; : , . ! ?``), falling
        back to word boundaries when a punctuation chunk is itself too long,
        and, for a single over-long word, to fixed-size character slices.
        """
        limit = self.settings.split_segment_tokens
        segments: List[str] = []
        current = ""
        for chunk in _PUNCT_SPLIT_RE.split(text):
            if not chunk:
                continue
            if self.count_tokens(chunk) > limit:
                if current:
                    segments.append(current)
                    current = ""
                segments.extend(self._split_words(chunk, limit))
                continue
            candidate = f"{current} {chunk}" if current else chunk
            if current and self.count_tokens(candidate) > limit:
                segments.append(current)
                current = chunk
            else:
                current = candidate
        if current:
            segments.append(current)
        return segments or [text]

    def _split_words(self, chunk: str, limit: int) -> List[str]:
        segments: List[str] = []
        current = ""
        for word in chunk.split(" "):
            if not word:
                continue
            if self.count_tokens(word) > limit:
                if current:
                    segments.append(current)
                    current = ""
                # One "word" longer than a segment: slice it by characters.
                # Each SentencePiece piece covers at least one character, so a
                # slice of `limit - 1` characters can never exceed the budget.
                step = max(limit - 1, 1)
                segments.extend(word[i : i + step] for i in range(0, len(word), step))
                continue
            candidate = f"{current} {word}" if current else word
            if current and self.count_tokens(candidate) > limit:
                segments.append(current)
                current = word
            else:
                current = candidate
        if current:
            segments.append(current)
        return segments

    # ── model calls ─────────────────────────────────────────────

    def _infer(self, texts: List[str]) -> List[str]:
        """One ``model.translate(list)`` call under the lock, with OOM fallback."""
        if not texts:
            return []
        with self._lock:
            longest = max(self.count_tokens(t) for t in texts)
            budget = self.budget.compute(longest_line=longest)
            while True:
                self._set_budget(budget)
                try:
                    outputs = self.model.translate(texts, beam=self.settings.beam)
                    break
                except Exception as exc:
                    if not is_oom_error(exc):
                        raise
                    self._oom_cleanup()
                    new_budget = self.budget.halve(budget, longest)
                    if new_budget is None:
                        raise
                    OOM_FALLBACKS.inc()
                    logger.warning(
                        "out of memory, retrying with a smaller token budget",
                        extra={"budget": budget, "new_budget": new_budget},
                    )
                    budget = new_budget
                    TOKEN_BUDGET.set(budget)
            if isinstance(outputs, str):
                outputs = [outputs]
            return list(outputs)

    def _infer_unique(self, texts: List[str]) -> List[str]:
        """Translate each distinct line once and map the results back."""
        unique = list(OrderedDict.fromkeys(texts))
        outputs = self._infer(unique)
        if len(outputs) != len(unique):
            raise RuntimeError(
                f"model returned {len(outputs)} outputs for {len(unique)} inputs"
            )
        table = dict(zip(unique, outputs))
        return [table[t] for t in texts]

    def _run(self, texts: List[str]) -> List[str]:
        if not texts:
            return []
        if self.settings.coalesce:
            return self._coalescer.submit(texts)
        return self._infer_unique(texts)

    def translate_one(self, text: str) -> str:
        """Reference path: exactly one model call for this line."""
        text = self.normalize(text)
        if not text:
            return ""
        with self._lock:
            budget = self.budget.compute(longest_line=self.count_tokens(text))
            self._set_budget(budget)
            return self.model.translate(text, beam=self.settings.beam)

    # ── public API ──────────────────────────────────────────────

    def translate_batch(self, texts: Sequence[str], batch_size: int = 32) -> List[str]:
        """Translate ``texts``; the result has the same length and order.

        ``batch_size`` is kept for API compatibility. ``1`` selects the
        reference path (one model call per line, no splitting, no
        de-duplication, no coalescing); any other value takes the batched
        path, where fairseq sizes the actual batches from the token budget.
        """
        normalized = [self.normalize(t) for t in texts]
        if batch_size == 1:
            return [self.translate_one(t) if t else "" for t in normalized]

        # plan: index → list of segment positions in `to_translate`
        to_translate: List[str] = []
        plan: List[Optional[List[int]]] = []
        threshold = self.settings.split_over_tokens
        for text in normalized:
            if not text:
                plan.append(None)
                continue
            if threshold > 0 and self.count_tokens(text) > threshold:
                segments = self.split_long(text)
                LINES_SPLIT.inc()
            else:
                segments = [text]
            positions = []
            for seg in segments:
                positions.append(len(to_translate))
                to_translate.append(seg)
            plan.append(positions)

        outputs = self._run(to_translate)
        if len(outputs) != len(to_translate):
            raise RuntimeError(
                f"model returned {len(outputs)} outputs for {len(to_translate)} inputs"
            )

        results: List[str] = []
        for positions in plan:
            if positions is None:
                results.append("")
            elif len(positions) == 1:
                results.append(outputs[positions[0]])
            else:
                results.append(" ".join(outputs[p] for p in positions).strip())
        return results

    def info(self) -> Dict:
        return {
            "token_budget": self.budget.last,
            "token_budget_mode": self.budget.mode,
            "oom_cap": self.budget.oom_cap,
            "fp16": self.settings.fp16,
            "beam_size": self.settings.beam,
            "coalesce_requests": self.settings.coalesce,
            "split_over_tokens": self.settings.split_over_tokens,
        }


# ── Singleton used by the routes ────────────────────────────────────


class FairseqTranslator:
    """Thread-safe singleton wrapper around the Fairseq LSTM model."""

    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
                    cls._instance._engine = None
        return cls._instance

    def load(self) -> None:
        if self._initialized:
            return

        start = time.perf_counter()
        model_dir = str(config.MODEL_DIR)
        settings = InferenceSettings.from_config()

        logger.info(
            "loading model",
            extra={
                "model_dir": model_dir,
                "checkpoint": config.CHECKPOINT_FILE,
                "data_dir": config.DATA_DIR,
                "bpe_model": config.BPE_MODEL,
            },
        )

        device = self._resolve_device()
        if device == "cpu":
            logger.warning("running on CPU (slower inference)")

        from fairseq.models.lstm import LSTMModel

        # The budget is rewritten before every call; this is just a sane
        # starting value.
        initial_budget = settings.max_tokens or (
            settings.cpu_max_tokens if device == "cpu" else settings.floor
        )
        self._model = LSTMModel.from_pretrained(
            model_name_or_path=model_dir,
            checkpoint_file=config.CHECKPOINT_FILE,
            data_name_or_path=str(config.MODEL_DIR / config.DATA_DIR),
            bpe="sentencepiece",
            sentencepiece_model=str(config.MODEL_DIR / config.BPE_MODEL),
            source_lang=config.SOURCE_LANG,
            target_lang=config.TARGET_LANG,
            beam=settings.beam,
            max_tokens=initial_budget,
        )

        if device == "cuda":
            self._model.cuda()
            logger.info("model moved to GPU (cuda:0)")
            if settings.fp16:
                self._model.half()
                logger.info("model converted to fp16")
        elif settings.fp16:
            logger.warning("FP16=true ignored on CPU")
            settings.fp16 = False

        self._model.eval()
        self._device = device
        self._engine = BatchTranslator(
            self._model, device, settings, lock=self._lock
        )
        self._initialized = True

        elapsed = time.perf_counter() - start
        MODEL_LOAD_TIME.set(elapsed)
        logger.info(
            "model loaded",
            extra={
                "load_time_s": round(elapsed, 2),
                "device": device,
                "beam": settings.beam,
                "fp16": settings.fp16,
                "max_tokens": "auto" if settings.max_tokens is None else settings.max_tokens,
                "coalesce_requests": settings.coalesce,
            },
        )

        for s in ["Que voulez-vous donc?", "Il estoit une fois un homme.", "Je ne sçay pas."]:
            self._engine.translate_one(s)
        logger.info("warm-up complete", extra={"token_budget": self._engine.budget.last})

    @staticmethod
    def _resolve_device() -> str:
        requested = config.DEVICE.lower()
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("DEVICE=cuda but CUDA is not available")
            return "cuda"
        if requested == "cpu":
            return "cpu"
        # auto
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def is_loaded(self) -> bool:
        return self._initialized

    @property
    def device(self) -> str:
        return getattr(self, "_device", "unknown")

    @property
    def engine(self) -> Optional[BatchTranslator]:
        return self._engine

    @property
    def device_info(self) -> dict:
        cuda_available = torch.cuda.is_available()
        info = {
            "device": self.device,
            "cuda_available": cuda_available,
            "device_config": config.DEVICE,
        }
        if cuda_available:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_mb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024)
        return info

    @property
    def inference_info(self) -> Optional[dict]:
        if self._engine is None:
            return None
        return self._engine.info()

    def translate(self, text: str) -> str:
        return self._engine.translate_one(text)

    def translate_batch(self, texts: List[str], batch_size: int = 32) -> List[str]:
        return self._engine.translate_batch(texts, batch_size)


translator = FairseqTranslator()
