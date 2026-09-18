import os
from pathlib import Path
from typing import Optional

# ── Base paths ──────────────────────────────────────────────────────
# Project root = parent of the app/ package
BASE_DIR = Path(os.getenv("BASE_DIR", Path(__file__).resolve().parent.parent))
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "freem_lstm_fairseq"))

# ── Model files (relative to MODEL_DIR) ────────────────────────────
CHECKPOINT_FILE: str = os.getenv("CHECKPOINT_FILE", "model/checkpoint_best.pt")
DATA_DIR: str = os.getenv("DATA_DIR", "data_norm_bin_4000")
BPE_MODEL: str = os.getenv("BPE_MODEL", "bpe_joint_4000.model")
SOURCE_LANG: str = os.getenv("SOURCE_LANG", "src")
TARGET_LANG: str = os.getenv("TARGET_LANG", "trg")

# ── Device ──────────────────────────────────────────────────────────
# "auto" = GPU if available else CPU | "cpu" = force CPU | "cuda" = force GPU
DEVICE: str = os.getenv("DEVICE", "auto")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _parse_max_tokens(raw: str) -> Optional[int]:
    """`auto` (or empty) → None, meaning "size the batch from free memory"."""
    raw = raw.strip().lower()
    if raw in ("", "auto"):
        return None
    return int(raw)


# ── Inference settings ──────────────────────────────────────────────
# Beam width. 5 is the reference setting; 1 is ~4x faster on CPU but
# changes a few outputs — never change it without a parity measurement.
BEAM_SIZE: int = int(os.getenv("BEAM_SIZE", "5"))

# Half-precision inference (GPU only). Off by default: it can change outputs.
FP16: bool = _env_bool("FP16", False)

# Token budget per model batch (fairseq `cfg.dataset.max_tokens`). This is NOT
# a maximum line length: it is the total number of source tokens fairseq packs
# into one batch. `auto` (default) recomputes it from free GPU memory before
# each request; an integer pins it (the old behaviour, e.g. `200`).
MAX_TOKENS: Optional[int] = _parse_max_tokens(os.getenv("MAX_TOKENS", "auto"))
# Budget used on CPU when MAX_TOKENS=auto (memory is not the constraint there).
CPU_MAX_TOKENS: int = int(os.getenv("CPU_MAX_TOKENS", "4000"))
# Bounds of the automatic budget on GPU.
MAX_TOKENS_FLOOR: int = int(os.getenv("MAX_TOKENS_FLOOR", "1024"))
MAX_TOKENS_CEIL: int = int(os.getenv("MAX_TOKENS_CEIL", "16000"))
# Share of the free GPU memory the batch may use.
GPU_MEMORY_FRACTION: float = float(os.getenv("GPU_MEMORY_FRACTION", "0.6"))
# Peak GPU bytes needed per source token in a batch, measured at the
# reference setting (beam 5, fp32). Scaled by beam and fp16 at run time.
# Default is a deliberately conservative estimate: run
# `scripts/parity_bench.py` on your GPU and lower it if the measured value
# (with margin) allows.
BYTES_PER_TOKEN: int = int(os.getenv("BYTES_PER_TOKEN", str(1024 * 1024)))
REFERENCE_BEAM: int = 5

# Lines longer than this (source tokens, SentencePiece + EOS) are split at
# punctuation into ~SPLIT_SEGMENT_TOKENS segments, translated separately and
# re-joined with a space. 200 = the old hard limit: only lines that used to
# crash the whole batch are touched.
SPLIT_OVER_TOKENS: int = int(os.getenv("SPLIT_OVER_TOKENS", "200"))
SPLIT_SEGMENT_TOKENS: int = int(os.getenv("SPLIT_SEGMENT_TOKENS", "40"))

# Merge concurrent /translate/batch requests into one model call.
COALESCE_REQUESTS: bool = _env_bool("COALESCE_REQUESTS", True)

DEFAULT_BATCH_SIZE: int = int(os.getenv("DEFAULT_BATCH_SIZE", "32"))

# ── Server settings ─────────────────────────────────────────────────
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))
WORKERS: int = int(os.getenv("WORKERS", "1"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")

# ── Timeouts (seconds) ─────────────────────────────────────────────
TRANSLATE_TIMEOUT: int = int(os.getenv("TRANSLATE_TIMEOUT", "30"))
BATCH_TIMEOUT: int = int(os.getenv("BATCH_TIMEOUT", "300"))

# ── App metadata ────────────────────────────────────────────────────
APP_VERSION: str = "1.1.0"
MODEL_VERSION: str = "freem-lstm-v1"
