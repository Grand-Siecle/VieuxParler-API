import os
from pathlib import Path

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

# ── Inference settings ──────────────────────────────────────────────
BEAM_SIZE: int = int(os.getenv("BEAM_SIZE", "5"))
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "200"))
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
APP_VERSION: str = "1.0.0"
MODEL_VERSION: str = "freem-lstm-v1"
