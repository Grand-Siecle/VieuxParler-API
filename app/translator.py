import re
import threading
import time
from typing import List

import torch

from . import config
from .metrics import logger, MODEL_LOAD_TIME


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
        return cls._instance

    def load(self) -> None:
        if self._initialized:
            return

        start = time.perf_counter()
        model_dir = str(config.MODEL_DIR)

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

        self._model = LSTMModel.from_pretrained(
            model_name_or_path=model_dir,
            checkpoint_file=config.CHECKPOINT_FILE,
            data_name_or_path=str(config.MODEL_DIR / config.DATA_DIR),
            bpe="sentencepiece",
            sentencepiece_model=str(config.MODEL_DIR / config.BPE_MODEL),
            source_lang=config.SOURCE_LANG,
            target_lang=config.TARGET_LANG,
            beam=config.BEAM_SIZE,
            max_tokens=config.MAX_TOKENS,
        )

        if device == "cuda":
            self._model.cuda()
            logger.info("model moved to GPU (cuda:0)")

        self._model.eval()
        self._device = device
        self._initialized = True

        elapsed = time.perf_counter() - start
        MODEL_LOAD_TIME.set(elapsed)
        logger.info("model loaded", extra={"load_time_s": round(elapsed, 2), "device": device})

        for s in ["Que voulez-vous donc?", "Il estoit une fois un homme.", "Je ne sçay pas."]:
            self._translate_raw(s)
        logger.info("warm-up complete")

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

    def _translate_raw(self, text: str) -> str:
        with self._lock:
            return self._model.translate(text)

    def _translate_batch_raw(self, texts: List[str]) -> List[str]:
        with self._lock:
            return [self._model.translate(t) for t in texts]

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def translate(self, text: str) -> str:
        text = self._normalize(text)
        if not text:
            return ""
        return self._translate_raw(text)

    def translate_batch(self, texts: List[str], batch_size: int = 32) -> List[str]:
        normalized = [self._normalize(t) for t in texts]
        results: List[str] = []
        for i in range(0, len(normalized), batch_size):
            chunk = normalized[i : i + batch_size]
            results.extend(self._translate_batch_raw(chunk))
        return results


translator = FairseqTranslator()
