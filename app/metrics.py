import logging
import sys

from pythonjsonlogger import jsonlogger
from prometheus_client import Counter, Histogram, Gauge


def _setup_logging(level: str = "info") -> logging.Logger:
    log = logging.getLogger("vieuxparler")
    log.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level"},
    )
    handler.setFormatter(formatter)
    log.handlers = [handler]
    return log


logger = _setup_logging()

TRANSLATION_REQUESTS = Counter(
    "translation_requests_total",
    "Total translation requests",
    ["endpoint", "status"],
)
TRANSLATION_DURATION = Histogram(
    "translation_duration_seconds",
    "Time spent on translation inference",
    ["endpoint"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)
BATCH_SIZE_PROCESSED = Gauge(
    "batch_size_processed",
    "Size of last processed batch",
)
MODEL_LOAD_TIME = Gauge(
    "model_load_time_seconds",
    "Time to load the model at startup",
)
