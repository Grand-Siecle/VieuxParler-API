# ============================================================
# Stage 1 – Install dependencies
# ============================================================
FROM python:3.9-slim-bullseye AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install PyTorch CPU first (keeps image small; swap index for CUDA if needed)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================
# Stage 2 – Runtime image
# ============================================================
FROM python:3.9-slim-bullseye

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy application code
COPY main.py ./
COPY app/ ./app/

# Logs volume
RUN mkdir -p /logs

ENV MODEL_DIR=/app/freem_lstm_fairseq \
    HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=1 \
    LOG_LEVEL=info \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
