# ============================================================
# Stage 1 – Install dependencies
# ============================================================
FROM python:3.9-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# 1. Pin pip and install build tools first
RUN pip install "pip<24.1" \
    && pip install --no-cache-dir setuptools wheel build Cython

# 2. Install fairseq and ML dependencies
RUN pip install --no-cache-dir \
    fairseq==0.12.2 \
    iopath \
    sentencepiece \
    sacrebleu \
    omegaconf==2.0.5 \
    gdown==4.2.0 \
    tensorboardX \
    numpy==1.25.2 \
    pandas \
    matplotlib

# 3. Install PyTorch CPU (replaces any torch pulled by fairseq)
RUN pip uninstall -y torch torchaudio 2>/dev/null; \
    pip install --no-cache-dir \
    torch==2.5.1 torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# 4. Install app dependencies (fastapi, uvicorn, etc.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================
# Stage 2 – Runtime image
# ============================================================
FROM python:3.9-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        unzip \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy application code
COPY main.py ./
COPY app/ ./app/
COPY scripts/ ./scripts/

# Download model
RUN curl -fL -o freem_lstm_fairseq.zip \
        "https://github.com/Grand-Siecle/test_modernisation/releases/download/fairseq-v0.1/freem_lstm_fairseq.zip" \
    && unzip -o freem_lstm_fairseq.zip \
    && rm freem_lstm_fairseq.zip \
    && ls -la freem_lstm_fairseq/model/checkpoint_best.pt

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
