# syntax=docker/dockerfile:1.4
# =============================================================================
# PitWall Intelligence — Backend Dockerfile (CPU-first)
#
# This image runs on CPU. For GPU builds, see Dockerfile.gpu.
# Multi-stage build: builder installs deps, final stage is lean.
# =============================================================================

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# System dependencies required for opencv-python-headless and numpy builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install PyTorch CPU build (avoids downloading 2+ GB CUDA wheels)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime system deps (OpenCV shared libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Non-root user for security
RUN useradd -m -u 1000 apex && \
    mkdir -p /app/logs /app/.hf_cache /app/demo_clips && \
    chown -R apex:apex /app

WORKDIR /app
USER apex

# Copy application code
COPY --chown=apex:apex app/ app/
COPY --chown=apex:apex scripts/ scripts/
COPY --chown=apex:apex .env.example .env.example

# Environment variables for the container
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000

# HuggingFace cache inside container (mount as volume in production)
ENV HF_HOME=/app/.hf_cache
ENV HF_HUB_CACHE=/app/.hf_cache/hub
ENV TRANSFORMERS_CACHE=/app/.hf_cache/hub
ENV TOKENIZERS_PARALLELISM=false

EXPOSE 8000

# Health check using our liveness endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch with uvicorn
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--loop", "asyncio", \
     "--log-level", "warning"]
