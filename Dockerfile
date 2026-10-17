# ==============================================================================
# CareerSynk Impact - FastAPI Inference Service
# Multi-Architecture Dockerfile: LinuxONE (linux/s390x) & x86_64 (linux/amd64)
# ==============================================================================

# Base image: Official multi-arch Python 3.13 slim with verified linux/s390x manifest
FROM python:3.13-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

WORKDIR /app

# Install minimal OS dependencies needed for building packages and health check curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt .

# Install dependencies (CPU-only PyTorch index to prevent CUDA bloat and ensure s390x/CPU compatibility)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy source code and processed artifacts only (STRICT: data/raw/ is excluded)
COPY src/ /app/src/
COPY data/processed/skill_graph.pkl /app/data/processed/skill_graph.pkl
COPY data/processed/baseline_relevance_model.pkl /app/data/processed/baseline_relevance_model.pkl

# Security: Create and run under an unprivileged non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose standard HTTP port
EXPOSE 8000

# Docker healthcheck querying the /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Bounded concurrency suitable for small footprint deployments (<350MB target)
# Single worker with capped concurrency to prevent memory spikes on s390x LPAR
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "50"]
