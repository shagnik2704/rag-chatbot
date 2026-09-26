# ==============================================================================
# Stage 1: Build Frontend Assets
# ==============================================================================
FROM node:22-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# ==============================================================================
# Stage 2: Production Python Backend & Runtime
# ==============================================================================
FROM python:3.11-slim AS production

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HF_HOME=/home/appuser/.cache/huggingface

WORKDIR /app

# Install system dependencies (curl for healthchecks, libpq for postgres)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create non-root application user first so we can prepare its home directory
RUN useradd -m -u 1001 appuser && \
    mkdir -p /home/appuser/.cache/huggingface /app/storage

# Install uv for fast, reliable dependency installation
RUN pip install --no-cache-dir uv

# Install lightweight CPU-only PyTorch first to prevent downloading 4.5GB of unused NVIDIA CUDA runtime
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# Copy dependency manifests
COPY pyproject.toml uv.lock ./

# Install Python dependencies globally
RUN uv pip install --system --no-cache -r pyproject.toml

# Pre-bake HuggingFace embedding model into image to eliminate cold-start lag
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# Copy source code, data, and entrypoint
COPY src/ ./src/
COPY data/ ./data/
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

# Copy compiled frontend from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Set ownership of all app files to appuser
RUN chown -R appuser:appuser /app /home/appuser

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]

CMD ["uvicorn", "src.app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
