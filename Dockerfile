# Multi-stage build for production-ready container
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip setuptools wheel
RUN pip install .

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r ngfx && useradd -r -g ngfx ngfx

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create application directory
WORKDIR /app

# Copy application code
COPY --chown=ngfx:ngfx src/ ./src/
COPY --chown=ngfx:ngfx tests/ ./tests/
COPY --chown=ngfx:ngfx scripts/ ./scripts/

# Create required directories
RUN mkdir -p /app/models /app/logs /app/mlruns /app/data \
    && chown -R ngfx:ngfx /app

# Switch to non-root user
USER ngfx

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "src.ngfx_predictor.api.main:app", "--host", "0.0.0.0", "--port", "8000"] 