# ─────────────────────────────────────────────────────────────────────────────
# ProcEx — Dockerfile
# Branch: gemma-mode
#
# Build context: run  docker build  from the root of the gemma-mode repo.
# The image contains everything the Celery worker needs to run the pipeline:
#   - Python 3.12
#   - FFmpeg
#   - Manim + full LaTeX (texlive)
#   - All Python dependencies from requirements.txt
#   - celery, redis, boto3 (deployment deps not in the original requirements.txt)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

# ── System dependencies ───────────────────────────────────────────────────────
# texlive-* packages are large (~1.5GB) but required by Manim for LaTeX rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    texlive-latex-recommended \
    texlive-fonts-extra \
    texlive-latex-extra \
    texlive-science \
    libcairo2-dev \
    libpango1.0-dev \
    pkg-config \
    python3-dev \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install deployment dependencies that may not be in the original requirements.txt
RUN pip install --no-cache-dir \
    manim \
    python-dotenv

# ── Copy project source ───────────────────────────────────────────────────────
COPY . .

# ── Output directory for generated videos ────────────────────────────────────
RUN mkdir -p /tmp/procex_output

# ── Default command ───────────────────────────────────────────────────────────
# Overridden by docker-compose for worker vs API roles.
# Workers use:  celery -A celery_app worker ...
# API uses:     uvicorn api_server:app ...
CMD ["celery", "-A", "celery_app", "worker", \
     "--loglevel=info", \
     "--concurrency=1", \
     "-Q", "procex,procex-premium"]