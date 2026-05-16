# ─────────────────────────────────────────────────────────────────────────────
# ProcEx — Dockerfile (RunPod Serverless)
# Branch: gemma-mode
#
# Differences from the Celery version:
#   - Entry point is  python handler.py  not  celery worker
#   - runpod SDK added to dependencies
#   - No celery, no redis broker needed
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    texlive \
    texlive-latex-extra \
    texlive-fonts-recommended \
    dvipng \
    cm-super \
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
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir manim python-dotenv runpod

# ── Copy project source ───────────────────────────────────────────────────────
COPY . .

# ── Output directory ──────────────────────────────────────────────────────────
RUN mkdir -p /tmp/procex_output

# ── RunPod Serverless entry point ─────────────────────────────────────────────
CMD ["python", "handler.py"]