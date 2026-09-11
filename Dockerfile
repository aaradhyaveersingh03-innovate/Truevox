FROM python:3.11-slim

# ffmpeg: decodes browser mic recordings (webm/m4a) that libsndfile can't read.
# libsndfile1: required by the soundfile/librosa audio-loading path.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# CPU-only torch build -- the default PyPI wheel drags in several GB of CUDA
# libraries that are useless (and too large for Render's build limits) on a
# CPU-only web instance.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HF_HOME=/app/hf_cache
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

# Render injects $PORT at runtime; gunicorn must bind to it (not a hardcoded port).
# --timeout 120: model loading + first inference can take longer than gunicorn's
# default 30s worker timeout, especially on a cold instance.
CMD gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1
