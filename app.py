"""
AI vs Human Voice Identifier - Backend API
-------------------------------------------
Serves the Hugging Face model garystafford/wav2vec2-deepfake-voice-detector
and powers the landing page in templates/index.html.

Setup:
    pip install flask flask-cors torch torchaudio transformers librosa soundfile matplotlib numpy

    Browser microphone recordings arrive as webm/Opus (or m4a on some
    Safari versions), which libsndfile (librosa's default backend) can't
    read. Decoding those falls back to piping the bytes through the
    ffmpeg binary directly, which needs ffmpeg installed and on PATH:
      - Windows: download a build from https://www.gyan.dev/ffmpeg/builds/
        (or `choco install ffmpeg`), then add its bin folder to PATH.
      - macOS: brew install ffmpeg
      - Linux: apt install ffmpeg (or your distro's equivalent)
    Plain file uploads (wav/mp3/flac/etc.) work without ffmpeg.

Run:
    python app.py
    -> Page:    http://localhost:5000/
    -> Analyze: http://localhost:5000/analyze  (POST, used by the page's "Run Detection" button)
    -> Predict: http://localhost:5000/predict  (POST, legacy single-shot endpoint)
    -> Health:  http://localhost:5000/health   (GET)

NOTE: The landing page's copy describes a hand-rolled CNN-on-spectrograms
pipeline (train.py / infer.py / features.py etc.) as illustrative text.
The actual classifier running underneath is the Hugging Face wav2vec2
model above -- /analyze reproduces the same chunk-then-aggregate flow
the page describes (5s windows, 50% overlap, mean aggregation) using
that model, and renders a real Mel spectrogram of your uploaded clip
for the on-page visualization. If you swap in a real custom CNN later,
this file is the place to change.

NOTE: If HF_HOME / HF_HUB_CACHE / TRANSFORMERS_CACHE point at a drive
that doesn't exist on this machine, downloads will fail with a WinError 3.
Either unset those env vars or point them at a folder that exists.
"""

import base64
import io
import os
import subprocess
import time

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_PATH = "garystafford/wav2vec2-deepfake-voice-detector"  # or "./ai_voice_model"
TARGET_SR = 16000
N_MELS = 128
CHUNK_SECONDS = 5.0
OVERLAP = 0.5
AGGREGATION = "mean"
SILENCE_TRIM_DB = 30

# Labels considered "AI-generated" vs "human" when reading the model's own
# id2label names -- adjust here if this model's labels differ.
AI_KEYWORDS = ("fake", "ai", "synthetic", "spoof", "generated", "deepfake")
HUMAN_KEYWORDS = ("real", "human", "bonafide", "genuine")

app = Flask(__name__)
CORS(app)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {device} ...")
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
model = AutoModelForAudioClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()
print("Model loaded. Labels:", model.config.id2label)

# Work out which output index is "AI" vs "human" from the model's own labels.
_id2label = model.config.id2label
_ai_idx = None
for idx, name in _id2label.items():
    low = name.lower()
    if any(k in low for k in AI_KEYWORDS):
        _ai_idx = int(idx)
        break
if _ai_idx is None:
    for idx, name in _id2label.items():
        low = name.lower()
        if any(k in low for k in HUMAN_KEYWORDS):
            _ai_idx = int(idx == 0)  # the other index is "AI"
            break
if _ai_idx is None:
    # Fall back to the common convention: index 1 = positive/fake class.
    _ai_idx = 1 if len(_id2label) > 1 else 0


def _ffmpeg_transcode_to_wav(audio_bytes: bytes) -> bytes:
    """Pipes arbitrary audio bytes through the ffmpeg binary and returns
    16kHz mono WAV bytes. ffmpeg auto-detects the real container from the
    stream itself, so no filename/content-type hint is needed."""
    cmd = [
        "ffmpeg", "-y", "-i", "pipe:0",
        "-ar", str(TARGET_SR), "-ac", "1", "-f", "wav", "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, input=audio_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg is not installed or not on PATH. Browser microphone "
            "recordings (webm/m4a) need ffmpeg to decode -- plain file "
            "uploads (wav/mp3/etc.) don't require it."
        )
    if proc.returncode != 0:
        stderr_tail = proc.stderr.decode(errors="ignore")[-500:]
        raise RuntimeError(f"ffmpeg failed to decode this audio file: {stderr_tail}")
    return proc.stdout


def decode_audio_bytes(audio_bytes: bytes, filename: str = "", content_type: str = "") -> np.ndarray:
    """Decode audio bytes to a mono float array at TARGET_SR.

    Tries libsndfile (via librosa) first -- fast, and covers wav/flac/ogg.
    libsndfile can't read webm/Opus (what browser MediaRecorder produces)
    or some m4a/mp3 variants, so on failure this falls back to piping the
    bytes through the ffmpeg binary, which auto-detects the container.
    Requires ffmpeg to be installed and on PATH for that fallback.
    """
    try:
        speech, _ = librosa.load(io.BytesIO(audio_bytes), sr=TARGET_SR, mono=True)
        return speech
    except Exception:
        pass

    wav_bytes = _ffmpeg_transcode_to_wav(audio_bytes)
    speech, _ = librosa.load(io.BytesIO(wav_bytes), sr=TARGET_SR, mono=True)
    return speech


def load_audio(audio_bytes: bytes, filename: str = "", content_type: str = ""):
    speech = decode_audio_bytes(audio_bytes, filename, content_type)
    speech, _ = librosa.effects.trim(speech, top_db=SILENCE_TRIM_DB)
    return speech


def chunk_audio(speech: np.ndarray):
    chunk_len = int(CHUNK_SECONDS * TARGET_SR)
    hop = int(chunk_len * (1 - OVERLAP))
    if len(speech) <= chunk_len:
        chunks = [np.pad(speech, (0, chunk_len - len(speech)))]
    else:
        chunks = []
        start = 0
        while start < len(speech):
            piece = speech[start:start + chunk_len]
            if len(piece) < chunk_len:
                piece = np.pad(piece, (0, chunk_len - len(piece)))
            chunks.append(piece)
            if start + chunk_len >= len(speech):
                break
            start += hop
    return chunks


def score_chunk(chunk: np.ndarray) -> float:
    """Returns the AI-probability for a single chunk."""
    inputs = feature_extractor(chunk, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
    return float(probs[_ai_idx])


def make_spectrogram_png(speech: np.ndarray) -> str:
    """Renders a Mel spectrogram of the (trimmed) audio and returns a data: URI."""
    mel = librosa.feature.melspectrogram(y=speech, sr=TARGET_SR, n_mels=N_MELS, power=2.0)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(8, 3), dpi=110)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    librosa.display.specshow(mel_db, sr=TARGET_SR, x_axis="time", y_axis="mel", ax=ax, cmap="magma")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(colors="#8b8fa3", labelsize=7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def aggregate(scores):
    if AGGREGATION == "median":
        return float(np.median(scores))
    if AGGREGATION == "vote":
        return float(np.mean([1.0 if s > 0.5 else 0.0 for s in scores]))
    return float(np.mean(scores))  # default: mean


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        model_path=MODEL_PATH,
        sample_rate=TARGET_SR,
        n_mels=N_MELS,
        chunk_seconds=CHUNK_SECONDS,
        aggregation=AGGREGATION,
    )


@app.route("/analyze", methods=["POST"])
def analyze_route():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file found in request (field name must be 'audio')"}), 400

    file = request.files["audio"]
    t0 = time.time()
    try:
        raw_bytes = file.read()
        speech = load_audio(raw_bytes, filename=file.filename, content_type=file.content_type)
        duration_sec = len(speech) / TARGET_SR

        chunks = chunk_audio(speech)
        chunk_scores = [score_chunk(c) for c in chunks]
        ai_score = aggregate(chunk_scores)

        spectrogram_png = make_spectrogram_png(speech)

        prediction = "AI" if ai_score > 0.5 else "HUMAN"
        confidence = ai_score if prediction == "AI" else (1 - ai_score)

        return jsonify({
            "prediction": prediction,
            "ai_score_pct": round(ai_score * 100, 2),
            "human_score_pct": round((1 - ai_score) * 100, 2),
            "confidence_pct": round(confidence * 100, 2),
            "num_chunks": len(chunks),
            "chunk_scores": [round(s, 4) for s in chunk_scores],
            "aggregation_method": AGGREGATION,
            "processing_time_sec": round(time.time() - t0, 3),
            "device": device,
            "spectrogram_png": spectrogram_png,
            "specs": {
                "duration_sec": round(duration_sec, 2),
                "original_sample_rate": TARGET_SR,
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict_route():
    """Legacy single-shot endpoint (whole clip, no chunking)."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file found in request (field name must be 'audio')"}), 400

    file = request.files["audio"]
    try:
        speech = decode_audio_bytes(file.read(), filename=file.filename, content_type=file.content_type)
        inputs = feature_extractor(speech, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs))
        return jsonify({
            "label": model.config.id2label[pred_id],
            "confidence": round(float(probs[pred_id]), 4),
            "all_scores": {
                model.config.id2label[i]: round(float(p), 4) for i, p in enumerate(probs)
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": device})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
