# Voice Origin — AI vs Human Voice Identifier

A small end-to-end project that classifies a short audio clip as **human-spoken** or **AI-generated**, using a pretrained Hugging Face model, a Flask API, and a browser frontend.

## How it works

```
index.html  --(upload audio)-->  Flask API (app.py)  --(inference)-->  Hugging Face model
   (frontend)                      (backend)                    wav2vec2-deepfake-voice-detector
```

1. The frontend lets a user drag/drop or upload an audio file.
2. It's POSTed to a local Flask API.
3. The API runs the file through a Wav2Vec2 model fine-tuned for real-vs-AI-generated speech classification, and returns a label and confidence score.

## Model

- **Model**: [`garystafford/wav2vec2-deepfake-voice-detector`](https://huggingface.co/garystafford/wav2vec2-deepfake-voice-detector)
- **Base architecture**: Wav2Vec2-XLSR, fine-tuned for binary audio classification (`real` vs `fake`)
- **Trained on**: synthetic speech from ElevenLabs, Amazon Polly, Hexgrad Kokoro, Hume AI, Speechify, and Luvvoice, vs. real human recordings

### Known limitation

This model was fine-tuned on modern TTS/voice-cloning platforms. It was also validated against [ASVspoof 2021](https://www.asvspoof.org/), which mostly contains older voice-conversion and replay attacks rather than modern TTS — accuracy on that dataset was noticeably lower on the `bonafide` (real) class than on `spoof`, likely due to this domain gap plus the codec-compressed telephony-style audio ASVspoof uses. If your use case is closer to detecting voice conversion/replay attacks specifically, consider swapping in a model trained for that task instead.

## Repository structure

```
.
├── ai_voice_detector_kaggle.ipynb   # Kaggle notebook: load model, test, export
├── app.py                           # Flask backend serving the model
├── index.html                       # Frontend upload UI
└── README.md
```

## Setup

### 1. Kaggle notebook (optional — for testing/exporting the model)

Upload `ai_voice_detector_kaggle.ipynb` to Kaggle and run it top to bottom. It will:
- Load the model from Hugging Face
- Run inference on sample audio
- Save and zip the model to Kaggle's output panel for download

### 2. Backend

```bash
pip install flask flask-cors torch torchaudio transformers librosa soundfile
python app.py
```

The API starts at `http://localhost:5000`.

- `POST /predict` — accepts `multipart/form-data` with an `audio` field, returns:
  ```json
  { "label": "real", "confidence": 0.97, "all_scores": { "real": 0.97, "fake": 0.03 } }
  ```
- `GET /health` — basic status check

By default, the backend pulls the model live from the Hugging Face Hub on startup. To use a locally downloaded copy instead (e.g. the zip exported from the Kaggle notebook), change `MODEL_PATH` in `app.py` to the local folder path.

### 3. Frontend

Open `index.html` in a browser (or serve it from any static host). Make sure `API_URL` inside the `<script>` tag points at wherever `app.py` is running.

## Requirements

- Python 3.10+
- ~2–3 GB free disk space (model weights + PyTorch)
- CPU is sufficient; GPU speeds up inference but isn't required

## Disclaimer

This is a demonstration project, not a production-grade fraud detection system. Model outputs are probabilistic and should not be treated as definitive proof of authenticity, especially on audio types or languages the model wasn't trained on.
