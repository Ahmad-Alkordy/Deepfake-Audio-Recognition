# 🛡️ VoiceShield — Deepfake Audio Detector

A deep learning system for detecting AI-generated and voice-cloned speech. Trained on the [Fake-or-Real (FOR)](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset) dataset using mel spectrogram analysis and a residual CNN.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=flat-square)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.89%25-brightgreen?style=flat-square)

---

## Demo

> Upload any speech audio file and the app will return a **REAL / FAKE** verdict with a confidence score, waveform, and mel spectrogram visualization.

---

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 99.89% |
| Precision | 99.93% |
| Recall | 99.85% |
| F1 Score | 99.89% |
| ROC-AUC | 100.00% |

Evaluated on the FOR `for-2sec` test set (n = 2,678 clips).

---

## How it works

1. **Audio loading** — input is resampled to 16 kHz mono
2. **Mel spectrogram** — 128 mel bins, log-scaled to dB, fixed to 87 time frames
3. **Z-score normalization** — per-sample `(x − mean) / std`
4. **Residual CNN** — stem → 3 ResBlock stages (32→64→128→256 channels) → global avg pool → classifier head
5. **Prediction** — sigmoid output gives a fake probability; classified FAKE if prob ≥ threshold (default 0.5)

AI-generated speech leaves artifacts in the spectrogram that are invisible in the raw waveform — unnatural formant transitions, absent breath noise, overly regular pitch. The CNN learns to detect these directly from the spectrogram image.

---

## Model architecture

```
Input: (1, 128, 87)  — 1 channel · 128 mel bins · 87 time frames

Stem      Conv3×3 (1→32) · BN · ReLU
Layer 1   ResidualBlock(32→64,  stride=2) × 2
Layer 2   ResidualBlock(64→128, stride=2) × 2
Layer 3   ResidualBlock(128→256,stride=2) × 2
GAP       AdaptiveAvgPool2d(1, 1)
Head      Dropout(0.4) → Linear(256→128) → ReLU → Dropout(0.3) → Linear(128→1)

Parameters: 2,790,369
Loss:       BCEWithLogitsLoss (pos_weight = n_real / n_fake)
Optimizer:  Adam lr=1e-4, weight_decay=1e-5
Scheduler:  ReduceLROnPlateau (factor=0.5, patience=2)
Stopped:    Epoch 16 of 30 (early stopping, patience=5)
```

---

## Project structure

```
voiceshield/
├── app.py                         # Streamlit web app
├── deepfake_audio_detector.ipynb  # Training notebook (Google Colab)
├── requirements.txt               # Python dependencies
├── packages.txt                   # System dependencies (for Streamlit Cloud)
└── README.md
```

---

## Run locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python -m streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501), upload your model weights (`.pth`) in the sidebar, and drop in an audio file.

---

## Training your own model

Open `deepfake_audio_detector.ipynb` in Google Colab with a T4 GPU.

**Before running:** Runtime → Change runtime type → T4 GPU

The notebook will:
1. Download the FOR dataset via the Kaggle API
2. Re-split it 70 / 15 / 15 %
3. Train the residual CNN with SpecAugment and early stopping
4. Evaluate on the test set and download the `.pth` weights

You'll need a `kaggle.json` API key — get it from your [Kaggle account settings](https://www.kaggle.com/settings).

---

## Dataset

**Fake-or-Real (FOR)** — [kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)

This project uses the `for-2sec` subset: every clip is trimmed to 2 seconds, giving a balanced set of real human speech and AI-generated speech from TTS systems.

| Split | Real | Fake |
|-------|------|------|
| Train | 6,254 | 6,254 |
| Val | 1,340 | 1,341 |
| Test | 1,338 | 1,340 |

---

## Limitations

- Trained on TTS systems from ~2019–2020. May not generalize to newer systems like ElevenLabs or VALL-E.
- Optimized for clean, close-mic speech. Performance may drop on noisy or phone-channel audio.
- Use as a screening tool, not a definitive forensic verdict.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Model inference |
| `librosa` | Audio loading and mel spectrogram extraction |
| `streamlit` | Web interface |
| `matplotlib` | Waveform and spectrogram plots |
| `numpy` | Numerical operations |
