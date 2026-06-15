# Deepfake Audio Detection

Binary classifier to detect whether a speech recording is **Genuine (Human)** or **Deepfake (AI-Generated)**.

---

## Methodology

### Preprocessing

- All audio resampled to **16 kHz**, mono
- Padded or trimmed to **4 seconds** (64,000 samples)

### Feature Extraction

- **LFCC** (Linear Frequency Cepstral Coefficients, 40 dims) — primary signal; deepfake artifacts are more visible in linearly-spaced frequency bands than log-spaced Mel bands
- **MFCC** (Mel Frequency Cepstral Coefficients, 40 dims) — complementary perceptual features
- Both stacked → **(80 × T)** feature map, per-feature normalised (zero mean, unit variance)
- Frame: 512-point FFT, 160-sample hop (10 ms)

### Model — LCNN (Light CNN)

```
Input (1, 80, T)
  └─ ConvBlock(32) → MaxPool /2
  └─ ConvBlock(64) → MaxPool /2
  └─ ConvBlock(96) → ConvBlock(96) → MaxPool /2
  └─ ConvBlock(128) → ConvBlock(128) → MaxPool /2
  └─ AdaptiveAvgPool(4×4)
  └─ Flatten → Linear(256) → Dropout(0.5) → Linear(2)
```

**ConvBlock** uses **Max-Feature-Map (MFM)** activation — doubles output channels then takes element-wise max across the two halves, acting as a built-in feature selector that suppresses noise.

### Training

- Loss: CrossEntropyLoss
- Optimiser: Adam (lr=1e-3, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR
- 30 epochs, batch size 64
- Best checkpoint saved by validation accuracy

---

## Dataset

[Fake-or-Real Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)  
Training split: `LA_norm/train/real/` and `LA_norm/train/fake/`

---

## Results

| Metric            | Value  | Threshold |
| ----------------- | ------ | --------- |
| Overall Accuracy  | 99.89% | ≥ 80% ✅  |
| F1 Score (macro)  | 99.89% | ≥ 80% ✅  |
| EER               | 0.11%  | ≤ 12% ✅  |
| AUC-ROC           | 1.0000 |           |
| Genuine Accuracy  | 99.85% | ≥ 75% ✅  |
| Deepfake Accuracy | 99.93% | ≥ 75% ✅  |

---

## Repository Structure

```
├── deepfake_audio_detection.ipynb  # Full training notebook
├── predict.py                      # CLI inference script
├── app.py                          # Streamlit web app
├── best_model.pt                   # Trained model weights
├── training_curves.png
├── confusion_matrix.png
├── roc_curve.png
└── README.md
```

---

## Quick Start

```bash
pip install torch librosa scikit-learn matplotlib streamlit

# Train (run notebook) or skip if best_model.pt is available

# CLI inference
python predict.py --audio path/to/audio.wav

# Web app
streamlit run app.py
```

---

## Streamlit App

- Upload `.wav` / `.flac` / `.mp3`
- Returns: **Genuine** or **Deepfake** label + confidence score
