# Deepfake Audio Detection

Binary classifier to detect whether a speech recording is **Genuine (Human)** or **Deepfake (AI-Generated)**.

**Task:** Binary classification — Genuine (Human) vs Deepfake (AI-Generated) speech  
**Dataset:** Fake-or-Real (FoR) Dataset — `for-norm/for-norm/training/`  
**Pipeline:** Audio → LFCC + MFCC Features → LCNN Model → Prediction

---

## Live Demo

[Click here to try the Streamlit app](https://deepfakeaudiodetection-alakvk5trbervyetxuqffu.streamlit.app)

---

## Pipeline Overview

```
Raw Audio (.wav/.flac)
    → Resample to 16kHz, Pad/Trim to 4s
        → LFCC (40 dims) + MFCC (40 dims) → stacked (80 x 401)
            → LCNN (Light CNN with MFM activation)
                → Softmax → Genuine / Deepfake + Confidence
```

---

## Methodology

### Preprocessing

- All audio resampled to 16 kHz (standard for speech processing)
- Padded or trimmed to exactly 4 seconds — fixed-size feature maps
- Frame: 512-point FFT, 160-sample hop (10 ms frame shift)

### Feature Extraction

**LFCC (Linear Frequency Cepstral Coefficients, 40 dims)**  
Uses a linearly-spaced filterbank instead of the log-spaced Mel filterbank. Deepfake artifacts from vocoders (WaveNet, Griffin-Lim) appear in high-frequency linear bands that Mel filterbanks compress or hide. LFCC is the primary discriminative feature for anti-spoofing.

**MFCC (Mel Frequency Cepstral Coefficients, 40 dims)**  
Standard perceptual speech features capturing the overall spectral envelope (vocal tract shape). Complementary to LFCC.

Both stacked vertically to form a **(80 x 401)** feature map, per-feature normalised (zero mean, unit variance).

### Model Architecture — LCNN (Light CNN)

LCNN is the ASVspoof 2019 baseline model, specifically designed for anti-spoofing tasks.

**Max-Feature-Map (MFM) Activation** — core innovation:
- Each Conv2d outputs `2 x channels`
- MFM takes element-wise max between the two halves
- Acts as a learnable filter that suppresses noise features
- Outperforms ReLU for spoofing detection because it selects the most informative feature map per spatial location

```
Input (1, 80, 401)
 └─ ConvBlock(32)  + MaxPool  →  (32, 40, 200)
 └─ ConvBlock(64)  + MaxPool  →  (64, 20, 100)
 └─ ConvBlock(96) x2 + MaxPool  →  (96, 10, 50)
 └─ ConvBlock(128) x2 + MaxPool  →  (128, 5, 25)
 └─ AdaptiveAvgPool(4x4)  →  (128, 4, 4)
 └─ Flatten → Linear(256) → Dropout(0.5) → Linear(2)
```

Total trainable parameters: 1,357,250

### Training Configuration

- Loss: CrossEntropyLoss
- Optimiser: Adam (lr=1e-3, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR (avoids sharp loss spikes near end of training)
- Epochs: 20
- Batch size: 64
- Checkpoint: saved on best validation accuracy

---

## Dataset

**Fake-or-Real (FoR) Dataset** — for-norm split  
Source: [kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)

| Split | Genuine | Deepfake | Total |
|-------|---------|----------|-------|
| Train | 20,208 | 20,193 | 40,401 |
| Val   | 4,040  | 4,040  | 8,080  |
| Test  | 2,693  | 2,694  | 5,387  |
| Total | 26,941 | 26,927 | 53,868 |

Class balance: 50% / 50% — perfectly balanced, no class weighting required.

---

## Results

| Metric | Value | Required Threshold |
|--------|-------|--------------------|
| Overall Accuracy | 99.89% | >= 80% |
| F1 Score (macro) | 99.89% | >= 80% |
| EER | 0.15% | <= 12% |
| AUC-ROC | 1.0000 | — |
| Genuine Accuracy | 99.85% | >= 75% |
| Deepfake Accuracy | 99.93% | >= 75% |

Confusion matrix: 6 errors out of 5,387 test samples.

---

## Repository Structure

```
deepfake_audio_detection/
├── notebook_final.ipynb          # Full training notebook with EDA
├── predict.py                    # CLI inference script
├── app.py                        # Streamlit web app
├── best_model.pt                 # Trained model weights
├── requirements.txt              # Dependencies
├── images/                       # Result plots
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── training_curves.png
│   ├── score_distribution.png
│   ├── eda_waveform.png
│   ├── eda_mel_spectrogram.png
│   ├── eda_mfcc.png
│   ├── eda_mfcc_means.png
│   └── eda_feature_map.png
└── README.md
```

---

## Quick Start

```bash
pip install torch librosa streamlit numpy scikit-learn matplotlib

# CLI inference
python predict.py --audio path/to/audio.wav

# Web app
streamlit run app.py
```

---

## Performance Notes

The high accuracy on the FoR for-norm dataset is consistent with published literature. The normalized version removes loudness variation, making vocoder artifacts clearly visible in LFCC features. LCNN with LFCC is the established baseline for this task and achieves near-perfect scores on this benchmark.
