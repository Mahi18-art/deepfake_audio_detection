"""
Deepfake Audio Detection — Streamlit Web App
Usage: streamlit run app.py
"""

import streamlit as st
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import tempfile
import os

# ── Config (must match notebook) ──────────────────────────────────────────────
SR          = 16000
DURATION    = 4.0
N_SAMPLES   = int(SR * DURATION)
N_MFCC      = 40
N_LFCC      = 40
HOP_LENGTH  = 160
N_FFT       = 512
MODEL_PATH  = 'best_model.pt'
T_EXPECTED  = None   # will be inferred or hardcoded below — set after training


# ── Model ─────────────────────────────────────────────────────────────────────
class MFM(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return torch.max(x1, x2)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 2, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch * 2)
        self.mfm  = MFM()

    def forward(self, x):
        return self.mfm(self.bn(self.conv(x)))


class LCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            nn.MaxPool2d(2, 2),
            ConvBlock(32, 64),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 96),
            ConvBlock(96, 96),
            nn.MaxPool2d(2, 2),
            ConvBlock(96, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Feature extraction ────────────────────────────────────────────────────────
def load_audio(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))
    else:
        y = y[:N_SAMPLES]
    return y


def extract_lfcc(y):
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)) ** 2
    lin_filters = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_LFCC,
                                       fmin=0, fmax=SR // 2, norm=None, htk=False)
    lin_spec = np.dot(lin_filters, S)
    lin_spec = np.where(lin_spec == 0, np.finfo(float).eps, lin_spec)
    return librosa.feature.mfcc(S=librosa.power_to_db(lin_spec), n_mfcc=N_LFCC)


def extract_features(path):
    y    = load_audio(path)
    lfcc = extract_lfcc(y)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC,
                                  hop_length=HOP_LENGTH, n_fft=N_FFT)
    feat = np.vstack([lfcc, mfcc]).astype(np.float32)
    feat = (feat - feat.mean(axis=1, keepdims=True)) / \
           (feat.std(axis=1, keepdims=True) + 1e-8)
    return feat


@st.cache_resource
def load_model():
    device = torch.device('cpu')
    m = LCNN(in_channels=1).to(device)
    if os.path.exists(MODEL_PATH):
        m.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        m.eval()
        return m, device
    return None, device


def predict(audio_path, model, device, t_expected=401):
    feat = extract_features(audio_path)
    if feat.shape[1] < t_expected:
        feat = np.pad(feat, ((0, 0), (0, t_expected - feat.shape[1])))
    else:
        feat = feat[:, :t_expected]
    tensor = torch.tensor(feat[np.newaxis, np.newaxis], dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).cpu().numpy()[0]
    pred  = int(np.argmax(probs))
    label = 'Deepfake (AI-Generated)' if pred == 1 else 'Genuine (Human)'
    return label, probs[pred] * 100, probs


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title='Deepfake Audio Detector', page_icon='🎙️', layout='centered')

st.title('🎙️ Deepfake Audio Detector')
st.markdown('Upload a `.wav` or `.flac` audio file to check if it is **Genuine (Human)** or **Deepfake (AI-Generated)**.')

model, device = load_model()

if model is None:
    st.error('⚠️ Model file `best_model.pt` not found. Train the model first using the notebook.')
    st.stop()

uploaded = st.file_uploader('Choose an audio file', type=['wav', 'flac', 'mp3'])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.audio(uploaded)

    with st.spinner('Analysing...'):
        try:
            # ── set T_EXPECTED to match your trained model's time dimension ──
            # If you saved T value during training, load it here.
            # Default 401 = (16000*4) / 160 + 1
            T_EXPECTED = 401
            label, confidence, probs = predict(tmp_path, model, device, T_EXPECTED)
        except Exception as e:
            st.error(f'Prediction failed: {e}')
            os.unlink(tmp_path)
            st.stop()

    os.unlink(tmp_path)

    color = '🟢' if 'Genuine' in label else '🔴'
    st.markdown(f'## {color} {label}')
    st.metric('Confidence', f'{confidence:.1f}%')

    st.markdown('---')
    col1, col2 = st.columns(2)
    col1.metric('Genuine Score',  f'{probs[0]*100:.1f}%')
    col2.metric('Deepfake Score', f'{probs[1]*100:.1f}%')

    st.progress(float(probs[1]))
    st.caption('Progress bar shows deepfake probability')
