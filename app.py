"""
Deepfake Audio Detection — Streamlit Web App
"""

import streamlit as st
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import tempfile
import os
import subprocess
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

st.set_page_config(
    page_title='Deepfake Audio Detector',
    page_icon='🎙️',
    layout='centered',
    initial_sidebar_state='collapsed'
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0f1117; }
    .hero-container {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem 1rem;
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid #2d3561;
    }
    .hero-title { font-size: 2.6rem; font-weight: 700; color: #ffffff; margin: 0.5rem 0 0.3rem 0; }
    .hero-subtitle { font-size: 1rem; color: #8892b0; margin: 0; }
    .result-genuine {
        background: linear-gradient(135deg, #0d2137 0%, #0a3d2e 100%);
        border: 2px solid #00d4aa; border-radius: 16px;
        padding: 2rem; text-align: center; margin: 1.5rem 0;
    }
    .result-fake {
        background: linear-gradient(135deg, #2d0a0a 0%, #3d1a0a 100%);
        border: 2px solid #ff4757; border-radius: 16px;
        padding: 2rem; text-align: center; margin: 1.5rem 0;
    }
    .result-label { font-size: 2rem; font-weight: 700; margin: 0.5rem 0; }
    .result-genuine .result-label { color: #00d4aa; }
    .result-fake .result-label { color: #ff4757; }
    .result-confidence { font-size: 1rem; color: #8892b0; margin: 0; }
    .score-card {
        background: #1a1f2e; border: 1px solid #2d3561;
        border-radius: 12px; padding: 1.2rem; text-align: center;
    }
    .score-label { font-size: 0.8rem; color: #8892b0; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.3rem; }
    .score-value { font-size: 1.8rem; font-weight: 700; margin: 0; }
    .info-box {
        background: #1a1f2e; border: 1px solid #2d3561;
        border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 1.5rem;
    }
    .info-title { font-size: 0.85rem; color: #64ffda; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem; }
    .info-text { font-size: 0.9rem; color: #8892b0; line-height: 1.6; margin: 0; }
    div[data-testid="stFileUploader"] {
        background: #1a1f2e; border: 2px dashed #2d3561; border-radius: 12px; padding: 1rem;
    }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
SR         = 16000
DURATION   = 4.0
N_SAMPLES  = int(SR * DURATION)
N_MFCC     = 40
N_LFCC     = 40
HOP_LENGTH = 160
N_FFT      = 512
MODEL_PATH = 'best_model.pt'
T_EXPECTED = 401

# ── Model ─────────────────────────────────────────────────────────────────────
class MFM(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return torch.max(x1, x2)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 2, 3, 1, 1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch * 2)
        self.mfm  = MFM()
    def forward(self, x):
        return self.mfm(self.bn(self.conv(x)))

class LCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32), nn.MaxPool2d(2, 2),
            ConvBlock(32, 64),          nn.MaxPool2d(2, 2),
            ConvBlock(64, 96), ConvBlock(96, 96), nn.MaxPool2d(2, 2),
            ConvBlock(96, 128), ConvBlock(128, 128), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(128 * 16, 256), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# ── Feature Extraction ────────────────────────────────────────────────────────
def load_audio(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    y = np.pad(y, (0, max(0, N_SAMPLES - len(y))))[:N_SAMPLES]
    return y

def extract_lfcc(y):
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)) ** 2
    lin_f = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_LFCC,
                                  fmin=0, fmax=SR//2, norm=None, htk=False)
    lin_s = np.dot(lin_f, S)
    lin_s = np.where(lin_s == 0, np.finfo(float).eps, lin_s)
    return librosa.feature.mfcc(S=librosa.power_to_db(lin_s), n_mfcc=N_LFCC)

def extract_features(path):
    y    = load_audio(path)
    lfcc = extract_lfcc(y)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC,
                                  hop_length=HOP_LENGTH, n_fft=N_FFT)
    feat = np.vstack([lfcc, mfcc]).astype(np.float32)
    feat = (feat - feat.mean(1, keepdims=True)) / (feat.std(1, keepdims=True) + 1e-8)
    if feat.shape[1] < T_EXPECTED:
        feat = np.pad(feat, ((0, 0), (0, T_EXPECTED - feat.shape[1])))
    return feat[:, :T_EXPECTED]

def plot_waveform(path):
    y, _ = librosa.load(path, sr=SR, duration=DURATION)
    t = np.linspace(0, len(y)/SR, len(y))
    fig, ax = plt.subplots(figsize=(8, 2))
    fig.patch.set_facecolor('#1a1f2e')
    ax.set_facecolor('#1a1f2e')
    ax.plot(t, y, color='#64ffda', linewidth=0.6, alpha=0.9)
    ax.set_xlabel('Time (s)', color='#8892b0', fontsize=9)
    ax.set_ylabel('Amplitude', color='#8892b0', fontsize=9)
    ax.tick_params(colors='#8892b0', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2d3561')
    ax.set_title('Waveform', color='#ccd6f6', fontsize=10, pad=8)
    plt.tight_layout()
    return fig

def plot_spectrogram(path):
    y, _ = librosa.load(path, sr=SR, duration=DURATION)
    S    = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=64)
    S_db = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(8, 2))
    fig.patch.set_facecolor('#1a1f2e')
    ax.set_facecolor('#1a1f2e')
    img = librosa.display.specshow(S_db, sr=SR, x_axis='time',
                                    y_axis='mel', ax=ax, cmap='magma')
    ax.set_title('Mel Spectrogram', color='#ccd6f6', fontsize=10, pad=8)
    ax.tick_params(colors='#8892b0', labelsize=8)
    ax.set_xlabel('Time (s)', color='#8892b0', fontsize=9)
    ax.set_ylabel('Hz', color='#8892b0', fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2d3561')
    plt.colorbar(img, ax=ax, format='%+2.0f dB').ax.tick_params(colors='#8892b0', labelsize=7)
    plt.tight_layout()
    return fig

@st.cache_resource
def load_model():
    device = torch.device('cpu')
    m = LCNN().to(device)
    if os.path.exists(MODEL_PATH):
        m.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        m.eval()
        return m, device
    return None, device

def predict(audio_path, model, device):
    feat   = extract_features(audio_path)
    tensor = torch.tensor(feat[np.newaxis, np.newaxis], dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).cpu().numpy()[0]
    pred  = int(np.argmax(probs))
    label = 'Deepfake (AI-Generated)' if pred == 1 else 'Genuine (Human)'
    return label, probs[pred] * 100, probs

def convert_to_wav(tmp_path, original_name):
    ext = os.path.splitext(original_name)[1].lower()
    if ext in ['.m4a', '.mp4', '.ogg', '.opus']:
        wav_path = tmp_path.replace(os.path.splitext(tmp_path)[1], '.wav')
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', tmp_path,
             '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', wav_path],
            capture_output=True
        )
        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
            return wav_path
        else:
            raise RuntimeError(f'ffmpeg conversion failed: {result.stderr.decode()[-200:]}')
    return tmp_path

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div style="font-size:3rem">🎙️</div>
    <div class="hero-title">Deepfake Audio Detector</div>
    <p class="hero-subtitle">AI-powered system to detect synthetic speech using LCNN + LFCC/MFCC features</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <div class="info-title">How it works</div>
    <p class="info-text">
    Upload any speech audio file. The model extracts <strong style="color:#64ffda">LFCC + MFCC features</strong>
    and runs them through a <strong style="color:#64ffda">Light CNN (LCNN)</strong> trained on 53,868 audio clips
    to determine if the voice is human or AI-generated.
    </p>
</div>
""", unsafe_allow_html=True)

model, device = load_model()

if model is None:
    st.error('Model file best_model.pt not found.')
    st.stop()

uploaded = st.file_uploader(
    '**Upload an audio file**',
    type=['wav', 'flac', 'mp3', 'm4a', 'ogg', 'opus'],
    help='Supported formats: WAV, FLAC, MP3, M4A, OGG'
)

if uploaded:
    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    try:
        tmp_path = convert_to_wav(tmp_path, uploaded.name)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    st.audio(uploaded)

    with st.spinner('Analysing audio...'):
        try:
            label, confidence, probs = predict(tmp_path, model, device)
            waveform_fig = plot_waveform(tmp_path)
            spec_fig     = plot_spectrogram(tmp_path)
        except Exception as e:
            st.error(f'Error: {e}')
            os.unlink(tmp_path)
            st.stop()

    os.unlink(tmp_path)

    is_genuine = 'Genuine' in label

    if is_genuine:
        st.markdown("""
        <div class="result-genuine">
            <div style="font-size:2.5rem">✅</div>
            <div class="result-label">Genuine (Human)</div>
            <p class="result-confidence">This audio appears to be from a real human speaker</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-fake">
            <div style="font-size:2.5rem">🚨</div>
            <div class="result-label">Deepfake (AI-Generated)</div>
            <p class="result-confidence">This audio appears to be synthetically generated</p>
        </div>
        """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        color = '#00d4aa' if is_genuine else '#ff4757'
        st.markdown(f"""
        <div class="score-card">
            <div class="score-label">Confidence</div>
            <div class="score-value" style="color:{color}">{confidence:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-label">Genuine Score</div>
            <div class="score-value" style="color:#00d4aa">{probs[0]*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-label">Deepfake Score</div>
            <div class="score-value" style="color:#ff4757">{probs[1]*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Audio Analysis")
    st.pyplot(waveform_fig)
    st.pyplot(spec_fig)
    plt.close('all')

    st.markdown("""
    <div class="info-box">
        <div class="info-title">Model Performance</div>
        <p class="info-text">
        Accuracy: <strong style="color:#64ffda">99.89%</strong> &nbsp;|&nbsp;
        F1 Score: <strong style="color:#64ffda">99.89%</strong> &nbsp;|&nbsp;
        EER: <strong style="color:#64ffda">0.15%</strong> &nbsp;|&nbsp;
        AUC-ROC: <strong style="color:#64ffda">1.0000</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
