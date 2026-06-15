"""
predict.py — Test the trained model on any audio file from command line.
Usage: python predict.py --audio path/to/file.wav [--model best_model.pt]
"""

import argparse
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

SR, DURATION, N_MFCC, N_LFCC, HOP_LENGTH, N_FFT = 16000, 4.0, 40, 40, 160, 512
N_SAMPLES  = int(SR * DURATION)
T_EXPECTED = 401  # (N_SAMPLES / HOP_LENGTH) + 1 — adjust if you changed config


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
    def forward(self, x): return self.mfm(self.bn(self.conv(x)))


class LCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32), nn.MaxPool2d(2, 2),
            ConvBlock(32, 64), nn.MaxPool2d(2, 2),
            ConvBlock(64, 96), ConvBlock(96, 96), nn.MaxPool2d(2, 2),
            ConvBlock(96, 128), ConvBlock(128, 128), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(128 * 16, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
    def forward(self, x): return self.classifier(self.features(x))


def extract_features(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    y = np.pad(y, (0, max(0, N_SAMPLES - len(y))))[:N_SAMPLES]
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)) ** 2
    lin_f = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_LFCC, fmin=0,
                                  fmax=SR//2, norm=None, htk=False)
    lin_s = np.where(np.dot(lin_f, S) == 0, np.finfo(float).eps, np.dot(lin_f, S))
    lfcc  = librosa.feature.mfcc(S=librosa.power_to_db(lin_s), n_mfcc=N_LFCC)
    mfcc  = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT)
    feat  = np.vstack([lfcc, mfcc]).astype(np.float32)
    feat  = (feat - feat.mean(1, keepdims=True)) / (feat.std(1, keepdims=True) + 1e-8)
    if feat.shape[1] < T_EXPECTED:
        feat = np.pad(feat, ((0,0),(0, T_EXPECTED - feat.shape[1])))
    return feat[:, :T_EXPECTED]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', required=True, help='Path to audio file')
    parser.add_argument('--model', default='best_model.pt', help='Path to .pt model')
    args = parser.parse_args()

    assert os.path.exists(args.audio), f'Audio not found: {args.audio}'
    assert os.path.exists(args.model), f'Model not found: {args.model}'

    device = torch.device('cpu')
    model  = LCNN().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    feat   = extract_features(args.audio)
    tensor = torch.tensor(feat[np.newaxis, np.newaxis], dtype=torch.float32)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).numpy()[0]

    pred  = int(np.argmax(probs))
    label = 'Deepfake (AI-Generated)' if pred == 1 else 'Genuine (Human)'
    print(f'\nFile      : {args.audio}')
    print(f'Prediction: {label}')
    print(f'Confidence: {probs[pred]*100:.1f}%')
    print(f'  Genuine  : {probs[0]*100:.1f}%')
    print(f'  Deepfake : {probs[1]*100:.1f}%\n')


if __name__ == '__main__':
    main()
