"""Task 4.1 – Anti-Spoofing Classifier (Bona Fide vs Spoof).

Pipeline
--------
Waveform -> LFCC (n_ceps=60, delta + delta-delta) -> 2-layer BiLSTM(128)
         -> attention pooling -> Linear(2).

LFCC (Linear Frequency Cepstral Coefficients) is chosen over MFCC
because the ASVspoof community has shown LFCC to be stronger against
neural vocoder artefacts in the upper spectrum, which is where our
YourTTS/MMS output differs most from real human speech.

Evaluation: Equal Error Rate (EER).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# --------------------------- LFCC feature ------------------------------

class LFCC(nn.Module):
    """Linear-filter-bank cepstral coefficients.

    We approximate the LFCC front-end from Sahidullah et al. (2015):
    pre-emphasis -> Hamming window -> FFT -> linear triangular filter
    bank (n=40) -> log -> DCT (keep first n_ceps).
    """

    def __init__(self, sr: int = 16_000, n_fft: int = 512, hop: int = 160,
                 n_filters: int = 40, n_ceps: int = 60, preemph: float = 0.97):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop
        self.preemph = preemph
        self.n_ceps = n_ceps
        fb = self._linear_filterbank(n_filters, n_fft, sr)
        self.register_buffer("fb", fb)
        self.register_buffer("dct", self._dct_matrix(n_filters, n_ceps))

    def _linear_filterbank(self, n_filters, n_fft, sr):
        edges = torch.linspace(0, sr / 2, n_filters + 2)
        bins = torch.floor((n_fft + 1) * edges / sr).long()
        fb = torch.zeros(n_filters, n_fft // 2 + 1)
        for m in range(1, n_filters + 1):
            l, c, r = bins[m - 1], bins[m], bins[m + 1]
            if c == l: c = l + 1
            if r == c: r = c + 1
            fb[m - 1, l:c] = torch.linspace(0, 1, c - l)
            fb[m - 1, c:r] = torch.linspace(1, 0, r - c)
        return fb

    def _dct_matrix(self, n_filters, n_ceps):
        n = torch.arange(n_filters).float()
        k = torch.arange(n_ceps).float().unsqueeze(-1)
        m = torch.cos(math.pi * k * (2 * n + 1) / (2 * n_filters))
        m[0] = m[0] / math.sqrt(2)
        return m * math.sqrt(2.0 / n_filters)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (B, T)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        x = torch.cat([wav[:, :1], wav[:, 1:] - self.preemph * wav[:, :-1]], dim=-1)
        win = torch.hamming_window(self.n_fft, device=x.device)
        spec = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop,
                          win_length=self.n_fft, window=win,
                          return_complex=True, center=True).abs() ** 2
        # (B, F, T') x (F_filters, F) -> (B, F_filters, T')
        mel = torch.matmul(self.fb, spec)
        logmel = torch.log(mel.clamp_min(1e-6))
        ceps = torch.matmul(self.dct, logmel)         # (B, n_ceps, T')
        d = torchaudio.functional.compute_deltas(ceps)
        dd = torchaudio.functional.compute_deltas(d)
        return torch.cat([ceps, d, dd], dim=1)        # (B, 3*n_ceps, T')


# --------------------------- model -----------------------------------

class AttnPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.v = nn.Linear(dim, 1)

    def forward(self, x):                    # (B, T, D)
        w = torch.softmax(self.v(x).squeeze(-1), dim=1)
        return (x * w.unsqueeze(-1)).sum(dim=1)


class CMClassifier(nn.Module):
    def __init__(self, n_ceps: int = 60, hidden: int = 128):
        super().__init__()
        self.feat = LFCC(n_ceps=n_ceps)
        self.bn = nn.BatchNorm1d(3 * n_ceps)
        self.rnn = nn.LSTM(3 * n_ceps, hidden, num_layers=2,
                           batch_first=True, bidirectional=True, dropout=0.1)
        self.pool = AttnPool(2 * hidden)
        self.fc = nn.Linear(2 * hidden, 2)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.feat(wav)
        x = self.bn(x).transpose(1, 2)      # (B, T, 3*n_ceps)
        h, _ = self.rnn(x)
        z = self.pool(h)
        return self.fc(z)                    # logits: [bona, spoof]

    @torch.no_grad()
    def score(self, wav: torch.Tensor) -> torch.Tensor:
        """Return the spoof likelihood p(spoof|x)."""
        return torch.softmax(self.forward(wav if wav.dim() == 2 else wav.unsqueeze(0)), dim=-1)[:, 1]


# --------------------------- EER ------------------------------------

def eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Classic EER via sweeping the decision threshold.

    scores : higher = more likely spoof.
    labels : 1 = spoof, 0 = bona fide.
    Returns (eer, threshold_at_eer).
    """
    order = np.argsort(-scores)
    s = scores[order]; y = labels[order]
    n_spoof = max(1, int(y.sum()))
    n_bona = max(1, len(y) - n_spoof)
    tpr = np.cumsum(y) / n_spoof                      # detection rate
    fpr = np.cumsum(1 - y) / n_bona                   # false alarm
    fnr = 1 - tpr
    # Find crossover between fpr and fnr.
    diff = fpr - fnr
    cross = np.where(np.diff(np.signbit(diff)))[0]
    if len(cross) == 0:
        idx = int(np.argmin(np.abs(diff)))
    else:
        idx = int(cross[0])
    return float((fpr[idx] + fnr[idx]) / 2), float(s[idx])


# ------------------------- training loop ----------------------------

def train_cm(bona_wavs: list[str], spoof_wavs: list[str], ckpt: str,
             epochs: int = 15, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Train the CM on a small manifest (good enough for EER < 10%
    when bona_wavs come from the 60 s reference & spoof_wavs from the
    TTS output in Task 3.3)."""

    model = CMClassifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    clip = int(2.0 * 16_000)

    def iter_batches(bs: int = 8):
        items = [(w, 0) for w in bona_wavs] + [(w, 1) for w in spoof_wavs]
        rng = np.random.default_rng(0)
        while True:
            rng.shuffle(items)
            wavs, ys = [], []
            for path, y in items:
                import soundfile as sf
                data, sr = sf.read(path, dtype="float32", always_2d=True)
                w = torch.from_numpy(data.T).mean(0)
                if sr != 16_000:
                    w = torchaudio.functional.resample(w, sr, 16_000)
                # random 2-s crop
                if w.shape[0] >= clip:
                    i = rng.integers(0, w.shape[0] - clip + 1)
                    w = w[i:i + clip]
                else:
                    w = F.pad(w, (0, clip - w.shape[0]))
                wavs.append(w); ys.append(y)
                if len(wavs) == bs:
                    yield torch.stack(wavs).to(device), torch.tensor(ys, device=device)
                    wavs, ys = [], []
            if wavs:
                yield torch.stack(wavs).to(device), torch.tensor(ys, device=device)

    it = iter_batches()
    steps_per_epoch = max(1, (len(bona_wavs) + len(spoof_wavs)) // 8)
    for ep in range(epochs):
        model.train()
        losses = []
        for _ in range(steps_per_epoch):
            x, y = next(it)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"[cm] epoch {ep:02d}  loss={np.mean(losses):.3f}")
    Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    return model


@torch.no_grad()
def evaluate_eer(model: CMClassifier, bona_wavs: list[str], spoof_wavs: list[str],
                 device: str = "cpu") -> float:
    model.to(device).eval()
    scores, labels = [], []
    for path, lbl in [(p, 0) for p in bona_wavs] + [(p, 1) for p in spoof_wavs]:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        w = torch.from_numpy(data.T).mean(0)
        if sr != 16_000:
            w = torchaudio.functional.resample(w, sr, 16_000)
        scores.append(float(model.score(w.to(device)).item()))
        labels.append(lbl)
    e, _ = eer(np.array(scores), np.array(labels))
    return e


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bona", nargs="+", required=True)
    ap.add_argument("--spoof", nargs="+", required=True)
    ap.add_argument("--ckpt", default="models/cm.pt")
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()
    model = train_cm(args.bona, args.spoof, args.ckpt, epochs=args.epochs)
    e = evaluate_eer(model, args.bona, args.spoof)
    print(f"[cm] EER={e*100:.2f}%")
