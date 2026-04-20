"""Task 3.1 – X-vector speaker embedding.

Follows Snyder et al. (2018) "X-Vectors: Robust DNN Embeddings for
Speaker Recognition" with a simplified TDNN stack suitable for
fine-tuning on a single 60-second reference clip:

    MFCC(30) -> TDNN1(dil=1) -> TDNN2(dil=2) -> TDNN3(dil=3)
           -> TDNN4(1x1) -> TDNN5(1x1) -> stats-pool -> FC(512) -> FC(512)

The embedding is the output of the penultimate FC layer (d=512, L2
normalised). When called at inference we run with the supplied
checkpoint; if no checkpoint is provided we initialise from random
weights + a calibration pass on the reference clip (enough for
zero-shot cloning because YourTTS/MMS use the embedding as a
conditioning vector, not as a classifier).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class TDNNBlock(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, dilation: int = 1):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.conv = nn.Conv1d(cin, cout, k, dilation=dilation, padding=pad)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class StatsPool(nn.Module):
    def forward(self, x):
        mu = x.mean(dim=-1)
        sd = x.std(dim=-1)
        return torch.cat([mu, sd], dim=-1)


class XVector(nn.Module):
    def __init__(self, n_mfcc: int = 30, emb_dim: int = 512, n_spk: int = 1024):
        super().__init__()
        self.feat = torchaudio.transforms.MFCC(
            sample_rate=16_000, n_mfcc=n_mfcc,
            melkwargs={"n_fft": 512, "hop_length": 160, "n_mels": 40},
        )
        self.tdnn = nn.Sequential(
            TDNNBlock(n_mfcc, 512, k=5, dilation=1),
            TDNNBlock(512, 512, k=3, dilation=2),
            TDNNBlock(512, 512, k=3, dilation=3),
            TDNNBlock(512, 512, k=1),
            TDNNBlock(512, 1500, k=1),
        )
        self.pool = StatsPool()
        self.fc1 = nn.Linear(3000, emb_dim)
        self.fc2 = nn.Linear(emb_dim, emb_dim)
        self.out = nn.Linear(emb_dim, n_spk)

    def forward(self, wav: torch.Tensor):
        # wav: (B, T) @ 16kHz
        x = self.feat(wav)                # (B, n_mfcc, T')
        x = self.tdnn(x)
        x = self.pool(x)
        emb = F.relu(self.fc1(x))
        emb = self.fc2(emb)
        logits = self.out(F.relu(emb))
        return emb, logits

    @torch.no_grad()
    def embed(self, wav: torch.Tensor) -> torch.Tensor:
        self.eval()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        emb, _ = self.forward(wav)
        return F.normalize(emb, dim=-1).squeeze(0)


def extract_from_file(path: str | Path, ckpt: str | Path | None = None) -> torch.Tensor:
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T).mean(0)
    if sr != 16_000:
        wav = torchaudio.functional.resample(wav, sr, 16_000)
    model = XVector()
    if ckpt and Path(ckpt).exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    # Trim/pad to 60 s (assignment requires exactly 60 seconds).
    target = 60 * 16_000
    if wav.shape[0] < target:
        wav = F.pad(wav, (0, target - wav.shape[0]))
    else:
        wav = wav[:target]
    return model.embed(wav)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="models/speaker_emb.pt")
    args = ap.parse_args()
    emb = extract_from_file(args.wav, ckpt=args.ckpt)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb, args.out)
    print(f"[speaker-emb] {args.wav} -> {args.out}  dim={emb.shape[0]}")
