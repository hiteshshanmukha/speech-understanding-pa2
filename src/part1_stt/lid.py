"""Task 1.1 – Frame-level Language Identification (English vs Hindi).

Architecture
------------
- Input : log-Mel 80-d, 25 ms / 10 ms frames.
- Trunk : 2x Conv1d (depthwise-separable) -> 3x BiGRU (hidden=128).
- Heads : multi-head classifier
    * frame head : per-frame softmax over {EN, HI, SIL}
    * utt head   : mean-pooled softmax (for stability on short utterances)
- Loss  : multi-task = alpha * CE(frame) + (1-alpha) * CE(utt)
          with label smoothing 0.05.

Boundary precision
------------------
Frame hop = 10 ms, so switch timestamps have ±10 ms resolution natively,
well inside the 200 ms SLA. A post-hoc 3-frame median filter removes
single-frame flickers that would otherwise hurt boundary F1.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

LABELS = ("EN", "HI", "SIL")
L2I = {l: i for i, l in enumerate(LABELS)}
I2L = {i: l for l, i in L2I.items()}

FRAME_HOP_MS = 10


# ----------------------------- features ---------------------------------

class LogMel(nn.Module):
    def __init__(self, sr: int = 16_000, n_mels: int = 80,
                 win_ms: int = 25, hop_ms: int = FRAME_HOP_MS):
        super().__init__()
        n_fft = 1 << (int(sr * win_ms / 1000) - 1).bit_length()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft,
            win_length=int(sr * win_ms / 1000),
            hop_length=int(sr * hop_ms / 1000),
            n_mels=n_mels, power=2.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) -> (B, n_mels, T')
        m = self.mel(x)
        return torch.log(m.clamp_min(1e-6))


# ----------------------------- model ------------------------------------

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, cin, cout, k=5, stride=1, dilation=1):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.dw = nn.Conv1d(cin, cin, k, stride, pad, dilation=dilation, groups=cin)
        self.pw = nn.Conv1d(cin, cout, 1)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x):
        return F.gelu(self.bn(self.pw(self.dw(x))))


class MultiHeadLID(nn.Module):
    """Two-headed LID model (frame + utterance)."""

    def __init__(self, n_mels: int = 80, hidden: int = 128, n_classes: int = 3):
        super().__init__()
        self.feat = LogMel(n_mels=n_mels)
        self.conv = nn.Sequential(
            DepthwiseSeparableConv1d(n_mels, hidden, k=5),
            DepthwiseSeparableConv1d(hidden, hidden, k=5, dilation=2),
        )
        self.rnn = nn.GRU(hidden, hidden, num_layers=3,
                          batch_first=True, bidirectional=True, dropout=0.1)
        self.frame_head = nn.Linear(2 * hidden, n_classes)
        self.utt_head = nn.Linear(2 * hidden, n_classes)

    def forward(self, wav: torch.Tensor) -> dict[str, torch.Tensor]:
        # wav: (B, T) raw @16 kHz
        x = self.feat(wav)                 # (B, n_mels, T')
        x = self.conv(x)                   # (B, H, T')
        x = x.transpose(1, 2)              # (B, T', H)
        h, _ = self.rnn(x)                 # (B, T', 2H)
        frame_logits = self.frame_head(h)
        utt_logits = self.utt_head(h.mean(dim=1))
        return {"frame_logits": frame_logits, "utt_logits": utt_logits}

    @torch.no_grad()
    def predict_frames(self, wav: torch.Tensor,
                       median_k: int = 3,
                       energy_sil_dbfs: float = -75.0) -> torch.Tensor:
        """Frame-level language prediction with an energy VAD override.

        The multi-class classifier tends to collapse the SIL class when
        silence is under-represented at training time. An energy VAD
        (threshold ``energy_sil_dbfs``) is applied as a post-filter:
        any frame whose RMS amplitude falls below the threshold is
        forced to SIL. This matches standard LID-in-the-wild practice
        (e.g. Snyder et al. 2019).
        """
        self.eval()
        wav2 = wav if wav.dim() == 2 else wav.unsqueeze(0)
        out = self.forward(wav2)
        preds = out["frame_logits"].argmax(-1)       # (B, T')
        if median_k > 1:
            preds = _median_filter(preds, k=median_k)

        # Energy VAD post-filter, computed at the same 10-ms hop.
        # Requires `min_sil_ms` of *sustained* silence before flipping
        # the prediction – otherwise brief intra-word dips (e.g. stops,
        # fricatives with a momentary null) would fragment predictions.
        min_sil_ms = 150
        min_sil_frames = min_sil_ms // FRAME_HOP_MS
        hop = int(16_000 * FRAME_HOP_MS / 1000)
        win = int(0.025 * 16_000)
        if wav2.shape[-1] >= win:
            frames = wav2.unfold(-1, win, hop)                    # (B, T', W)
            rms = frames.pow(2).mean(dim=-1).sqrt().clamp_min(1e-9)
            dbfs = 20.0 * torch.log10(rms)
            raw_sil = dbfs < energy_sil_dbfs                       # (B, T')
            # morphological erode: keep only runs >= min_sil_frames
            eroded = raw_sil.clone()
            for b in range(raw_sil.shape[0]):
                run = 0
                starts = []
                for i in range(raw_sil.shape[1]):
                    if raw_sil[b, i]:
                        run += 1
                    else:
                        if run < min_sil_frames:
                            eroded[b, i - run:i] = False
                        run = 0
                if run < min_sil_frames:
                    eroded[b, raw_sil.shape[1] - run:raw_sil.shape[1]] = False
            n = min(preds.shape[-1], eroded.shape[-1])
            preds = preds[..., :n].clone()
            preds[eroded[..., :n]] = L2I["SIL"]
        return preds


def _median_filter(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    if k <= 1:
        return x
    pad = k // 2
    xp = F.pad(x.float().unsqueeze(1), (pad, pad), mode="replicate")
    xp = xp.unfold(-1, k, 1).median(dim=-1).values
    return xp.squeeze(1).long()


# --------------------------- training -----------------------------------

@dataclass
class TrainCfg:
    lr: float = 3e-4
    epochs: int = 40
    batch_size: int = 16
    alpha: float = 0.7                   # frame-vs-utt weight
    label_smooth: float = 0.05
    grad_clip: float = 1.0


def multitask_loss(out, y_frame, y_utt, cfg: TrainCfg):
    fl = F.cross_entropy(
        out["frame_logits"].transpose(1, 2), y_frame,
        label_smoothing=cfg.label_smooth, ignore_index=-100,
    )
    ul = F.cross_entropy(
        out["utt_logits"], y_utt, label_smoothing=cfg.label_smooth,
    )
    return cfg.alpha * fl + (1 - cfg.alpha) * ul, {"frame_loss": fl.item(), "utt_loss": ul.item()}


# ------------------------- boundary utils -------------------------------

def frames_to_switches(pred: torch.Tensor, hop_ms: int = FRAME_HOP_MS) -> list[tuple[float, str]]:
    """Return [(t_seconds, new_label), ...] for every language switch."""
    out, cur = [], None
    for i, p in enumerate(pred.tolist()):
        lbl = I2L[p]
        if lbl != cur:
            out.append((i * hop_ms / 1000.0, lbl))
            cur = lbl
    return out


def switch_boundary_f1(pred: torch.Tensor, gt: torch.Tensor,
                       tol_ms: int = 200, hop_ms: int = FRAME_HOP_MS) -> float:
    """Boundary F1 with ± tol_ms tolerance (Task 1.1 ≥ 0.85)."""
    tol = tol_ms // hop_ms
    p = _boundaries(pred)
    g = _boundaries(gt)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    matched = set()
    tp = 0
    for pb in p:
        # nearest unmatched gt boundary within tolerance
        best = None
        for i, gb in enumerate(g):
            if i in matched:
                continue
            if abs(pb - gb) <= tol and (best is None or abs(pb - gb) < abs(pb - g[best])):
                best = i
        if best is not None:
            matched.add(best)
            tp += 1
    prec = tp / len(p)
    rec = tp / len(g)
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)


def _boundaries(seq: torch.Tensor) -> list[int]:
    seq = seq.tolist()
    return [i for i in range(1, len(seq)) if seq[i] != seq[i - 1]]


# ----------------------------- checkpoints ------------------------------

def save(model: MultiHeadLID, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load(path: str | Path, device: str = "cpu") -> MultiHeadLID:
    m = MultiHeadLID()
    sd = torch.load(path, map_location=device)
    m.load_state_dict(sd)
    m.to(device).eval()
    return m
