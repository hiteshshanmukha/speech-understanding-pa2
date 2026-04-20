"""Training harness for the multi-head LID.

The harness accepts a JSONL manifest where every row is:

    {"wav": "path/to/clip.wav",
     "segments": [[start_s, end_s, "EN"|"HI"|"SIL"], ...]}

It works with real data (Hindi-English code-switched corpora such as
MUCS 2021 or the IIT-M Code-Switch) as well as with the synthetic
mixer below, which lets us smoke-test the whole stack without network
access.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from .lid import (
    FRAME_HOP_MS, L2I, MultiHeadLID, TrainCfg, multitask_loss, save,
    switch_boundary_f1,
)
from sklearn.metrics import f1_score


# -------------------------- manifest dataset ---------------------------

class ManifestLID(Dataset):
    """Each __getitem__ returns a random 5-s crop of a source clip.

    `samples_per_epoch` multiplies the logical dataset length so that
    one epoch sees many random crops even when the manifest has few
    (long) clips – without this, 5 × 10-min clips produce 5 crops per
    epoch which is too few for batched SGD.
    """

    def __init__(self, manifest: str | Path, sr: int = 16_000,
                 clip_seconds: float = 5.0, samples_per_epoch: int = 512):
        self.rows = [json.loads(l) for l in Path(manifest).read_text("utf-8").splitlines() if l.strip()]
        self.sr = sr
        self.clip = int(clip_seconds * sr)
        self.hop = int(sr * FRAME_HOP_MS / 1000)
        self.samples_per_epoch = max(samples_per_epoch, len(self.rows))

    def __len__(self):
        return self.samples_per_epoch

    def _segments_to_frames(self, segs, n_frames):
        y = np.full(n_frames, -100, dtype=np.int64)      # ignore gaps
        for s, e, lbl in segs:
            i0 = int(s * self.sr // self.hop)
            i1 = int(e * self.sr // self.hop)
            y[i0:i1] = L2I[lbl]
        return y

    def __getitem__(self, idx):
        import soundfile as sf
        row = self.rows[idx % len(self.rows)]
        data, sr = sf.read(row["wav"], dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T).mean(0)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        # random crop to fixed length
        if wav.shape[0] > self.clip:
            i = random.randint(0, wav.shape[0] - self.clip)
            wav = wav[i:i + self.clip]
            seg_off = i / self.sr
            segs = [[max(0, s - seg_off), e - seg_off, l]
                    for s, e, l in row["segments"]
                    if e - seg_off > 0 and s - seg_off < self.clip / self.sr]
        else:
            pad = self.clip - wav.shape[0]
            wav = torch.nn.functional.pad(wav, (0, pad))
            segs = row["segments"]
        n_frames = wav.shape[0] // self.hop
        y_frame = self._segments_to_frames(segs, n_frames)
        # majority-vote utterance label ignoring SIL
        valid = y_frame[y_frame >= 0]
        y_utt = int(np.bincount(valid, minlength=len(L2I)).argmax()) if valid.size else L2I["SIL"]
        return wav.float(), torch.from_numpy(y_frame), torch.tensor(y_utt)


def collate(batch):
    wavs, y_frames, y_utts = zip(*batch)
    return (torch.stack(wavs), torch.stack(y_frames), torch.stack(y_utts))


# ----------------------- synthetic data generator ----------------------

def make_synth_clip(dur_s: float = 5.0, sr: int = 16_000, seed: int = 0):
    """Two-tone-plus-formant synthetic clip: EN=~200 Hz F0, HI=~300 Hz F0.

    This is ONLY to let the training loop run without a real corpus.
    """
    rng = np.random.default_rng(seed)
    n = int(dur_s * sr)
    x = np.zeros(n, dtype=np.float32)
    segments = []
    t = 0.0
    while t < dur_s - 0.3:
        dur = rng.uniform(0.4, 1.2)
        end = min(dur_s, t + dur)
        lbl = rng.choice(["EN", "HI"])
        i0, i1 = int(t * sr), int(end * sr)
        f0 = 200.0 if lbl == "EN" else 310.0
        phase = np.linspace(0, 2 * math.pi * f0 * (end - t), i1 - i0, endpoint=False)
        x[i0:i1] = 0.3 * np.sin(phase) + 0.1 * np.sin(2 * phase) + 0.05 * rng.standard_normal(i1 - i0)
        segments.append([float(t), float(end), lbl])
        t = end
    return torch.from_numpy(x), segments


def build_synth_manifest(out_dir: str | Path, n: int = 64, seed: int = 0):
    import soundfile as sf
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        wav, segs = make_synth_clip(seed=seed + i)
        path = out_dir / f"synth_{i:04d}.wav"
        sf.write(str(path), wav.numpy(), 16_000, subtype="PCM_16")
        rows.append({"wav": str(path), "segments": segs})
    manifest = out_dir / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows), "utf-8")
    return manifest


# ------------------------------- training ------------------------------

def train(manifest: str | Path, ckpt: str | Path, cfg: TrainCfg = TrainCfg(),
          device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    ds = ManifestLID(manifest)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    collate_fn=collate, num_workers=0)
    model = MultiHeadLID().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    print(f"[lid-train] {len(ds)} clips, cfg={asdict(cfg)}, device={device}")

    for epoch in range(cfg.epochs):
        model.train()
        losses = []
        for wav, y_frame, y_utt in dl:
            wav, y_frame, y_utt = wav.to(device), y_frame.to(device), y_utt.to(device)
            # Align label frames to model output frames. MelSpectrogram
            # with center=True can emit one extra frame, so we either
            # trim or right-pad the labels to match.
            out = model(wav)
            T_out = out["frame_logits"].shape[1]
            if y_frame.shape[1] < T_out:
                pad = T_out - y_frame.shape[1]
                y_frame = torch.nn.functional.pad(y_frame, (0, pad), value=-100)
            else:
                y_frame = y_frame[:, :T_out]
            loss, parts = multitask_loss(out, y_frame, y_utt, cfg)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            losses.append(loss.item())
        sched.step()

        f1, bf1 = eval_epoch(model, dl, device)
        print(f"epoch {epoch:02d}  loss={np.mean(losses):.3f}  frame-F1={f1:.3f}  boundary-F1={bf1:.3f}")

    save(model, ckpt)
    print(f"[lid-train] saved -> {ckpt}")


@torch.no_grad()
def eval_epoch(model, dl, device):
    model.eval()
    all_pred, all_gt = [], []
    bf1s = []
    for wav, y_frame, _ in dl:
        wav, y_frame = wav.to(device), y_frame.to(device)
        pred = model.predict_frames(wav, median_k=3)
        T = min(pred.shape[1], y_frame.shape[1])
        mask = y_frame[:, :T] >= 0
        all_pred.append(pred[:, :T][mask].cpu().numpy())
        all_gt.append(y_frame[:, :T][mask].cpu().numpy())
        for p, g in zip(pred, y_frame):
            bf1s.append(switch_boundary_f1(p[:T].cpu(), g[:T].cpu()))
    p = np.concatenate(all_pred)
    g = np.concatenate(all_gt)
    f1 = f1_score(g, p, average="macro")
    return f1, float(np.mean(bf1s))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="JSONL manifest; omit to use synth")
    ap.add_argument("--ckpt", default="models/lid.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--synth-dir", default="data/synth_lid")
    ap.add_argument("--synth-n", type=int, default=64)
    args = ap.parse_args()

    manifest = args.manifest or build_synth_manifest(args.synth_dir, n=args.synth_n)
    train(manifest, args.ckpt, TrainCfg(epochs=args.epochs))
