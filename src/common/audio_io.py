"""Shared audio I/O + resampling helpers.

On Windows, torchaudio.load / .save try to go through torchcodec + FFmpeg
which often aren't shipped with the Python build. We therefore use
soundfile (libsndfile) as the default backend and expose a uniform
interface that returns torch.float32 tensors.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


TARGET_SR = 16_000
OUTPUT_SR = 22_050


def _sf_load(path: str | Path, mono: bool = True) -> tuple[torch.Tensor, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)                    # (C, T)
    if mono and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav, sr


def load_wav(path: str | Path, sr: int = TARGET_SR, mono: bool = True) -> torch.Tensor:
    """Load a WAV as a float32 tensor, shape (T,). Resample to `sr`."""
    wav, orig_sr = _sf_load(path, mono=mono)
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    return wav.squeeze(0).contiguous()


def load_wav_keep_sr(path: str | Path, mono: bool = True) -> tuple[torch.Tensor, int]:
    wav, sr = _sf_load(path, mono=mono)
    return wav.squeeze(0).contiguous(), sr


def save_wav(path: str | Path, wav: torch.Tensor | np.ndarray, sr: int = OUTPUT_SR) -> None:
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()
    wav = np.asarray(wav, dtype=np.float32)
    peak = float(np.max(np.abs(wav)) or 1.0)
    if peak > 1.0:
        wav = wav / peak
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, sr, subtype="PCM_16")


def snr_db(clean: torch.Tensor, noisy: torch.Tensor) -> float:
    """SNR between `clean` and its noisy counterpart (same length)."""
    clean = clean.float()
    noise = noisy.float() - clean
    num = torch.sum(clean ** 2) + 1e-12
    den = torch.sum(noise ** 2) + 1e-12
    return float(10.0 * torch.log10(num / den))
