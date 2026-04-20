"""Task 1.3 – Denoising & Normalization.

Primary path: DeepFilterNet (`deepfilternet`) when the package is importable.
Fallback: multi-band spectral subtraction with a Wiener-style gain floor.
Both return a torch.float32 waveform at the input sample rate.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torchaudio


def _noise_estimate(mag: torch.Tensor, n_init_frames: int = 6) -> torch.Tensor:
    """Estimate noise magnitude from the first `n_init_frames` frames."""
    n_init_frames = min(n_init_frames, mag.shape[-1])
    return mag[..., :n_init_frames].mean(dim=-1, keepdim=True)


def spectral_subtraction(
    waveform: torch.Tensor,
    sr: int = 16_000,
    n_fft: int = 512,
    hop: int = 128,
    over_sub: float = 1.8,
    floor: float = 0.05,
) -> torch.Tensor:
    """Classic Boll (1979) spectral subtraction with a Wiener floor.

    Parameters
    ----------
    waveform : (T,) float32
    over_sub : over-subtraction factor alpha (typ. 1.5–2.5 for babble).
    floor    : spectral floor beta to avoid musical noise.
    """
    assert waveform.dim() == 1, "Expected (T,) mono waveform"
    window = torch.hann_window(n_fft, device=waveform.device)
    spec = torch.stft(
        waveform, n_fft=n_fft, hop_length=hop, window=window,
        return_complex=True, center=True,
    )
    mag = spec.abs()
    phase = torch.angle(spec)

    noise_mag = _noise_estimate(mag)                    # (F,1)
    sub = mag - over_sub * noise_mag
    # Wiener-style flooring: keep a fraction of the original magnitude.
    floored = torch.maximum(sub, floor * mag)

    cleaned = torch.polar(floored, phase)
    y = torch.istft(
        cleaned, n_fft=n_fft, hop_length=hop, window=window,
        length=waveform.shape[-1],
    )
    # peak-normalise to -1 dBFS to counter the gain loss from subtraction
    peak = y.abs().max().clamp_min(1e-6)
    return (y / peak) * math.pow(10.0, -1.0 / 20.0)


def denoise(waveform: torch.Tensor, sr: int = 16_000, backend: str = "auto") -> torch.Tensor:
    """Denoise `waveform` using the best available backend.

    backend: 'deepfilternet' | 'spectral' | 'auto'
    """
    if backend in ("auto", "deepfilternet"):
        try:
            from df.enhance import enhance, init_df  # type: ignore
            model, df_state, _ = init_df()
            x = waveform.unsqueeze(0).cpu().numpy().astype(np.float32)
            # DeepFilterNet expects 48 kHz internally; it resamples automatically.
            y = enhance(model, df_state, x)
            return torch.from_numpy(y).squeeze(0).float()
        except Exception:
            if backend == "deepfilternet":
                raise
    return spectral_subtraction(waveform, sr=sr)


def normalize_loudness(waveform: torch.Tensor, target_dbfs: float = -23.0) -> torch.Tensor:
    """Simple RMS normalisation to a target dBFS (EBU-R128 approximation)."""
    rms = waveform.pow(2).mean().sqrt().clamp_min(1e-6)
    target_rms = 10.0 ** (target_dbfs / 20.0)
    return waveform * (target_rms / rms)


if __name__ == "__main__":
    import argparse, soundfile as sf
    parser = argparse.ArgumentParser()
    parser.add_argument("input_wav")
    parser.add_argument("output_wav")
    parser.add_argument("--backend", default="auto")
    args = parser.parse_args()

    data, sr = sf.read(args.input_wav, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T).mean(0)
    cleaned = denoise(wav, sr=sr, backend=args.backend)
    cleaned = normalize_loudness(cleaned)
    sf.write(args.output_wav, cleaned.numpy(), sr, subtype="PCM_16")
    print(f"[denoise] {args.input_wav} -> {args.output_wav}  sr={sr}  backend={args.backend}")
