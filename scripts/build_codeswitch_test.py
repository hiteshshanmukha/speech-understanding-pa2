"""Build a synthetic Hindi/English code-switched test clip.

Uses held-out clips from the LID training corpus (the ones NOT picked
by prepare_lid_data.py) to avoid train/test leak. Alternates between
Hindi and English utterances with small pauses, targeting a specific
total duration, and writes:

    data/test_cs/synth_code_switch.wav   16-kHz mono WAV (large, gitignored)
    results/synth_segments.json          [[start_s, end_s, "EN"|"HI"], ...]
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def iter_librispeech_flacs():
    root = Path("data/_cache_librispeech/LibriSpeech/dev-clean")
    return sorted(root.rglob("*.flac"))


def iter_fleurs_rows():
    from datasets import load_dataset, Audio
    ds = load_dataset("google/fleurs", "hi_in", split="train",
                      cache_dir="data/_cache_fleurs")
    ds = ds.cast_column("audio", Audio(decode=False))
    return ds


def read_clip(src) -> tuple[np.ndarray, int]:
    if isinstance(src, Path):
        w, sr = sf.read(str(src), dtype="float32", always_2d=False)
    else:
        audio = src["audio"]
        if audio.get("path") and Path(audio["path"]).exists():
            w, sr = sf.read(audio["path"], dtype="float32", always_2d=False)
        else:
            w, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
    if w.ndim > 1:
        w = w.mean(axis=-1)
    if sr != 16_000:
        import torch, torchaudio
        t = torch.from_numpy(w).unsqueeze(0)
        t = torchaudio.functional.resample(t, sr, 16_000).squeeze(0)
        w = t.numpy()
    return w.astype(np.float32), 16_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-wav", default="data/test_cs/synth_code_switch.wav")
    ap.add_argument("--out-segs", default="results/synth_segments.json")
    ap.add_argument("--target-s", type=float, default=120.0,
                    help="Total target length in seconds")
    ap.add_argument("--pause-s", type=float, default=0.3,
                    help="Silence between switches")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    sr = 16_000
    pause = np.zeros(int(args.pause_s * sr), dtype=np.float32)

    ls = iter_librispeech_flacs()
    # use a shuffled tail so we avoid the prefix picked by prepare_lid_data.py
    rng.shuffle(ls); ls_pool = ls[400:800]
    hi = list(iter_fleurs_rows())
    rng.shuffle(hi); hi_pool = hi[400:800]

    audio: list[np.ndarray] = []
    segments: list[list] = []
    t_cur = 0.0
    k_en = k_hi = 0

    # Alternate EN, HI, EN, HI until we reach target
    lang = "EN"
    while t_cur < args.target_s:
        try:
            if lang == "EN":
                wav, _ = read_clip(ls_pool[k_en % len(ls_pool)]); k_en += 1
            else:
                wav, _ = read_clip(hi_pool[k_hi % len(hi_pool)]); k_hi += 1
        except Exception as exc:
            print(f"[warn] skip: {exc}")
            lang = "HI" if lang == "EN" else "EN"
            continue
        # Trim very long FLEURS clips so switches happen often enough
        max_len = int(sr * min(8.0, max(3.0, 10 - t_cur % 10)))
        if wav.shape[0] > max_len:
            wav = wav[:max_len]
        dur = wav.shape[0] / sr
        audio.append(wav)
        segments.append([float(t_cur), float(t_cur + dur), lang])
        t_cur += dur
        # pause
        audio.append(pause)
        segments.append([float(t_cur), float(t_cur + args.pause_s), "SIL"])
        t_cur += args.pause_s
        lang = "HI" if lang == "EN" else "EN"

    wav = np.concatenate(audio).astype(np.float32)
    # peak normalise
    peak = float(np.max(np.abs(wav)) or 1.0)
    wav = wav / peak * 0.9
    Path(args.out_wav).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out_wav, wav, sr, subtype="PCM_16")
    Path(args.out_segs).write_text(json.dumps(segments, indent=2))
    print(f"[synth] wrote {len(segments)} segments, {t_cur:.1f}s -> {args.out_wav}")


if __name__ == "__main__":
    main()
