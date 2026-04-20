"""Build synthetic code-switched LID training clips.

Concatenates LibriSpeech EN + FLEURS hi_in HI utterances in random
order with real silence gaps so the model sees:
  * intra-clip language switches (the missing ingredient last time)
  * silence / noise between utterances (mapped to the SIL class)

Each synthetic clip is 30-60 s long, written as 16-kHz mono PCM-16
WAV, with a precise per-segment label list attached.

Output:
    data/lid_cs_train/clip_XXXX.wav
    data/lid_cs_train/manifest.jsonl    (same shape as ManifestLID)
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def iter_librispeech() -> list[Path]:
    root = Path("data/_cache_librispeech/LibriSpeech/dev-clean")
    return sorted(root.rglob("*.flac"))


def iter_fleurs():
    from datasets import load_dataset, Audio
    ds = load_dataset("google/fleurs", "hi_in", split="train",
                      cache_dir="data/_cache_fleurs")
    return list(ds.cast_column("audio", Audio(decode=False)))


def read_clip(src, min_dur: float = 2.0,
              max_dur: float = 7.0) -> np.ndarray | None:
    if isinstance(src, Path):
        try:
            w, sr = sf.read(str(src), dtype="float32", always_2d=False)
        except Exception:
            return None
    else:
        audio = src["audio"]
        try:
            if audio.get("path") and Path(audio["path"]).exists():
                w, sr = sf.read(audio["path"], dtype="float32", always_2d=False)
            else:
                w, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
        except Exception:
            return None
    if w.ndim > 1:
        w = w.mean(axis=-1)
    if sr != 16_000:
        import torch, torchaudio
        t = torch.from_numpy(w).unsqueeze(0)
        t = torchaudio.functional.resample(t, sr, 16_000).squeeze(0)
        w = t.numpy()
    if w.shape[0] / 16_000 < min_dur:
        return None
    if w.shape[0] / 16_000 > max_dur:
        start = random.randint(0, w.shape[0] - int(max_dur * 16_000))
        w = w[start:start + int(max_dur * 16_000)]
    # peak normalise each utterance so concatenation doesn't jump
    peak = float(np.max(np.abs(w)) or 1.0)
    return (w / peak * 0.7).astype(np.float32)


def make_silence(sr: int, dur: float, noise_db: float = -40.0) -> np.ndarray:
    """Low-level Gaussian noise (not pure digital zero) – more realistic SIL."""
    n = int(dur * sr)
    amp = 10.0 ** (noise_db / 20.0)
    return (amp * np.random.randn(n)).astype(np.float32)


def build_clip(en_pool: list[Path], hi_pool: list[dict],
               target_s: float, rng: random.Random,
               sr: int = 16_000) -> tuple[np.ndarray, list[list]] | None:
    segs: list[list] = []
    parts: list[np.ndarray] = []
    t = 0.0
    # start with an initial silence
    sil = make_silence(sr, rng.uniform(0.2, 0.5))
    parts.append(sil); segs.append([t, t + sil.shape[0] / sr, "SIL"]); t = segs[-1][1]

    # alternate labels but with random runs of 1-3 utterances per language
    lang = rng.choice(["EN", "HI"])
    while t < target_s:
        run = rng.randint(1, 3)
        for _ in range(run):
            src = rng.choice(en_pool) if lang == "EN" else rng.choice(hi_pool)
            w = read_clip(src)
            if w is None:
                continue
            parts.append(w); segs.append([t, t + w.shape[0] / sr, lang])
            t = segs[-1][1]
            # short intra-run pause
            sil = make_silence(sr, rng.uniform(0.10, 0.25))
            parts.append(sil); segs.append([t, t + sil.shape[0] / sr, "SIL"])
            t = segs[-1][1]
        # longer inter-run pause
        sil = make_silence(sr, rng.uniform(0.3, 0.6))
        parts.append(sil); segs.append([t, t + sil.shape[0] / sr, "SIL"])
        t = segs[-1][1]
        lang = "HI" if lang == "EN" else "EN"

    if not parts:
        return None
    return np.concatenate(parts).astype(np.float32), segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clips", type=int, default=120)
    ap.add_argument("--target-s", type=float, default=45.0)
    ap.add_argument("--out-dir", default="data/lid_cs_train")
    ap.add_argument("--manifest", default="data/lid_cs_train/manifest.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    en = iter_librispeech()
    rng.shuffle(en)
    en_pool = en[:800]

    hi_all = iter_fleurs()
    rng.shuffle(hi_all)
    hi_pool = hi_all[:800]

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(args.n_clips):
        clip = build_clip(en_pool, hi_pool, args.target_s, rng)
        if clip is None:
            continue
        wav, segs = clip
        path = out_dir / f"cs_{i:04d}.wav"
        sf.write(str(path), wav, 16_000, subtype="PCM_16")
        rows.append({"wav": str(path), "segments": segs})
        if (i + 1) % 10 == 0:
            print(f"[cs-train] {i+1}/{args.n_clips}", flush=True)

    Path(args.manifest).write_text("\n".join(json.dumps(r) for r in rows), "utf-8")
    print(f"[cs-train] wrote {len(rows)} clips -> {args.manifest}")


if __name__ == "__main__":
    main()
