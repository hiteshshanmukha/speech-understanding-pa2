"""Build a balanced EN/HI LID training set.

English  : LibriSpeech dev-clean (direct tar download + FLAC via soundfile).
Hindi    : google/fleurs hi_in (HuggingFace datasets, with decode=False to
           avoid torchaudio/torchcodec and read with soundfile).

Each monolingual utterance is written as a 16-kHz mono PCM-16 WAV and
added to a JSONL manifest compatible with `lid_train.ManifestLID` as a
single full-length segment.

We intentionally treat LID as an acoustic task at training time
(pure-EN vs pure-HI) and let the frame-level median filter resolve
switches at inference. Training on monolingual corpora produces a
much cleaner decision boundary than Whisper-based weak labels on a
~97%-English lecture.
"""
from __future__ import annotations

import argparse
import io
import json
import random
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf


LS_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"


def _download(url: str, dest: Path, label: str = "") -> None:
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"[{label}] cache hit {dest} ({dest.stat().st_size/1e6:.0f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{label}] downloading {url}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        total = 0
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            if total % (10 * 1024 * 1024) == 0:
                print(f"[{label}]   {total/1e6:.0f} MB", flush=True)
    print(f"[{label}] done ({dest.stat().st_size/1e6:.0f} MB)")


def _extract_ls(tarball: Path, root: Path) -> Path:
    """Extract (or reuse) LibriSpeech tar; returns dev-clean root."""
    ls_root = root / "LibriSpeech" / "dev-clean"
    if ls_root.exists() and any(ls_root.rglob("*.flac")):
        print(f"[EN] already extracted -> {ls_root}")
        return ls_root
    print(f"[EN] extracting {tarball}")
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(root)
    return ls_root


def _resample(wav: np.ndarray, sr: int, target: int = 16_000) -> np.ndarray:
    if sr == target:
        return wav
    import torch, torchaudio
    t = torch.from_numpy(wav).unsqueeze(0)
    t = torchaudio.functional.resample(t, sr, target).squeeze(0)
    return t.numpy()


def prepare_english(out_dir: Path, n_clips: int, min_dur: float = 3.0) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path("data/_cache_librispeech")
    root.mkdir(parents=True, exist_ok=True)
    tarball = root / "dev-clean.tar.gz"
    _download(LS_URL, tarball, label="EN")
    ls_root = _extract_ls(tarball, root)
    flacs = sorted(ls_root.rglob("*.flac"))
    print(f"[EN] flac files found: {len(flacs)}")

    rng = random.Random(0)
    rng.shuffle(flacs)
    paths: list[str] = []
    for flac in flacs:
        try:
            wav, sr = sf.read(str(flac), dtype="float32", always_2d=False)
        except Exception as exc:
            print(f"[EN] skip {flac.name}: {exc}")
            continue
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        if wav.shape[0] / sr < min_dur:
            continue
        wav = _resample(wav, sr)
        out = out_dir / f"{flac.stem}.wav"
        sf.write(str(out), wav.astype(np.float32), 16_000, subtype="PCM_16")
        paths.append(str(out))
        if len(paths) >= n_clips:
            break
    print(f"[EN] wrote {len(paths)} clips -> {out_dir}")
    return paths


def prepare_hindi(out_dir: Path, n_clips: int, min_dur: float = 3.0) -> list[str]:
    from datasets import load_dataset, Audio

    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("google/fleurs", "hi_in", split="train",
                      cache_dir="data/_cache_fleurs")
    # Return raw file paths / bytes so we can decode with soundfile
    # (torchaudio fails on Windows without FFmpeg/torchcodec).
    ds = ds.cast_column("audio", Audio(decode=False))
    print(f"[HI] fleurs hi_in size={len(ds)}")
    paths: list[str] = []
    for i, row in enumerate(ds):
        audio = row["audio"]
        src = audio.get("path") or None
        buf = audio.get("bytes") or None
        try:
            if src and Path(src).exists():
                wav, sr = sf.read(src, dtype="float32", always_2d=False)
            elif buf:
                wav, sr = sf.read(io.BytesIO(buf), dtype="float32", always_2d=False)
            else:
                continue
        except Exception as exc:
            print(f"[HI] skip row {i}: {exc}")
            continue
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        if wav.shape[0] / sr < min_dur:
            continue
        wav = _resample(wav, sr)
        p = out_dir / f"hi_{i:06d}.wav"
        sf.write(str(p), wav.astype(np.float32), 16_000, subtype="PCM_16")
        paths.append(str(p))
        if len(paths) >= n_clips:
            break
    print(f"[HI] wrote {len(paths)} clips -> {out_dir}")
    return paths


def build_manifest(en_paths: list[str], hi_paths: list[str], out: Path):
    rows = []
    for p in en_paths:
        d, sr = sf.read(p, dtype="float32", always_2d=True)
        dur = d.shape[0] / sr
        rows.append({"wav": p, "segments": [[0.0, dur, "EN"]]})
    for p in hi_paths:
        d, sr = sf.read(p, dtype="float32", always_2d=True)
        dur = d.shape[0] / sr
        rows.append({"wav": p, "segments": [[0.0, dur, "HI"]]})
    random.Random(0).shuffle(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows), "utf-8")
    print(f"[manifest] rows={len(rows)}  ({len(en_paths)} EN + {len(hi_paths)} HI)  -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-lang", type=int, default=400)
    ap.add_argument("--out-dir", default="data/lid_train")
    ap.add_argument("--manifest", default="data/lid_train/manifest.jsonl")
    args = ap.parse_args()

    en_dir = Path(args.out_dir) / "en"
    hi_dir = Path(args.out_dir) / "hi"
    en = prepare_english(en_dir, args.n_per_lang)
    hi = prepare_hindi(hi_dir, args.n_per_lang)
    build_manifest(en, hi, Path(args.manifest))


if __name__ == "__main__":
    main()
