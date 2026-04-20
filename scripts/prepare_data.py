"""Download the source lecture from YouTube, slice training + inference clips.

Usage:
    python scripts/prepare_data.py

Produces:
    data/raw/source.wav              # full audio, 16-kHz mono
    data/train_clips/clip_XX.wav     # 10-min training slices (<2h mark)
    data/infer/original_segment.wav  # 10-min inference clip [2:20:00, 2:30:00]
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

YT_URL = "https://youtu.be/ZPUtA3W-7_I"

# (start_hms, end_hms, basename) – 10-minute clips, avoiding the inference range.
TRAIN_RANGES = [
    ("00:00:30", "00:10:30", "clip_00"),
    ("00:30:00", "00:40:00", "clip_01"),
    ("01:00:00", "01:10:00", "clip_02"),
    ("01:30:00", "01:40:00", "clip_03"),
    ("02:00:00", "02:10:00", "clip_04"),
]
INFER_RANGE = ("02:20:00", "02:30:00", "original_segment")


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


def download(url: str, out_wav: Path) -> None:
    if out_wav.exists() and out_wav.stat().st_size > 10_000_000:
        print(f"[skip] {out_wav} already present ({out_wav.stat().st_size/1e6:.1f} MB)")
        return
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    tmp_stem = out_wav.with_suffix("")
    run([
        "yt-dlp",
        # prefer m4a medium (140) then webm opus medium (251), then any audio
        "-f", "140/251/bestaudio",
        "--extract-audio",
        "--audio-format", "wav", "--audio-quality", "0",
        "--postprocessor-args", "ffmpeg:-ac 1 -ar 16000",
        "-o", f"{tmp_stem}.%(ext)s", url,
    ])


def slice_clip(src: Path, start: str, end: str, out: Path) -> None:
    if out.exists() and out.stat().st_size > 1_000_000:
        print(f"[skip] {out} already present")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg", "-y", "-ss", start, "-to", end, "-i", str(src),
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(out),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=YT_URL)
    ap.add_argument("--raw", default="data/raw/source.wav")
    ap.add_argument("--train-dir", default="data/train_clips")
    ap.add_argument("--infer-dir", default="data/infer")
    args = ap.parse_args()

    raw = Path(args.raw)
    download(args.url, raw)

    for start, end, name in TRAIN_RANGES:
        slice_clip(raw, start, end, Path(args.train_dir) / f"{name}.wav")

    s, e, name = INFER_RANGE
    slice_clip(raw, s, e, Path(args.infer_dir) / f"{name}.wav")

    print("\n[done] source:", raw)
    print("       training clips:", args.train_dir)
    print("       inference clip:", Path(args.infer_dir) / f"{name}.wav")


if __name__ == "__main__":
    main()
