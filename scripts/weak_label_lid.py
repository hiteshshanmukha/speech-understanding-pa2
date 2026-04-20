"""Weak-label training clips for LID using Whisper language detection.

Batched version: we collect N non-overlapping 3-s chunks of audio,
stack their log-mel features into one batch, run Whisper's encoder
once, then a single decoder step to read the language-token logits.

This is ~10× faster than a per-window loop and lets us process each
10-min clip in < 30 s on CPU with whisper-tiny.

Output manifest (`data/train_clips/manifest.jsonl`) is compatible
with `lid_train.ManifestLID`:

    {"wav": "path.wav", "segments": [[start_s, end_s, "EN"|"HI"|"SIL"], ...]}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


WINDOW_S = 3.0           # chunk size for language detection
BATCH = 16               # how many chunks per encoder call
SIL_RMS = 0.005


def load_whisper(model_id: str, device: str):
    from transformers import AutoProcessor, WhisperForConditionalGeneration
    proc = AutoProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device).eval()
    en_id = proc.tokenizer.convert_tokens_to_ids("<|en|>")
    hi_id = proc.tokenizer.convert_tokens_to_ids("<|hi|>")
    sot_id = proc.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    return proc, model, en_id, hi_id, sot_id


@torch.no_grad()
def batched_detect(proc, model, sot_id: int, en_id: int, hi_id: int,
                   chunks: list[np.ndarray], device: str) -> list[tuple[float, float]]:
    """Return [(p_en, p_hi), ...] aligned with input chunks."""
    if not chunks:
        return []
    results: list[tuple[float, float]] = []
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        inputs = proc(batch, sampling_rate=16_000, return_tensors="pt")
        feats = inputs["input_features"].to(device)           # (B, 80, 3000)
        enc = model.get_encoder()(feats).last_hidden_state     # (B, T, D)
        dec = torch.full((feats.shape[0], 1), sot_id,
                         dtype=torch.long, device=device)
        logits = model(encoder_outputs=(enc,),
                       decoder_input_ids=dec).logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        for p in probs:
            results.append((float(p[en_id]), float(p[hi_id])))
    return results


def label_clip(wav_path: Path, proc, model, sot_id, en_id, hi_id,
               device: str) -> list[list]:
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    wav = data.mean(axis=1)
    if sr != 16_000:
        import torchaudio
        w = torch.from_numpy(wav).unsqueeze(0)
        w = torchaudio.functional.resample(w, sr, 16_000).squeeze(0).numpy()
        wav, sr = w, 16_000

    win = int(WINDOW_S * sr)
    chunks, bounds, silent_mask = [], [], []
    for i in range(0, wav.shape[0] - win + 1, win):
        chunk = wav[i:i + win]
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        bounds.append((i / sr, (i + win) / sr))
        if rms < SIL_RMS:
            silent_mask.append(True)
            chunks.append(None)
        else:
            silent_mask.append(False)
            chunks.append(chunk)

    nonsil_chunks = [c for c in chunks if c is not None]
    probs = batched_detect(proc, model, sot_id, en_id, hi_id, nonsil_chunks, device)

    segs: list[list] = []
    p_iter = iter(probs)
    for (t0, t1), chunk in zip(bounds, chunks):
        if chunk is None:
            lbl = "SIL"
        else:
            p_en, p_hi = next(p_iter)
            lbl = "EN" if p_en >= p_hi else "HI"
        if segs and segs[-1][2] == lbl and abs(segs[-1][1] - t0) < 1e-3:
            segs[-1][1] = t1
        else:
            segs.append([t0, t1, lbl])
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-dir", default="data/train_clips")
    ap.add_argument("--out", default="data/train_clips/manifest.jsonl")
    ap.add_argument("--model", default="openai/whisper-tiny")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    proc, model, en_id, hi_id, sot_id = load_whisper(args.model, args.device)
    print(f"[weak-label] whisper={args.model} device={args.device}  batch={BATCH}")

    rows = []
    for wav in sorted(Path(args.clips_dir).glob("*.wav")):
        print(f"[weak-label] {wav.name} ...", flush=True)
        segs = label_clip(wav, proc, model, sot_id, en_id, hi_id, args.device)
        counts = {"EN": 0.0, "HI": 0.0, "SIL": 0.0}
        for s, e, l in segs:
            counts[l] += e - s
        print(f"   {wav.name}: segments={len(segs)}  EN={counts['EN']:.0f}s"
              f"  HI={counts['HI']:.0f}s  SIL={counts['SIL']:.0f}s", flush=True)
        rows.append({"wav": str(wav), "segments": segs})

    Path(args.out).write_text("\n".join(json.dumps(r) for r in rows), "utf-8")
    print(f"[weak-label] -> {args.out}  ({len(rows)} clips)")


if __name__ == "__main__":
    main()
