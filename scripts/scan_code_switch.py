"""Scan a range of `data/raw/source.wav` for Hindi/English content.

Samples a 3-s clip every STRIDE seconds inside [START, END] and asks
Whisper which language it is. Prints a compact per-sample report plus
a suggestion for the 10-min window with the highest Hindi fraction.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="data/raw/source.wav")
    ap.add_argument("--start", type=float, default=2 * 3600 + 20 * 60,
                    help="start seconds (default 2:20:00)")
    ap.add_argument("--end", type=float, default=2 * 3600 + 54 * 60,
                    help="end seconds (default 2:54:00)")
    ap.add_argument("--stride", type=float, default=30.0,
                    help="seconds between probes")
    ap.add_argument("--win", type=float, default=5.0,
                    help="probe window length in seconds")
    ap.add_argument("--model", default="openai/whisper-small")
    args = ap.parse_args()

    from transformers import AutoProcessor, WhisperForConditionalGeneration
    proc = AutoProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model).eval()
    en_id = proc.tokenizer.convert_tokens_to_ids("<|en|>")
    hi_id = proc.tokenizer.convert_tokens_to_ids("<|hi|>")
    sot_id = proc.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

    info = sf.info(str(args.wav))
    sr = info.samplerate
    print(f"wav sr={sr}  dur={info.duration:.1f}s")
    assert sr == 16_000

    # Build batch of probe chunks
    probes = []
    ts = []
    t = args.start
    while t + args.win <= args.end:
        wav, _ = sf.read(str(args.wav), dtype="float32",
                         start=int(t * sr), stop=int((t + args.win) * sr),
                         always_2d=True)
        probes.append(wav.mean(axis=1).astype(np.float32))
        ts.append(t)
        t += args.stride

    print(f"probes: {len(probes)}")

    # Batched encoder + one-step decoder
    BATCH = 16
    results = []
    with torch.no_grad():
        for i in range(0, len(probes), BATCH):
            batch = probes[i:i + BATCH]
            feats = proc(batch, sampling_rate=16_000,
                         return_tensors="pt").input_features
            enc = model.get_encoder()(feats).last_hidden_state
            dec = torch.full((feats.shape[0], 1), sot_id, dtype=torch.long)
            logits = model(encoder_outputs=(enc,),
                           decoder_input_ids=dec).logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            for t, p in zip(ts[i:i + BATCH], probs):
                p_en = float(p[en_id]); p_hi = float(p[hi_id])
                results.append((t, p_en, p_hi))

    for t, p_en, p_hi in results:
        h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
        top = "HI" if p_hi > p_en else "EN"
        print(f"  {h:02d}:{m:02d}:{s:02d}  EN={p_en:.2f}  HI={p_hi:.2f}  -> {top}")

    # Slide a 10-min (600 s) window with the given stride and pick the
    # window with the highest Hindi fraction.
    stride = args.stride
    win_n = int(600 / stride)
    best = (0.0, args.start)
    for i in range(len(results) - win_n + 1):
        chunk = results[i:i + win_n]
        hi_frac = sum(1 for _, p_en, p_hi in chunk if p_hi > p_en) / len(chunk)
        if hi_frac > best[0]:
            best = (hi_frac, chunk[0][0])
    hi_frac, t0 = best
    h = int(t0 // 3600); m = int((t0 % 3600) // 60); s = int(t0 % 60)
    print(f"\n[best 10-min window] starts at {h:02d}:{m:02d}:{s:02d}  "
          f"Hindi fraction = {hi_frac*100:.0f}%")


if __name__ == "__main__":
    main()
