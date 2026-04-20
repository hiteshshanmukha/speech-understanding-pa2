"""Evaluate LID + STT on the synthetic code-switched clip.

Prints:
  * frame-level F1 (macro) against ground-truth segments
  * boundary F1 with ±200 ms tolerance (Task 1.1 target ≥ 0.85)
  * code-switch confusion matrix (EN↔HI, EN↔SIL, HI↔SIL)
  * the Whisper+LM transcript
  * sample of IPA + Santhali translation
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.part1_stt.denoise import denoise, normalize_loudness
from src.part1_stt.lid import (
    FRAME_HOP_MS, I2L, L2I, frames_to_switches, load as load_lid,
    switch_boundary_f1,
)
from src.part2_phonetic.hinglish_g2p import text_to_ipa
from src.part2_phonetic.parallel_corpus import Entry, translate


def seg_to_frames(segments: list[list], n_frames: int, sr: int = 16_000) -> np.ndarray:
    hop = int(sr * FRAME_HOP_MS / 1000)
    y = np.full(n_frames, -100, dtype=np.int64)
    for s, e, lbl in segments:
        i0 = int(s * sr // hop); i1 = int(e * sr // hop)
        y[i0:i1] = L2I[lbl]
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="data/test_cs/synth_code_switch.wav")
    ap.add_argument("--segs", default="results/synth_segments.json")
    ap.add_argument("--lid", default="models/lid.pt")
    ap.add_argument("--lm", default="models/ngram_lm.pkl")
    ap.add_argument("--whisper", default="openai/whisper-small")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    data, sr = sf.read(args.wav, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T).mean(0)
    assert sr == 16_000

    # --- 1. denoise (for Whisper only; LID sees raw audio to match
    #        training-time distribution which was peak-normalised
    #        per utterance, not denoised/loudness-normalised)
    clean = normalize_loudness(denoise(wav))
    clean_path = out / "clean.wav"
    sf.write(clean_path, clean.numpy(), sr, subtype="PCM_16")

    # --- 2. LID on raw waveform ----------------------------------------
    lid = load_lid(args.lid)
    chunk = 30 * sr
    frames = []
    for i in range(0, wav.shape[0], chunk):
        pred = lid.predict_frames(wav[i:i + chunk].unsqueeze(0), median_k=3).squeeze(0)
        frames.append(pred)
    pred = torch.cat(frames)
    gt_segs = json.loads(Path(args.segs).read_text("utf-8"))
    gt_frames = torch.from_numpy(seg_to_frames(gt_segs, pred.shape[0], sr=sr))

    # frame F1 & boundary F1 (ignore frames labelled -100 = padding)
    mask = gt_frames >= 0
    p = pred[mask].cpu().numpy()
    g = gt_frames[mask].cpu().numpy()
    macro = f1_score(g, p, average="macro", labels=[0, 1, 2], zero_division=0)
    bF1 = switch_boundary_f1(pred[mask], gt_frames[mask])
    cm = confusion_matrix(g, p, labels=[0, 1, 2])

    report = {
        "frame_macro_F1": float(macro),
        "boundary_F1_200ms": float(bF1),
        "labels": ["EN", "HI", "SIL"],
        "confusion_matrix_rows_gt_cols_pred": cm.tolist(),
    }
    (out / "lid_eval.json").write_text(json.dumps(report, indent=2))
    print(f"[LID] frame macro-F1 = {macro:.3f}")
    print(f"[LID] boundary F1 (±200 ms) = {bF1:.3f}")
    print(f"[LID] confusion matrix (rows=gt EN/HI/SIL, cols=pred):")
    for row in cm:
        print("  ", row.tolist())

    switches = frames_to_switches(pred, hop_ms=FRAME_HOP_MS)
    (out / "switches.json").write_text(json.dumps(switches, indent=2))
    print(f"[LID] predicted switch points: {len(switches)}  (ground truth: {len(gt_segs)})")

    # --- 3. Whisper + LM bias -----------------------------------------
    print("[STT] transcribing with Whisper + LM bias...")
    from src.part1_stt.decode import TranscribeCfg, transcribe
    cfg = TranscribeCfg(model_id=args.whisper, lm_lambda=0.4, beam_size=5)
    text = transcribe(str(clean_path), args.lm, cfg=cfg)
    (out / "transcript.txt").write_text(text, "utf-8")

    # --- 4. IPA + translation ------------------------------------------
    ipa = text_to_ipa(text)
    (out / "ipa.txt").write_text(ipa, "utf-8")

    corpus_path = Path("data/parallel_corpus.json")
    if corpus_path.exists():
        corpus = {k: Entry(**v) for k, v in json.loads(corpus_path.read_text("utf-8")).items()}
        (out / "translation.txt").write_text(translate(text, corpus, target="deva"), "utf-8")

    # Avoid UnicodeEncodeError on Windows cp1252 consoles by asciifying
    safe_text = text[:400].encode("ascii", "backslashreplace").decode("ascii")
    safe_ipa = ipa[:200].encode("ascii", "backslashreplace").decode("ascii")
    print(f"[STT] transcript ({len(text)} chars), first 400:\n{safe_text}")
    print(f"[IPA] first 200: {safe_ipa}")


if __name__ == "__main__":
    main()
