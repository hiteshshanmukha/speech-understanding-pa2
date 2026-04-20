"""Run the STT half of the pipeline on the inference clip.

Produces (under data/infer/):
    lecture_clean.wav       – denoised 16-kHz inference audio
    switches.json           – frame-level LID + merged code-switch timeline
    transcript.txt          – Whisper-v3 output biased by the custom KN LM
    ipa.txt                 – unified IPA string
    translation.txt         – Santhali (Devanagari) translation
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import load_wav, save_wav
from src.part1_stt.denoise import denoise, normalize_loudness
from src.part1_stt.lid import (
    FRAME_HOP_MS, frames_to_switches, load as load_lid,
)
from src.part2_phonetic.hinglish_g2p import text_to_ipa
from src.part2_phonetic.parallel_corpus import build_corpus, translate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="data/infer/original_segment.wav")
    ap.add_argument("--out-dir", default="data/infer")
    ap.add_argument("--lid", default="models/lid.pt")
    ap.add_argument("--lm", default="models/ngram_lm.pkl")
    ap.add_argument("--whisper", default="openai/whisper-large-v3")
    ap.add_argument("--lm-lambda", type=float, default=0.4)
    ap.add_argument("--beam", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) denoise ---------------------------------------------------------
    print("[1/5] denoise")
    wav = load_wav(args.wav, sr=16_000)
    cleaned = normalize_loudness(denoise(wav))
    clean_path = out_dir / "lecture_clean.wav"
    save_wav(clean_path, cleaned, sr=16_000)

    # 2) LID frame-level -------------------------------------------------
    print("[2/5] LID frame prediction")
    lid = load_lid(args.lid)
    # process in ≤30-s chunks to keep memory in check for a 10-min clip
    chunk = 30 * 16_000
    frames = []
    for i in range(0, cleaned.shape[0], chunk):
        sub = cleaned[i:i + chunk]
        pred = lid.predict_frames(sub.unsqueeze(0), median_k=3).squeeze(0)
        frames.append(pred)
    pred = torch.cat(frames, dim=0)
    switches = frames_to_switches(pred, hop_ms=FRAME_HOP_MS)
    (out_dir / "switches.json").write_text(json.dumps(switches, indent=2))
    print(f"   {len(switches)} code-switch points")

    # 3) transcribe ------------------------------------------------------
    print("[3/5] Whisper + N-gram LM logit bias")
    from src.part1_stt.decode import TranscribeCfg, transcribe
    cfg = TranscribeCfg(model_id=args.whisper,
                        lm_lambda=args.lm_lambda, beam_size=args.beam)
    text = transcribe(str(clean_path), args.lm, cfg=cfg)
    (out_dir / "transcript.txt").write_text(text, "utf-8")
    print(f"   transcript ({len(text)} chars)")

    # 4) IPA G2P ---------------------------------------------------------
    print("[4/5] Hinglish IPA")
    ipa = text_to_ipa(text)
    (out_dir / "ipa.txt").write_text(ipa, "utf-8")

    # 5) translation -----------------------------------------------------
    print("[5/5] Santhali translation")
    corpus_path = Path("data/parallel_corpus.json")
    if corpus_path.exists():
        from src.part2_phonetic.parallel_corpus import Entry
        corpus = {k: Entry(**v) for k, v in
                  json.loads(corpus_path.read_text("utf-8")).items()}
    else:
        corpus = build_corpus(corpus_path)
    tr = translate(text, corpus, target="deva")
    (out_dir / "translation.txt").write_text(tr, "utf-8")

    print(f"[done] STT artefacts in {out_dir}")


if __name__ == "__main__":
    main()
