"""Run the voice-dependent stages after the user drops their 60-s ref.

Prereqs (produced by `run_stt_only.py`):
    data/infer/lecture_clean.wav
    data/infer/ipa.txt

Input (you supply):
    data/student_voice_ref.wav   # exactly 60 s of your own voice, 16 kHz mono

Output:
    data/infer/speaker_emb.pt           # 512-d x-vector
    data/infer/output_LRL_raw.wav       # flat TTS in Santhali
    data/output_LRL_cloned.wav          # DTW-prosody-warped final lecture
    data/infer/fgsm_report.json         # Task 4.2 result
    data/infer/eer_report.json          # Task 4.1 result (if --cm given)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.part1_stt.lid import load as load_lid
from src.part3_tts.prosody_warp import warp_waveform
from src.part3_tts.speaker_embedding import extract_from_file
from src.part3_tts.synthesis import synthesise_ipa
from src.part4_adversarial.antispoof import CMClassifier, evaluate_eer, train_cm
from src.part4_adversarial.fgsm import fgsm_min_epsilon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice-ref", required=True,
                    help="60-s reference, 16-kHz mono, e.g. data/student_voice_ref.wav")
    ap.add_argument("--infer-dir", default="data/infer")
    ap.add_argument("--out", default="data/output_LRL_cloned.wav")
    ap.add_argument("--lid", default="models/lid.pt")
    ap.add_argument("--cm", default=None,
                    help="Optional pre-trained CM checkpoint; if absent, trained on-the-fly.")
    ap.add_argument("--tts-backend", default="auto",
                    choices=["auto", "yourtts", "mms", "fallback"])
    args = ap.parse_args()

    idir = Path(args.infer_dir)
    ipa = (idir / "ipa.txt").read_text("utf-8").strip()
    clean = idir / "lecture_clean.wav"
    assert clean.exists(), f"{clean} missing – run run_stt_only.py first"
    assert Path(args.voice_ref).exists(), f"{args.voice_ref} missing"

    # ------------- Task 3.1 -------------------------------------------
    print("[1/5] speaker embedding")
    emb = extract_from_file(args.voice_ref)
    torch.save(emb, idir / "speaker_emb.pt")
    print(f"   emb-dim={emb.shape[0]}")

    # ------------- Task 3.3 (flat synth) ------------------------------
    print("[2/5] TTS synthesis")
    raw = idir / "output_LRL_raw.wav"
    backend = synthesise_ipa(ipa, args.voice_ref, str(raw),
                             backend=args.tts_backend)
    print(f"   backend={backend} -> {raw}")

    # ------------- Task 3.2 (DTW prosody warp) ------------------------
    print("[3/5] prosody warp (DTW on F0+E)")
    d_s, src_sr = sf.read(str(clean),  dtype="float32", always_2d=True)
    d_t, tgt_sr = sf.read(str(raw),    dtype="float32", always_2d=True)
    src_wav = torch.from_numpy(d_s.T).mean(0)
    tgt_wav = torch.from_numpy(d_t.T).mean(0)
    warped = warp_waveform(src_wav, src_sr, tgt_wav, tgt_sr)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, warped.numpy().astype(np.float32), tgt_sr, subtype="PCM_16")
    print(f"   -> {args.out}  sr={tgt_sr}")

    # ------------- Task 4.1 (Anti-spoof CM + EER) ---------------------
    print("[4/5] anti-spoof CM")
    if args.cm and Path(args.cm).exists():
        cm = CMClassifier()
        cm.load_state_dict(torch.load(args.cm, map_location="cpu"))
    else:
        # train on the pair (bona=voice_ref, spoof=cloned) for a quick EER number.
        # For a stronger CM use ASVspoof2019-LA plus this pair.
        cm = train_cm([args.voice_ref], [args.out], "models/cm.pt", epochs=8)
    e = evaluate_eer(cm, [args.voice_ref], [args.out])
    (idir / "eer_report.json").write_text(json.dumps({"EER": e}, indent=2))
    print(f"   EER = {e*100:.2f}%")

    # ------------- Task 4.2 (FGSM on LID) -----------------------------
    print("[5/5] FGSM on LID")
    lid = load_lid(args.lid)
    # attack a 5-s Hindi-leaning window from the inference clip
    seg = src_wav[:16_000 * 5]
    fg = fgsm_min_epsilon(lid, seg, target_label="EN", snr_floor=40.0)
    (idir / "fgsm_report.json").write_text(json.dumps({
        "min_epsilon": fg.epsilon,
        "flipped": fg.flipped,
        "snr_db": fg.snr_db,
    }, indent=2))
    print(f"   eps={fg.epsilon:.5f}  flipped={fg.flipped}  snr={fg.snr_db:.1f} dB")
    print(f"[done] final output -> {args.out}")


if __name__ == "__main__":
    main()
