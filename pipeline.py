"""End-to-end runner for the Speech Understanding PA-2 pipeline.

Usage
-----
Train artefacts (LID + n-gram LM) once:

    python pipeline.py train-lid   --out models/lid.pt
    python pipeline.py train-lm    --out models/ngram_lm.pkl
    python pipeline.py build-corpus --out data/parallel_corpus.json

Run the full inference chain on the supplied lecture + reference:

    python pipeline.py run \
        --lecture    data/original_segment.wav \
        --voice-ref  data/student_voice_ref.wav \
        --out        data/output_LRL_cloned.wav \
        --lm         models/ngram_lm.pkl \
        --lid        models/lid.pt \
        --cm         models/cm.pt           # optional

The runner writes:
  * data/transcript.txt         – Whisper+LM code-switched transcript
  * data/ipa.txt                – unified IPA sequence
  * data/translation.txt        – Santhali translation (Ol Chiki)
  * data/output_LRL_raw.wav     – pre-warp TTS output
  * data/output_LRL_cloned.wav  – prosody-warped final lecture
  * data/confusion_matrix.json  – code-switch boundary confusion stats
  * data/fgsm_report.json       – minimum epsilon from Task 4.2
  * data/eer_report.json        – CM EER from Task 4.1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio

from src.common import load_wav, save_wav
from src.part1_stt.denoise import denoise, normalize_loudness
from src.part1_stt.lid import (
    FRAME_HOP_MS, I2L, MultiHeadLID, frames_to_switches,
    load as load_lid_ckpt, switch_boundary_f1,
)
from src.part1_stt.lid_train import build_synth_manifest, train as train_lid
from src.part1_stt.ngram_lm import KneserNeyLM, build_syllabus_lm
from src.part2_phonetic.hinglish_g2p import text_to_ipa
from src.part2_phonetic.parallel_corpus import build_corpus, translate
from src.part3_tts.speaker_embedding import extract_from_file
from src.part3_tts.prosody_warp import warp_waveform
from src.part3_tts.synthesis import synthesise_ipa, OUT_SR
from src.part4_adversarial.antispoof import CMClassifier, evaluate_eer, train_cm
from src.part4_adversarial.fgsm import fgsm_min_epsilon


# ------------------------------------------------------------------ sub-cmds

def cmd_train_lid(args):
    manifest = args.manifest or build_synth_manifest(args.synth_dir, n=args.synth_n)
    from src.part1_stt.lid import TrainCfg as _Cfg
    train_lid(manifest, args.out, _Cfg(epochs=args.epochs))


def cmd_train_lm(args):
    extra = []
    if args.extra:
        extra = Path(args.extra).read_text("utf-8").splitlines()
    lm = build_syllabus_lm(args.out, extra_texts=extra)
    print(f"[lm] vocab={len(lm.vocab)} -> {args.out}")


def cmd_build_corpus(args):
    c = build_corpus(args.out)
    print(f"[corpus] rows={len(c)} -> {args.out}")


def cmd_run(args):
    """The full 4-part pipeline on a single lecture + voice reference."""
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Part I.3: denoise + normalise -----------------------------------
    print("[1/7] denoise")
    lecture = load_wav(args.lecture, sr=16_000)
    cleaned = normalize_loudness(denoise(lecture))
    cleaned_path = out_dir / "lecture_clean.wav"
    save_wav(cleaned_path, cleaned, sr=16_000)

    # --- Part I.1: frame-level LID + switch timestamps ------------------
    print("[2/7] LID")
    lid = load_lid_ckpt(args.lid)
    pred = lid.predict_frames(cleaned.unsqueeze(0), median_k=3).squeeze(0)
    switches = frames_to_switches(pred, hop_ms=FRAME_HOP_MS)
    (out_dir / "switches.json").write_text(json.dumps(switches, indent=2))

    # --- Part I.2: Whisper + n-gram LM transcription --------------------
    print("[3/7] transcribe (Whisper + LM logit bias)")
    try:
        from src.part1_stt.decode import TranscribeCfg, transcribe
        transcript = transcribe(str(cleaned_path), args.lm,
                                cfg=TranscribeCfg(beam_size=args.beam,
                                                  lm_lambda=args.lm_lambda))
    except Exception as exc:
        transcript = f"[WARN transcribe failed: {exc}]"
    (out_dir / "transcript.txt").write_text(transcript, "utf-8")

    # --- Part II.1: IPA conversion --------------------------------------
    print("[4/7] IPA G2P")
    ipa = text_to_ipa(transcript)
    (out_dir / "ipa.txt").write_text(ipa, "utf-8")

    # --- Part II.2: dictionary translation ------------------------------
    print("[5/7] translation")
    corpus_path = Path("data/parallel_corpus.json")
    if corpus_path.exists():
        corpus = {
            k: __import__("src.part2_phonetic.parallel_corpus", fromlist=["Entry"]).Entry(**v)
            for k, v in json.loads(corpus_path.read_text("utf-8")).items()
        }
    else:
        corpus = build_corpus(corpus_path)
    lrl_text = translate(transcript, corpus, target="deva")
    (out_dir / "translation.txt").write_text(lrl_text, "utf-8")

    # --- Part III.1: speaker embedding ----------------------------------
    print("[6/7] speaker embedding")
    emb = extract_from_file(args.voice_ref)
    torch.save(emb, out_dir / "speaker_emb.pt")

    # --- Part III.3: TTS ------------------------------------------------
    raw_tts = out_dir / "output_LRL_raw.wav"
    backend = synthesise_ipa(ipa, args.voice_ref, str(raw_tts), backend=args.tts_backend)
    print(f"       tts-backend={backend}")

    # --- Part III.2: DTW prosody warping --------------------------------
    print("[7/7] prosody warp")
    import soundfile as sf
    d_s, src_sr = sf.read(str(cleaned_path), dtype="float32", always_2d=True)
    d_t, tgt_sr = sf.read(str(raw_tts),      dtype="float32", always_2d=True)
    src_wav = torch.from_numpy(d_s.T).mean(0)
    tgt_wav = torch.from_numpy(d_t.T).mean(0)
    warped = warp_waveform(src_wav, src_sr, tgt_wav, tgt_sr)
    save_wav(args.out, warped, sr=tgt_sr)

    # --- Part IV: adversarial + spoof reports ---------------------------
    print("[IV] adversarial + spoof")
    # 4.1 CM: if checkpoint given and at least one spoof wav exists
    if args.cm:
        try:
            cm = CMClassifier(); cm.load_state_dict(torch.load(args.cm, map_location="cpu"))
            e = evaluate_eer(cm, [args.voice_ref], [args.out])
            (out_dir / "eer_report.json").write_text(json.dumps({"eer": e}, indent=2))
            print(f"       CM EER={e*100:.2f}%")
        except Exception as exc:
            (out_dir / "eer_report.json").write_text(json.dumps({"error": str(exc)}))

    # 4.2 FGSM on the LID
    try:
        fg = fgsm_min_epsilon(lid, cleaned[:16_000 * 5], target_label="EN")
        (out_dir / "fgsm_report.json").write_text(json.dumps({
            "min_epsilon": fg.epsilon,
            "flipped": fg.flipped,
            "snr_db": fg.snr_db,
        }, indent=2))
        print(f"       FGSM eps={fg.epsilon:.5f}  flipped={fg.flipped}  snr={fg.snr_db:.1f}")
    except Exception as exc:
        (out_dir / "fgsm_report.json").write_text(json.dumps({"error": str(exc)}))

    print(f"[done] output -> {args.out}")


# --------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description="Speech Understanding PA-2 pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("train-lid")
    s.add_argument("--manifest", default=None)
    s.add_argument("--synth-dir", default="data/synth_lid")
    s.add_argument("--synth-n", type=int, default=64)
    s.add_argument("--epochs", type=int, default=40)
    s.add_argument("--out", default="models/lid.pt")
    s.set_defaults(func=cmd_train_lid)

    s = sub.add_parser("train-lm")
    s.add_argument("--extra", default=None)
    s.add_argument("--out", default="models/ngram_lm.pkl")
    s.set_defaults(func=cmd_train_lm)

    s = sub.add_parser("build-corpus")
    s.add_argument("--out", default="data/parallel_corpus.json")
    s.set_defaults(func=cmd_build_corpus)

    s = sub.add_parser("run")
    s.add_argument("--lecture", required=True)
    s.add_argument("--voice-ref", required=True)
    s.add_argument("--out", default="data/output_LRL_cloned.wav")
    s.add_argument("--lm", default="models/ngram_lm.pkl")
    s.add_argument("--lid", default="models/lid.pt")
    s.add_argument("--cm", default=None)
    s.add_argument("--tts-backend", default="auto",
                   choices=["auto", "yourtts", "mms", "fallback"])
    s.add_argument("--beam", type=int, default=5)
    s.add_argument("--lm-lambda", type=float, default=0.4)
    s.set_defaults(func=cmd_run)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
