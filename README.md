# Speech Understanding – Programming Assignment 2

Code-switched (Hinglish) lecture transcription → IPA translation →
zero-shot voice cloning into a Low-Resource Language (**Maithili**,
one of the three LRLs explicitly allowed by the statement) →
anti-spoofing & adversarial robustness, all in **Python + PyTorch**.

Source lecture URL: <https://youtu.be/ZPUtA3W-7_I>
Inference segment: first 10 min of the user-supplied
`data/infer/inference_audio.m4a` (10-min slice of the 02:20–02:54
window in the source video).

## Repository layout

```
assignment2/
├── pipeline.py                 # unified entry point (train + run)
├── scripts/
│   ├── prepare_data.py         # yt-dlp + ffmpeg: download the source lecture
│   ├── prepare_lid_data.py     # LibriSpeech EN + FLEURS hi_in HI
│   ├── build_codeswitch_train.py   # synthetic CS clips for LID training
│   ├── build_codeswitch_test.py    # held-out synth CS test clip
│   ├── eval_on_codeswitch.py   # LID frame/boundary F1 + confusion matrix
│   ├── run_stt_only.py         # Part I-II on inference clip
│   ├── run_voice_stages.py     # Part III-IV given a voice ref
│   ├── compute_wer.py          # WER helper against user-supplied ref
│   ├── scan_code_switch.py     # Whisper-based language probe
│   └── weak_label_lid.py       # (legacy) Whisper weak-labelling
├── requirements.txt
├── IMPLEMENTATION_NOTES.md     # 1-page non-obvious design choices
├── report/report.tex           # IEEE two-column report
├── src/
│   ├── part1_stt/              # denoising, LID, n-gram LM, constrained decode
│   │   ├── denoise.py
│   │   ├── lid.py              # MultiHeadLID + energy-VAD post-filter
│   │   ├── lid_train.py
│   │   ├── lid_whisper.py      # Whisper-based LID reference (reporting)
│   │   ├── ngram_lm.py         # modified Kneser-Ney 3-gram
│   │   └── decode.py           # chunked Whisper + NgramLogitBias
│   ├── part2_phonetic/
│   │   ├── hinglish_g2p.py     # script-aware G2P → unified IPA
│   │   ├── ipa_to_deva.py      # IPA → Devanagari back-transliteration
│   │   └── parallel_corpus.py  # 500-row Hinglish → Santhali + Devanagari
│   ├── part3_tts/
│   │   ├── speaker_embedding.py   # x-vector TDNN
│   │   ├── prosody_warp.py     # F0/E extraction + DTW + envelope warp
│   │   └── synthesis.py        # YourTTS → MMS → formant fallback
│   ├── part4_adversarial/
│   │   ├── antispoof.py        # LFCC + BiLSTM + attention pool + EER
│   │   └── fgsm.py             # targeted FGSM with binary search
│   └── common/audio_io.py      # soundfile-backed WAV IO (Windows-safe)
├── data/
│   ├── infer/original_segment.wav  # 10-min inference INPUT
│   ├── student_voice_ref.wav       # 60-s voice reference INPUT
│   └── parallel_corpus.json        # 500-row technical dictionary
├── results/                        # all pipeline outputs (see results/README.md)
│   ├── output_LRL_cloned.wav       # **FINAL 22.05-kHz LRL lecture**
│   ├── transcript.txt, ipa.txt, translation.txt, maithili_text.txt
│   ├── switches.json, switches_whisper.json
│   ├── lid_confusion.json, synth_segments.json
│   ├── metrics_final.json, fgsm_report.json
│   └── README.md
└── models/
    ├── lid.pt                  # trained LID
    ├── ngram_lm.pkl            # modified KN 3-gram
    └── cm.pt                   # anti-spoof CM
```

## End-to-end reproduction (full pipeline)

```bash
# 0. Environment
python -m venv .venv && .venv\Scripts\activate        # or: source .venv/bin/activate
pip install -r requirements.txt

# 1. LID training data - LibriSpeech EN + FLEURS hi_in HI (≈800 clips)
python scripts/prepare_lid_data.py --n-per-lang 400

# 2. Add 120 synthetic code-switched training clips (EN/HI/SIL labelled)
python scripts/build_codeswitch_train.py --n-clips 120 --target-s 45

# 3. Merge + train LID on the combined 920-clip manifest
python -c "from pathlib import Path; \
  Path('data/lid_combined.jsonl').write_text( \
    '\n'.join(l for p in ['data/lid_train/manifest.jsonl', \
                           'data/lid_cs_train/manifest.jsonl'] \
             for l in Path(p).read_text('utf-8').splitlines() if l.strip()), 'utf-8')"
python pipeline.py train-lid --manifest data/lid_combined.jsonl \
                             --out models/lid.pt --epochs 30

# 4. N-gram LM + parallel corpus
python pipeline.py train-lm     --out models/ngram_lm.pkl
python pipeline.py build-corpus --out data/parallel_corpus.json

# 5. (Optional) held-out CS test + eval (produces the confusion matrix)
python scripts/build_codeswitch_test.py --target-s 120
python scripts/eval_on_codeswitch.py --whisper openai/whisper-small

# 6. Inference - drop your 10-min audio at data/infer/original_segment.wav
#    and your 60-s voice reference at data/student_voice_ref.wav, then:
python scripts/run_stt_only.py \
       --wav data/infer/original_segment.wav \
       --whisper openai/whisper-small
python scripts/run_voice_stages.py --voice-ref data/student_voice_ref.wav
#   -> results/{transcript, ipa, translation, maithili_text}.txt
#   -> results/{switches, switches_whisper, fgsm_report, metrics_final}.json
#   -> results/output_LRL_cloned.wav        (final 22.05-kHz LRL lecture)
```

## Part-by-part reference

### Part I – Robust Code-Switched Transcription
| Task | File | Notes |
|---|---|---|
| 1.1 Frame-level LID | [src/part1_stt/lid.py](src/part1_stt/lid.py), [lid_train.py](src/part1_stt/lid_train.py) | 2×Conv + 3×BiGRU, frame + utt heads, energy-VAD post-filter. |
| 1.2 Constrained decoding | [src/part1_stt/ngram_lm.py](src/part1_stt/ngram_lm.py), [decode.py](src/part1_stt/decode.py) | Modified Kneser–Ney 3-gram on the course syllabus, shallow-fusion bias at λ=0.4 over top-64 candidates + hard +2.0 boost for 20 technical terms. |
| 1.3 Denoising | [src/part1_stt/denoise.py](src/part1_stt/denoise.py) | DeepFilterNet + spectral-subtraction fallback with Wiener-style floor. |

### Part II – Phonetic Mapping & Translation
| Task | File | Notes |
|---|---|---|
| 2.1 Hinglish IPA G2P | [src/part2_phonetic/hinglish_g2p.py](src/part2_phonetic/hinglish_g2p.py) | Script-aware segmenter + Devanagari→IPA with schwa deletion + English pronunciation dict. |
| 2.2 Parallel corpus | [src/part2_phonetic/parallel_corpus.py](src/part2_phonetic/parallel_corpus.py) | 150 seed + compositional expansion → 500 rows with Ol Chiki + Devanagari + Hindi + English. |
| 2.3 IPA → Devanagari | [src/part2_phonetic/ipa_to_deva.py](src/part2_phonetic/ipa_to_deva.py) | Phoneme-level back-transliterator for MMS-TTS input. |

### Part III – Zero-Shot Cross-Lingual Voice Cloning
| Task | File | Notes |
|---|---|---|
| 3.1 Speaker embedding | [src/part3_tts/speaker_embedding.py](src/part3_tts/speaker_embedding.py) | TDNN x-vector, 512-d L2-normalised, exactly 60 s input. |
| 3.2 Prosody warping | [src/part3_tts/prosody_warp.py](src/part3_tts/prosody_warp.py) | F0 (NCCF) + RMS energy, Sakoe–Chiba DTW on [log F0, log E], coarse-grid downsampling for 10-min audio. |
| 3.3 Synthesis | [src/part3_tts/synthesis.py](src/part3_tts/synthesis.py) | **Meta MMS-Maithili** → 22.05 kHz upsample; YourTTS and formant fallbacks available. |

### Part IV – Adversarial Robustness & Spoofing
| Task | File | Notes |
|---|---|---|
| 4.1 Anti-Spoof CM | [src/part4_adversarial/antispoof.py](src/part4_adversarial/antispoof.py) | LFCC(60) + Δ + ΔΔ → BiLSTM → attention pool → softmax, EER sweep. |
| 4.2 FGSM on LID | [src/part4_adversarial/fgsm.py](src/part4_adversarial/fgsm.py) | Targeted FGSM with binary search over ε, SNR > 40 dB constraint. |

## Passing criteria – headline metrics

| Metric | Target | Observed | Status |
|---|---|---|---|
| LID frame macro-F1 (synth CS test) | - | **0.833** | ✓ |
| LID boundary F1 ±200 ms (synth CS test) | ≥ 0.85 | **0.780** | near |
| MCD vs lecturer, flat MMS | < 8.0 | **9.20 dB** | near |
| **EER** | **< 10 %** | **5.00 %** | **✓** |
| **FGSM ε @ SNR > 40 dB** | report | **no flip** | **✓ robust** |
| TTS output sample rate | ≥ 22.05 kHz | **22.05 kHz** | ✓ |

WER needs a ground-truth transcript - drop one at `results/ref_transcript.txt` and run [scripts/compute_wer.py](scripts/compute_wer.py).

### Ablation: Prosody warping vs flat synthesis (MCD-DTW, dB)

| Reference | Flat MMS | + DTW warp | Δ |
|---:|---:|---:|---:|
| `student_voice_ref.wav` | 13.00 | 16.22 | +3.22 |
| `lecture_clean.wav`     |  9.20 | 11.61 | +2.41 |

The warp correctly transfers F0+energy; the resample-based
pitch shifter distorts the spectral envelope - a phase-vocoder /
TD-PSOLA pitch shifter is the primary future-work item (see report).

### LID confusion matrix on the held-out synthetic CS test

```
                pred EN    pred HI   pred SIL
gt EN           6 448       249        59
gt HI           1 192      3 280      161
gt SIL             36        21       564
```

HI→EN is the dominant residual error (FLEURS is acoustically softer
than LibriSpeech; classifier leans toward the cleaner class). SIL
recall 91 % - the energy-VAD post-filter did its job.

## Report + implementation note
- [report/report.tex](report/report.tex) – full IEEE two-column report.
- [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) – one non-obvious design choice per task (mandatory 1-pager).

## Citations
See the report bibliography for full references to Whisper
(Radford 2023), YourTTS (Casanova 2022), Meta MMS (Pratap 2023),
x-vector (Snyder 2018), Kneser–Ney (Chen & Goodman 1998), Sakoe &
Chiba (1978), Boll (1979), Sahidullah (2015), Goodfellow et al.
(2015), FLEURS (Conneau 2023), MUCS-2021 (Diwan 2021).
