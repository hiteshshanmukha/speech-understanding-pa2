# Pipeline outputs

All artefacts produced by the Speech Understanding PA-2 pipeline on
the user-supplied 10-minute inference audio + 60-second voice
reference.

| File | Produced by | What it is |
|---|---|---|
| `output_LRL_cloned.wav` | Part 3.2+3.3 | **Final 430.7-s Maithili lecture @ 22.05 kHz** — MMS-TTS synthesis + DTW prosody warp. |
| `transcript.txt` | Part 1.2 | Whisper-small + N-gram LM biased transcript of the 10-min inference clip. |
| `ipa.txt` | Part 2.1 | Unified IPA string (Hinglish G2P output). |
| `translation.txt` | Part 2.2 | Dictionary-translated transcript (Devanagari where the 500-row corpus has an entry, else source). |
| `maithili_text.txt` | Part 2.3 | IPA → Devanagari back-transliteration, the actual input fed to MMS-TTS. |
| `switches.json` | Part 1.1 | Our custom LID timeline `[(t_seconds, label), ...]` on the inference clip. |
| `switches_whisper.json` | Part 1.1 (reference) | Whisper-built-in LID timeline for the same clip — apples-to-apples reference that the inference audio is ~75 % Hindi. |
| `metrics_final.json` | Parts 3–4 | Final MCD, EER, TTS backend metadata. |
| `fgsm_report.json` | Part 4.2 | Minimum ε to flip + SNR sweep (robustness proof at SNR > 40 dB). |
| `lid_confusion.json` | Part 1.1 eval | Frame-level confusion matrix on the held-out synthetic CS test (reported in Table II of the report). |
| `synth_segments.json` | Test fixture | Ground-truth segments for the synthetic CS test clip used to compute `lid_confusion.json`. |

Inputs (not in this folder — stay in `data/`):
* `data/infer/original_segment.wav` — 10-min inference clip
* `data/student_voice_ref.wav` — 60-s voice reference
