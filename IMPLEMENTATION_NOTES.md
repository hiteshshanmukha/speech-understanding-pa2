# Implementation Notes – One Non-Obvious Design Choice per Task

Mandatory 1-page supplement per the assignment. One *non-obvious*
decision per task, focused on trade-offs a quick code-scan would miss.

---

## Part I

**Task 1.1 — Multi-Head LID.**
The "multi-head" aspect is *temporally resolved*: a frame head
(argmax at a 10-ms hop) alongside a globally-pooled utterance head,
jointly trained with α = 0.7 on the frame loss. A single utterance
loss collapses mid-utterance boundaries; a single frame loss is
unstable on sub-1-s runs. A 3-frame median filter is applied only at
inference so gradients stay smooth but single-frame flicker is
suppressed before the ±200-ms SLA F1 is computed.

Also non-obvious: the model was initially trained only on
monolingual LibriSpeech-EN + FLEURS-HI clips and collapsed the SIL
class entirely (0 SIL predictions at test time). We added
(a) 120 **synthetic code-switched training clips** that expose the
model to EN↔HI transitions with real silence gaps, and
(b) an **energy-VAD post-filter** that overrides the classifier with
SIL when RMS < −75 dBFS sustained for ≥ 150 ms. Together these lifted
frame macro-F1 from 0.52 → 0.83 and SIL recall from 0 % → 91 % on
held-out code-switched audio.

**Task 1.2 — Constrained Decoding.**
Rather than biasing the full vocabulary on every step (intractable,
$O(|V|)$ extra cost per time-step), we re-score only the **top-K = 64
acoustic candidates**. Whisper's posteriors are sharp, so the ranking
of tokens outside the top-K is irrelevant, yet the N-gram term bias
still kicks in on the hypotheses that matter. The per-word log-prob
is also **smeared across BPE sub-tokens** so that the first token of
`cep|strum` isn't unfairly penalised by a low unigram prior.
Additionally, Whisper special tokens (`<|en|>`, `<|notimestamps|>`,
...) are explicitly skipped to preserve language / timestamp
tagging — otherwise biasing can corrupt the decoded structure.

**Task 1.3 — Denoising.**
The spectral-subtraction fallback uses a **Wiener-style floor
β·|X(ω)|** instead of a hard lower bound. Hard flooring produces the
characteristic "musical noise" that amplifies classroom environments;
a fraction-of-signal floor preserves voiced residuals and avoids
hallucinating harmonics.

---

## Part II

**Task 2.1 — Hinglish IPA.**
Schwa-deletion in Devanagari is applied **only in medial C-ə-C-V
contexts**, not to every non-final schwa. This matches how Hindi is
actually read aloud (*namaste → [nə̆m.əs.t̪eː]*) and keeps surface
durations consistent; global deletion collapses whole syllables and
breaks downstream TTS alignment.

**Task 2.2 — Parallel Corpus.**
The corpus is deliberately **compositional**: 150 hand-authored seed
rows plus a programmatic `modifier × head` expansion over seed
atoms. This (a) reaches the 500-row quota without fabricating new
roots and (b) guarantees IPA round-trip correctness because every
compound is built from already-verified atoms.

Extra for Part II: because MMS-Maithili's tokenizer accepts only
Devanagari, we ship a **phoneme-level IPA → Devanagari
back-transliterator** that re-introduces the inherent schwa 'अ' on
bare consonants and chooses matra vs independent vowel forms. This
is the inverse of the G2P so a round-trip stays consistent.

---

## Part III

**Task 3.1 — Speaker Embedding.**
The 60-s reference is **exactly 60 s**, not "roughly": shorter clips
are padded, longer clips are truncated. The x-vector statistics pool
would otherwise leak the recording length as an implicit feature,
which breaks downstream CM evaluation.

**Task 3.2 — DTW Prosody Warping.**
Two non-obvious choices:
1. The DTW operates on **joint [log F₀, log E]**, not F₀ alone.
   Energy carries stress and pause information that disambiguates
   emphatic vs declarative intonations with identical pitch.
2. The 10-ms contour for a 10-min lecture has ≈60 000 frames;
   pure-Python O(n²) DP would take hours. We **down-sample both
   contours to ≤600 points**, run DTW in the coarse grid, then
   interpolate the frame alignment back to full resolution. This is
   what lets the pipeline finish in minutes instead of hours.

**Task 3.3 — Synthesis backend order.**
Priority is YourTTS → MMS → formant fallback. YourTTS can do genuine
**zero-shot speaker cloning from a WAV** (the strict MCD target
assumes that ability). MMS covers ~1 000 languages including Maithili
but has a fixed speaker; we therefore treat its output as a "flat"
voice and apply the Part 3.2 warp with the student's x-vector to
transfer prosodic identity. Empty-chunk detection (input with zero
Devanagari tokens) was added after a long run crashed on the
`_relative_position_to_absolute_position` attention kernel inside
VITS — the punctuation-only tail of the transcript tokenises to an
empty sequence.

---

## Part IV

**Task 4.1 — Anti-Spoofing CM.**
LFCC is chosen over MFCC specifically for the **linear band above
4 kHz**. Neural vocoders (including our MMS output) tend to leave
checkerboard artefacts in 4–8 kHz; Mel warping squashes that band to
a handful of coefficients, whereas LFCC keeps it at full resolution.
That makes LFCC a strictly better discriminator against the exact
kind of spoof we produce.

**Task 4.2 — FGSM.**
The search is **binary, not grid**: the feasible ε set is convex in
both the flipping and the SNR constraint, so log₂(ε_max/tol) ≈ 18
iterations locate the minimum passing ε. A grid search would need
10×–100× more forward passes and still miss the knee. The attack is
\emph{targeted} toward `EN` so the gradient sign is *subtracted*, not
added — a small sign-flip that's easy to miss in a copy-paste FGSM.

Also non-obvious: FGSM on real-world audio requires care with the
baseline. If the LID already predicts the target class, every
perturbation looks like an attack success at ε ≈ 0. Our reporting
therefore pulls the attack from a ground-truth **Hindi** chunk of
the synthetic CS test (LID correctly predicts HI there) so the
minimum-ε number actually measures robustness.
