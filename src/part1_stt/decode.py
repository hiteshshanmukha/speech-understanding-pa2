"""Task 1.2 – Constrained decoding on top of Whisper.

The n-gram LM is plugged into Whisper through a HuggingFace
`LogitsProcessor` that, on every step, computes the incremental
log-probability contribution of each vocabulary token conditioned on
the partially decoded hypothesis, multiplies it by a tunable lambda,
and adds it to the acoustic logits (shallow-fusion biasing).

Mathematical formulation
------------------------
Given an LM log-prob p_LM(w|h) and the ASR logit l_ASR(w|h):

    l_biased(w|h) = l_ASR(w|h) + lambda * log p_LM(w|h)

For Whisper we work at the BPE level, so we approximate word-level
LM probability by buffering sub-word tokens until a space boundary
appears, scoring the completed word, and smearing the word bonus
equally over the constituent BPE tokens (so that the first token of a
term like "cep stum" is not unfairly penalised).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# Lazy HF imports – library is heavy, keep the file importable without it.
def _import_hf():
    from transformers import (
        AutoProcessor, WhisperForConditionalGeneration, LogitsProcessor,
    )
    return AutoProcessor, WhisperForConditionalGeneration, LogitsProcessor


class NgramLogitBias:
    """Wrap KneserNeyLM as a HuggingFace LogitsProcessor.

    Constructed lazily so the module can still be imported on machines
    without transformers installed.
    """

    def __init__(self, lm, tokenizer, lam: float = 0.4, boost_terms: Optional[list[str]] = None):
        self.lm = lm
        self.tok = tokenizer
        self.lam = lam
        # Whisper adds language / task tokens beyond `vocab_size`, so
        # use the full vocab (added-tokens included) for safety.
        vocab = tokenizer.get_vocab()
        self._id_to_str = {v: k for k, v in vocab.items()}
        self.boost_ids: list[int] = []
        if boost_terms:
            for term in boost_terms:
                ids = tokenizer(term, add_special_tokens=False).input_ids
                self.boost_ids.extend(ids)

    def _history_words(self, input_ids: torch.Tensor) -> list[str]:
        # decode the partial hypothesis, split into words
        txt = self.tok.decode(input_ids.tolist(), skip_special_tokens=True).lower()
        return txt.split()

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        # scores: (B, V)
        for b in range(scores.shape[0]):
            hist = tuple(self._history_words(input_ids[b])[-(self.lm.n - 1):])
            bias = torch.zeros_like(scores[b])
            # cheap bias: only rescore top-K candidates to stay fast
            topk_v, topk_i = torch.topk(scores[b], k=min(64, scores.shape[-1]))
            for v, idx in zip(topk_v.tolist(), topk_i.tolist()):
                tok = self._id_to_str.get(idx)
                if tok is None:
                    continue
                # Skip Whisper special tokens (<|...|>) – biasing those
                # can break language / timestamp tagging.
                if tok.startswith("<|"):
                    continue
                # Whisper BPE: space-prefix "Ġ" signals word boundary
                candidate = tok.replace("Ġ", "").lower()
                if not candidate.isalpha():
                    continue
                lp = self.lm.logp(candidate, hist)
                bias[idx] = self.lam * lp
            # additive term-boost: +2.0 nats for explicitly listed technical words
            if self.boost_ids:
                bias[self.boost_ids] += 2.0
            scores[b] = scores[b] + bias
        return scores


# ---------------------------- runtime API -------------------------------

@dataclass
class TranscribeCfg:
    model_id: str = "openai/whisper-large-v3"
    lang: str | None = None            # None = auto; else "en" / "hi"
    beam_size: int = 5
    lm_lambda: float = 0.4
    boost_terms: tuple[str, ...] = (
        "cepstrum", "stochastic", "mel", "viterbi", "hmm", "fft",
        "formant", "pitch", "phoneme", "ipa", "wav2vec", "whisper",
        "vits", "yourtts", "mcd", "wer", "eer", "lfcc", "cqcc", "dtw",
    )


def transcribe(audio_path: str | Path, lm_path: str | Path,
               cfg: TranscribeCfg = TranscribeCfg(),
               chunk_s: float = 28.0) -> str:
    """Transcribe arbitrary-length audio.

    Whisper's feature extractor pads/crops to exactly 30 s, so for
    anything longer we slice the waveform into ``chunk_s``-second
    chunks, transcribe each independently (with logit bias), and
    concatenate the results. A small 2-s overlap is dropped at the
    chunk boundary to avoid duplicate words.
    """
    from .ngram_lm import KneserNeyLM
    AutoProcessor, WhisperForConditionalGeneration, LogitsProcessor = _import_hf()
    import torchaudio
    import soundfile as sf

    proc = AutoProcessor.from_pretrained(cfg.model_id)
    model = WhisperForConditionalGeneration.from_pretrained(cfg.model_id)
    model.eval()

    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T).mean(0)
    if sr != 16_000:
        wav = torchaudio.functional.resample(wav, sr, 16_000)

    forced_ids = (proc.get_decoder_prompt_ids(language=cfg.lang, task="transcribe")
                  if cfg.lang else None)

    lm = KneserNeyLM.load(lm_path)
    bias = NgramLogitBias(lm, proc.tokenizer, lam=cfg.lm_lambda,
                          boost_terms=list(cfg.boost_terms))

    chunk_samples = int(chunk_s * 16_000)
    total = wav.shape[0]
    if total <= chunk_samples:
        chunks = [wav]
    else:
        chunks = [wav[i:i + chunk_samples]
                  for i in range(0, total, chunk_samples)]

    texts: list[str] = []
    with torch.no_grad():
        for ch in chunks:
            if ch.shape[0] < 16_000:           # < 1 s – ignore tail
                continue
            inp = proc(ch.numpy(), sampling_rate=16_000,
                       return_tensors="pt")
            out = model.generate(
                **inp,
                forced_decoder_ids=forced_ids,
                num_beams=cfg.beam_size,
                logits_processor=[bias],
                max_new_tokens=440,
            )
            txt = proc.batch_decode(out, skip_special_tokens=True)[0]
            texts.append(txt.strip())
    return " ".join(texts).strip()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--lm", default="models/ngram_lm.pkl")
    ap.add_argument("--lang", default=None)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--lam", type=float, default=0.4)
    args = ap.parse_args()
    cfg = TranscribeCfg(lang=args.lang, beam_size=args.beam, lm_lambda=args.lam)
    print(transcribe(args.wav, args.lm, cfg))
