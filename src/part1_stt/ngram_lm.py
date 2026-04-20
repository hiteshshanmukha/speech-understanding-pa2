"""Custom n-gram LM trained on the speech-course syllabus.

Uses modified Kneser-Ney smoothing with a trigram back-off chain
(3-gram -> 2-gram -> 1-gram -> OOV floor). We stay dependency-free so
the same class can be shipped as the `models/ngram_lm.pt` artifact and
re-loaded by the Whisper logit-bias processor without any extra
libraries.

Math (Kneser-Ney, modified)
---------------------------
P_KN(w_i | w_{i-n+1}^{i-1}) = max(c(h w_i) - D_n, 0) / c(h)
                            + gamma(h) * P_KN(w_i | w_{i-n+2}^{i-1})

with D_n derived from n_1(n)/(n_1(n) + 2 n_2(n)) and
gamma(h) = D_n * |{w : c(h w) > 0}| / c(h).

The continuation probability for the lowest order is

    P_cont(w) = |{v : c(v w) > 0}| / |{(v, w') : c(v w') > 0}|.
"""
from __future__ import annotations

import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path


_WORD = re.compile(r"[a-zA-Z\u0900-\u097F]+(?:'[a-z]+)?|[0-9]+")
_BOS, _EOS, _UNK = "<s>", "</s>", "<unk>"


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(text)]


class KneserNeyLM:
    """Modified Kneser-Ney tri-gram LM."""

    def __init__(self, n: int = 3):
        self.n = n
        self.ngrams: list[Counter] = [Counter() for _ in range(n + 1)]
        self.context_types: list[defaultdict] = [
            defaultdict(set) for _ in range(n + 1)
        ]
        self.vocab: set[str] = set()
        self.D: list[float] = [0.0] * (n + 1)
        self._cont_numer: defaultdict = defaultdict(set)
        self._cont_denom: int = 0

    # ----------- fitting -----------

    def fit(self, corpus_texts: list[str]):
        for text in corpus_texts:
            toks = [_BOS] * (self.n - 1) + tokenize(text) + [_EOS]
            self.vocab.update(toks)
            for k in range(1, self.n + 1):
                for i in range(len(toks) - k + 1):
                    ng = tuple(toks[i:i + k])
                    self.ngrams[k][ng] += 1
                    if k >= 2:
                        self.context_types[k][ng[:-1]].add(ng[-1])
            # lowest-order continuation stats
            for i in range(len(toks) - 1):
                self._cont_numer[toks[i + 1]].add(toks[i])
            self._cont_denom += max(0, len(toks) - 1)

        # discounts D via n_1 / (n_1 + 2 n_2) per order
        for k in range(1, self.n + 1):
            n1 = sum(1 for c in self.ngrams[k].values() if c == 1)
            n2 = sum(1 for c in self.ngrams[k].values() if c == 2)
            self.D[k] = n1 / (n1 + 2 * n2) if (n1 + 2 * n2) else 0.5

    # ----------- scoring -----------

    def _pcont(self, w: str) -> float:
        num = len(self._cont_numer.get(w, ()))
        return (num + 1) / (self._cont_denom + len(self.vocab) + 1)

    def _prob(self, w: str, hist: tuple[str, ...]) -> float:
        if not hist:
            return self._pcont(w)
        k = len(hist) + 1
        h = hist
        ch = self.ngrams[k - 1].get(h, 0)
        if ch == 0:
            return self._prob(w, hist[1:])
        D = self.D[k]
        num = max(self.ngrams[k].get(h + (w,), 0) - D, 0.0)
        gamma = D * len(self.context_types[k].get(h, ())) / ch
        return num / ch + gamma * self._prob(w, hist[1:])

    def logp(self, w: str, hist: tuple[str, ...]) -> float:
        hist = hist[-(self.n - 1):]
        p = self._prob(w, hist)
        return math.log(max(p, 1e-12))

    def score_sequence(self, tokens: list[str]) -> float:
        pad = [_BOS] * (self.n - 1)
        toks = pad + tokens + [_EOS]
        lp = 0.0
        for i in range(self.n - 1, len(toks)):
            lp += self.logp(toks[i], tuple(toks[i - self.n + 1:i]))
        return lp

    # ----------- I/O -----------

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "n": self.n,
                    "ngrams": self.ngrams,
                    "context_types": {
                        k: {h: list(s) for h, s in ct.items()}
                        for k, ct in enumerate(self.context_types)
                    },
                    "vocab": list(self.vocab),
                    "D": self.D,
                    "cont_numer": {w: list(s) for w, s in self._cont_numer.items()},
                    "cont_denom": self._cont_denom,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "KneserNeyLM":
        with open(path, "rb") as f:
            st = pickle.load(f)
        lm = cls(n=st["n"])
        lm.ngrams = st["ngrams"]
        lm.context_types = [defaultdict(set) for _ in range(lm.n + 1)]
        for k, ct in st["context_types"].items():
            for h, lst in ct.items():
                lm.context_types[int(k)][h] = set(lst)
        lm.vocab = set(st["vocab"])
        lm.D = st["D"]
        lm._cont_numer = defaultdict(set, {w: set(v) for w, v in st["cont_numer"].items()})
        lm._cont_denom = st["cont_denom"]
        return lm


# ------------------------ syllabus corpus --------------------------------

# Seed corpus – speech-course syllabus / typical Indian UG slide-deck
# vocabulary. Extend with course notes to improve bias. Kept as a
# constant so the model is reproducible from source code alone.
SYLLABUS_SENTENCES: list[str] = [
    "this lecture introduces stochastic processes and their role in speech modelling",
    "the cepstrum is the inverse Fourier transform of the log magnitude spectrum",
    "mel frequency cepstral coefficients are derived from the mel filter bank",
    "hidden Markov models assume a Markov chain of latent states emitting observations",
    "dynamic time warping aligns two sequences of different lengths",
    "the Viterbi algorithm finds the most likely state sequence",
    "formants are resonances of the vocal tract",
    "pitch detection uses autocorrelation or cepstral peak picking",
    "phoneme recognition is the first stage of many speech recognisers",
    "language identification distinguishes between English and Hindi",
    "code switching refers to alternation between two languages within an utterance",
    "connectionist temporal classification trains acoustic models without alignment",
    "attention based encoder decoder architectures dominate modern recognition",
    "transformer models use multi head self attention",
    "word error rate measures substitution insertion and deletion errors",
    "mel cepstral distortion evaluates synthesised speech quality",
    "equal error rate is where false acceptance and false rejection cross",
    "fundamental frequency is the rate of vocal fold vibration",
    "short time Fourier transform gives a time frequency representation",
    "spectral subtraction reduces additive background noise",
    "Wiener filtering uses knowledge of signal and noise power spectra",
    "overlap add synthesis reconstructs the waveform from frames",
    "griffin lim iterative phase reconstruction is used when phase is missing",
    "vocoders convert acoustic features to waveforms",
    "WaveNet is an autoregressive neural vocoder",
    "HiFi GAN is a generative adversarial vocoder",
    "VITS end to end architecture combines flows and adversarial training",
    "YourTTS supports zero shot multilingual voice cloning",
    "speaker embedding captures speaker identity in a compact vector",
    "x vectors are statistics pooling embeddings from TDNN",
    "d vectors are utterance embeddings from a speaker classifier",
    "prosody includes rhythm stress and intonation",
    "G2P converts graphemes to phonemes using rules or neural models",
    "IPA is the international phonetic alphabet",
    "low resource languages lack parallel corpora and pronunciation lexicons",
    "transfer learning adapts models trained on high resource data",
    "self supervised learning exploits unlabelled audio",
    "wav2vec 2 learns representations via contrastive prediction",
    "Whisper is a robust multilingual ASR model",
    "constrained beam search restricts hypotheses to allowed tokens",
    "logit biasing adds scalar offsets to chosen vocabulary items",
    "FGSM is the fast gradient sign method for adversarial attacks",
    "anti spoofing detects synthesised or replayed speech",
    "LFCC are linear frequency cepstral coefficients",
    "CQCC uses the constant Q transform for anti spoofing",
    "the Mel scale approximates human auditory frequency resolution",
    "pre emphasis flattens the spectral tilt of voiced speech",
    "windowing reduces spectral leakage",
    "the short time energy measures frame level loudness",
    "pitch contours carry prosodic information",
    "f zero estimation is critical for prosody warping",
    "Hinglish mixes English and Hindi within a single utterance",
    "teachers often switch to Hindi for explanations and English for technical terms",
    "the syllabus covers acoustic phonetics and digital signal processing",
    "mid term projects require a pipeline from feature extraction to synthesis",
    "anti aliasing filtering precedes down sampling",
    "Nyquist theorem bounds the maximum representable frequency",
    "linear predictive coding models the vocal tract as an all pole filter",
    "filter bank energies are robust features",
    "maximum likelihood estimation fits parameters to data",
    "Bayesian inference computes posterior distributions",
    "EM algorithm iteratively optimises latent variable models",
    "Gaussian mixture models represent density as a weighted sum of Gaussians",
    "i vector is a low dimensional speaker factor representation",
    "zero shot cloning generates speech in an unseen speaker voice",
    "the final lecture covers evaluation metrics WER MCD and EER",
    "adversarial robustness measures perturbation epsilon for misclassification",
    "semantic translation preserves meaning across languages",
    "Indian languages include Hindi Bengali Tamil Telugu Santhali Maithili",
    "acoustic model and language model scores are combined in decoding",
    "kaldi wav2vec2 and whisper are common ASR toolkits",
    "evaluation on a held out test set avoids overfitting",
    "regularisation prevents overfitting through dropout or weight decay",
]


def build_syllabus_lm(out_path: str | Path, extra_texts: list[str] | None = None,
                      n: int = 3) -> KneserNeyLM:
    lm = KneserNeyLM(n=n)
    lm.fit(SYLLABUS_SENTENCES + list(extra_texts or []))
    lm.save(out_path)
    return lm


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/ngram_lm.pkl")
    ap.add_argument("--extra", help="optional newline-delimited extra text")
    args = ap.parse_args()
    extra = []
    if args.extra:
        extra = Path(args.extra).read_text("utf-8").splitlines()
    lm = build_syllabus_lm(args.out, extra_texts=extra)
    print(f"[ngram-lm] vocab={len(lm.vocab)} -> {args.out}")
    # quick sanity score
    print("cepstrum score:", lm.score_sequence(tokenize("the cepstrum is important")))
    print("random score :", lm.score_sequence(tokenize("banana rainbow pizza tuesday")))
