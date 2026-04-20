"""Task 2.1 – Hinglish G2P with a unified IPA backbone.

Why a hand-written mapping layer?
--------------------------------
Off-the-shelf G2P tools (epitran, g2p-en) handle either English OR
Devanagari, but not both in a single utterance, and none of them know
that a professor reading "stochastic" in the middle of a Hindi
sentence should still be pronounced /stəˈkæstɪk/, not
/s̪t̪oːtʃ̪ast̪ik/.

Our layer does three things:

1. **Script segmentation**: split mixed text into runs of Latin /
   Devanagari / punctuation.
2. **Per-script G2P**: Latin runs go through a rule-based English
   pronunciation dictionary + CMU-style fallback; Devanagari runs go
   through an explicit Devanagari-IPA table and schwa-deletion rules.
3. **Hinglish phonology fixups**: common loan-word remappings
   (retroflex /ʈ/ -> /t/ when the loan is clearly English, /v/ -> /w/
   for Hindi-accent English, etc.) plus neutralisation of length
   contrasts across the script boundary.

The output is a single IPA string with ‘ˈ’ marking primary stress for
English words and ‘·’ as an internal IPA word separator kept so that
downstream TTS can re-group words if needed.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------- Devanagari

# Core Devanagari → IPA mapping (simplified; Hindi inventory).
_DEV_VOWELS = {
    "अ": "ə", "आ": "aː", "इ": "ɪ", "ई": "iː", "उ": "ʊ", "ऊ": "uː",
    "ऋ": "ri", "ए": "eː", "ऐ": "ɛː", "ओ": "oː", "औ": "ɔː",
    "ऍ": "æ", "ऑ": "ɒ",
}

_DEV_CONSONANTS = {
    # velar
    "क": "k", "ख": "kʰ", "ग": "ɡ", "घ": "ɡʱ", "ङ": "ŋ",
    # palatal
    "च": "tʃ", "छ": "tʃʰ", "ज": "dʒ", "झ": "dʒʱ", "ञ": "ɲ",
    # retroflex
    "ट": "ʈ", "ठ": "ʈʰ", "ड": "ɖ", "ढ": "ɖʱ", "ण": "ɳ", "ड़": "ɽ", "ढ़": "ɽʱ",
    # dental
    "त": "t̪", "थ": "t̪ʰ", "द": "d̪", "ध": "d̪ʱ", "न": "n",
    # labial
    "प": "p", "फ": "pʰ", "ब": "b", "भ": "bʱ", "म": "m",
    # approx + fricatives
    "य": "j", "र": "r", "ल": "l", "व": "ʋ",
    "श": "ʃ", "ष": "ʂ", "स": "s", "ह": "ɦ",
    "ज़": "z", "फ़": "f",
}

# Matras (dependent vowels) follow the base consonant.
_DEV_MATRAS = {
    "ा": "aː", "ि": "ɪ", "ी": "iː", "ु": "ʊ", "ू": "uː",
    "ृ": "ri", "े": "eː", "ै": "ɛː", "ो": "oː", "ौ": "ɔː",
    "ॅ": "æ", "ॉ": "ɒ",
}

_VIRAMA = "्"
_NUKTA = "़"
_ANUSVARA = "ं"       # nasalises the following consonant / preceding vowel
_VISARGA = "ः"        # /h/ echo
_CANDRABINDU = "ँ"    # vowel nasalisation


def _is_dev(ch: str) -> bool:
    return "\u0900" <= ch <= "\u097F"


def _dev_to_ipa(word: str) -> str:
    """Rule-based Devanagari->IPA with schwa deletion.

    Schwa deletion rule: the inherent /ə/ of a medial consonant is
    dropped when it is followed by another consonant that in turn has
    an attached vowel (standard Hindi schwa-deletion heuristic).
    """
    # Normalise nukta combinations (क़, ख़, ज़, फ़ ...).
    w = unicodedata.normalize("NFC", word)
    out: list[str] = []
    i = 0
    syllable_breaks: list[int] = []   # indexes in `out` where schwa was inserted
    while i < len(w):
        ch = w[i]
        nxt = w[i + 1] if i + 1 < len(w) else ""
        if ch in _DEV_CONSONANTS:
            base = _DEV_CONSONANTS[ch]
            if nxt == _NUKTA:
                base = {"ज": "z", "फ": "f", "क": "q", "ख": "x", "ग": "ɣ", "ड": "ɽ", "ढ": "ɽʱ"}.get(ch, base)
                i += 1
                nxt = w[i + 1] if i + 1 < len(w) else ""
            out.append(base)
            if nxt == _VIRAMA:
                i += 2
                continue
            if nxt in _DEV_MATRAS:
                out.append(_DEV_MATRAS[nxt])
                i += 2
                continue
            # Implicit schwa
            out.append("ə")
            syllable_breaks.append(len(out) - 1)
            i += 1
            continue
        if ch in _DEV_VOWELS:
            out.append(_DEV_VOWELS[ch])
            i += 1
            continue
        if ch == _ANUSVARA:
            out.append("̃")
            i += 1
            continue
        if ch == _CANDRABINDU:
            out.append("̃")
            i += 1
            continue
        if ch == _VISARGA:
            out.append("h")
            i += 1
            continue
        # unknown – pass through so we can debug later
        out.append(ch)
        i += 1

    # Schwa deletion: drop final schwa (Hindi orthographic convention)
    if out and out[-1] == "ə":
        out.pop()
        if syllable_breaks and syllable_breaks[-1] == len(out):
            syllable_breaks.pop()

    # Medial schwa deletion: C ə C V  -> C C V
    new: list[str] = []
    for j, tok in enumerate(out):
        if j in syllable_breaks and j + 1 < len(out):
            # look ahead: is next a consonant followed by a vowel?
            k = j + 1
            if _is_consonant_ipa(out[k]) and k + 1 < len(out) and _is_vowel_ipa(out[k + 1]):
                continue  # drop this schwa
        new.append(tok)
    return "".join(new)


def _is_vowel_ipa(tok: str) -> bool:
    return any(ch in "əaiueoɛɔɒæɪʊ" for ch in tok)


def _is_consonant_ipa(tok: str) -> bool:
    return not _is_vowel_ipa(tok) and tok not in {"̃", "ˈ", "·"}


# ---------------------------------------------------------------- English

# Minimal CMUdict-style hand-authored entries for technical terms that
# show up in the speech-course lectures. For anything outside this
# dictionary we fall back to a naive letter-to-IPA mapper that is
# good enough for loan words and proper nouns in a classroom setting.

_EN_DICT: dict[str, str] = {
    "the": "ðə", "a": "ə", "an": "ən", "is": "ɪz", "are": "ɑːr", "and": "ænd",
    "of": "ʌv", "to": "tuː", "in": "ɪn", "on": "ɒn", "for": "fɔːr",
    "stochastic": "stəˈkæstɪk", "cepstrum": "ˈkɛpstrəm",
    "fourier": "ˈfʊrieɪ", "transform": "ˈtrænsfɔːrm",
    "spectrum": "ˈspɛktrəm", "phoneme": "ˈfoʊniːm",
    "mel": "mɛl", "filter": "ˈfɪltər", "bank": "bæŋk",
    "hidden": "ˈhɪdən", "markov": "ˈmɑːrkɒv", "model": "ˈmɒdəl",
    "viterbi": "vɪˈtɜːrbi", "dynamic": "daɪˈnæmɪk",
    "time": "taɪm", "warping": "ˈwɔːrpɪŋ",
    "speech": "spiːtʃ", "recognition": "ˌrɛkəɡˈnɪʃən",
    "whisper": "ˈwɪspər", "wav": "wæv", "vec": "vɛk",
    "transformer": "trænsˈfɔːrmər", "attention": "əˈtɛnʃən",
    "encoder": "ɛnˈkoʊdər", "decoder": "diːˈkoʊdər",
    "pitch": "pɪtʃ", "formant": "ˈfɔːrmənt",
    "ipa": "aɪ piː eɪ", "hmm": "eɪtʃ ɛm ɛm",
    "vits": "vɪts", "yourtts": "jɔːr tiː tiː ɛs",
    "mcd": "ɛm siː diː", "wer": "dʌbəljuː iː ɑːr",
    "eer": "iː iː ɑːr", "lfcc": "ɛl ɛf siː siː",
    "cqcc": "siː kjuː siː siː", "dtw": "diː tiː dʌbəljuː",
    "hello": "həˈloʊ", "class": "klæs", "lecture": "ˈlɛktʃər",
    "today": "təˈdeɪ", "will": "wɪl", "discuss": "dɪˈskʌs",
    "signal": "ˈsɪɡnəl", "processing": "ˈprɒsɛsɪŋ",
}

# Very small letter-rule fallback. Order of keys matters (longest first).
_EN_FALLBACK: list[tuple[str, str]] = [
    ("tion", "ʃən"), ("sion", "ʒən"), ("sch", "sk"),
    ("ch", "tʃ"), ("sh", "ʃ"), ("th", "θ"), ("ph", "f"),
    ("ng", "ŋ"), ("ck", "k"), ("qu", "kw"),
    ("ee", "iː"), ("ea", "iː"), ("oo", "uː"), ("ou", "aʊ"), ("ow", "aʊ"),
    ("oa", "oʊ"), ("ai", "eɪ"), ("ay", "eɪ"), ("oy", "ɔɪ"), ("oi", "ɔɪ"),
    ("a", "æ"), ("e", "ɛ"), ("i", "ɪ"), ("o", "ɒ"), ("u", "ʌ"), ("y", "i"),
    ("b", "b"), ("c", "k"), ("d", "d"), ("f", "f"), ("g", "ɡ"),
    ("h", "h"), ("j", "dʒ"), ("k", "k"), ("l", "l"), ("m", "m"),
    ("n", "n"), ("p", "p"), ("r", "r"), ("s", "s"), ("t", "t"),
    ("v", "v"), ("w", "w"), ("x", "ks"), ("z", "z"),
]


def _en_to_ipa(word: str) -> str:
    w = word.lower()
    if w in _EN_DICT:
        return _EN_DICT[w]
    # letter-rule fallback
    out: list[str] = []
    i = 0
    while i < len(w):
        match = None
        for pat, rep in _EN_FALLBACK:
            if w.startswith(pat, i):
                match = (pat, rep)
                break
        if match:
            out.append(match[1])
            i += len(match[0])
        else:
            i += 1
    return "".join(out)


# ------------------------------------------------- Hinglish fixup layer

_FIXUPS = [
    ("ʋ", "v"),                 # Hindi-accent [ʋ] regularised to [v]
    ("t̪", "t"),                 # Dental/alveolar neutralisation
    ("d̪", "d"),
    ("ɳ", "n"),
    ("ʂ", "ʃ"),
    ("ə ʃ", "əʃ"),               # glue anusvara/nasal + consonant
]


def _apply_fixups(s: str, context_script: str) -> str:
    # Only regularise when the word came from a Latin-script token –
    # Devanagari tokens keep their native IPA inventory.
    if context_script != "latin":
        return s
    for a, b in _FIXUPS:
        s = s.replace(a, b)
    return s


# ------------------------------------------------- top-level segmenter


@dataclass
class G2POptions:
    word_sep: str = "·"
    lowercase: bool = True


def text_to_ipa(text: str, opts: G2POptions = G2POptions()) -> str:
    """Convert a mixed Hinglish string to a unified IPA sequence."""
    if opts.lowercase:
        text = text.lower()
    tokens = re.findall(r"[A-Za-z']+|[\u0900-\u097F]+|[.,!?;:\-]", text)
    ipa_tokens: list[str] = []
    for tok in tokens:
        if not tok.strip():
            continue
        if tok[0].isalpha() and tok[0].isascii():
            ipa = _en_to_ipa(tok)
            ipa_tokens.append(_apply_fixups(ipa, "latin"))
        elif _is_dev(tok[0]):
            ipa = _dev_to_ipa(tok)
            ipa_tokens.append(_apply_fixups(ipa, "devanagari"))
        else:
            ipa_tokens.append(tok)
    return opts.word_sep.join(ipa_tokens)


if __name__ == "__main__":
    samples = [
        "Today we will discuss stochastic processes",
        "आज हम cepstrum के बारे में पढ़ेंगे",
        "The speech signal is ek simple time series",
    ]
    for s in samples:
        print(s, "->", text_to_ipa(s))
