"""Back-transliterate IPA → Devanagari.

Used for feeding MMS-Maithili TTS, whose tokenizer only accepts
Devanagari characters. The mapping is intentionally phoneme-level: we
inspect the IPA string one grapheme cluster at a time and emit the
closest Devanagari consonant-vowel pair, re-introducing the schwa ‘अ’
when a bare consonant appears without a following vowel.

This is the inverse of the forward G2P in ``hinglish_g2p.py`` and
re-uses the same phone tables so that a round-trip
``text → IPA → Deva`` stays consistent.
"""
from __future__ import annotations

import re


# IPA vowel -> (independent form, matra form)
_VOWEL: dict[str, tuple[str, str]] = {
    "ə":  ("अ", ""),
    "a":  ("अ", ""),
    "aː": ("आ", "ा"),
    "ɑ":  ("आ", "ा"),
    "ɪ":  ("इ", "ि"),
    "i":  ("इ", "ि"),
    "iː": ("ई", "ी"),
    "ʊ":  ("उ", "ु"),
    "u":  ("उ", "ु"),
    "uː": ("ऊ", "ू"),
    "e":  ("ए", "े"),
    "eː": ("ए", "े"),
    "ɛ":  ("ऐ", "ै"),
    "ɛː": ("ऐ", "ै"),
    "æ":  ("ऐ", "ै"),
    "o":  ("ओ", "ो"),
    "oː": ("ओ", "ो"),
    "ɒ":  ("औ", "ौ"),
    "ɔ":  ("औ", "ौ"),
    "ɔː": ("औ", "ौ"),
    "ʌ":  ("अ", ""),
}

_DIPHTHONGS: dict[str, tuple[str, str]] = {
    "eɪ": ("ए", "े"),
    "aɪ": ("आइ", "ाइ"),
    "ɔɪ": ("आइ", "ाइ"),
    "aʊ": ("आउ", "ाउ"),
    "oʊ": ("ओ", "ो"),
}

_CONSONANT: dict[str, str] = {
    "p": "प", "pʰ": "फ", "b": "ब", "bʱ": "भ", "m": "म",
    "t": "ट", "t̪": "त", "tʰ": "ठ", "t̪ʰ": "थ", "d": "ड", "d̪": "द",
    "dʱ": "ध", "d̪ʱ": "ध", "n": "न", "ɳ": "ण",
    "k": "क", "kʰ": "ख", "ɡ": "ग", "ɡʱ": "घ", "ŋ": "ङ",
    "tʃ": "च", "tʃʰ": "छ", "dʒ": "ज", "dʒʱ": "झ", "ɲ": "ञ",
    "ʈ": "ट", "ʈʰ": "ठ", "ɖ": "ड", "ɖʱ": "ढ",
    "f": "फ़", "v": "व", "ʋ": "व",
    "s": "स", "z": "ज़", "ʃ": "श", "ʂ": "ष",
    "h": "ह", "ɦ": "ह",
    "j": "य", "r": "र", "ɽ": "ड़", "l": "ल", "w": "व",
    "θ": "थ", "ð": "द",
}

_SEPS = re.compile(r"[·\s]+")

# Canonical multi-char IPA units first, then single characters.
_ORDERED_KEYS: list[str] = sorted(
    list(_CONSONANT.keys()) + list(_DIPHTHONGS.keys()) + list(_VOWEL.keys()),
    key=lambda s: -len(s),
)


def _consume(ipa: str, i: int) -> tuple[str, int, str]:
    """Return (kind, new_i, value) where kind ∈ {'C','V','D','SKIP'}."""
    for k in _ORDERED_KEYS:
        if ipa.startswith(k, i):
            if k in _CONSONANT:
                return "C", i + len(k), _CONSONANT[k]
            if k in _DIPHTHONGS:
                return "D", i + len(k), k
            if k in _VOWEL:
                return "V", i + len(k), k
    return "SKIP", i + 1, ""


def ipa_to_devanagari(ipa: str) -> str:
    """Convert an IPA string (with `·` word separators) to Devanagari.

    The output is suitable for a char-tokenised Devanagari TTS such as
    MMS-Maithili. Bare consonants get the implicit schwa ‘अ’;
    consonants followed by a vowel get the matra form; initial vowels
    get the independent form. Unknown characters are dropped so the
    TTS tokenizer never sees an out-of-alphabet symbol.
    """
    out: list[str] = []
    i = 0
    n = len(ipa)
    pending_consonant = False          # emitted a bare consonant, next vowel should be matra
    while i < n:
        ch = ipa[i]
        if _SEPS.match(ch):
            if pending_consonant:
                pending_consonant = False
            out.append(" ")
            i += 1
            continue
        if ch in ",.!?;:-":
            out.append(ch)
            i += 1
            continue

        kind, new_i, val = _consume(ipa, i)
        if kind == "C":
            if pending_consonant:
                out.append("अ")                 # previous bare C gets schwa
            out.append(val)
            pending_consonant = True
        elif kind == "V":
            indep, matra = _VOWEL[val]
            if pending_consonant:
                out.append(matra)
                pending_consonant = False
            else:
                out.append(indep)
        elif kind == "D":
            indep, matra = _DIPHTHONGS[val]
            if pending_consonant:
                out.append(matra)
                pending_consonant = False
            else:
                out.append(indep)
        # else: SKIP
        i = new_i

    if pending_consonant:
        out.append("अ")
    return "".join(out).strip()


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    samples = [
        "ðə·kɛpstrəm·ɪz·ɪmpɒrtænt",
        "bʌt·ɪll·nɛvɛr·bɛ·lækɪŋ·ɪn·mi·ɛffɒrts",
        "aːdʒ·ɦəm·ˈkɛpstrəm·keː·baːreː·meː·pɽʱeːɡeː",
    ]
    for s in samples:
        print(s, "->", ipa_to_devanagari(s))
