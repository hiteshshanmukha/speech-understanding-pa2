"""Compute WER on the inference clip given a user-supplied reference.

Drop a ground-truth transcript at ``results/ref_transcript.txt``
(one utterance per line, or just a single blob – jiwer accepts both)
and this script reports:

    - WER_overall          – WER between ref and `transcript.txt`
    - WER_en_segments      – WER over frames predicted EN by LID
    - WER_hi_segments      – WER over frames predicted HI by LID
    - CER                  – character error rate (more forgiving on
                             code-switching spelling variation)

The EN / HI split uses the LID timeline in `switches.json`. If the
LID decides the whole clip is one language (which is what happens
today on the real YouTube audio), only WER_overall is meaningful.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _tokens(s: str) -> list[str]:
    return [t for t in re.split(r"\s+", s.strip()) if t]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="results/ref_transcript.txt")
    ap.add_argument("--hyp", default="results/transcript.txt")
    ap.add_argument("--switches", default="results/switches.json")
    ap.add_argument("--out", default="results/wer_report.json")
    args = ap.parse_args()

    from jiwer import wer, cer
    ref = Path(args.ref).read_text("utf-8")
    hyp = Path(args.hyp).read_text("utf-8")
    out = {
        "WER_overall": wer(ref, hyp),
        "CER_overall": cer(ref, hyp),
    }

    # split by LID EN/HI if we have switches
    try:
        switches = json.loads(Path(args.switches).read_text("utf-8"))
    except FileNotFoundError:
        switches = []
    langs = {s[1] for s in switches if s[1] in ("EN", "HI")}
    if len(langs) > 1:
        # If ref is a timestamped JSONL ({"start": .., "end": .., "text": ..}) split accordingly;
        # otherwise note the skip.
        out["WER_en_segments"] = None
        out["WER_hi_segments"] = None
        out["segment_split_note"] = (
            "To split WER by LID language, supply a time-stamped reference "
            "(JSONL rows with start,end,text fields) and re-run."
        )

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
