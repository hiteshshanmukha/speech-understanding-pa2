"""Whisper-based LID fallback for real-world noisy recordings.

Task 1.1 requires a custom LID model, which we provide in `lid.py`.
That model generalises well on the FLEURS / LibriSpeech distribution
(train + synthetic CS test: frame-F1 0.833, boundary-F1 0.780) but
the YouTube classroom recording we were handed is noisier and
speaker-distant; our custom LID's FLEURS-trained Hindi representation
does not transfer to that distribution out-of-the-box.

As a reporting-only companion (the custom LID remains the "LID under
test" for Tasks 1.1 / 4.2), this module produces a frame-level EN /
HI / SIL timeline by probing Whisper's built-in language-detection
head on short windows:

    window_s = 3.0, hop_s = 1.0
    for every window: softmax over <|en|>, <|hi|> after one decoder step.
    mark SIL when RMS < sil_dbfs, else argmax(EN, HI).
    interpolate back to the 10-ms LID grid.

Whisper is orders of magnitude larger than our 0.5M-parameter LID and
has seen orders of magnitude more noisy speech, so it is useful as a
sanity reference and, in the report, as the "teacher" distribution we
would distil our LID onto if extended training were allowed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .lid import FRAME_HOP_MS, L2I


WINDOW_S = 3.0
HOP_S = 1.0
SIL_DBFS = -55.0


def _dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(x ** 2)) + 1e-9)
    return 20.0 * np.log10(rms)


@torch.no_grad()
def whisper_lid_frames(wav_path: str | Path,
                       model_id: str = "openai/whisper-tiny",
                       batch: int = 16,
                       device: str = "cpu") -> tuple[torch.Tensor, int]:
    """Return (frame-level labels, n_frames)."""
    from transformers import AutoProcessor, WhisperForConditionalGeneration

    proc = AutoProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device).eval()
    en_id = proc.tokenizer.convert_tokens_to_ids("<|en|>")
    hi_id = proc.tokenizer.convert_tokens_to_ids("<|hi|>")
    sot_id = proc.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

    wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1).astype(np.float32)
    assert sr == 16_000, "this helper assumes 16-kHz input"
    win = int(WINDOW_S * sr); hop = int(HOP_S * sr)
    hop_10ms = int(sr * FRAME_HOP_MS / 1000)
    n_frames = wav.shape[0] // hop_10ms

    # Collect probe windows + coarse labels (one per window).
    probes, ts, sils = [], [], []
    for t0 in range(0, max(1, wav.shape[0] - win + 1), hop):
        chunk = wav[t0:t0 + win]
        ts.append(t0 / sr)
        if _dbfs(chunk) < SIL_DBFS:
            sils.append(True)
            probes.append(None)
        else:
            sils.append(False)
            probes.append(chunk)

    # Batch Whisper encoder + 1-step decoder over non-silent probes.
    nonsil = [p for p in probes if p is not None]
    results: list[tuple[float, float]] = []
    for i in range(0, len(nonsil), batch):
        b = nonsil[i:i + batch]
        feats = proc(b, sampling_rate=sr,
                     return_tensors="pt").input_features.to(device)
        enc = model.get_encoder()(feats).last_hidden_state
        dec = torch.full((feats.shape[0], 1), sot_id,
                         dtype=torch.long, device=device)
        logits = model(encoder_outputs=(enc,),
                       decoder_input_ids=dec).logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        for p in probs:
            results.append((float(p[en_id]), float(p[hi_id])))

    # Assemble per-window labels, then splat onto the 10-ms grid.
    win_labels: list[int] = []
    r_it = iter(results)
    for is_sil in sils:
        if is_sil:
            win_labels.append(L2I["SIL"])
        else:
            p_en, p_hi = next(r_it)
            win_labels.append(L2I["EN"] if p_en >= p_hi else L2I["HI"])

    # Each window decides ``HOP_S`` seconds = HOP_S/0.01 frames.
    frames_per_hop = int(HOP_S / (FRAME_HOP_MS / 1000))
    frame_preds = np.full(n_frames, L2I["EN"], dtype=np.int64)
    for k, lbl in enumerate(win_labels):
        i0 = k * frames_per_hop
        i1 = min(n_frames, i0 + frames_per_hop)
        frame_preds[i0:i1] = lbl
    return torch.from_numpy(frame_preds), n_frames


if __name__ == "__main__":
    import argparse, json, sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", default="results/switches_whisper.json")
    ap.add_argument("--model", default="openai/whisper-tiny")
    args = ap.parse_args()

    pred, n = whisper_lid_frames(args.wav, model_id=args.model)
    from .lid import frames_to_switches
    switches = frames_to_switches(pred, hop_ms=FRAME_HOP_MS)
    Path(args.out).write_text(json.dumps(switches, indent=2))
    print(f"[whisper-lid] {n} frames, {len(switches)} switches -> {args.out}")
