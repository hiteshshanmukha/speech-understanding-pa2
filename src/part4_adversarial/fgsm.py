"""Task 4.2 – FGSM adversarial attack on the LID classifier.

Goal: find the smallest L∞ perturbation epsilon that flips the LID
prediction on a Hindi clip to 'English' while keeping the SNR between
the clean and perturbed signal above 40 dB.

We implement a line search over epsilon: for each trial epsilon we

    delta = epsilon * sign( dL / dx )

where L = cross-entropy(LID(x + delta), y_target='EN'). If the attack
succeeds and SNR(x, x+delta) > 40 dB we record that epsilon and keep
shrinking; otherwise we enlarge. Binary search gives us the minimum
passing epsilon in O(log(eps_max/tol)) iterations.

The waveform is clipped to [-1, 1] after each step so the output is
still a legal 16-bit WAV.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..part1_stt.lid import L2I, MultiHeadLID
from ..common.audio_io import snr_db


@dataclass
class FGSMResult:
    epsilon: float
    flipped: bool
    snr_db: float
    adv: torch.Tensor


def _fgsm_step(model: MultiHeadLID, wav: torch.Tensor, target: int,
               epsilon: float) -> torch.Tensor:
    wav = wav.detach().clone().requires_grad_(True)
    out = model(wav.unsqueeze(0))
    # Encourage the utterance prediction to become `target` (EN).
    logits = out["utt_logits"]
    loss = F.cross_entropy(logits, torch.tensor([target], device=wav.device))
    grad = torch.autograd.grad(loss, wav)[0]
    # minimise loss -> step in -sign(grad) (targeted)
    adv = wav.detach() - epsilon * grad.sign()
    return adv.clamp_(-1.0, 1.0)


def flips_prediction(model: MultiHeadLID, wav: torch.Tensor,
                     target: int) -> bool:
    with torch.no_grad():
        out = model(wav.unsqueeze(0))
    return int(out["utt_logits"].argmax(-1)) == target


def fgsm_min_epsilon(model: MultiHeadLID, wav: torch.Tensor,
                     target_label: str = "EN",
                     snr_floor: float = 40.0,
                     eps_lo: float = 1e-5, eps_hi: float = 1e-1,
                     n_iters: int = 18) -> FGSMResult:
    """Binary search for the smallest epsilon that (a) flips the
    prediction and (b) keeps SNR above `snr_floor`. If the latter
    constraint is unreachable the closest passing epsilon is returned
    and `flipped=False`."""
    target = L2I[target_label]
    model.eval()
    best: FGSMResult | None = None
    lo, hi = eps_lo, eps_hi
    for _ in range(n_iters):
        mid = (lo + hi) / 2
        adv = _fgsm_step(model, wav, target, mid)
        flipped = flips_prediction(model, adv, target)
        s = snr_db(wav, adv)
        if flipped and s >= snr_floor:
            best = FGSMResult(epsilon=mid, flipped=True, snr_db=s, adv=adv)
            hi = mid
        else:
            lo = mid
    if best is None:
        # fall back to the largest trial so the caller still has a waveform
        adv = _fgsm_step(model, wav, target, hi)
        best = FGSMResult(
            epsilon=hi, flipped=flips_prediction(model, adv, target),
            snr_db=snr_db(wav, adv), adv=adv,
        )
    return best


if __name__ == "__main__":
    import argparse
    import torchaudio
    import soundfile as sf
    from ..part1_stt.lid import load as load_lid

    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--ckpt", default="models/lid.pt")
    ap.add_argument("--target", default="EN")
    ap.add_argument("--out", default="data/adv.wav")
    args = ap.parse_args()
    model = load_lid(args.ckpt)
    data, sr = sf.read(args.wav, dtype="float32", always_2d=True)
    w = torch.from_numpy(data.T).mean(0)
    if sr != 16_000:
        w = torchaudio.functional.resample(w, sr, 16_000)
    r = fgsm_min_epsilon(model, w, target_label=args.target)
    sf.write(args.out, r.adv.detach().numpy(), 16_000, subtype="PCM_16")
    print(f"[fgsm] eps={r.epsilon:.5f}  flipped={r.flipped}  snr={r.snr_db:.1f} dB -> {args.out}")
