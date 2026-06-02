# Held-out tau gate — 2-fold CV on the 18 color exprs (iKUN seed0)

Split: seqmap_colorA.txt (9) + seqmap_colorB.txt (9), sorted-alternating. Rerank each fold
ALONE (ship base, sharedweight seed0), sweep tau; pick tau on train fold, apply to val fold.
ship pooled = 44.561.

| tau | A-alone pooled | A subset | B-alone pooled | B subset |
|-----|------|------|------|------|
|0.205| 44.641 | 33.2 | 44.443 | 29.4 |
|0.210| 44.763 | 35.3 | 44.478 | 26.6 |
|0.216| 44.854 | 37.0 | 44.534 | 23.5 |
|0.220| 44.786 | 32.1 | 44.579 | 24.2 |

- tau ROBUST: A-opt 0.216, B-opt 0.220 (Δ=0.004) → threshold generalizes, NOT overfit.
- CV held-out full-color gain = +0.198 (val A @ tauB=0.220 → +0.225; val B @ tauA=0.216 → −0.027).
- vs in-sample (tau=0.216, full color) +0.263 → overfit penalty +0.065 (~25%).
- Shrink is uneven per-expr pooled leverage (fold A carries +0.293, fold B +0.018), NOT tau overfit.

Honest stack = Path B n=3 +0.339 + rerank held-out +0.198 = +0.537 (in-sample stack was +0.603).
GATE PASS: gain stays POS, tau robust, stack clears +0.5.
