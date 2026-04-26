# Multi-Sequence Eval: exp37_stage_b2_orb_omf28d

## What this measures
AUC = probability that a randomly chosen GT-matching track gets a higher cosine score than a randomly chosen non-matching track, for a given expression. 0.50 = chance, 1.00 = perfect. < 0.50 means inverted.

## Headline
- Mean AUC (micro, pooled across 3 seqs): **0.624** (over 33 expressions)
- Mean AUC (macro, per-seq averaged):     **0.577** ± 0.030 (over 12 expressions in ≥2 seqs)
- Seq-0011 only (legacy, for continuity): **—**

## Per-expression breakdown
| Expression | 0005 | 0011 | 0013 | macro μ ± σ | micro | GT counts |
|---|---|---|---|---|---|---|
| vehicles in front of ours | 0.854 | — | — | 0.854 ± — | 0.854 | 295/0/0 |
| cars in front of ours | 0.854 | — | — | 0.854 ± — | 0.854 | 295/0/0 |
| right cars which are parking | — | 0.827 | — | 0.827 ± — | 0.827 | 0/1056/0 |
| right vehicles which are parking | — | 0.827 | — | 0.827 ± — | 0.827 | 0/1056/0 |
| left cars which are parking | 0.354 | 0.808 | — | 0.581 ± 0.227 | 0.718 | 169/1375/0 |
| left vehicles which are parking | 0.355 | 0.808 | — | 0.581 ± 0.227 | 0.718 | 169/1375/0 |
| same direction cars in the left | 0.681 | — | — | 0.681 ± — | 0.681 | 215/0/0 |
| same direction vehicles in the left | 0.681 | — | — | 0.681 ± — | 0.681 | 215/0/0 |
| left cars in the same direction of ours | 0.681 | — | — | 0.681 ± — | 0.681 | 215/0/0 |
| left vehicles in the same direction of ours | 0.681 | — | — | 0.681 ± — | 0.681 | 215/0/0 |
| vehicles which are braking | 0.676 | — | — | 0.676 ± — | 0.676 | 295/0/0 |
| cars which are braking | 0.676 | — | — | 0.676 ± — | 0.676 | 295/0/0 |
| vehicles which are faster than ours | — | 0.675 | — | 0.675 ± — | 0.675 | 0/371/0 |
| cars which are faster than ours | — | 0.671 | — | 0.671 ± — | 0.671 | 0/371/0 |
| moving left pedestrian | — | — | 0.645 | 0.645 ± — | 0.645 | 0/0/253 |
| moving right pedestrian | — | — | 0.642 | 0.642 ± — | 0.642 | 0/0/261 |
| moving pedestrian | — | 0.579 | — | 0.579 ± — | 0.579 | 0/88/0 |
| moving cars | 0.403 | 0.659 | — | 0.531 ± 0.128 | 0.577 | 569/765/0 |
| moving vehicles | 0.403 | 0.659 | — | 0.531 ± 0.128 | 0.577 | 569/765/0 |
| parking cars | — | 0.576 | — | 0.576 ± — | 0.576 | 0/2851/0 |
| parking vehicles | — | 0.576 | — | 0.576 ± — | 0.576 | 0/2851/0 |
| counter direction cars in the left | 0.663 | 0.544 | — | 0.604 ± 0.059 | 0.557 | 526/1692/0 |
| counter direction vehicles in the left | 0.662 | 0.542 | — | 0.602 ± 0.060 | 0.555 | 526/1692/0 |
| vehicles in the same direction of ours | 0.520 | 0.690 | — | 0.605 ± 0.085 | 0.554 | 510/371/0 |
| cars in the same direction of ours | 0.520 | 0.690 | — | 0.605 ± 0.085 | 0.554 | 510/371/0 |
| left cars in the counter direction of ours | 0.661 | 0.540 | — | 0.600 ± 0.061 | 0.554 | 526/1692/0 |
| left vehicles in the counter direction of ours | 0.661 | 0.539 | — | 0.600 ± 0.061 | 0.554 | 526/1692/0 |
| cars in the counter direction of ours | 0.619 | 0.460 | — | 0.540 ± 0.079 | 0.494 | 526/1692/0 |
| vehicles in the counter direction of ours | 0.618 | 0.459 | — | 0.539 ± 0.080 | 0.493 | 526/1692/0 |
| cars in horizon direction | — | 0.490 | — | 0.490 ± — | 0.490 | 0/178/0 |
| vehicles in horizon direction | — | 0.489 | — | 0.489 ± — | 0.489 | 0/178/0 |
| turning cars | — | 0.452 | — | 0.452 ± — | 0.452 | 0/30/0 |
| turning vehicles | — | 0.452 | — | 0.452 ± — | 0.452 | 0/30/0 |
