# Multi-Sequence Eval: v1train_F5_nbrmean

## What this measures
AUC = probability that a randomly chosen GT-matching track gets a higher cosine score than a randomly chosen non-matching track, for a given expression. 0.50 = chance, 1.00 = perfect. < 0.50 means inverted.

## Headline
- Mean AUC (micro, pooled across 3 seqs): **0.756** (over 33 expressions)
- Mean AUC (macro, per-seq averaged):     **0.809** ± 0.060 (over 12 expressions in ≥2 seqs)
- Seq-0011 only (legacy, for continuity): **—**

## Per-expression breakdown
| Expression | 0005 | 0011 | 0013 | macro μ ± σ | micro | GT counts |
|---|---|---|---|---|---|---|
| cars in front of ours | 0.992 | — | — | 0.992 ± — | 0.992 | 295/0/0 |
| vehicles in front of ours | 0.992 | — | — | 0.992 ± — | 0.992 | 295/0/0 |
| cars which are braking | 0.971 | — | — | 0.971 ± — | 0.971 | 295/0/0 |
| vehicles which are braking | 0.971 | — | — | 0.971 ± — | 0.971 | 295/0/0 |
| right vehicles which are parking | — | 0.940 | — | 0.940 ± — | 0.940 | 0/1056/0 |
| right cars which are parking | — | 0.940 | — | 0.940 ± — | 0.940 | 0/1056/0 |
| left vehicles in the counter direction of ours | 0.800 | 0.955 | — | 0.878 ± 0.077 | 0.903 | 526/1692/0 |
| counter direction cars in the left | 0.801 | 0.954 | — | 0.877 ± 0.077 | 0.902 | 526/1692/0 |
| left cars in the counter direction of ours | 0.801 | 0.953 | — | 0.877 ± 0.076 | 0.902 | 526/1692/0 |
| counter direction vehicles in the left | 0.800 | 0.953 | — | 0.877 ± 0.076 | 0.902 | 526/1692/0 |
| moving right pedestrian | — | — | 0.853 | 0.853 ± — | 0.853 | 0/0/261 |
| vehicles in the same direction of ours | 0.758 | 0.940 | — | 0.849 ± 0.091 | 0.803 | 510/371/0 |
| cars in the same direction of ours | 0.758 | 0.940 | — | 0.849 ± 0.091 | 0.803 | 510/371/0 |
| left vehicles which are parking | 0.650 | 0.878 | — | 0.764 ± 0.114 | 0.799 | 169/1375/0 |
| left cars which are parking | 0.649 | 0.878 | — | 0.764 ± 0.114 | 0.799 | 169/1375/0 |
| moving left pedestrian | — | — | 0.797 | 0.797 ± — | 0.797 | 0/0/253 |
| moving cars | 0.788 | 0.729 | — | 0.758 ± 0.030 | 0.758 | 569/765/0 |
| moving vehicles | 0.787 | 0.728 | — | 0.758 ± 0.029 | 0.758 | 569/765/0 |
| vehicles in the counter direction of ours | 0.800 | 0.662 | — | 0.731 ± 0.069 | 0.704 | 526/1692/0 |
| cars in the counter direction of ours | 0.800 | 0.662 | — | 0.731 ± 0.069 | 0.704 | 526/1692/0 |
| vehicles in horizon direction | — | 0.692 | — | 0.692 ± — | 0.692 | 0/178/0 |
| cars in horizon direction | — | 0.692 | — | 0.692 ± — | 0.692 | 0/178/0 |
| left cars in the same direction of ours | 0.634 | — | — | 0.634 ± — | 0.634 | 215/0/0 |
| same direction cars in the left | 0.633 | — | — | 0.633 ± — | 0.633 | 215/0/0 |
| left vehicles in the same direction of ours | 0.632 | — | — | 0.632 ± — | 0.632 | 215/0/0 |
| same direction vehicles in the left | 0.632 | — | — | 0.632 ± — | 0.632 | 215/0/0 |
| turning vehicles | — | 0.619 | — | 0.619 ± — | 0.619 | 0/30/0 |
| turning cars | — | 0.618 | — | 0.618 ± — | 0.618 | 0/30/0 |
| cars which are faster than ours | — | 0.583 | — | 0.583 ± — | 0.583 | 0/371/0 |
| vehicles which are faster than ours | — | 0.580 | — | 0.580 ± — | 0.580 | 0/371/0 |
| parking cars | — | 0.489 | — | 0.489 ± — | 0.489 | 0/2851/0 |
| parking vehicles | — | 0.489 | — | 0.489 ± — | 0.489 | 0/2851/0 |
| moving pedestrian | — | 0.466 | — | 0.466 ± — | 0.466 | 0/88/0 |
