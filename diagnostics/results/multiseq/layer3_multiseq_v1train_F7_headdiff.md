# Multi-Sequence Eval: v1train_F7_headdiff

## What this measures
AUC = probability that a randomly chosen GT-matching track gets a higher cosine score than a randomly chosen non-matching track, for a given expression. 0.50 = chance, 1.00 = perfect. < 0.50 means inverted.

## Headline
- Mean AUC (micro, pooled across 3 seqs): **0.772** (over 33 expressions)
- Mean AUC (macro, per-seq averaged):     **0.831** ± 0.059 (over 12 expressions in ≥2 seqs)
- Seq-0011 only (legacy, for continuity): **—**

## Per-expression breakdown
| Expression | 0005 | 0011 | 0013 | macro μ ± σ | micro | GT counts |
|---|---|---|---|---|---|---|
| cars in front of ours | 0.996 | — | — | 0.996 ± — | 0.996 | 295/0/0 |
| vehicles in front of ours | 0.996 | — | — | 0.996 ± — | 0.996 | 295/0/0 |
| cars which are braking | 0.983 | — | — | 0.983 ± — | 0.983 | 295/0/0 |
| vehicles which are braking | 0.983 | — | — | 0.983 ± — | 0.983 | 295/0/0 |
| right vehicles which are parking | — | 0.935 | — | 0.935 ± — | 0.935 | 0/1056/0 |
| right cars which are parking | — | 0.935 | — | 0.935 ± — | 0.935 | 0/1056/0 |
| left cars in the counter direction of ours | 0.820 | 0.969 | — | 0.894 ± 0.075 | 0.910 | 526/1692/0 |
| counter direction cars in the left | 0.819 | 0.969 | — | 0.894 ± 0.075 | 0.910 | 526/1692/0 |
| counter direction vehicles in the left | 0.819 | 0.969 | — | 0.894 ± 0.075 | 0.910 | 526/1692/0 |
| left vehicles in the counter direction of ours | 0.820 | 0.969 | — | 0.894 ± 0.074 | 0.910 | 526/1692/0 |
| moving right pedestrian | — | — | 0.851 | 0.851 ± — | 0.851 | 0/0/261 |
| vehicles in the same direction of ours | 0.779 | 0.980 | — | 0.879 ± 0.100 | 0.805 | 510/371/0 |
| cars in the same direction of ours | 0.779 | 0.980 | — | 0.879 ± 0.100 | 0.805 | 510/371/0 |
| moving left pedestrian | — | — | 0.791 | 0.791 ± — | 0.791 | 0/0/253 |
| left cars which are parking | 0.641 | 0.889 | — | 0.765 ± 0.124 | 0.780 | 169/1375/0 |
| left vehicles which are parking | 0.641 | 0.889 | — | 0.765 ± 0.124 | 0.780 | 169/1375/0 |
| vehicles in the counter direction of ours | 0.812 | 0.745 | — | 0.779 ± 0.033 | 0.764 | 526/1692/0 |
| cars in the counter direction of ours | 0.812 | 0.745 | — | 0.778 ± 0.033 | 0.764 | 526/1692/0 |
| moving vehicles | 0.787 | 0.761 | — | 0.774 ± 0.013 | 0.751 | 569/765/0 |
| moving cars | 0.785 | 0.761 | — | 0.773 ± 0.012 | 0.750 | 569/765/0 |
| turning cars | — | 0.731 | — | 0.731 ± — | 0.731 | 0/30/0 |
| turning vehicles | — | 0.731 | — | 0.731 ± — | 0.731 | 0/30/0 |
| vehicles in horizon direction | — | 0.674 | — | 0.674 ± — | 0.674 | 0/178/0 |
| cars in horizon direction | — | 0.671 | — | 0.671 ± — | 0.671 | 0/178/0 |
| same direction cars in the left | 0.644 | — | — | 0.644 ± — | 0.644 | 215/0/0 |
| same direction vehicles in the left | 0.644 | — | — | 0.644 ± — | 0.644 | 215/0/0 |
| left vehicles in the same direction of ours | 0.642 | — | — | 0.642 ± — | 0.642 | 215/0/0 |
| left cars in the same direction of ours | 0.642 | — | — | 0.642 ± — | 0.642 | 215/0/0 |
| vehicles which are faster than ours | — | 0.612 | — | 0.612 ± — | 0.612 | 0/371/0 |
| cars which are faster than ours | — | 0.610 | — | 0.610 ± — | 0.610 | 0/371/0 |
| parking cars | — | 0.533 | — | 0.533 ± — | 0.533 | 0/2851/0 |
| parking vehicles | — | 0.532 | — | 0.532 ± — | 0.532 | 0/2851/0 |
| moving pedestrian | — | 0.504 | — | 0.504 ± — | 0.504 | 0/88/0 |
