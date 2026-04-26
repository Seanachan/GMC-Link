# Multi-Sequence Eval: v1train_F2_heading

## What this measures
AUC = probability that a randomly chosen GT-matching track gets a higher cosine score than a randomly chosen non-matching track, for a given expression. 0.50 = chance, 1.00 = perfect. < 0.50 means inverted.

## Headline
- Mean AUC (micro, pooled across 3 seqs): **0.762** (over 33 expressions)
- Mean AUC (macro, per-seq averaged):     **0.827** ± 0.068 (over 12 expressions in ≥2 seqs)
- Seq-0011 only (legacy, for continuity): **—**

## Per-expression breakdown
| Expression | 0005 | 0011 | 0013 | macro μ ± σ | micro | GT counts |
|---|---|---|---|---|---|---|
| vehicles in front of ours | 0.988 | — | — | 0.988 ± — | 0.988 | 295/0/0 |
| cars in front of ours | 0.988 | — | — | 0.988 ± — | 0.988 | 295/0/0 |
| cars which are braking | 0.973 | — | — | 0.973 ± — | 0.973 | 295/0/0 |
| vehicles which are braking | 0.973 | — | — | 0.973 ± — | 0.973 | 295/0/0 |
| right vehicles which are parking | — | 0.936 | — | 0.936 ± — | 0.936 | 0/1056/0 |
| right cars which are parking | — | 0.936 | — | 0.936 ± — | 0.936 | 0/1056/0 |
| left vehicles in the counter direction of ours | 0.825 | 0.953 | — | 0.889 ± 0.064 | 0.898 | 526/1692/0 |
| counter direction vehicles in the left | 0.825 | 0.952 | — | 0.889 ± 0.064 | 0.897 | 526/1692/0 |
| left cars in the counter direction of ours | 0.825 | 0.951 | — | 0.888 ± 0.063 | 0.897 | 526/1692/0 |
| counter direction cars in the left | 0.825 | 0.951 | — | 0.888 ± 0.063 | 0.896 | 526/1692/0 |
| moving right pedestrian | — | — | 0.845 | 0.845 ± — | 0.845 | 0/0/261 |
| cars in the same direction of ours | 0.791 | 0.981 | — | 0.886 ± 0.095 | 0.802 | 510/371/0 |
| vehicles in the same direction of ours | 0.791 | 0.981 | — | 0.886 ± 0.095 | 0.802 | 510/371/0 |
| moving cars | 0.873 | 0.787 | — | 0.830 ± 0.043 | 0.801 | 569/765/0 |
| moving vehicles | 0.873 | 0.787 | — | 0.830 ± 0.043 | 0.801 | 569/765/0 |
| moving left pedestrian | — | — | 0.786 | 0.786 ± — | 0.786 | 0/0/253 |
| left vehicles which are parking | 0.601 | 0.883 | — | 0.742 ± 0.141 | 0.764 | 169/1375/0 |
| left cars which are parking | 0.601 | 0.883 | — | 0.742 ± 0.141 | 0.764 | 169/1375/0 |
| cars in the counter direction of ours | 0.818 | 0.640 | — | 0.729 ± 0.089 | 0.695 | 526/1692/0 |
| vehicles in the counter direction of ours | 0.818 | 0.640 | — | 0.729 ± 0.089 | 0.695 | 526/1692/0 |
| cars in horizon direction | — | 0.692 | — | 0.692 ± — | 0.692 | 0/178/0 |
| vehicles in horizon direction | — | 0.692 | — | 0.692 ± — | 0.692 | 0/178/0 |
| left cars in the same direction of ours | 0.673 | — | — | 0.673 ± — | 0.673 | 215/0/0 |
| same direction cars in the left | 0.673 | — | — | 0.673 ± — | 0.673 | 215/0/0 |
| left vehicles in the same direction of ours | 0.673 | — | — | 0.673 ± — | 0.673 | 215/0/0 |
| same direction vehicles in the left | 0.672 | — | — | 0.672 ± — | 0.672 | 215/0/0 |
| turning cars | — | 0.671 | — | 0.671 ± — | 0.671 | 0/30/0 |
| turning vehicles | — | 0.671 | — | 0.671 ± — | 0.671 | 0/30/0 |
| cars which are faster than ours | — | 0.604 | — | 0.604 ± — | 0.604 | 0/371/0 |
| vehicles which are faster than ours | — | 0.601 | — | 0.601 ± — | 0.601 | 0/371/0 |
| parking cars | — | 0.496 | — | 0.496 ± — | 0.496 | 0/2851/0 |
| parking vehicles | — | 0.496 | — | 0.496 ± — | 0.496 | 0/2851/0 |
| moving pedestrian | — | 0.400 | — | 0.400 ± — | 0.400 | 0/88/0 |
