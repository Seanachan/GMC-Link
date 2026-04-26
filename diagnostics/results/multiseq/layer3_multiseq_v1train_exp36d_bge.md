# Multi-Sequence Eval: v1train_exp36d_bge

## What this measures
AUC = probability that a randomly chosen GT-matching track gets a higher cosine score than a randomly chosen non-matching track, for a given expression. 0.50 = chance, 1.00 = perfect. < 0.50 means inverted.

## Headline
- Mean AUC (micro, pooled across 3 seqs): **0.735** (over 33 expressions)
- Mean AUC (macro, per-seq averaged):     **0.802** ± 0.052 (over 12 expressions in ≥2 seqs)
- Seq-0011 only (legacy, for continuity): **—**

## Per-expression breakdown
| Expression | 0005 | 0011 | 0013 | macro μ ± σ | micro | GT counts |
|---|---|---|---|---|---|---|
| vehicles in front of ours | 1.000 | — | — | 1.000 ± — | 1.000 | 295/0/0 |
| cars in front of ours | 1.000 | — | — | 1.000 ± — | 1.000 | 295/0/0 |
| right cars which are parking | — | 0.933 | — | 0.933 ± — | 0.933 | 0/1056/0 |
| right vehicles which are parking | — | 0.933 | — | 0.933 ± — | 0.933 | 0/1056/0 |
| counter direction cars in the left | 0.790 | 0.925 | — | 0.858 ± 0.068 | 0.873 | 526/1692/0 |
| left vehicles in the counter direction of ours | 0.789 | 0.924 | — | 0.857 ± 0.067 | 0.872 | 526/1692/0 |
| left cars in the counter direction of ours | 0.790 | 0.919 | — | 0.854 ± 0.065 | 0.869 | 526/1692/0 |
| counter direction vehicles in the left | 0.787 | 0.920 | — | 0.853 ± 0.066 | 0.869 | 526/1692/0 |
| moving left pedestrian | — | — | 0.852 | 0.852 ± — | 0.852 | 0/0/253 |
| vehicles which are braking | 0.844 | — | — | 0.844 ± — | 0.844 | 295/0/0 |
| cars which are braking | 0.842 | — | — | 0.842 ± — | 0.842 | 295/0/0 |
| moving right pedestrian | — | — | 0.810 | 0.810 ± — | 0.810 | 0/0/261 |
| left cars which are parking | 0.681 | 0.882 | — | 0.781 ± 0.101 | 0.794 | 169/1375/0 |
| left vehicles which are parking | 0.684 | 0.879 | — | 0.782 ± 0.098 | 0.791 | 169/1375/0 |
| vehicles in the same direction of ours | 0.739 | 0.950 | — | 0.845 ± 0.105 | 0.781 | 510/371/0 |
| cars in the same direction of ours | 0.740 | 0.948 | — | 0.844 ± 0.104 | 0.780 | 510/371/0 |
| moving cars | 0.687 | 0.798 | — | 0.742 ± 0.055 | 0.766 | 569/765/0 |
| moving vehicles | 0.684 | 0.792 | — | 0.738 ± 0.054 | 0.761 | 569/765/0 |
| vehicles in the counter direction of ours | 0.794 | 0.677 | — | 0.736 ± 0.058 | 0.713 | 526/1692/0 |
| cars in the counter direction of ours | 0.795 | 0.668 | — | 0.732 ± 0.063 | 0.708 | 526/1692/0 |
| turning cars | — | 0.700 | — | 0.700 ± — | 0.700 | 0/30/0 |
| turning vehicles | — | 0.673 | — | 0.673 ± — | 0.673 | 0/30/0 |
| cars in horizon direction | — | 0.672 | — | 0.672 ± — | 0.672 | 0/178/0 |
| vehicles in horizon direction | — | 0.670 | — | 0.670 ± — | 0.670 | 0/178/0 |
| vehicles which are faster than ours | — | 0.572 | — | 0.572 ± — | 0.572 | 0/371/0 |
| cars which are faster than ours | — | 0.559 | — | 0.559 ± — | 0.559 | 0/371/0 |
| left vehicles in the same direction of ours | 0.552 | — | — | 0.552 ± — | 0.552 | 215/0/0 |
| parking vehicles | — | 0.550 | — | 0.550 ± — | 0.550 | 0/2851/0 |
| parking cars | — | 0.546 | — | 0.546 ± — | 0.546 | 0/2851/0 |
| same direction vehicles in the left | 0.542 | — | — | 0.542 ± — | 0.542 | 215/0/0 |
| left cars in the same direction of ours | 0.538 | — | — | 0.538 ± — | 0.538 | 215/0/0 |
| same direction cars in the left | 0.537 | — | — | 0.537 ± — | 0.537 | 215/0/0 |
| moving pedestrian | — | 0.351 | — | 0.351 ± — | 0.351 | 0/88/0 |
