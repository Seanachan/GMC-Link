# Multi-Sequence Eval: v1train_exp36e_curriculum

## What this measures
AUC = probability that a randomly chosen GT-matching track gets a higher cosine score than a randomly chosen non-matching track, for a given expression. 0.50 = chance, 1.00 = perfect. < 0.50 means inverted.

## Headline
- Mean AUC (micro, pooled across 3 seqs): **0.762** (over 33 expressions)
- Mean AUC (macro, per-seq averaged):     **0.840** ± 0.063 (over 12 expressions in ≥2 seqs)
- Seq-0011 only (legacy, for continuity): **—**

## Per-expression breakdown
| Expression | 0005 | 0011 | 0013 | macro μ ± σ | micro | GT counts |
|---|---|---|---|---|---|---|
| cars which are braking | 0.994 | — | — | 0.994 ± — | 0.994 | 295/0/0 |
| vehicles which are braking | 0.994 | — | — | 0.994 ± — | 0.994 | 295/0/0 |
| cars in front of ours | 0.993 | — | — | 0.993 ± — | 0.993 | 295/0/0 |
| vehicles in front of ours | 0.992 | — | — | 0.992 ± — | 0.992 | 295/0/0 |
| right cars which are parking | — | 0.934 | — | 0.934 ± — | 0.934 | 0/1056/0 |
| right vehicles which are parking | — | 0.934 | — | 0.934 ± — | 0.934 | 0/1056/0 |
| counter direction vehicles in the left | 0.843 | 0.967 | — | 0.905 ± 0.062 | 0.911 | 526/1692/0 |
| left vehicles in the counter direction of ours | 0.843 | 0.967 | — | 0.905 ± 0.062 | 0.911 | 526/1692/0 |
| counter direction cars in the left | 0.843 | 0.966 | — | 0.905 ± 0.062 | 0.911 | 526/1692/0 |
| left cars in the counter direction of ours | 0.844 | 0.965 | — | 0.904 ± 0.060 | 0.910 | 526/1692/0 |
| vehicles in the same direction of ours | 0.789 | 0.988 | — | 0.888 ± 0.100 | 0.802 | 510/371/0 |
| cars in the same direction of ours | 0.789 | 0.988 | — | 0.888 ± 0.100 | 0.802 | 510/371/0 |
| moving right pedestrian | — | — | 0.796 | 0.796 ± — | 0.796 | 0/0/261 |
| moving cars | 0.868 | 0.782 | — | 0.825 ± 0.043 | 0.791 | 569/765/0 |
| moving vehicles | 0.867 | 0.782 | — | 0.825 ± 0.042 | 0.791 | 569/765/0 |
| left cars which are parking | 0.619 | 0.895 | — | 0.757 ± 0.138 | 0.777 | 169/1375/0 |
| left vehicles which are parking | 0.619 | 0.894 | — | 0.757 ± 0.138 | 0.776 | 169/1375/0 |
| moving left pedestrian | — | — | 0.758 | 0.758 ± — | 0.758 | 0/0/253 |
| cars in the counter direction of ours | 0.843 | 0.684 | — | 0.764 ± 0.079 | 0.730 | 526/1692/0 |
| vehicles in the counter direction of ours | 0.843 | 0.682 | — | 0.763 ± 0.080 | 0.729 | 526/1692/0 |
| vehicles in horizon direction | — | 0.684 | — | 0.684 ± — | 0.684 | 0/178/0 |
| cars in horizon direction | — | 0.682 | — | 0.682 ± — | 0.682 | 0/178/0 |
| turning cars | — | 0.656 | — | 0.656 ± — | 0.656 | 0/30/0 |
| turning vehicles | — | 0.655 | — | 0.655 ± — | 0.655 | 0/30/0 |
| same direction vehicles in the left | 0.640 | — | — | 0.640 ± — | 0.640 | 215/0/0 |
| left vehicles in the same direction of ours | 0.638 | — | — | 0.638 ± — | 0.638 | 215/0/0 |
| same direction cars in the left | 0.638 | — | — | 0.638 ± — | 0.638 | 215/0/0 |
| left cars in the same direction of ours | 0.637 | — | — | 0.637 ± — | 0.637 | 215/0/0 |
| cars which are faster than ours | — | 0.607 | — | 0.607 ± — | 0.607 | 0/371/0 |
| vehicles which are faster than ours | — | 0.605 | — | 0.605 ± — | 0.605 | 0/371/0 |
| parking cars | — | 0.524 | — | 0.524 ± — | 0.524 | 0/2851/0 |
| parking vehicles | — | 0.523 | — | 0.523 ± — | 0.523 | 0/2851/0 |
| moving pedestrian | — | 0.407 | — | 0.407 ± — | 0.407 | 0/88/0 |
