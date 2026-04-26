# Multi-Sequence Eval: v1train_F9_density

## What this measures
AUC = probability that a randomly chosen GT-matching track gets a higher cosine score than a randomly chosen non-matching track, for a given expression. 0.50 = chance, 1.00 = perfect. < 0.50 means inverted.

## Headline
- Mean AUC (micro, pooled across 3 seqs): **0.781** (over 33 expressions)
- Mean AUC (macro, per-seq averaged):     **0.828** ± 0.067 (over 12 expressions in ≥2 seqs)
- Seq-0011 only (legacy, for continuity): **—**

## Per-expression breakdown
| Expression | 0005 | 0011 | 0013 | macro μ ± σ | micro | GT counts |
|---|---|---|---|---|---|---|
| cars in front of ours | 0.978 | — | — | 0.978 ± — | 0.978 | 295/0/0 |
| vehicles in front of ours | 0.978 | — | — | 0.978 ± — | 0.978 | 295/0/0 |
| vehicles which are braking | 0.977 | — | — | 0.977 ± — | 0.977 | 295/0/0 |
| cars which are braking | 0.977 | — | — | 0.977 ± — | 0.977 | 295/0/0 |
| right vehicles which are parking | — | 0.938 | — | 0.938 ± — | 0.938 | 0/1056/0 |
| right cars which are parking | — | 0.938 | — | 0.938 ± — | 0.938 | 0/1056/0 |
| left vehicles in the counter direction of ours | 0.828 | 0.949 | — | 0.889 ± 0.060 | 0.904 | 526/1692/0 |
| counter direction vehicles in the left | 0.828 | 0.948 | — | 0.888 ± 0.060 | 0.904 | 526/1692/0 |
| left cars in the counter direction of ours | 0.829 | 0.948 | — | 0.888 ± 0.060 | 0.903 | 526/1692/0 |
| counter direction cars in the left | 0.828 | 0.946 | — | 0.887 ± 0.059 | 0.902 | 526/1692/0 |
| moving right pedestrian | — | — | 0.861 | 0.861 ± — | 0.861 | 0/0/261 |
| cars in the same direction of ours | 0.834 | 0.958 | — | 0.896 ± 0.062 | 0.824 | 510/371/0 |
| vehicles in the same direction of ours | 0.834 | 0.958 | — | 0.896 ± 0.062 | 0.824 | 510/371/0 |
| turning cars | — | 0.787 | — | 0.787 ± — | 0.787 | 0/30/0 |
| turning vehicles | — | 0.787 | — | 0.787 ± — | 0.787 | 0/30/0 |
| moving cars | 0.863 | 0.757 | — | 0.810 ± 0.053 | 0.787 | 569/765/0 |
| moving vehicles | 0.863 | 0.757 | — | 0.810 ± 0.053 | 0.787 | 569/765/0 |
| left vehicles which are parking | 0.620 | 0.895 | — | 0.757 ± 0.138 | 0.785 | 169/1375/0 |
| left cars which are parking | 0.619 | 0.895 | — | 0.757 ± 0.138 | 0.785 | 169/1375/0 |
| moving left pedestrian | — | — | 0.783 | 0.783 ± — | 0.783 | 0/0/253 |
| left cars in the same direction of ours | 0.738 | — | — | 0.738 ± — | 0.738 | 215/0/0 |
| same direction cars in the left | 0.738 | — | — | 0.738 ± — | 0.738 | 215/0/0 |
| same direction vehicles in the left | 0.738 | — | — | 0.738 ± — | 0.738 | 215/0/0 |
| left vehicles in the same direction of ours | 0.738 | — | — | 0.738 ± — | 0.738 | 215/0/0 |
| vehicles in horizon direction | — | 0.708 | — | 0.708 ± — | 0.708 | 0/178/0 |
| cars in horizon direction | — | 0.708 | — | 0.708 ± — | 0.708 | 0/178/0 |
| vehicles in the counter direction of ours | 0.831 | 0.626 | — | 0.729 ± 0.103 | 0.687 | 526/1692/0 |
| cars in the counter direction of ours | 0.831 | 0.625 | — | 0.728 ± 0.103 | 0.687 | 526/1692/0 |
| vehicles which are faster than ours | — | 0.565 | — | 0.565 ± — | 0.565 | 0/371/0 |
| cars which are faster than ours | — | 0.563 | — | 0.563 ± — | 0.563 | 0/371/0 |
| parking cars | — | 0.513 | — | 0.513 ± — | 0.513 | 0/2851/0 |
| parking vehicles | — | 0.513 | — | 0.513 ± — | 0.513 | 0/2851/0 |
| moving pedestrian | — | 0.474 | — | 0.474 ± — | 0.474 | 0/88/0 |
