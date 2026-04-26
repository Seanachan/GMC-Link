# Multi-Sequence Eval: v1train_exp36b

## What this measures
AUC = probability that a randomly chosen GT-matching track gets a higher cosine score than a randomly chosen non-matching track, for a given expression. 0.50 = chance, 1.00 = perfect. < 0.50 means inverted.

## Headline
- Mean AUC (micro, pooled across 3 seqs): **0.754** (over 33 expressions)
- Mean AUC (macro, per-seq averaged):     **0.809** ± 0.081 (over 12 expressions in ≥2 seqs)
- Seq-0011 only (legacy, for continuity): **—**

## Per-expression breakdown
| Expression | 0005 | 0011 | 0013 | macro μ ± σ | micro | GT counts |
|---|---|---|---|---|---|---|
| vehicles in front of ours | 0.999 | — | — | 0.999 ± — | 0.999 | 295/0/0 |
| cars in front of ours | 0.999 | — | — | 0.999 ± — | 0.999 | 295/0/0 |
| cars which are braking | 0.963 | — | — | 0.963 ± — | 0.963 | 295/0/0 |
| vehicles which are braking | 0.959 | — | — | 0.959 ± — | 0.959 | 295/0/0 |
| right vehicles which are parking | — | 0.938 | — | 0.938 ± — | 0.938 | 0/1056/0 |
| moving left pedestrian | — | — | 0.937 | 0.937 ± — | 0.937 | 0/0/253 |
| right cars which are parking | — | 0.936 | — | 0.936 ± — | 0.936 | 0/1056/0 |
| left cars in the counter direction of ours | 0.802 | 0.909 | — | 0.856 ± 0.053 | 0.855 | 526/1692/0 |
| left vehicles in the counter direction of ours | 0.806 | 0.887 | — | 0.847 ± 0.041 | 0.843 | 526/1692/0 |
| counter direction vehicles in the left | 0.806 | 0.882 | — | 0.844 ± 0.038 | 0.839 | 526/1692/0 |
| counter direction cars in the left | 0.796 | 0.882 | — | 0.839 ± 0.043 | 0.839 | 526/1692/0 |
| vehicles in the same direction of ours | 0.844 | 0.989 | — | 0.917 ± 0.073 | 0.833 | 510/371/0 |
| cars in the same direction of ours | 0.843 | 0.992 | — | 0.917 ± 0.074 | 0.832 | 510/371/0 |
| turning cars | — | 0.808 | — | 0.808 ± — | 0.808 | 0/30/0 |
| turning vehicles | — | 0.798 | — | 0.798 ± — | 0.798 | 0/30/0 |
| moving cars | 0.875 | 0.804 | — | 0.839 ± 0.036 | 0.793 | 569/765/0 |
| left vehicles which are parking | 0.579 | 0.917 | — | 0.748 ± 0.169 | 0.791 | 169/1375/0 |
| left cars which are parking | 0.578 | 0.910 | — | 0.744 ± 0.166 | 0.784 | 169/1375/0 |
| moving vehicles | 0.851 | 0.788 | — | 0.819 ± 0.032 | 0.777 | 569/765/0 |
| vehicles in horizon direction | — | 0.695 | — | 0.695 ± — | 0.695 | 0/178/0 |
| cars in horizon direction | — | 0.691 | — | 0.691 ± — | 0.691 | 0/178/0 |
| vehicles which are faster than ours | — | 0.652 | — | 0.652 ± — | 0.652 | 0/371/0 |
| cars which are faster than ours | — | 0.631 | — | 0.631 ± — | 0.631 | 0/371/0 |
| same direction vehicles in the left | 0.627 | — | — | 0.627 ± — | 0.627 | 215/0/0 |
| left cars in the same direction of ours | 0.624 | — | — | 0.624 ± — | 0.624 | 215/0/0 |
| same direction cars in the left | 0.619 | — | — | 0.619 ± — | 0.619 | 215/0/0 |
| cars in the counter direction of ours | 0.810 | 0.526 | — | 0.668 ± 0.142 | 0.617 | 526/1692/0 |
| left vehicles in the same direction of ours | 0.615 | — | — | 0.615 ± — | 0.615 | 215/0/0 |
| vehicles in the counter direction of ours | 0.814 | 0.520 | — | 0.667 ± 0.147 | 0.613 | 526/1692/0 |
| moving pedestrian | — | 0.542 | — | 0.542 ± — | 0.542 | 0/88/0 |
| moving right pedestrian | — | — | 0.541 | 0.541 ± — | 0.541 | 0/0/261 |
| parking vehicles | — | 0.453 | — | 0.453 ± — | 0.453 | 0/2851/0 |
| parking cars | — | 0.448 | — | 0.448 ± — | 0.448 | 0/2851/0 |
