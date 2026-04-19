# Multi-Sequence Eval: v1train_hninfo_beta0.5

## What this measures
AUC = probability that a randomly chosen GT-matching track gets a higher cosine score than a randomly chosen non-matching track, for a given expression. 0.50 = chance, 1.00 = perfect. < 0.50 means inverted.

## Headline
- Mean AUC (micro, pooled across 3 seqs): **0.753** (over 33 expressions)
- Mean AUC (macro, per-seq averaged):     **0.818** ± 0.064 (over 12 expressions in ≥2 seqs)
- Seq-0011 only (legacy, for continuity): **—**

## Per-expression breakdown
| Expression | 0005 | 0011 | 0013 | macro μ ± σ | micro | GT counts |
|---|---|---|---|---|---|---|
| cars which are braking | 0.991 | — | — | 0.991 ± — | 0.991 | 295/0/0 |
| vehicles which are braking | 0.991 | — | — | 0.991 ± — | 0.991 | 295/0/0 |
| cars in front of ours | 0.974 | — | — | 0.974 ± — | 0.974 | 295/0/0 |
| vehicles in front of ours | 0.973 | — | — | 0.973 ± — | 0.973 | 295/0/0 |
| right cars which are parking | — | 0.928 | — | 0.928 ± — | 0.928 | 0/1056/0 |
| right vehicles which are parking | — | 0.928 | — | 0.928 ± — | 0.928 | 0/1056/0 |
| counter direction cars in the left | 0.817 | 0.906 | — | 0.861 ± 0.044 | 0.860 | 526/1692/0 |
| left vehicles in the counter direction of ours | 0.818 | 0.904 | — | 0.861 ± 0.043 | 0.860 | 526/1692/0 |
| counter direction vehicles in the left | 0.817 | 0.901 | — | 0.859 ± 0.042 | 0.857 | 526/1692/0 |
| left cars in the counter direction of ours | 0.818 | 0.899 | — | 0.859 ± 0.041 | 0.857 | 526/1692/0 |
| vehicles in the same direction of ours | 0.809 | 0.987 | — | 0.898 ± 0.089 | 0.812 | 510/371/0 |
| cars in the same direction of ours | 0.809 | 0.986 | — | 0.898 ± 0.089 | 0.812 | 510/371/0 |
| moving cars | 0.853 | 0.797 | — | 0.825 ± 0.028 | 0.799 | 569/765/0 |
| moving vehicles | 0.853 | 0.797 | — | 0.825 ± 0.028 | 0.799 | 569/765/0 |
| moving left pedestrian | — | — | 0.787 | 0.787 ± — | 0.787 | 0/0/253 |
| left vehicles which are parking | 0.602 | 0.861 | — | 0.732 ± 0.129 | 0.757 | 169/1375/0 |
| left cars which are parking | 0.602 | 0.860 | — | 0.731 ± 0.129 | 0.757 | 169/1375/0 |
| moving right pedestrian | — | — | 0.709 | 0.709 ± — | 0.709 | 0/0/261 |
| same direction vehicles in the left | 0.706 | — | — | 0.706 ± — | 0.706 | 215/0/0 |
| left vehicles in the same direction of ours | 0.705 | — | — | 0.705 ± — | 0.705 | 215/0/0 |
| same direction cars in the left | 0.705 | — | — | 0.705 ± — | 0.705 | 215/0/0 |
| left cars in the same direction of ours | 0.705 | — | — | 0.705 ± — | 0.705 | 215/0/0 |
| cars in the counter direction of ours | 0.791 | 0.673 | — | 0.732 ± 0.059 | 0.699 | 526/1692/0 |
| vehicles in the counter direction of ours | 0.790 | 0.671 | — | 0.731 ± 0.059 | 0.697 | 526/1692/0 |
| vehicles in horizon direction | — | 0.684 | — | 0.684 ± — | 0.684 | 0/178/0 |
| cars in horizon direction | — | 0.683 | — | 0.683 ± — | 0.683 | 0/178/0 |
| turning cars | — | 0.657 | — | 0.657 ± — | 0.657 | 0/30/0 |
| turning vehicles | — | 0.656 | — | 0.656 ± — | 0.656 | 0/30/0 |
| vehicles which are faster than ours | — | 0.534 | — | 0.534 ± — | 0.534 | 0/371/0 |
| cars which are faster than ours | — | 0.533 | — | 0.533 ± — | 0.533 | 0/371/0 |
| parking vehicles | — | 0.507 | — | 0.507 ± — | 0.507 | 0/2851/0 |
| parking cars | — | 0.507 | — | 0.507 ± — | 0.507 | 0/2851/0 |
| moving pedestrian | — | 0.408 | — | 0.408 ± — | 0.408 | 0/88/0 |
