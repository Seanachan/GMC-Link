# Multi-Sequence Eval: exp37_stage_a2_recoverpose

## What this measures
AUC = probability that a randomly chosen GT-matching track gets a higher cosine score than a randomly chosen non-matching track, for a given expression. 0.50 = chance, 1.00 = perfect. < 0.50 means inverted.

## Headline
- Mean AUC (micro, pooled across 3 seqs): **0.743** (over 33 expressions)
- Mean AUC (macro, per-seq averaged):     **0.767** ± 0.121 (over 12 expressions in ≥2 seqs)
- Seq-0011 only (legacy, for continuity): **—**

## Per-expression breakdown
| Expression | 0005 | 0011 | 0013 | macro μ ± σ | micro | GT counts |
|---|---|---|---|---|---|---|
| cars in front of ours | 0.978 | — | — | 0.978 ± — | 0.978 | 295/0/0 |
| vehicles in front of ours | 0.978 | — | — | 0.978 ± — | 0.978 | 295/0/0 |
| right vehicles which are parking | — | 0.952 | — | 0.952 ± — | 0.952 | 0/1056/0 |
| right cars which are parking | — | 0.952 | — | 0.952 ± — | 0.952 | 0/1056/0 |
| vehicles which are braking | 0.897 | — | — | 0.897 ± — | 0.897 | 295/0/0 |
| cars which are braking | 0.897 | — | — | 0.897 ± — | 0.897 | 295/0/0 |
| left cars in the counter direction of ours | 0.821 | 0.928 | — | 0.874 ± 0.054 | 0.892 | 526/1692/0 |
| counter direction vehicles in the left | 0.821 | 0.927 | — | 0.874 ± 0.053 | 0.892 | 526/1692/0 |
| counter direction cars in the left | 0.821 | 0.927 | — | 0.874 ± 0.053 | 0.891 | 526/1692/0 |
| left vehicles in the counter direction of ours | 0.821 | 0.925 | — | 0.873 ± 0.052 | 0.890 | 526/1692/0 |
| moving right pedestrian | — | — | 0.856 | 0.856 ± — | 0.856 | 0/0/261 |
| vehicles in the same direction of ours | 0.783 | 0.953 | — | 0.868 ± 0.085 | 0.836 | 510/371/0 |
| cars in the same direction of ours | 0.783 | 0.953 | — | 0.868 ± 0.085 | 0.836 | 510/371/0 |
| left vehicles which are parking | 0.640 | 0.880 | — | 0.760 ± 0.120 | 0.799 | 169/1375/0 |
| left cars which are parking | 0.640 | 0.879 | — | 0.760 ± 0.120 | 0.799 | 169/1375/0 |
| cars in horizon direction | — | 0.748 | — | 0.748 ± — | 0.748 | 0/178/0 |
| vehicles in horizon direction | — | 0.748 | — | 0.748 ± — | 0.748 | 0/178/0 |
| moving left pedestrian | — | — | 0.719 | 0.719 ± — | 0.719 | 0/0/253 |
| turning vehicles | — | 0.706 | — | 0.706 ± — | 0.706 | 0/30/0 |
| turning cars | — | 0.706 | — | 0.706 ± — | 0.706 | 0/30/0 |
| left vehicles in the same direction of ours | 0.675 | — | — | 0.675 ± — | 0.675 | 215/0/0 |
| left cars in the same direction of ours | 0.675 | — | — | 0.675 ± — | 0.675 | 215/0/0 |
| same direction cars in the left | 0.674 | — | — | 0.674 ± — | 0.674 | 215/0/0 |
| same direction vehicles in the left | 0.674 | — | — | 0.674 ± — | 0.674 | 215/0/0 |
| vehicles in the counter direction of ours | 0.772 | 0.575 | — | 0.674 ± 0.098 | 0.635 | 526/1692/0 |
| cars in the counter direction of ours | 0.771 | 0.575 | — | 0.673 ± 0.098 | 0.635 | 526/1692/0 |
| moving vehicles | 0.453 | 0.656 | — | 0.555 ± 0.102 | 0.614 | 569/765/0 |
| moving cars | 0.452 | 0.655 | — | 0.554 ± 0.102 | 0.614 | 569/765/0 |
| vehicles which are faster than ours | — | 0.483 | — | 0.483 ± — | 0.483 | 0/371/0 |
| cars which are faster than ours | — | 0.482 | — | 0.482 ± — | 0.482 | 0/371/0 |
| parking vehicles | — | 0.463 | — | 0.463 ± — | 0.463 | 0/2851/0 |
| parking cars | — | 0.463 | — | 0.463 ± — | 0.463 | 0/2851/0 |
| moving pedestrian | — | 0.456 | — | 0.456 ± — | 0.456 | 0/88/0 |
