# Multi-Sequence Eval: v1train_stage1

## What this measures
AUC = probability that a randomly chosen GT-matching track gets a higher cosine score than a randomly chosen non-matching track, for a given expression. 0.50 = chance, 1.00 = perfect. < 0.50 means inverted.

## Headline
- Mean AUC (micro, pooled across 3 seqs): **0.779** (over 33 expressions)
- Mean AUC (macro, per-seq averaged):     **0.838** ± 0.064 (over 12 expressions in ≥2 seqs)
- Seq-0011 only (legacy, for continuity): **—**

## Per-expression breakdown
| Expression | 0005 | 0011 | 0013 | macro μ ± σ | micro | GT counts |
|---|---|---|---|---|---|---|
| cars which are braking | 0.991 | — | — | 0.991 ± — | 0.991 | 295/0/0 |
| vehicles which are braking | 0.991 | — | — | 0.991 ± — | 0.991 | 295/0/0 |
| cars in front of ours | 0.982 | — | — | 0.982 ± — | 0.982 | 295/0/0 |
| vehicles in front of ours | 0.981 | — | — | 0.981 ± — | 0.981 | 295/0/0 |
| right cars which are parking | — | 0.937 | — | 0.937 ± — | 0.937 | 0/1056/0 |
| right vehicles which are parking | — | 0.937 | — | 0.937 ± — | 0.937 | 0/1056/0 |
| left vehicles in the counter direction of ours | 0.837 | 0.965 | — | 0.901 ± 0.064 | 0.912 | 526/1692/0 |
| left cars in the counter direction of ours | 0.837 | 0.964 | — | 0.901 ± 0.063 | 0.912 | 526/1692/0 |
| counter direction vehicles in the left | 0.837 | 0.963 | — | 0.900 ± 0.063 | 0.912 | 526/1692/0 |
| counter direction cars in the left | 0.837 | 0.963 | — | 0.900 ± 0.063 | 0.911 | 526/1692/0 |
| moving right pedestrian | — | — | 0.846 | 0.846 ± — | 0.846 | 0/0/261 |
| vehicles in the same direction of ours | 0.805 | 0.976 | — | 0.890 ± 0.086 | 0.800 | 510/371/0 |
| cars in the same direction of ours | 0.804 | 0.975 | — | 0.890 ± 0.086 | 0.800 | 510/371/0 |
| moving cars | 0.871 | 0.786 | — | 0.828 ± 0.042 | 0.796 | 569/765/0 |
| moving vehicles | 0.871 | 0.786 | — | 0.828 ± 0.042 | 0.796 | 569/765/0 |
| moving left pedestrian | — | — | 0.789 | 0.789 ± — | 0.789 | 0/0/253 |
| left vehicles which are parking | 0.606 | 0.895 | — | 0.751 ± 0.144 | 0.782 | 169/1375/0 |
| left cars which are parking | 0.606 | 0.895 | — | 0.751 ± 0.145 | 0.782 | 169/1375/0 |
| vehicles in the counter direction of ours | 0.828 | 0.691 | — | 0.759 ± 0.068 | 0.732 | 526/1692/0 |
| cars in the counter direction of ours | 0.827 | 0.691 | — | 0.759 ± 0.068 | 0.731 | 526/1692/0 |
| same direction vehicles in the left | 0.726 | — | — | 0.726 ± — | 0.726 | 215/0/0 |
| left vehicles in the same direction of ours | 0.725 | — | — | 0.725 ± — | 0.725 | 215/0/0 |
| same direction cars in the left | 0.725 | — | — | 0.725 ± — | 0.725 | 215/0/0 |
| left cars in the same direction of ours | 0.725 | — | — | 0.725 ± — | 0.725 | 215/0/0 |
| turning cars | — | 0.720 | — | 0.720 ± — | 0.720 | 0/30/0 |
| turning vehicles | — | 0.720 | — | 0.720 ± — | 0.720 | 0/30/0 |
| vehicles in horizon direction | — | 0.719 | — | 0.719 ± — | 0.719 | 0/178/0 |
| cars in horizon direction | — | 0.718 | — | 0.718 ± — | 0.718 | 0/178/0 |
| cars which are faster than ours | — | 0.612 | — | 0.612 ± — | 0.612 | 0/371/0 |
| vehicles which are faster than ours | — | 0.611 | — | 0.611 ± — | 0.611 | 0/371/0 |
| parking vehicles | — | 0.493 | — | 0.493 ± — | 0.493 | 0/2851/0 |
| parking cars | — | 0.493 | — | 0.493 ± — | 0.493 | 0/2851/0 |
| moving pedestrian | — | 0.409 | — | 0.409 ± — | 0.409 | 0/88/0 |
