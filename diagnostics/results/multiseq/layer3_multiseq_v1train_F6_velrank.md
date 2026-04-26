# Multi-Sequence Eval: v1train_F6_velrank

## What this measures
AUC = probability that a randomly chosen GT-matching track gets a higher cosine score than a randomly chosen non-matching track, for a given expression. 0.50 = chance, 1.00 = perfect. < 0.50 means inverted.

## Headline
- Mean AUC (micro, pooled across 3 seqs): **0.770** (over 33 expressions)
- Mean AUC (macro, per-seq averaged):     **0.835** ± 0.062 (over 12 expressions in ≥2 seqs)
- Seq-0011 only (legacy, for continuity): **—**

## Per-expression breakdown
| Expression | 0005 | 0011 | 0013 | macro μ ± σ | micro | GT counts |
|---|---|---|---|---|---|---|
| cars in front of ours | 0.998 | — | — | 0.998 ± — | 0.998 | 295/0/0 |
| vehicles in front of ours | 0.998 | — | — | 0.998 ± — | 0.998 | 295/0/0 |
| cars which are braking | 0.979 | — | — | 0.979 ± — | 0.979 | 295/0/0 |
| vehicles which are braking | 0.979 | — | — | 0.979 ± — | 0.979 | 295/0/0 |
| right cars which are parking | — | 0.936 | — | 0.936 ± — | 0.936 | 0/1056/0 |
| right vehicles which are parking | — | 0.936 | — | 0.936 ± — | 0.936 | 0/1056/0 |
| counter direction vehicles in the left | 0.848 | 0.953 | — | 0.901 ± 0.053 | 0.902 | 526/1692/0 |
| counter direction cars in the left | 0.848 | 0.953 | — | 0.900 ± 0.053 | 0.902 | 526/1692/0 |
| left vehicles in the counter direction of ours | 0.848 | 0.953 | — | 0.900 ± 0.052 | 0.902 | 526/1692/0 |
| left cars in the counter direction of ours | 0.848 | 0.951 | — | 0.900 ± 0.052 | 0.901 | 526/1692/0 |
| moving right pedestrian | — | — | 0.856 | 0.856 ± — | 0.856 | 0/0/261 |
| cars in the same direction of ours | 0.782 | 0.974 | — | 0.878 ± 0.096 | 0.794 | 510/371/0 |
| vehicles in the same direction of ours | 0.782 | 0.974 | — | 0.878 ± 0.096 | 0.794 | 510/371/0 |
| moving cars | 0.849 | 0.778 | — | 0.813 ± 0.035 | 0.786 | 569/765/0 |
| moving vehicles | 0.848 | 0.778 | — | 0.813 ± 0.035 | 0.786 | 569/765/0 |
| moving left pedestrian | — | — | 0.776 | 0.776 ± — | 0.776 | 0/0/253 |
| left vehicles which are parking | 0.622 | 0.890 | — | 0.756 ± 0.134 | 0.769 | 169/1375/0 |
| left cars which are parking | 0.622 | 0.890 | — | 0.756 ± 0.134 | 0.769 | 169/1375/0 |
| vehicles in horizon direction | — | 0.740 | — | 0.740 ± — | 0.740 | 0/178/0 |
| cars in horizon direction | — | 0.740 | — | 0.740 ± — | 0.740 | 0/178/0 |
| vehicles in the counter direction of ours | 0.838 | 0.681 | — | 0.759 ± 0.079 | 0.726 | 526/1692/0 |
| cars in the counter direction of ours | 0.838 | 0.681 | — | 0.759 ± 0.079 | 0.726 | 526/1692/0 |
| turning vehicles | — | 0.714 | — | 0.714 ± — | 0.714 | 0/30/0 |
| turning cars | — | 0.713 | — | 0.713 ± — | 0.713 | 0/30/0 |
| same direction vehicles in the left | 0.674 | — | — | 0.674 ± — | 0.674 | 215/0/0 |
| left vehicles in the same direction of ours | 0.674 | — | — | 0.674 ± — | 0.674 | 215/0/0 |
| same direction cars in the left | 0.673 | — | — | 0.673 ± — | 0.673 | 215/0/0 |
| left cars in the same direction of ours | 0.673 | — | — | 0.673 ± — | 0.673 | 215/0/0 |
| cars which are faster than ours | — | 0.604 | — | 0.604 ± — | 0.604 | 0/371/0 |
| vehicles which are faster than ours | — | 0.602 | — | 0.602 ± — | 0.602 | 0/371/0 |
| parking cars | — | 0.517 | — | 0.517 ± — | 0.517 | 0/2851/0 |
| parking vehicles | — | 0.517 | — | 0.517 ± — | 0.517 | 0/2851/0 |
| moving pedestrian | — | 0.367 | — | 0.367 ± — | 0.367 | 0/88/0 |
