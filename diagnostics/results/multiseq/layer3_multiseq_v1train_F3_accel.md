# Multi-Sequence Eval: v1train_F3_accel

## What this measures
AUC = probability that a randomly chosen GT-matching track gets a higher cosine score than a randomly chosen non-matching track, for a given expression. 0.50 = chance, 1.00 = perfect. < 0.50 means inverted.

## Headline
- Mean AUC (micro, pooled across 3 seqs): **0.779** (over 33 expressions)
- Mean AUC (macro, per-seq averaged):     **0.830** ± 0.060 (over 12 expressions in ≥2 seqs)
- Seq-0011 only (legacy, for continuity): **—**

## Per-expression breakdown
| Expression | 0005 | 0011 | 0013 | macro μ ± σ | micro | GT counts |
|---|---|---|---|---|---|---|
| cars in front of ours | 0.993 | — | — | 0.993 ± — | 0.993 | 295/0/0 |
| vehicles in front of ours | 0.993 | — | — | 0.993 ± — | 0.993 | 295/0/0 |
| vehicles which are braking | 0.987 | — | — | 0.987 ± — | 0.987 | 295/0/0 |
| cars which are braking | 0.987 | — | — | 0.987 ± — | 0.987 | 295/0/0 |
| right cars which are parking | — | 0.931 | — | 0.931 ± — | 0.931 | 0/1056/0 |
| right vehicles which are parking | — | 0.931 | — | 0.931 ± — | 0.931 | 0/1056/0 |
| left vehicles in the counter direction of ours | 0.830 | 0.955 | — | 0.893 ± 0.062 | 0.902 | 526/1692/0 |
| counter direction cars in the left | 0.830 | 0.953 | — | 0.891 ± 0.062 | 0.901 | 526/1692/0 |
| left cars in the counter direction of ours | 0.831 | 0.952 | — | 0.892 ± 0.061 | 0.901 | 526/1692/0 |
| counter direction vehicles in the left | 0.830 | 0.952 | — | 0.891 ± 0.061 | 0.900 | 526/1692/0 |
| moving right pedestrian | — | — | 0.855 | 0.855 ± — | 0.855 | 0/0/261 |
| cars in the same direction of ours | 0.781 | 0.987 | — | 0.884 ± 0.103 | 0.806 | 510/371/0 |
| vehicles in the same direction of ours | 0.780 | 0.987 | — | 0.884 ± 0.103 | 0.806 | 510/371/0 |
| left cars which are parking | 0.614 | 0.892 | — | 0.753 ± 0.139 | 0.779 | 169/1375/0 |
| left vehicles which are parking | 0.614 | 0.892 | — | 0.753 ± 0.139 | 0.779 | 169/1375/0 |
| moving cars | 0.819 | 0.772 | — | 0.796 ± 0.023 | 0.772 | 569/765/0 |
| moving vehicles | 0.819 | 0.772 | — | 0.796 ± 0.023 | 0.772 | 569/765/0 |
| turning cars | — | 0.761 | — | 0.761 ± — | 0.761 | 0/30/0 |
| turning vehicles | — | 0.760 | — | 0.760 ± — | 0.760 | 0/30/0 |
| moving left pedestrian | — | — | 0.751 | 0.751 ± — | 0.751 | 0/0/253 |
| vehicles in the counter direction of ours | 0.824 | 0.706 | — | 0.765 ± 0.059 | 0.741 | 526/1692/0 |
| cars in the counter direction of ours | 0.824 | 0.706 | — | 0.765 ± 0.059 | 0.741 | 526/1692/0 |
| vehicles in horizon direction | — | 0.732 | — | 0.732 ± — | 0.732 | 0/178/0 |
| cars in horizon direction | — | 0.732 | — | 0.732 ± — | 0.732 | 0/178/0 |
| left cars in the same direction of ours | 0.688 | — | — | 0.688 ± — | 0.688 | 215/0/0 |
| left vehicles in the same direction of ours | 0.688 | — | — | 0.688 ± — | 0.688 | 215/0/0 |
| same direction vehicles in the left | 0.688 | — | — | 0.688 ± — | 0.688 | 215/0/0 |
| same direction cars in the left | 0.687 | — | — | 0.687 ± — | 0.687 | 215/0/0 |
| cars which are faster than ours | — | 0.654 | — | 0.654 ± — | 0.654 | 0/371/0 |
| vehicles which are faster than ours | — | 0.652 | — | 0.652 ± — | 0.652 | 0/371/0 |
| parking cars | — | 0.507 | — | 0.507 ± — | 0.507 | 0/2851/0 |
| parking vehicles | — | 0.507 | — | 0.507 ± — | 0.507 | 0/2851/0 |
| moving pedestrian | — | 0.437 | — | 0.437 ± — | 0.437 | 0/88/0 |
