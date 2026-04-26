# Multi-Sequence Eval: v1v2train

## What this measures
AUC = probability that a randomly chosen GT-matching track gets a higher cosine score than a randomly chosen non-matching track, for a given expression. 0.50 = chance, 1.00 = perfect. < 0.50 means inverted.

## Headline
- Mean AUC (micro, pooled across 3 seqs): **0.767** (over 33 expressions)
- Mean AUC (macro, per-seq averaged):     **0.843** ± 0.047 (over 12 expressions in ≥2 seqs)
- Seq-0011 only (legacy, for continuity): **—**

## Per-expression breakdown
| Expression | 0005 | 0011 | 0013 | macro μ ± σ | micro | GT counts |
|---|---|---|---|---|---|---|
| cars in front of ours | 1.000 | — | — | 1.000 ± — | 1.000 | 295/0/0 |
| vehicles in front of ours | 1.000 | — | — | 1.000 ± — | 1.000 | 295/0/0 |
| cars which are braking | 0.985 | — | — | 0.985 ± — | 0.985 | 295/0/0 |
| vehicles which are braking | 0.984 | — | — | 0.984 ± — | 0.984 | 295/0/0 |
| counter direction cars in the left | 0.815 | 0.972 | — | 0.894 ± 0.079 | 0.914 | 526/1692/0 |
| counter direction vehicles in the left | 0.819 | 0.972 | — | 0.896 ± 0.077 | 0.914 | 526/1692/0 |
| left vehicles in the counter direction of ours | 0.812 | 0.972 | — | 0.892 ± 0.080 | 0.914 | 526/1692/0 |
| left cars in the counter direction of ours | 0.811 | 0.974 | — | 0.892 ± 0.081 | 0.913 | 526/1692/0 |
| right vehicles which are parking | — | 0.910 | — | 0.910 ± — | 0.910 | 0/1056/0 |
| right cars which are parking | — | 0.908 | — | 0.908 ± — | 0.908 | 0/1056/0 |
| vehicles which are faster than ours | — | 0.899 | — | 0.899 ± — | 0.899 | 0/371/0 |
| cars in the counter direction of ours | 0.828 | 0.846 | — | 0.837 ± 0.009 | 0.837 | 526/1692/0 |
| moving cars | 0.863 | 0.869 | — | 0.866 ± 0.003 | 0.834 | 569/765/0 |
| vehicles in the counter direction of ours | 0.820 | 0.840 | — | 0.830 ± 0.010 | 0.831 | 526/1692/0 |
| moving right pedestrian | — | — | 0.825 | 0.825 ± — | 0.825 | 0/0/261 |
| left vehicles which are parking | 0.590 | 0.928 | — | 0.759 ± 0.169 | 0.813 | 169/1375/0 |
| moving vehicles | 0.830 | 0.839 | — | 0.835 ± 0.005 | 0.806 | 569/765/0 |
| left cars which are parking | 0.578 | 0.927 | — | 0.753 ± 0.175 | 0.806 | 169/1375/0 |
| vehicles in the same direction of ours | 0.751 | 0.914 | — | 0.832 ± 0.081 | 0.755 | 510/371/0 |
| cars in the same direction of ours | 0.739 | 0.920 | — | 0.830 ± 0.091 | 0.744 | 510/371/0 |
| cars which are faster than ours | — | 0.730 | — | 0.730 ± — | 0.730 | 0/371/0 |
| moving pedestrian | — | 0.701 | — | 0.701 ± — | 0.701 | 0/88/0 |
| vehicles in horizon direction | — | 0.645 | — | 0.645 ± — | 0.645 | 0/178/0 |
| cars in horizon direction | — | 0.639 | — | 0.639 ± — | 0.639 | 0/178/0 |
| turning cars | — | 0.631 | — | 0.631 ± — | 0.631 | 0/30/0 |
| turning vehicles | — | 0.598 | — | 0.598 ± — | 0.598 | 0/30/0 |
| moving left pedestrian | — | — | 0.597 | 0.597 ± — | 0.597 | 0/0/253 |
| same direction cars in the left | 0.553 | — | — | 0.553 ± — | 0.553 | 215/0/0 |
| left cars in the same direction of ours | 0.551 | — | — | 0.551 ± — | 0.551 | 215/0/0 |
| parking vehicles | — | 0.538 | — | 0.538 ± — | 0.538 | 0/2851/0 |
| same direction vehicles in the left | 0.523 | — | — | 0.523 ± — | 0.523 | 215/0/0 |
| parking cars | — | 0.523 | — | 0.523 ± — | 0.523 | 0/2851/0 |
| left vehicles in the same direction of ours | 0.502 | — | — | 0.502 ± — | 0.502 | 215/0/0 |
