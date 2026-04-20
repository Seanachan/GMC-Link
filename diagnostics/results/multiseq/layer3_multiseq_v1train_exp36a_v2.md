# Multi-Sequence Eval: v1train_exp36a_v2

## What this measures
AUC = probability that a randomly chosen GT-matching track gets a higher cosine score than a randomly chosen non-matching track, for a given expression. 0.50 = chance, 1.00 = perfect. < 0.50 means inverted.

## Headline
- Mean AUC (micro, pooled across 3 seqs): **0.741** (over 33 expressions)
- Mean AUC (macro, per-seq averaged):     **0.815** ± 0.046 (over 12 expressions in ≥2 seqs)
- Seq-0011 only (legacy, for continuity): **—**

## Per-expression breakdown
| Expression | 0005 | 0011 | 0013 | macro μ ± σ | micro | GT counts |
|---|---|---|---|---|---|---|
| vehicles in front of ours | 0.998 | — | — | 0.998 ± — | 0.998 | 295/0/0 |
| cars in front of ours | 0.998 | — | — | 0.998 ± — | 0.998 | 295/0/0 |
| right cars which are parking | — | 0.950 | — | 0.950 ± — | 0.950 | 0/1056/0 |
| right vehicles which are parking | — | 0.948 | — | 0.948 ± — | 0.948 | 0/1056/0 |
| vehicles which are braking | 0.923 | — | — | 0.923 ± — | 0.923 | 295/0/0 |
| cars which are braking | 0.917 | — | — | 0.917 ± — | 0.917 | 295/0/0 |
| left cars in the counter direction of ours | 0.809 | 0.923 | — | 0.866 ± 0.057 | 0.873 | 526/1692/0 |
| left vehicles in the counter direction of ours | 0.812 | 0.921 | — | 0.866 ± 0.054 | 0.873 | 526/1692/0 |
| counter direction vehicles in the left | 0.805 | 0.920 | — | 0.862 ± 0.058 | 0.871 | 526/1692/0 |
| counter direction cars in the left | 0.804 | 0.916 | — | 0.860 ± 0.056 | 0.869 | 526/1692/0 |
| moving left pedestrian | — | — | 0.821 | 0.821 ± — | 0.821 | 0/0/253 |
| moving cars | 0.745 | 0.836 | — | 0.790 ± 0.045 | 0.796 | 569/765/0 |
| moving vehicles | 0.742 | 0.837 | — | 0.789 ± 0.047 | 0.796 | 569/765/0 |
| vehicles in the same direction of ours | 0.761 | 0.941 | — | 0.851 ± 0.090 | 0.791 | 510/371/0 |
| cars in the same direction of ours | 0.760 | 0.937 | — | 0.848 ± 0.088 | 0.787 | 510/371/0 |
| left cars which are parking | 0.671 | 0.889 | — | 0.780 ± 0.109 | 0.785 | 169/1375/0 |
| left vehicles which are parking | 0.673 | 0.887 | — | 0.780 ± 0.107 | 0.783 | 169/1375/0 |
| turning cars | — | 0.761 | — | 0.761 ± — | 0.761 | 0/30/0 |
| turning vehicles | — | 0.750 | — | 0.750 ± — | 0.750 | 0/30/0 |
| cars in the counter direction of ours | 0.807 | 0.687 | — | 0.747 ± 0.060 | 0.722 | 526/1692/0 |
| vehicles in the counter direction of ours | 0.810 | 0.674 | — | 0.742 ± 0.068 | 0.714 | 526/1692/0 |
| moving right pedestrian | — | — | 0.628 | 0.628 ± — | 0.628 | 0/0/261 |
| vehicles in horizon direction | — | 0.624 | — | 0.624 ± — | 0.624 | 0/178/0 |
| cars in horizon direction | — | 0.621 | — | 0.621 ± — | 0.621 | 0/178/0 |
| moving pedestrian | — | 0.609 | — | 0.609 ± — | 0.609 | 0/88/0 |
| parking vehicles | — | 0.552 | — | 0.552 ± — | 0.552 | 0/2851/0 |
| vehicles which are faster than ours | — | 0.551 | — | 0.551 ± — | 0.551 | 0/371/0 |
| parking cars | — | 0.549 | — | 0.549 ± — | 0.549 | 0/2851/0 |
| cars which are faster than ours | — | 0.538 | — | 0.538 ± — | 0.538 | 0/371/0 |
| same direction vehicles in the left | 0.514 | — | — | 0.514 ± — | 0.514 | 215/0/0 |
| same direction cars in the left | 0.513 | — | — | 0.513 ± — | 0.513 | 215/0/0 |
| left cars in the same direction of ours | 0.509 | — | — | 0.509 ± — | 0.509 | 215/0/0 |
| left vehicles in the same direction of ours | 0.508 | — | — | 0.508 ± — | 0.508 | 215/0/0 |
