# Multi-Sequence Eval: v1train_exp36b_long

## What this measures
AUC = probability that a randomly chosen GT-matching track gets a higher cosine score than a randomly chosen non-matching track, for a given expression. 0.50 = chance, 1.00 = perfect. < 0.50 means inverted.

## Headline
- Mean AUC (micro, pooled across 3 seqs): **0.761** (over 33 expressions)
- Mean AUC (macro, per-seq averaged):     **0.821** ± 0.070 (over 12 expressions in ≥2 seqs)
- Seq-0011 only (legacy, for continuity): **—**

## Per-expression breakdown
| Expression | 0005 | 0011 | 0013 | macro μ ± σ | micro | GT counts |
|---|---|---|---|---|---|---|
| cars which are braking | 1.000 | — | — | 1.000 ± — | 1.000 | 295/0/0 |
| vehicles which are braking | 1.000 | — | — | 1.000 ± — | 1.000 | 295/0/0 |
| cars in front of ours | 0.996 | — | — | 0.996 ± — | 0.996 | 295/0/0 |
| vehicles in front of ours | 0.992 | — | — | 0.992 ± — | 0.992 | 295/0/0 |
| moving left pedestrian | — | — | 0.941 | 0.941 ± — | 0.941 | 0/0/253 |
| right vehicles which are parking | — | 0.924 | — | 0.924 ± — | 0.924 | 0/1056/0 |
| right cars which are parking | — | 0.919 | — | 0.919 ± — | 0.919 | 0/1056/0 |
| counter direction cars in the left | 0.762 | 0.925 | — | 0.844 ± 0.081 | 0.859 | 526/1692/0 |
| left vehicles in the counter direction of ours | 0.770 | 0.912 | — | 0.841 ± 0.071 | 0.855 | 526/1692/0 |
| cars in the same direction of ours | 0.848 | 0.988 | — | 0.918 ± 0.070 | 0.848 | 510/371/0 |
| vehicles in the same direction of ours | 0.842 | 0.988 | — | 0.915 ± 0.073 | 0.848 | 510/371/0 |
| left cars in the counter direction of ours | 0.777 | 0.899 | — | 0.838 ± 0.061 | 0.847 | 526/1692/0 |
| counter direction vehicles in the left | 0.783 | 0.895 | — | 0.839 ± 0.056 | 0.846 | 526/1692/0 |
| moving cars | 0.909 | 0.831 | — | 0.870 ± 0.039 | 0.835 | 569/765/0 |
| moving vehicles | 0.901 | 0.834 | — | 0.867 ± 0.034 | 0.835 | 569/765/0 |
| left cars which are parking | 0.622 | 0.900 | — | 0.761 ± 0.139 | 0.806 | 169/1375/0 |
| left vehicles which are parking | 0.593 | 0.901 | — | 0.747 ± 0.154 | 0.797 | 169/1375/0 |
| same direction vehicles in the left | 0.742 | — | — | 0.742 ± — | 0.742 | 215/0/0 |
| left vehicles in the same direction of ours | 0.733 | — | — | 0.733 ± — | 0.733 | 215/0/0 |
| left cars in the same direction of ours | 0.730 | — | — | 0.730 ± — | 0.730 | 215/0/0 |
| same direction cars in the left | 0.703 | — | — | 0.703 ± — | 0.703 | 215/0/0 |
| vehicles in the counter direction of ours | 0.824 | 0.604 | — | 0.714 ± 0.110 | 0.667 | 526/1692/0 |
| cars in horizon direction | — | 0.666 | — | 0.666 ± — | 0.666 | 0/178/0 |
| vehicles in horizon direction | — | 0.653 | — | 0.653 ± — | 0.653 | 0/178/0 |
| cars in the counter direction of ours | 0.829 | 0.578 | — | 0.703 ± 0.125 | 0.650 | 526/1692/0 |
| moving right pedestrian | — | — | 0.624 | 0.624 ± — | 0.624 | 0/0/261 |
| vehicles which are faster than ours | — | 0.620 | — | 0.620 ± — | 0.620 | 0/371/0 |
| turning vehicles | — | 0.608 | — | 0.608 ± — | 0.608 | 0/30/0 |
| turning cars | — | 0.607 | — | 0.607 ± — | 0.607 | 0/30/0 |
| cars which are faster than ours | — | 0.583 | — | 0.583 ± — | 0.583 | 0/371/0 |
| parking vehicles | — | 0.490 | — | 0.490 ± — | 0.490 | 0/2851/0 |
| parking cars | — | 0.468 | — | 0.468 ± — | 0.468 | 0/2851/0 |
| moving pedestrian | — | 0.436 | — | 0.436 ± — | 0.436 | 0/88/0 |
