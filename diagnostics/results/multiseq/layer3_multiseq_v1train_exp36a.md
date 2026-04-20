# Multi-Sequence Eval: v1train_exp36a

## What this measures
AUC = probability that a randomly chosen GT-matching track gets a higher cosine score than a randomly chosen non-matching track, for a given expression. 0.50 = chance, 1.00 = perfect. < 0.50 means inverted.

## Headline
- Mean AUC (micro, pooled across 3 seqs): **0.747** (over 33 expressions)
- Mean AUC (macro, per-seq averaged):     **0.797** ± 0.057 (over 12 expressions in ≥2 seqs)
- Seq-0011 only (legacy, for continuity): **—**

## Per-expression breakdown
| Expression | 0005 | 0011 | 0013 | macro μ ± σ | micro | GT counts |
|---|---|---|---|---|---|---|
| cars in front of ours | 1.000 | — | — | 1.000 ± — | 1.000 | 295/0/0 |
| vehicles in front of ours | 1.000 | — | — | 1.000 ± — | 1.000 | 295/0/0 |
| right cars which are parking | — | 0.950 | — | 0.950 ± — | 0.950 | 0/1056/0 |
| right vehicles which are parking | — | 0.950 | — | 0.950 ± — | 0.950 | 0/1056/0 |
| cars which are braking | 0.921 | — | — | 0.921 ± — | 0.921 | 295/0/0 |
| vehicles which are braking | 0.914 | — | — | 0.914 ± — | 0.914 | 295/0/0 |
| left cars in the counter direction of ours | 0.783 | 0.913 | — | 0.848 ± 0.065 | 0.862 | 526/1692/0 |
| left vehicles in the counter direction of ours | 0.785 | 0.909 | — | 0.847 ± 0.062 | 0.860 | 526/1692/0 |
| counter direction vehicles in the left | 0.784 | 0.908 | — | 0.846 ± 0.062 | 0.859 | 526/1692/0 |
| counter direction cars in the left | 0.787 | 0.907 | — | 0.847 ± 0.060 | 0.859 | 526/1692/0 |
| moving left pedestrian | — | — | 0.825 | 0.825 ± — | 0.825 | 0/0/253 |
| cars in the same direction of ours | 0.766 | 0.957 | — | 0.861 ± 0.096 | 0.798 | 510/371/0 |
| vehicles in the same direction of ours | 0.762 | 0.955 | — | 0.858 ± 0.096 | 0.793 | 510/371/0 |
| left cars which are parking | 0.661 | 0.882 | — | 0.772 ± 0.110 | 0.783 | 169/1375/0 |
| left vehicles which are parking | 0.668 | 0.877 | — | 0.772 ± 0.105 | 0.779 | 169/1375/0 |
| turning cars | — | 0.766 | — | 0.766 ± — | 0.766 | 0/30/0 |
| moving vehicles | 0.701 | 0.781 | — | 0.741 ± 0.040 | 0.759 | 569/765/0 |
| moving cars | 0.701 | 0.780 | — | 0.741 ± 0.039 | 0.758 | 569/765/0 |
| moving pedestrian | — | 0.732 | — | 0.732 ± — | 0.732 | 0/88/0 |
| turning vehicles | — | 0.732 | — | 0.732 ± — | 0.732 | 0/30/0 |
| moving right pedestrian | — | — | 0.720 | 0.720 ± — | 0.720 | 0/0/261 |
| vehicles in the counter direction of ours | 0.795 | 0.643 | — | 0.719 ± 0.076 | 0.690 | 526/1692/0 |
| cars in the counter direction of ours | 0.795 | 0.634 | — | 0.714 ± 0.080 | 0.683 | 526/1692/0 |
| cars in horizon direction | — | 0.617 | — | 0.617 ± — | 0.617 | 0/178/0 |
| vehicles in horizon direction | — | 0.617 | — | 0.617 ± — | 0.617 | 0/178/0 |
| left cars in the same direction of ours | 0.587 | — | — | 0.587 ± — | 0.587 | 215/0/0 |
| left vehicles in the same direction of ours | 0.581 | — | — | 0.581 ± — | 0.581 | 215/0/0 |
| same direction cars in the left | 0.568 | — | — | 0.568 ± — | 0.568 | 215/0/0 |
| same direction vehicles in the left | 0.566 | — | — | 0.566 ± — | 0.566 | 215/0/0 |
| cars which are faster than ours | — | 0.558 | — | 0.558 ± — | 0.558 | 0/371/0 |
| vehicles which are faster than ours | — | 0.558 | — | 0.558 ± — | 0.558 | 0/371/0 |
| parking cars | — | 0.505 | — | 0.505 ± — | 0.505 | 0/2851/0 |
| parking vehicles | — | 0.504 | — | 0.504 ± — | 0.504 | 0/2851/0 |
