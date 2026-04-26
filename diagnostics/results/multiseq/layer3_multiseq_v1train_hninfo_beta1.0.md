# Multi-Sequence Eval: v1train_hninfo_beta1.0

## What this measures
AUC = probability that a randomly chosen GT-matching track gets a higher cosine score than a randomly chosen non-matching track, for a given expression. 0.50 = chance, 1.00 = perfect. < 0.50 means inverted.

## Headline
- Mean AUC (micro, pooled across 3 seqs): **0.746** (over 33 expressions)
- Mean AUC (macro, per-seq averaged):     **0.810** ± 0.063 (over 12 expressions in ≥2 seqs)
- Seq-0011 only (legacy, for continuity): **—**

## Per-expression breakdown
| Expression | 0005 | 0011 | 0013 | macro μ ± σ | micro | GT counts |
|---|---|---|---|---|---|---|
| cars which are braking | 0.987 | — | — | 0.987 ± — | 0.987 | 295/0/0 |
| vehicles which are braking | 0.987 | — | — | 0.987 ± — | 0.987 | 295/0/0 |
| cars in front of ours | 0.971 | — | — | 0.971 ± — | 0.971 | 295/0/0 |
| vehicles in front of ours | 0.971 | — | — | 0.971 ± — | 0.971 | 295/0/0 |
| right vehicles which are parking | — | 0.924 | — | 0.924 ± — | 0.924 | 0/1056/0 |
| right cars which are parking | — | 0.924 | — | 0.924 ± — | 0.924 | 0/1056/0 |
| counter direction cars in the left | 0.805 | 0.880 | — | 0.843 ± 0.038 | 0.837 | 526/1692/0 |
| left vehicles in the counter direction of ours | 0.805 | 0.878 | — | 0.841 ± 0.036 | 0.836 | 526/1692/0 |
| left cars in the counter direction of ours | 0.805 | 0.875 | — | 0.840 ± 0.035 | 0.834 | 526/1692/0 |
| counter direction vehicles in the left | 0.804 | 0.875 | — | 0.839 ± 0.036 | 0.834 | 526/1692/0 |
| vehicles in the same direction of ours | 0.809 | 0.988 | — | 0.899 ± 0.089 | 0.816 | 510/371/0 |
| cars in the same direction of ours | 0.809 | 0.988 | — | 0.899 ± 0.089 | 0.816 | 510/371/0 |
| moving cars | 0.853 | 0.800 | — | 0.827 ± 0.026 | 0.803 | 569/765/0 |
| moving vehicles | 0.853 | 0.800 | — | 0.827 ± 0.026 | 0.803 | 569/765/0 |
| moving left pedestrian | — | — | 0.785 | 0.785 ± — | 0.785 | 0/0/253 |
| moving right pedestrian | — | — | 0.758 | 0.758 ± — | 0.758 | 0/0/261 |
| left vehicles which are parking | 0.599 | 0.849 | — | 0.724 ± 0.125 | 0.750 | 169/1375/0 |
| left cars which are parking | 0.598 | 0.848 | — | 0.723 ± 0.125 | 0.749 | 169/1375/0 |
| cars in the counter direction of ours | 0.785 | 0.680 | — | 0.732 ± 0.053 | 0.702 | 526/1692/0 |
| vehicles in the counter direction of ours | 0.784 | 0.679 | — | 0.732 ± 0.053 | 0.701 | 526/1692/0 |
| same direction vehicles in the left | 0.698 | — | — | 0.698 ± — | 0.698 | 215/0/0 |
| same direction cars in the left | 0.697 | — | — | 0.697 ± — | 0.697 | 215/0/0 |
| left vehicles in the same direction of ours | 0.697 | — | — | 0.697 ± — | 0.697 | 215/0/0 |
| left cars in the same direction of ours | 0.696 | — | — | 0.696 ± — | 0.696 | 215/0/0 |
| vehicles in horizon direction | — | 0.679 | — | 0.679 ± — | 0.679 | 0/178/0 |
| cars in horizon direction | — | 0.678 | — | 0.678 ± — | 0.678 | 0/178/0 |
| turning cars | — | 0.606 | — | 0.606 ± — | 0.606 | 0/30/0 |
| turning vehicles | — | 0.606 | — | 0.606 ± — | 0.606 | 0/30/0 |
| vehicles which are faster than ours | — | 0.518 | — | 0.518 ± — | 0.518 | 0/371/0 |
| cars which are faster than ours | — | 0.516 | — | 0.516 ± — | 0.516 | 0/371/0 |
| parking cars | — | 0.511 | — | 0.511 ± — | 0.511 | 0/2851/0 |
| parking vehicles | — | 0.511 | — | 0.511 ± — | 0.511 | 0/2851/0 |
| moving pedestrian | — | 0.415 | — | 0.415 ± — | 0.415 | 0/88/0 |
