# Multi-Sequence Eval: v1train_temporal

## What this measures
AUC = probability that a randomly chosen GT-matching track gets a higher cosine score than a randomly chosen non-matching track, for a given expression. 0.50 = chance, 1.00 = perfect. < 0.50 means inverted.

## Headline
- Mean AUC (micro, pooled across 3 seqs): **0.747** (over 33 expressions)
- Mean AUC (macro, per-seq averaged):     **0.828** ± 0.077 (over 12 expressions in ≥2 seqs)
- Seq-0011 only (legacy, for continuity): **0.770**

## Per-expression breakdown
| Expression | 0005 | 0011 | 0013 | macro μ ± σ | micro | GT counts |
|---|---|---|---|---|---|---|
| cars in front of ours | 0.999 | — | — | 0.999 ± — | 0.999 | 295/0/0 |
| cars which are braking | 0.999 | — | — | 0.999 ± — | 0.999 | 295/0/0 |
| vehicles which are braking | 0.999 | — | — | 0.999 ± — | 0.999 | 295/0/0 |
| vehicles in front of ours | 0.998 | — | — | 0.998 ± — | 0.998 | 295/0/0 |
| right vehicles which are parking | — | 0.924 | — | 0.924 ± — | 0.924 | 0/1056/0 |
| right cars which are parking | — | 0.922 | — | 0.922 ± — | 0.922 | 0/1056/0 |
| counter direction cars in the left | 0.827 | 0.937 | — | 0.882 ± 0.055 | 0.883 | 526/1692/0 |
| left cars in the counter direction of ours | 0.826 | 0.936 | — | 0.881 ± 0.055 | 0.883 | 526/1692/0 |
| counter direction vehicles in the left | 0.826 | 0.935 | — | 0.881 ± 0.054 | 0.881 | 526/1692/0 |
| left vehicles in the counter direction of ours | 0.828 | 0.928 | — | 0.878 ± 0.050 | 0.877 | 526/1692/0 |
| moving vehicles | 0.928 | 0.834 | — | 0.881 ± 0.047 | 0.845 | 569/765/0 |
| moving cars | 0.930 | 0.827 | — | 0.879 ± 0.052 | 0.842 | 569/765/0 |
| moving right pedestrian | — | — | 0.827 | 0.827 ± — | 0.827 | 0/0/261 |
| turning vehicles | — | 0.822 | — | 0.822 ± — | 0.822 | 0/30/0 |
| turning cars | — | 0.811 | — | 0.811 ± — | 0.811 | 0/30/0 |
| cars in the same direction of ours | 0.776 | 0.990 | — | 0.883 ± 0.107 | 0.797 | 510/371/0 |
| vehicles in the same direction of ours | 0.775 | 0.987 | — | 0.881 ± 0.106 | 0.794 | 510/371/0 |
| moving left pedestrian | — | — | 0.792 | 0.792 ± — | 0.792 | 0/0/253 |
| left vehicles which are parking | 0.599 | 0.899 | — | 0.749 ± 0.150 | 0.776 | 169/1375/0 |
| left cars which are parking | 0.594 | 0.899 | — | 0.747 ± 0.153 | 0.775 | 169/1375/0 |
| vehicles in horizon direction | — | 0.668 | — | 0.668 ± — | 0.668 | 0/178/0 |
| cars in horizon direction | — | 0.665 | — | 0.665 ± — | 0.665 | 0/178/0 |
| cars in the counter direction of ours | 0.809 | 0.583 | — | 0.696 ± 0.113 | 0.651 | 526/1692/0 |
| vehicles in the counter direction of ours | 0.811 | 0.576 | — | 0.694 ± 0.118 | 0.647 | 526/1692/0 |
| cars which are faster than ours | — | 0.584 | — | 0.584 ± — | 0.584 | 0/371/0 |
| vehicles which are faster than ours | — | 0.578 | — | 0.578 ± — | 0.578 | 0/371/0 |
| parking vehicles | — | 0.525 | — | 0.525 ± — | 0.525 | 0/2851/0 |
| parking cars | — | 0.524 | — | 0.524 ± — | 0.524 | 0/2851/0 |
| same direction vehicles in the left | 0.503 | — | — | 0.503 ± — | 0.503 | 215/0/0 |
| same direction cars in the left | 0.502 | — | — | 0.502 ± — | 0.502 | 215/0/0 |
| left vehicles in the same direction of ours | 0.495 | — | — | 0.495 ± — | 0.495 | 215/0/0 |
| left cars in the same direction of ours | 0.493 | — | — | 0.493 ± — | 0.493 | 215/0/0 |
| moving pedestrian | — | 0.360 | — | 0.360 ± — | 0.360 | 0/88/0 |
