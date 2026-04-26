# Multi-Sequence Eval: v1train_F1_speed

## What this measures
AUC = probability that a randomly chosen GT-matching track gets a higher cosine score than a randomly chosen non-matching track, for a given expression. 0.50 = chance, 1.00 = perfect. < 0.50 means inverted.

## Headline
- Mean AUC (micro, pooled across 3 seqs): **0.771** (over 33 expressions)
- Mean AUC (macro, per-seq averaged):     **0.827** ± 0.063 (over 12 expressions in ≥2 seqs)
- Seq-0011 only (legacy, for continuity): **—**

## Per-expression breakdown
| Expression | 0005 | 0011 | 0013 | macro μ ± σ | micro | GT counts |
|---|---|---|---|---|---|---|
| vehicles in front of ours | 0.987 | — | — | 0.987 ± — | 0.987 | 295/0/0 |
| cars in front of ours | 0.987 | — | — | 0.987 ± — | 0.987 | 295/0/0 |
| cars which are braking | 0.975 | — | — | 0.975 ± — | 0.975 | 295/0/0 |
| vehicles which are braking | 0.975 | — | — | 0.975 ± — | 0.975 | 295/0/0 |
| right vehicles which are parking | — | 0.938 | — | 0.938 ± — | 0.938 | 0/1056/0 |
| right cars which are parking | — | 0.938 | — | 0.938 ± — | 0.938 | 0/1056/0 |
| left vehicles in the counter direction of ours | 0.821 | 0.953 | — | 0.887 ± 0.066 | 0.900 | 526/1692/0 |
| counter direction cars in the left | 0.821 | 0.952 | — | 0.887 ± 0.066 | 0.900 | 526/1692/0 |
| counter direction vehicles in the left | 0.821 | 0.951 | — | 0.886 ± 0.065 | 0.899 | 526/1692/0 |
| left cars in the counter direction of ours | 0.822 | 0.951 | — | 0.886 ± 0.064 | 0.899 | 526/1692/0 |
| moving right pedestrian | — | — | 0.809 | 0.809 ± — | 0.809 | 0/0/261 |
| cars in the same direction of ours | 0.800 | 0.973 | — | 0.887 ± 0.087 | 0.807 | 510/371/0 |
| vehicles in the same direction of ours | 0.800 | 0.973 | — | 0.887 ± 0.087 | 0.807 | 510/371/0 |
| moving left pedestrian | — | — | 0.768 | 0.768 ± — | 0.768 | 0/0/253 |
| left vehicles which are parking | 0.619 | 0.881 | — | 0.750 ± 0.131 | 0.763 | 169/1375/0 |
| left cars which are parking | 0.619 | 0.881 | — | 0.750 ± 0.131 | 0.763 | 169/1375/0 |
| moving cars | 0.864 | 0.743 | — | 0.804 ± 0.060 | 0.763 | 569/765/0 |
| moving vehicles | 0.864 | 0.743 | — | 0.804 ± 0.060 | 0.763 | 569/765/0 |
| cars in the counter direction of ours | 0.822 | 0.673 | — | 0.748 ± 0.074 | 0.719 | 526/1692/0 |
| vehicles in the counter direction of ours | 0.822 | 0.672 | — | 0.747 ± 0.075 | 0.719 | 526/1692/0 |
| turning cars | — | 0.714 | — | 0.714 ± — | 0.714 | 0/30/0 |
| turning vehicles | — | 0.712 | — | 0.712 ± — | 0.712 | 0/30/0 |
| cars in horizon direction | — | 0.708 | — | 0.708 ± — | 0.708 | 0/178/0 |
| vehicles in horizon direction | — | 0.708 | — | 0.708 ± — | 0.708 | 0/178/0 |
| same direction vehicles in the left | 0.701 | — | — | 0.701 ± — | 0.701 | 215/0/0 |
| left vehicles in the same direction of ours | 0.701 | — | — | 0.701 ± — | 0.701 | 215/0/0 |
| left cars in the same direction of ours | 0.700 | — | — | 0.700 ± — | 0.700 | 215/0/0 |
| same direction cars in the left | 0.700 | — | — | 0.700 ± — | 0.700 | 215/0/0 |
| vehicles which are faster than ours | — | 0.607 | — | 0.607 ± — | 0.607 | 0/371/0 |
| cars which are faster than ours | — | 0.606 | — | 0.606 ± — | 0.606 | 0/371/0 |
| moving pedestrian | — | 0.516 | — | 0.516 ± — | 0.516 | 0/88/0 |
| parking cars | — | 0.500 | — | 0.500 ± — | 0.500 | 0/2851/0 |
| parking vehicles | — | 0.500 | — | 0.500 ± — | 0.500 | 0/2851/0 |
