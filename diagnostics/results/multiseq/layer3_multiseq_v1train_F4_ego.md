# Multi-Sequence Eval: v1train_F4_ego

## What this measures
AUC = probability that a randomly chosen GT-matching track gets a higher cosine score than a randomly chosen non-matching track, for a given expression. 0.50 = chance, 1.00 = perfect. < 0.50 means inverted.

## Headline
- Mean AUC (micro, pooled across 3 seqs): **0.757** (over 33 expressions)
- Mean AUC (macro, per-seq averaged):     **0.810** ± 0.074 (over 12 expressions in ≥2 seqs)
- Seq-0011 only (legacy, for continuity): **—**

## Per-expression breakdown
| Expression | 0005 | 0011 | 0013 | macro μ ± σ | micro | GT counts |
|---|---|---|---|---|---|---|
| cars in front of ours | 0.991 | — | — | 0.991 ± — | 0.991 | 295/0/0 |
| vehicles in front of ours | 0.990 | — | — | 0.990 ± — | 0.990 | 295/0/0 |
| vehicles which are braking | 0.977 | — | — | 0.977 ± — | 0.977 | 295/0/0 |
| cars which are braking | 0.977 | — | — | 0.977 ± — | 0.977 | 295/0/0 |
| right cars which are parking | — | 0.941 | — | 0.941 ± — | 0.941 | 0/1056/0 |
| right vehicles which are parking | — | 0.941 | — | 0.941 ± — | 0.941 | 0/1056/0 |
| counter direction cars in the left | 0.833 | 0.949 | — | 0.891 ± 0.058 | 0.896 | 526/1692/0 |
| counter direction vehicles in the left | 0.834 | 0.948 | — | 0.891 ± 0.057 | 0.896 | 526/1692/0 |
| left cars in the counter direction of ours | 0.835 | 0.945 | — | 0.890 ± 0.055 | 0.895 | 526/1692/0 |
| left vehicles in the counter direction of ours | 0.835 | 0.945 | — | 0.890 ± 0.055 | 0.894 | 526/1692/0 |
| moving right pedestrian | — | — | 0.839 | 0.839 ± — | 0.839 | 0/0/261 |
| cars in the same direction of ours | 0.803 | 0.921 | — | 0.862 ± 0.059 | 0.793 | 510/371/0 |
| vehicles in the same direction of ours | 0.803 | 0.921 | — | 0.862 ± 0.059 | 0.792 | 510/371/0 |
| moving left pedestrian | — | — | 0.782 | 0.782 ± — | 0.782 | 0/0/253 |
| turning cars | — | 0.763 | — | 0.763 ± — | 0.763 | 0/30/0 |
| turning vehicles | — | 0.762 | — | 0.762 ± — | 0.762 | 0/30/0 |
| left vehicles which are parking | 0.631 | 0.871 | — | 0.751 ± 0.120 | 0.749 | 169/1375/0 |
| left cars which are parking | 0.631 | 0.871 | — | 0.751 ± 0.120 | 0.749 | 169/1375/0 |
| cars in horizon direction | — | 0.734 | — | 0.734 ± — | 0.734 | 0/178/0 |
| vehicles in horizon direction | — | 0.733 | — | 0.733 ± — | 0.733 | 0/178/0 |
| moving vehicles | 0.804 | 0.710 | — | 0.757 ± 0.047 | 0.732 | 569/765/0 |
| moving cars | 0.804 | 0.710 | — | 0.757 ± 0.047 | 0.731 | 569/765/0 |
| left vehicles in the same direction of ours | 0.667 | — | — | 0.667 ± — | 0.667 | 215/0/0 |
| same direction vehicles in the left | 0.667 | — | — | 0.667 ± — | 0.667 | 215/0/0 |
| left cars in the same direction of ours | 0.667 | — | — | 0.667 ± — | 0.667 | 215/0/0 |
| same direction cars in the left | 0.667 | — | — | 0.667 ± — | 0.667 | 215/0/0 |
| vehicles in the counter direction of ours | 0.834 | 0.581 | — | 0.707 ± 0.126 | 0.662 | 526/1692/0 |
| cars in the counter direction of ours | 0.833 | 0.581 | — | 0.707 ± 0.126 | 0.662 | 526/1692/0 |
| cars which are faster than ours | — | 0.620 | — | 0.620 ± — | 0.620 | 0/371/0 |
| vehicles which are faster than ours | — | 0.620 | — | 0.620 ± — | 0.620 | 0/371/0 |
| moving pedestrian | — | 0.395 | — | 0.395 ± — | 0.395 | 0/88/0 |
| parking vehicles | — | 0.392 | — | 0.392 ± — | 0.392 | 0/2851/0 |
| parking cars | — | 0.391 | — | 0.391 ± — | 0.391 | 0/2851/0 |
