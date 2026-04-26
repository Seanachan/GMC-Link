# Multi-Sequence Eval: exp37_stage_c2_orb

## What this measures
AUC = probability that a randomly chosen GT-matching track gets a higher cosine score than a randomly chosen non-matching track, for a given expression. 0.50 = chance, 1.00 = perfect. < 0.50 means inverted.

## Headline
- Mean AUC (micro, pooled across 3 seqs): **0.755** (over 33 expressions)
- Mean AUC (macro, per-seq averaged):     **0.814** ± 0.070 (over 12 expressions in ≥2 seqs)
- Seq-0011 only (legacy, for continuity): **—**

## Per-expression breakdown
| Expression | 0005 | 0011 | 0013 | macro μ ± σ | micro | GT counts |
|---|---|---|---|---|---|---|
| cars which are braking | 0.987 | — | — | 0.987 ± — | 0.987 | 295/0/0 |
| vehicles which are braking | 0.986 | — | — | 0.986 ± — | 0.986 | 295/0/0 |
| vehicles in front of ours | 0.985 | — | — | 0.985 ± — | 0.985 | 295/0/0 |
| cars in front of ours | 0.985 | — | — | 0.985 ± — | 0.985 | 295/0/0 |
| right vehicles which are parking | — | 0.939 | — | 0.939 ± — | 0.939 | 0/1056/0 |
| right cars which are parking | — | 0.939 | — | 0.939 ± — | 0.939 | 0/1056/0 |
| counter direction cars in the left | 0.826 | 0.946 | — | 0.886 ± 0.060 | 0.892 | 526/1692/0 |
| left vehicles in the counter direction of ours | 0.826 | 0.943 | — | 0.884 ± 0.059 | 0.890 | 526/1692/0 |
| left cars in the counter direction of ours | 0.826 | 0.942 | — | 0.884 ± 0.058 | 0.890 | 526/1692/0 |
| counter direction vehicles in the left | 0.825 | 0.942 | — | 0.884 ± 0.058 | 0.890 | 526/1692/0 |
| moving right pedestrian | — | — | 0.871 | 0.871 ± — | 0.871 | 0/0/261 |
| cars in the same direction of ours | 0.806 | 0.940 | — | 0.873 ± 0.067 | 0.802 | 510/371/0 |
| vehicles in the same direction of ours | 0.806 | 0.940 | — | 0.873 ± 0.067 | 0.801 | 510/371/0 |
| moving left pedestrian | — | — | 0.787 | 0.787 ± — | 0.787 | 0/0/253 |
| moving cars | 0.856 | 0.707 | — | 0.781 ± 0.075 | 0.743 | 569/765/0 |
| moving vehicles | 0.856 | 0.706 | — | 0.781 ± 0.075 | 0.743 | 569/765/0 |
| vehicles in horizon direction | — | 0.741 | — | 0.741 ± — | 0.741 | 0/178/0 |
| cars in horizon direction | — | 0.741 | — | 0.741 ± — | 0.741 | 0/178/0 |
| left cars which are parking | 0.625 | 0.856 | — | 0.740 ± 0.116 | 0.739 | 169/1375/0 |
| left vehicles which are parking | 0.625 | 0.856 | — | 0.740 ± 0.116 | 0.739 | 169/1375/0 |
| turning cars | — | 0.727 | — | 0.727 ± — | 0.727 | 0/30/0 |
| turning vehicles | — | 0.727 | — | 0.727 ± — | 0.727 | 0/30/0 |
| left cars in the same direction of ours | 0.680 | — | — | 0.680 ± — | 0.680 | 215/0/0 |
| same direction cars in the left | 0.679 | — | — | 0.679 ± — | 0.679 | 215/0/0 |
| left vehicles in the same direction of ours | 0.679 | — | — | 0.679 ± — | 0.679 | 215/0/0 |
| same direction vehicles in the left | 0.679 | — | — | 0.679 ± — | 0.679 | 215/0/0 |
| cars in the counter direction of ours | 0.822 | 0.616 | — | 0.719 ± 0.103 | 0.676 | 526/1692/0 |
| vehicles in the counter direction of ours | 0.822 | 0.616 | — | 0.719 ± 0.103 | 0.676 | 526/1692/0 |
| cars which are faster than ours | — | 0.587 | — | 0.587 ± — | 0.587 | 0/371/0 |
| vehicles which are faster than ours | — | 0.587 | — | 0.587 ± — | 0.587 | 0/371/0 |
| moving pedestrian | — | 0.460 | — | 0.460 ± — | 0.460 | 0/88/0 |
| parking vehicles | — | 0.341 | — | 0.341 ± — | 0.341 | 0/2851/0 |
| parking cars | — | 0.341 | — | 0.341 ± — | 0.341 | 0/2851/0 |
