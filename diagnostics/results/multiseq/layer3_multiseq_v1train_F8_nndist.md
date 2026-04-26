# Multi-Sequence Eval: v1train_F8_nndist

## What this measures
AUC = probability that a randomly chosen GT-matching track gets a higher cosine score than a randomly chosen non-matching track, for a given expression. 0.50 = chance, 1.00 = perfect. < 0.50 means inverted.

## Headline
- Mean AUC (micro, pooled across 3 seqs): **0.771** (over 33 expressions)
- Mean AUC (macro, per-seq averaged):     **0.819** ± 0.071 (over 12 expressions in ≥2 seqs)
- Seq-0011 only (legacy, for continuity): **—**

## Per-expression breakdown
| Expression | 0005 | 0011 | 0013 | macro μ ± σ | micro | GT counts |
|---|---|---|---|---|---|---|
| vehicles in front of ours | 0.985 | — | — | 0.985 ± — | 0.985 | 295/0/0 |
| cars in front of ours | 0.985 | — | — | 0.985 ± — | 0.985 | 295/0/0 |
| vehicles which are braking | 0.965 | — | — | 0.965 ± — | 0.965 | 295/0/0 |
| cars which are braking | 0.965 | — | — | 0.965 ± — | 0.965 | 295/0/0 |
| right cars which are parking | — | 0.919 | — | 0.919 ± — | 0.919 | 0/1056/0 |
| right vehicles which are parking | — | 0.919 | — | 0.919 ± — | 0.919 | 0/1056/0 |
| left vehicles in the counter direction of ours | 0.826 | 0.949 | — | 0.887 ± 0.062 | 0.895 | 526/1692/0 |
| counter direction vehicles in the left | 0.825 | 0.948 | — | 0.887 ± 0.062 | 0.894 | 526/1692/0 |
| left cars in the counter direction of ours | 0.826 | 0.948 | — | 0.887 ± 0.061 | 0.894 | 526/1692/0 |
| counter direction cars in the left | 0.825 | 0.947 | — | 0.886 ± 0.061 | 0.894 | 526/1692/0 |
| moving right pedestrian | — | — | 0.871 | 0.871 ± — | 0.871 | 0/0/261 |
| turning vehicles | — | 0.801 | — | 0.801 ± — | 0.801 | 0/30/0 |
| turning cars | — | 0.801 | — | 0.801 ± — | 0.801 | 0/30/0 |
| cars in the same direction of ours | 0.797 | 0.946 | — | 0.872 ± 0.074 | 0.800 | 510/371/0 |
| vehicles in the same direction of ours | 0.797 | 0.946 | — | 0.872 ± 0.074 | 0.800 | 510/371/0 |
| moving left pedestrian | — | — | 0.798 | 0.798 ± — | 0.798 | 0/0/253 |
| moving cars | 0.873 | 0.761 | — | 0.817 ± 0.056 | 0.784 | 569/765/0 |
| moving vehicles | 0.873 | 0.761 | — | 0.817 ± 0.056 | 0.783 | 569/765/0 |
| left vehicles which are parking | 0.626 | 0.878 | — | 0.752 ± 0.126 | 0.762 | 169/1375/0 |
| left cars which are parking | 0.626 | 0.878 | — | 0.752 ± 0.126 | 0.762 | 169/1375/0 |
| same direction cars in the left | 0.714 | — | — | 0.714 ± — | 0.714 | 215/0/0 |
| left cars in the same direction of ours | 0.714 | — | — | 0.714 ± — | 0.714 | 215/0/0 |
| same direction vehicles in the left | 0.714 | — | — | 0.714 ± — | 0.714 | 215/0/0 |
| left vehicles in the same direction of ours | 0.713 | — | — | 0.713 ± — | 0.713 | 215/0/0 |
| cars in the counter direction of ours | 0.809 | 0.594 | — | 0.701 ± 0.107 | 0.659 | 526/1692/0 |
| vehicles in the counter direction of ours | 0.809 | 0.594 | — | 0.701 ± 0.107 | 0.659 | 526/1692/0 |
| cars in horizon direction | — | 0.652 | — | 0.652 ± — | 0.652 | 0/178/0 |
| vehicles in horizon direction | — | 0.652 | — | 0.652 ± — | 0.652 | 0/178/0 |
| vehicles which are faster than ours | — | 0.596 | — | 0.596 ± — | 0.596 | 0/371/0 |
| cars which are faster than ours | — | 0.588 | — | 0.588 ± — | 0.588 | 0/371/0 |
| parking cars | — | 0.512 | — | 0.512 ± — | 0.512 | 0/2851/0 |
| parking vehicles | — | 0.512 | — | 0.512 ± — | 0.512 | 0/2851/0 |
| moving pedestrian | — | 0.469 | — | 0.469 ± — | 0.469 | 0/88/0 |
