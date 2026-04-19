# Multi-Sequence Eval: v1train_hninfo_beta2.0

## What this measures
AUC = probability that a randomly chosen GT-matching track gets a higher cosine score than a randomly chosen non-matching track, for a given expression. 0.50 = chance, 1.00 = perfect. < 0.50 means inverted.

## Headline
- Mean AUC (micro, pooled across 3 seqs): **0.733** (over 33 expressions)
- Mean AUC (macro, per-seq averaged):     **0.802** ± 0.064 (over 12 expressions in ≥2 seqs)
- Seq-0011 only (legacy, for continuity): **—**

## Per-expression breakdown
| Expression | 0005 | 0011 | 0013 | macro μ ± σ | micro | GT counts |
|---|---|---|---|---|---|---|
| cars which are braking | 0.985 | — | — | 0.985 ± — | 0.985 | 295/0/0 |
| vehicles which are braking | 0.985 | — | — | 0.985 ± — | 0.985 | 295/0/0 |
| cars in front of ours | 0.972 | — | — | 0.972 ± — | 0.972 | 295/0/0 |
| vehicles in front of ours | 0.972 | — | — | 0.972 ± — | 0.972 | 295/0/0 |
| right vehicles which are parking | — | 0.921 | — | 0.921 ± — | 0.921 | 0/1056/0 |
| right cars which are parking | — | 0.921 | — | 0.921 ± — | 0.921 | 0/1056/0 |
| vehicles in the same direction of ours | 0.813 | 0.989 | — | 0.901 ± 0.088 | 0.817 | 510/371/0 |
| cars in the same direction of ours | 0.813 | 0.989 | — | 0.901 ± 0.088 | 0.817 | 510/371/0 |
| moving vehicles | 0.861 | 0.804 | — | 0.832 ± 0.029 | 0.809 | 569/765/0 |
| moving cars | 0.861 | 0.804 | — | 0.832 ± 0.029 | 0.809 | 569/765/0 |
| counter direction cars in the left | 0.798 | 0.841 | — | 0.819 ± 0.022 | 0.809 | 526/1692/0 |
| left vehicles in the counter direction of ours | 0.798 | 0.841 | — | 0.819 ± 0.022 | 0.809 | 526/1692/0 |
| counter direction vehicles in the left | 0.797 | 0.839 | — | 0.818 ± 0.021 | 0.807 | 526/1692/0 |
| left cars in the counter direction of ours | 0.797 | 0.837 | — | 0.817 ± 0.020 | 0.805 | 526/1692/0 |
| moving left pedestrian | — | — | 0.782 | 0.782 ± — | 0.782 | 0/0/253 |
| left cars which are parking | 0.592 | 0.833 | — | 0.713 ± 0.120 | 0.737 | 169/1375/0 |
| left vehicles which are parking | 0.592 | 0.833 | — | 0.712 ± 0.120 | 0.736 | 169/1375/0 |
| moving right pedestrian | — | — | 0.732 | 0.732 ± — | 0.732 | 0/0/261 |
| vehicles in the counter direction of ours | 0.777 | 0.679 | — | 0.728 ± 0.049 | 0.696 | 526/1692/0 |
| cars in the counter direction of ours | 0.777 | 0.678 | — | 0.727 ± 0.049 | 0.696 | 526/1692/0 |
| same direction vehicles in the left | 0.693 | — | — | 0.693 ± — | 0.693 | 215/0/0 |
| same direction cars in the left | 0.693 | — | — | 0.693 ± — | 0.693 | 215/0/0 |
| left vehicles in the same direction of ours | 0.692 | — | — | 0.692 ± — | 0.692 | 215/0/0 |
| left cars in the same direction of ours | 0.692 | — | — | 0.692 ± — | 0.692 | 215/0/0 |
| vehicles in horizon direction | — | 0.662 | — | 0.662 ± — | 0.662 | 0/178/0 |
| cars in horizon direction | — | 0.661 | — | 0.661 ± — | 0.661 | 0/178/0 |
| turning cars | — | 0.520 | — | 0.520 ± — | 0.520 | 0/30/0 |
| turning vehicles | — | 0.518 | — | 0.518 ± — | 0.518 | 0/30/0 |
| parking cars | — | 0.513 | — | 0.513 ± — | 0.513 | 0/2851/0 |
| parking vehicles | — | 0.513 | — | 0.513 ± — | 0.513 | 0/2851/0 |
| vehicles which are faster than ours | — | 0.505 | — | 0.505 ± — | 0.505 | 0/371/0 |
| cars which are faster than ours | — | 0.502 | — | 0.502 ± — | 0.502 | 0/371/0 |
| moving pedestrian | — | 0.399 | — | 0.399 ± — | 0.399 | 0/88/0 |
