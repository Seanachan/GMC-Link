# Multi-Sequence Eval: Comparison Across Weights

## What AUC means here
AUC = probability that a randomly chosen GT-matching track gets a higher cosine score than a randomly chosen non-matching track, for a given expression. 0.50 = chance, 1.00 = perfect. < 0.50 means inverted.

**Held-out sequences:** 0005, 0011, 0013

| model_tag | mean_auc_micro | mean_auc_macro ± std | best_seq | worst_seq | max_gap |
|---|---|---|---|---|---|
| exp37_stage_b2_orb_omf28d | 0.624 | 0.577 ± 0.030 | 0013: 0.643 | 0011: 0.609 | 0.034 |
