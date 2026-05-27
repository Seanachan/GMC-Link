# Multi-Sequence Eval: Comparison Across Weights

## What AUC means here
AUC = probability that a randomly chosen GT-matching track gets a higher cosine score than a randomly chosen non-matching track, for a given expression. 0.50 = chance, 1.00 = perfect. < 0.50 means inverted.

**Held-out sequences:** 0005, 0011, 0013

| model_tag | mean_auc_micro | mean_auc_macro ± std | best_seq | worst_seq | max_gap |
|---|---|---|---|---|---|
| v1train_world_xy_seed1 | 0.759 | 0.829 ± 0.069 | 0013: 0.809 | 0005: 0.779 | 0.030 |
| v1train_world_xy_seed2 | 0.756 | 0.818 ± 0.072 | 0013: 0.830 | 0011: 0.768 | 0.061 |
| v1train_world_xy_seed0 | 0.752 | 0.829 ± 0.072 | 0013: 0.832 | 0011: 0.765 | 0.067 |
