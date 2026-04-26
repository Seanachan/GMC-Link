# Multi-Sequence Eval: Comparison Across Weights

## What AUC means here
AUC = probability that a randomly chosen GT-matching track gets a higher cosine score than a randomly chosen non-matching track, for a given expression. 0.50 = chance, 1.00 = perfect. < 0.50 means inverted.

**Held-out sequences:** 0005, 0011, 0013

| model_tag | mean_auc_micro | mean_auc_macro ± std | best_seq | worst_seq | max_gap |
|---|---|---|---|---|---|
| v1train_stage1 | 0.779 | 0.838 ± 0.064 | 0005: 0.821 | 0011: 0.779 | 0.042 |
| v1v2train | 0.767 | 0.843 ± 0.047 | 0011: 0.813 | 0013: 0.711 | 0.102 |
| v1train_exp36e_curriculum | 0.762 | 0.840 ± 0.063 | 0005: 0.807 | 0011: 0.773 | 0.034 |
| v1train_exp36b_long | 0.761 | 0.821 ± 0.070 | 0005: 0.818 | 0011: 0.749 | 0.069 |
| v1train_exp36b | 0.754 | 0.809 ± 0.081 | 0005: 0.790 | 0013: 0.739 | 0.051 |
| v1train_exp36a | 0.747 | 0.797 ± 0.057 | 0013: 0.773 | 0005: 0.756 | 0.016 |
| v1train_exp36a_v2 | 0.741 | 0.815 ± 0.046 | 0011: 0.775 | 0013: 0.725 | 0.050 |
| v1train_exp36d_bge | 0.735 | 0.802 ± 0.052 | 0013: 0.831 | 0005: 0.741 | 0.091 |
