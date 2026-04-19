# Multi-Sequence Eval: Comparison Across Weights

## What AUC means here
AUC = probability that a randomly chosen GT-matching track gets a higher cosine score than a randomly chosen non-matching track, for a given expression. 0.50 = chance, 1.00 = perfect. < 0.50 means inverted.

**Held-out sequences:** 0005, 0011, 0013

| model_tag | mean_auc_micro | mean_auc_macro ± std | best_seq | worst_seq | max_gap |
|---|---|---|---|---|---|
| v1train_F9_density | 0.781 | 0.828 ± 0.067 | 0005: 0.823 | 0011: 0.772 | 0.051 |
| v1train_F3_accel | 0.779 | 0.830 ± 0.060 | 0005: 0.805 | 0011: 0.788 | 0.017 |
| v1train_stage1 | 0.779 | 0.838 ± 0.064 | 0005: 0.821 | 0011: 0.779 | 0.042 |
| v1train_F7_headdiff | 0.772 | 0.831 ± 0.059 | 0013: 0.821 | 0011: 0.787 | 0.034 |
| v1train_F1_speed | 0.771 | 0.827 ± 0.063 | 0005: 0.811 | 0011: 0.774 | 0.037 |
| v1train_F8_nndist | 0.771 | 0.819 ± 0.071 | 0013: 0.835 | 0011: 0.764 | 0.071 |
| v1train_F6_velrank | 0.770 | 0.835 ± 0.062 | 0013: 0.816 | 0011: 0.776 | 0.040 |
| v1train_F2_heading | 0.762 | 0.827 ± 0.068 | 0013: 0.815 | 0011: 0.765 | 0.051 |
| v1train_F4_ego | 0.757 | 0.810 ± 0.074 | 0013: 0.810 | 0011: 0.750 | 0.061 |
| v1train_F5_nbrmean | 0.756 | 0.809 ± 0.060 | 0013: 0.825 | 0011: 0.754 | 0.071 |
| v1train_hninfo_beta0.5 | 0.753 | 0.818 ± 0.064 | 0005: 0.806 | 0013: 0.748 | 0.058 |
| v1train_temporal | 0.747 | 0.828 ± 0.077 | 0013: 0.810 | 0011: 0.770 | 0.039 |
| v1train_hninfo_beta1.0 | 0.746 | 0.810 ± 0.063 | 0005: 0.801 | 0011: 0.740 | 0.060 |
| v1train_hninfo_beta2.0 | 0.733 | 0.802 ± 0.064 | 0005: 0.798 | 0011: 0.722 | 0.076 |
