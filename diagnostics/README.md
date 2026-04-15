# GMC-Link Diagnostic Suite

Five independent diagnostic layers for identifying root causes of poor motion-language similarity discrimination. Run them in order (Layer 1 → 5) — each layer builds on the insights of the previous one.

## Prerequisites

```bash
pip install -e .          # install gmc_link package
# Ensure refer-kitti/ symlink or directory exists at project root
# Ensure NeuralSORT/ track directory exists for Layers 4-5
# Ensure trained weights exist (e.g. gmc_link_weights_v1train.pth)
```

## Quick Start

```bash
# Run all layers sequentially (V1 split, seq 0011)
python diagnostics/diag_training_convergence.py --weights gmc_link_weights_v1train.pth
python diagnostics/diag_embedding_space.py      --weights gmc_link_weights_v1train.pth
python diagnostics/diag_gt_cosine_distributions.py --weights gmc_link_weights_v1train.pth --seq 0011
python diagnostics/diag_train_vs_inference_gap.py  --weights gmc_link_weights_v1train.pth --seq 0011
python diagnostics/diag_live_inference_scores.py   --weights gmc_link_weights_v1train.pth --seq 0011

# Generate combined summary figure from Layer 4+5 results
python diagnostics/plot_diagnostics.py
```

## Layer Descriptions

| Layer | Script | What it checks | Key output |
|-------|--------|---------------|------------|
| 1 | `diag_training_convergence.py` | Did training converge? Loss, retrieval accuracy, positive/negative cosine histograms | `layer1_training_convergence.png` |
| 2 | `diag_embedding_space.py` | Is the embedding space structured or collapsed? Intra/inter-class cosine, cross-modal alignment, per-dim variance | `layer2_embedding_space.png` |
| 3 | `diag_gt_cosine_distributions.py` | Do GT tracks score higher than non-GT using ground-truth centroids? Per-expression AUC | `layer3_gt_cosine_<seq>.png` + `.npz` |
| 4 | `diag_train_vs_inference_gap.py` | Does the live inference distribution match training? Per-dimension mean shift, variance ratio | `layer4_distribution_gap_<seq>.png` + `.npz` |
| 5 | `diag_live_inference_scores.py` | End-to-end: do NeuralSORT tracks get discriminative scores? GT vs non-GT cosine + sigmoid scores, AUC | `layer5_live_scores_<seq>.png` + `.npz` |

## Interpretation Guide

**Layer 1** — If accuracy is near-random and loss plateaus high, training itself failed. Check the loss function, data pipeline, or learning rate.

**Layer 2** — If per-dim variance is < 1e-4, embeddings have collapsed (all inputs map to the same point). If intra-inter gap < 0.05, classes overlap heavily. If cross-modal cosine < 0.2, motion and language projections are misaligned.

**Layer 3** — This is the "ideal conditions" test: GT centroids, no tracker noise. If AUC is near 0.5 here, the model fundamentally cannot distinguish matching vs. non-matching motion patterns. Fix the model/loss before looking at tracker issues.

**Layer 4** — Large standardized shifts (|delta_mu| / sigma_train > 1.0) or extreme variance ratios indicate the model sees very different inputs at inference vs. training. Common causes: outlier features, different ego-motion characteristics, missing normalization.

**Layer 5** — The end-to-end test. Compare with Layer 3 AUC — if Layer 3 is good but Layer 5 drops, tracker noise is the bottleneck. If both are bad, the problem is upstream (model/loss).

## Output

All results are saved to `diagnostics/results/`:
- `.png` files: visualizations for each layer
- `.npz` files: raw data for further analysis or cross-layer comparison
- `diagnostic_summary.png`: combined 2x2 overview (from `plot_diagnostics.py`)

## Adding New Diagnostics

Each diagnostic script is self-contained. To add a new layer:
1. Create `diagnostics/diag_<name>.py`
2. Use relative paths and accept `--weights`, `--seq` CLI args
3. Save outputs to `diagnostics/results/`
4. Include both a text report (printed) and a visualization (saved as PNG)
