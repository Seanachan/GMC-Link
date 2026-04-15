#!/usr/bin/env python3
"""
Layer 2: Embedding Space Quality
=================================
Checks whether the trained model's embedding space has useful structure
or has collapsed. Computes intra-class vs inter-class cosine distances
and embedding variance.

Usage:
    python diagnostics/diag_embedding_space.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from collections import defaultdict
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gmc_link.alignment import MotionLanguageAligner
from gmc_link.dataset import build_training_data
from gmc_link.text_utils import TextEncoder

# ── V1 config ────────────────────────────────────────────────────────────
DATA_ROOT = "refer-kitti"
WEIGHTS_PATH = "gmc_link_weights_v1train.pth"
TRAIN_SEQUENCES = [
    "0001", "0002", "0003", "0004", "0006",
    "0007", "0008", "0009", "0010", "0012",
    "0014", "0015", "0016", "0018", "0020",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=WEIGHTS_PATH)
    parser.add_argument("--max-classes", type=int, default=50,
                        help="Max expression classes to analyze (for speed)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Weights: {args.weights}")

    # ── Load model ────────────────────────────────────────────────────
    model = MotionLanguageAligner(motion_dim=13, lang_dim=384, embed_dim=256).to(device)
    checkpoint = torch.load(args.weights, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # ── Build training data ───────────────────────────────────────────
    encoder = TextEncoder(device=str(device))
    all_motions, all_languages, all_labels = build_training_data(
        data_root=DATA_ROOT, sequences=TRAIN_SEQUENCES, text_encoder=encoder,
    )
    print(f"Total samples: {len(all_motions)}")

    # ── Group by expression class ─────────────────────────────────────
    class_samples = defaultdict(list)
    class_lang = {}
    for i, label in enumerate(all_labels):
        class_samples[label].append(i)
        if label not in class_lang:
            class_lang[label] = all_languages[i]

    classes = sorted(class_samples.keys())
    if len(classes) > args.max_classes:
        # Sample classes with most samples for stability
        classes = sorted(classes, key=lambda c: -len(class_samples[c]))[:args.max_classes]
    print(f"Analyzing {len(classes)} expression classes")

    # ── Encode all samples ────────────────────────────────────────────
    motion_arr = np.array(all_motions, dtype=np.float32)
    lang_arr = np.array(all_languages, dtype=np.float32)

    with torch.no_grad():
        motion_t = torch.tensor(motion_arr, device=device)
        lang_t = torch.tensor(lang_arr, device=device)
        # Process in chunks to avoid OOM
        chunk = 2048
        motion_embs = []
        lang_embs = []
        for start in range(0, len(motion_arr), chunk):
            end = min(start + chunk, len(motion_arr))
            m_emb, l_emb = model.encode(motion_t[start:end], lang_t[start:end])
            motion_embs.append(m_emb.cpu().numpy())
            lang_embs.append(l_emb.cpu().numpy())
        motion_embs = np.concatenate(motion_embs, axis=0)
        lang_embs = np.concatenate(lang_embs, axis=0)

    # ── Analysis ──────────────────────────────────────────────────────

    # 1. Embedding norm check (should be ~1.0 if L2-normalized)
    motion_norms = np.linalg.norm(motion_embs, axis=1)
    lang_norms = np.linalg.norm(lang_embs, axis=1)

    # 2. Mode collapse check: variance of embeddings
    motion_var = np.var(motion_embs, axis=0).mean()  # mean variance across dims
    lang_var = np.var(lang_embs, axis=0).mean()

    # 3. Intra-class vs inter-class cosine distances
    intra_cosines = []
    inter_cosines = []
    class_centroids = {}

    for cls in classes:
        idxs = class_samples[cls]
        if len(idxs) < 2:
            continue
        embs = motion_embs[idxs]
        centroid = embs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        class_centroids[cls] = centroid

        # Intra-class: pairwise cosine within class (sample up to 100 pairs)
        n = len(embs)
        if n > 50:
            sample_idx = np.random.choice(n, 50, replace=False)
            embs_sample = embs[sample_idx]
        else:
            embs_sample = embs
        cos_mat = embs_sample @ embs_sample.T
        mask = np.triu(np.ones(len(embs_sample), dtype=bool), k=1)
        intra_cosines.extend(cos_mat[mask].tolist())

    # Inter-class: pairwise cosine between centroids
    centroid_list = list(class_centroids.values())
    for i in range(len(centroid_list)):
        for j in range(i + 1, len(centroid_list)):
            inter_cosines.append(float(centroid_list[i] @ centroid_list[j]))

    intra = np.array(intra_cosines)
    inter = np.array(inter_cosines)

    # 4. Motion→Language cross-modal alignment per class
    cross_cosines = []
    for cls in classes:
        idxs = class_samples[cls]
        m_emb = motion_embs[idxs]
        l_emb = lang_embs[idxs[0:1]]  # all same expression → same lang embedding
        cos = (m_emb @ l_emb.T).flatten()
        cross_cosines.extend(cos.tolist())
    cross = np.array(cross_cosines)

    # ── Report ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LAYER 2: EMBEDDING SPACE QUALITY REPORT")
    print("=" * 60)

    print("\n  Embedding Norms (expect ~1.0 if L2-normalized):")
    print(f"    Motion: mean={motion_norms.mean():.4f} std={motion_norms.std():.6f}")
    print(f"    Language: mean={lang_norms.mean():.4f} std={lang_norms.std():.6f}")

    print(f"\n  Mean Per-Dim Variance (higher = more spread, 0 = collapse):")
    print(f"    Motion embeddings:  {motion_var:.6f}")
    print(f"    Language embeddings: {lang_var:.6f}")

    print(f"\n  Intra-Class Cosine (motion↔motion, same expression):")
    print(f"    mean={intra.mean():.4f}  std={intra.std():.4f}  "
          f"[{intra.min():.4f}, {intra.max():.4f}]")

    print(f"  Inter-Class Cosine (centroid↔centroid, different expressions):")
    print(f"    mean={inter.mean():.4f}  std={inter.std():.4f}  "
          f"[{inter.min():.4f}, {inter.max():.4f}]")

    print(f"  Intra - Inter gap: {intra.mean() - inter.mean():.4f}")

    print(f"\n  Cross-Modal Cosine (motion→language, same expression):")
    print(f"    mean={cross.mean():.4f}  std={cross.std():.4f}  "
          f"[{cross.min():.4f}, {cross.max():.4f}]")

    # Interpretation
    print()
    if motion_var < 1e-4:
        print("  ⚠ MOTION COLLAPSE: motion embeddings have near-zero variance.")
        print("    All inputs map to ~same point → cosine sim is meaningless.")
    elif motion_var < 1e-3:
        print("  ⚠ Low motion embedding variance — partial collapse risk.")
    else:
        print("  ✓ Motion embeddings show reasonable spread.")

    if intra.mean() - inter.mean() < 0.05:
        print("  ⚠ Intra/inter gap is tiny — classes overlap heavily.")
    elif intra.mean() - inter.mean() < 0.15:
        print("  ⚠ Modest class separation — some structure but noisy.")
    else:
        print("  ✓ Good class separation in embedding space.")

    if cross.mean() < 0.2:
        print("  ⚠ Low cross-modal alignment — motion and language are not well aligned.")
    else:
        print("  ✓ Reasonable cross-modal alignment.")

    print("=" * 60)

    # ── Visualization ─────────────────────────────────────────────
    os.makedirs("diagnostics/results", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Layer 2: Embedding Space Quality", fontsize=12, fontweight="bold")

    # Left: intra vs inter class cosine histograms
    axes[0].hist(intra, bins=60, alpha=0.7, label=f"Intra-class μ={intra.mean():.3f}",
                 color="#27ae60", density=True)
    axes[0].hist(inter, bins=60, alpha=0.5, label=f"Inter-class μ={inter.mean():.3f}",
                 color="#e74c3c", density=True)
    axes[0].set_xlabel("Cosine Similarity")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Intra vs Inter Class (Motion)")
    axes[0].legend(fontsize=8)

    # Middle: cross-modal cosine histogram
    axes[1].hist(cross, bins=60, alpha=0.7, color="#3498db", density=True)
    axes[1].axvline(cross.mean(), color="red", linestyle="--",
                    label=f"μ={cross.mean():.3f}")
    axes[1].set_xlabel("Cosine Similarity")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Cross-Modal (Motion→Language)")
    axes[1].legend(fontsize=8)

    # Right: per-dim variance bar chart
    motion_per_dim_var = np.var(motion_embs, axis=0)
    axes[2].bar(range(len(motion_per_dim_var)), sorted(motion_per_dim_var, reverse=True),
                color="#9b59b6", alpha=0.7)
    axes[2].set_xlabel("Embedding Dimension (sorted)")
    axes[2].set_ylabel("Variance")
    axes[2].set_title(f"Motion Embedding Variance (mean={motion_var:.5f})")
    axes[2].set_yscale("log")

    plt.tight_layout()
    plot_path = "diagnostics/results/layer2_embedding_space.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved: {plot_path}")


if __name__ == "__main__":
    main()
