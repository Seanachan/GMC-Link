#!/usr/bin/env bash
# Hard-Negative Mining Finetune driver for Exp 34.
#
# For each β in {0.5, 1.0, 2.0}:
#   1. Finetune v1train_stage1.pth with HN-InfoNCE for 30 epochs at LR=1e-4.
#   2. Run the Layer 3 diagnostic on each held-out seq (0005, 0011, 0013).
#   3. Relocate the .npz into diagnostics/results/multiseq/ with the HN tag.
#
# Finally, invoke aggregate_multiseq.py with a 14-weight list (11 Exp 33
# baselines + 3 HN finetunes) to regenerate layer3_multiseq_comparison.md.
#
# Spec: docs/superpowers/specs/2026-04-18-hard-negative-mining-stage1-design.md
# Plan: docs/superpowers/plans/2026-04-18-hard-negative-mining-stage1.md
#
# Usage: bash run_hn_finetune.sh

set -euo pipefail

PY=~/miniconda/envs/RMOT/bin/python
DIAG=diagnostics/diag_gt_cosine_distributions.py
RESULTS_DIR=diagnostics/results
MULTISEQ_DIR="${RESULTS_DIR}/multiseq"
mkdir -p "${MULTISEQ_DIR}"

BETAS=(0.5 1.0 2.0)
SEQS=(0005 0011 0013)
STAGE1_WEIGHTS=gmc_link_weights_v1train_stage1.pth

# Pre-flight
if [[ ! -f "${STAGE1_WEIGHTS}" ]]; then
  echo "ERROR: stage1 weights not found: ${STAGE1_WEIGHTS}" >&2
  exit 1
fi

# ── Training loop: one finetune per β ──────────────────────────────
for beta in "${BETAS[@]}"; do
  tag="v1train_hninfo_beta${beta}"
  save_path="gmc_link_weights_${tag}.pth"
  echo "============================================================"
  echo "Finetune β=${beta} → ${save_path}"
  echo "============================================================"
  "${PY}" -m gmc_link.train \
    --split v1 \
    --loss hninfo --beta "${beta}" \
    --resume "${STAGE1_WEIGHTS}" \
    --epochs 30 --lr 1e-4 --batch-size 256 \
    --save-path "${save_path}"
done

# ── Evaluation loop: diagnostic per (seq, β) ───────────────────────
for seq in "${SEQS[@]}"; do
  echo "============================================================"
  echo "Diagnostic | seq ${seq}"
  echo "============================================================"
  for beta in "${BETAS[@]}"; do
    tag="v1train_hninfo_beta${beta}"
    weights="gmc_link_weights_${tag}.pth"
    echo "--- ${seq} / ${tag} ---"
    "${PY}" "${DIAG}" --weights "${weights}" --seq "${seq}"
    src="${RESULTS_DIR}/layer3_gt_cosine_${seq}.npz"
    dst="${MULTISEQ_DIR}/layer3_${seq}_${tag}.npz"
    if [[ ! -f "${src}" ]]; then
      echo "ERROR: expected ${src} but diagnostic did not produce it" >&2
      exit 2
    fi
    mv "${src}" "${dst}"
    src_png="${RESULTS_DIR}/layer3_gt_cosine_${seq}.png"
    if [[ -f "${src_png}" ]]; then
      mv "${src_png}" "${MULTISEQ_DIR}/layer3_${seq}_${tag}.png"
    fi
  done
done

# ── Aggregate: 11 baseline weights + 3 HN weights = 14 total ───────
ALL_WEIGHTS=(
  "v1train_stage1=gmc_link_weights_v1train_stage1.pth"
  "v1train_F1_speed=gmc_link_weights_v1train_F1_speed.pth"
  "v1train_F2_heading=gmc_link_weights_v1train_F2_heading.pth"
  "v1train_F3_accel=gmc_link_weights_v1train_F3_accel.pth"
  "v1train_F4_ego=gmc_link_weights_v1train_F4_ego.pth"
  "v1train_F5_nbrmean=gmc_link_weights_v1train_F5_nbrmean.pth"
  "v1train_F6_velrank=gmc_link_weights_v1train_F6_velrank.pth"
  "v1train_F7_headdiff=gmc_link_weights_v1train_F7_headdiff.pth"
  "v1train_F8_nndist=gmc_link_weights_v1train_F8_nndist.pth"
  "v1train_F9_density=gmc_link_weights_v1train_F9_density.pth"
  "v1train_temporal=gmc_link_weights_v1train_temporal.pth"
)
for beta in "${BETAS[@]}"; do
  tag="v1train_hninfo_beta${beta}"
  ALL_WEIGHTS+=("${tag}=gmc_link_weights_${tag}.pth")
done

LEGACY=(
  "v1train_stage1=0.779"
  "v1train_temporal=0.770"
)

echo "============================================================"
echo "Aggregating 14 weights × 3 seqs → multiseq reports"
echo "============================================================"
"${PY}" diagnostics/aggregate_multiseq.py \
  --results-dir "${MULTISEQ_DIR}" \
  --output-dir "${MULTISEQ_DIR}" \
  --weights "${ALL_WEIGHTS[@]}" \
  --seqs "${SEQS[@]}" \
  --legacy-seq-0011 "${LEGACY[@]}"

echo "Done. See ${MULTISEQ_DIR}/layer3_multiseq_comparison.md"
