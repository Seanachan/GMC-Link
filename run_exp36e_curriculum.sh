#!/usr/bin/env bash
# Exp 36E: curriculum pretraining (--stage curriculum).
# Stage 1: 100 epochs on group labels (coarse). Stage 2: 50 epochs on
# expression labels, resume stage1, lr *= 0.1.
#
# Tests whether curriculum ordering (coarse -> fine) breaks the 0.779
# V1 ceiling. Eighth lever; first to use >5 training epochs.

set -euo pipefail

PY=~/miniconda/envs/RMOT/bin/python
DIAG=diagnostics/diag_gt_cosine_distributions.py
RESULTS_DIR=diagnostics/results
MULTISEQ_DIR="${RESULTS_DIR}/multiseq"
SEQS=(0005 0011 0013)
SAVE_BASE="gmc_link_weights_v1train_exp36e"
CURRICULUM_WEIGHTS="${SAVE_BASE}_curriculum.pth"
TAG="v1train_exp36e_curriculum"

echo "[$(date +%H:%M:%S)] Training curriculum: stage1 (100ep, group) + stage2 (50ep, expr)"
"${PY}" -m gmc_link.train \
    --split v1 \
    --stage curriculum \
    --lr 3e-4 --batch-size 256 \
    --save-path "${SAVE_BASE}.pth"

echo "[$(date +%H:%M:%S)] Evaluating curriculum weights on held-out seqs"
for seq in "${SEQS[@]}"; do
    "${PY}" "${DIAG}" --weights "${CURRICULUM_WEIGHTS}" --seq "${seq}" 2>&1 | tail -6
    mv "${RESULTS_DIR}/layer3_gt_cosine_${seq}.npz" \
       "${MULTISEQ_DIR}/layer3_${seq}_${TAG}.npz"
    if [[ -f "${RESULTS_DIR}/layer3_gt_cosine_${seq}.png" ]]; then
        mv "${RESULTS_DIR}/layer3_gt_cosine_${seq}.png" \
           "${MULTISEQ_DIR}/layer3_${seq}_${TAG}.png"
    fi
done

echo "[$(date +%H:%M:%S)] Aggregating 8-way"
ALL_WEIGHTS=(
    "v1train_stage1=gmc_link_weights_v1train_stage1.pth"
    "v1train_exp36a=gmc_link_weights_v1train_exp36a.pth"
    "v1train_exp36a_v2=gmc_link_weights_v1train_exp36a_v2.pth"
    "v1train_exp36b=gmc_link_weights_v1train_exp36b.pth"
    "v1train_exp36b_long=gmc_link_weights_v1train_exp36b_long.pth"
    "v1v2train=gmc_link_weights_v1v2train.pth"
    "v1train_exp36d_bge=gmc_link_weights_v1train_exp36d_bge.pth"
    "${TAG}=${CURRICULUM_WEIGHTS}"
)
"${PY}" diagnostics/aggregate_multiseq.py \
    --results-dir "${MULTISEQ_DIR}" \
    --output-dir "${MULTISEQ_DIR}" \
    --weights "${ALL_WEIGHTS[@]}" \
    --seqs "${SEQS[@]}"

echo "[$(date +%H:%M:%S)] Done. See ${MULTISEQ_DIR}/layer3_multiseq_comparison.md"
