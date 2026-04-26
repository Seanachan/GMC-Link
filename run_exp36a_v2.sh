#!/usr/bin/env bash
# Exp 36A-v2: spec-correct accel_multiscale = (v_k(t) - v_k(t-k))/k
# Same 25D as exp36a but true temporal derivative instead of scale-difference.
# Rules out impl-vs-spec confound before declaring features non-informative.

set -euo pipefail

PY=~/miniconda/envs/RMOT/bin/python
DIAG=diagnostics/diag_gt_cosine_distributions.py
RESULTS_DIR=diagnostics/results
MULTISEQ_DIR="${RESULTS_DIR}/multiseq"
SEQS=(0005 0011 0013)
TAG="v1train_exp36a_v2"
WEIGHTS="gmc_link_weights_${TAG}.pth"

echo "[$(date +%H:%M:%S)] Training ${WEIGHTS} (13D + spec-correct accel_multiscale + heading_sincos = 25D)"
"${PY}" -m gmc_link.train \
    --split v1 \
    --extra-features accel_multiscale,heading_sincos \
    --epochs 5 --lr 3e-4 --batch-size 128 \
    --save-path "${WEIGHTS}"

echo "[$(date +%H:%M:%S)] Evaluating on held-out seqs"
for seq in "${SEQS[@]}"; do
    "${PY}" "${DIAG}" --weights "${WEIGHTS}" --seq "${seq}" 2>&1 | tail -6
    mv "${RESULTS_DIR}/layer3_gt_cosine_${seq}.npz" \
       "${MULTISEQ_DIR}/layer3_${seq}_${TAG}.npz"
    if [[ -f "${RESULTS_DIR}/layer3_gt_cosine_${seq}.png" ]]; then
        mv "${RESULTS_DIR}/layer3_gt_cosine_${seq}.png" \
           "${MULTISEQ_DIR}/layer3_${seq}_${TAG}.png"
    fi
done

echo "[$(date +%H:%M:%S)] Aggregating against stage1 and exp36a"
ALL_WEIGHTS=(
    "v1train_stage1=gmc_link_weights_v1train_stage1.pth"
    "v1train_exp36a=gmc_link_weights_v1train_exp36a.pth"
    "${TAG}=${WEIGHTS}"
)
"${PY}" diagnostics/aggregate_multiseq.py \
    --results-dir "${MULTISEQ_DIR}" \
    --output-dir "${MULTISEQ_DIR}" \
    --weights "${ALL_WEIGHTS[@]}" \
    --seqs "${SEQS[@]}"

echo "[$(date +%H:%M:%S)] Done. See ${MULTISEQ_DIR}/layer3_multiseq_comparison.md"
