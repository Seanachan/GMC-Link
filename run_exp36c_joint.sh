#!/usr/bin/env bash
# Exp 36C: V1+V2 joint training on 13D MLP.
# Tests supervision expansion (paraphrase-augmented V2 labels, same bbox tracks)
# as the data-bound lever after 36A/A-v2/B/B-long ruled out features + arch.
#
# V2 held-out seqs {0005, 0011, 0013} are filtered inside train.py (--split v1v2).

set -euo pipefail

PY=~/miniconda/envs/RMOT/bin/python
DIAG=diagnostics/diag_gt_cosine_distributions.py
RESULTS_DIR=diagnostics/results
MULTISEQ_DIR="${RESULTS_DIR}/multiseq"
SEQS=(0005 0011 0013)
TAG="v1v2train"
WEIGHTS="gmc_link_weights_${TAG}.pth"

echo "[$(date +%H:%M:%S)] Training ${WEIGHTS} (13D MLP, V1+V2 joint, heldout-filtered)"
"${PY}" -m gmc_link.train \
    --split v1v2 \
    --epochs 5 --lr 3e-4 --batch-size 256 \
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

echo "[$(date +%H:%M:%S)] Aggregating 6-way (stage1 + 4x exp36 + joint)"
ALL_WEIGHTS=(
    "v1train_stage1=gmc_link_weights_v1train_stage1.pth"
    "v1train_exp36a=gmc_link_weights_v1train_exp36a.pth"
    "v1train_exp36a_v2=gmc_link_weights_v1train_exp36a_v2.pth"
    "v1train_exp36b=gmc_link_weights_v1train_exp36b.pth"
    "v1train_exp36b_long=gmc_link_weights_v1train_exp36b_long.pth"
    "${TAG}=${WEIGHTS}"
)
"${PY}" diagnostics/aggregate_multiseq.py \
    --results-dir "${MULTISEQ_DIR}" \
    --output-dir "${MULTISEQ_DIR}" \
    --weights "${ALL_WEIGHTS[@]}" \
    --seqs "${SEQS[@]}"

echo "[$(date +%H:%M:%S)] Done. See ${MULTISEQ_DIR}/layer3_multiseq_comparison.md"
