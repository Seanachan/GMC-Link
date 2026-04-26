#!/usr/bin/env bash
# Exp 36D: language encoder swap MiniLM-L6 (384D) -> BGE-base (768D).
# Tests whether the language-side is the ceiling after five motion-side
# experiments (36A/A-v2/B/B-long/C) all hit < 0.79 micro.
#
# Identical training recipe to stage1 baseline except for the encoder.

set -euo pipefail

PY=~/miniconda/envs/RMOT/bin/python
DIAG=diagnostics/diag_gt_cosine_distributions.py
RESULTS_DIR=diagnostics/results
MULTISEQ_DIR="${RESULTS_DIR}/multiseq"
SEQS=(0005 0011 0013)
TAG="v1train_exp36d_bge"
WEIGHTS="gmc_link_weights_${TAG}.pth"

echo "[$(date +%H:%M:%S)] Training ${WEIGHTS} (13D MLP, BGE-base 768D, V1 only)"
"${PY}" -m gmc_link.train \
    --split v1 \
    --text-encoder BAAI/bge-base-en-v1.5 \
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

echo "[$(date +%H:%M:%S)] Aggregating 7-way"
ALL_WEIGHTS=(
    "v1train_stage1=gmc_link_weights_v1train_stage1.pth"
    "v1train_exp36a=gmc_link_weights_v1train_exp36a.pth"
    "v1train_exp36a_v2=gmc_link_weights_v1train_exp36a_v2.pth"
    "v1train_exp36b=gmc_link_weights_v1train_exp36b.pth"
    "v1train_exp36b_long=gmc_link_weights_v1train_exp36b_long.pth"
    "v1v2train=gmc_link_weights_v1v2train.pth"
    "${TAG}=${WEIGHTS}"
)
"${PY}" diagnostics/aggregate_multiseq.py \
    --results-dir "${MULTISEQ_DIR}" \
    --output-dir "${MULTISEQ_DIR}" \
    --weights "${ALL_WEIGHTS[@]}" \
    --seqs "${SEQS[@]}"

echo "[$(date +%H:%M:%S)] Done. See ${MULTISEQ_DIR}/layer3_multiseq_comparison.md"
