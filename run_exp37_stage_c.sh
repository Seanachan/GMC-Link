#!/usr/bin/env bash
# Exp 37 Stage C — Structural ego-velocity concat ablation.
#
# C1 (no conditioning): same as Stage B winner — reused as baseline.
# C2 (concat): train with --extra-features ego_velocity_concat (+ Stage B feats if any).
#
# Depends on Stage A/B memos being written to
#   diagnostics/results/exp37/stage_a_winner   (e.g., "orb" | "recoverpose")
#   diagnostics/results/exp37/stage_b_features (e.g., "omf_stats" | "" )
#
# Spec: §§2 H_C, 3, 7 (Stage C), 10
# Plan: Task 13

set -euo pipefail

PY=~/miniconda/envs/RMOT/bin/python
DIAG=diagnostics/diag_gt_cosine_distributions.py
RESULTS_DIR=diagnostics/results
OUT="${RESULTS_DIR}/exp37"

if [[ ! -f "${OUT}/stage_a_winner" ]]; then
  echo "ERROR: ${OUT}/stage_a_winner missing — finish Task 8 first." >&2
  exit 1
fi
EGO=$(cat "${OUT}/stage_a_winner" | tr -d '[:space:]')
BFEAT=$(cat "${OUT}/stage_b_features" 2>/dev/null | tr -d '[:space:]' || true)

if [[ -n "${BFEAT}" ]]; then
  FEAT_ARG="--extra-features ${BFEAT},ego_velocity_concat"
else
  FEAT_ARG="--extra-features ego_velocity_concat"
fi

C2_TAG="exp37_stage_c2_${EGO}"
C2_W="gmc_link_weights_${C2_TAG}.pth"

echo "============================================================"
echo "Training C2: --ego ${EGO} ${FEAT_ARG} → ${C2_W}"
echo "============================================================"
"${PY}" -m gmc_link.train \
  --split v1 \
  --ego "${EGO}" \
  ${FEAT_ARG} \
  --epochs 100 \
  --lr 1e-3 \
  --batch-size 128 \
  --save-path "${C2_W}"

# Eval on held-out seqs
SEQS=(0005 0011 0013)
for seq in "${SEQS[@]}"; do
  echo "--- ${seq} / ${C2_TAG} ---"
  "${PY}" "${DIAG}" --weights "${C2_W}" --seq "${seq}"
  src="${RESULTS_DIR}/layer3_gt_cosine_${seq}.npz"
  dst="${OUT}/layer3_${seq}_${C2_TAG}.npz"
  mv "${src}" "${dst}"
  src_png="${RESULTS_DIR}/layer3_gt_cosine_${seq}.png"
  [[ -f "${src_png}" ]] && mv "${src_png}" "${OUT}/layer3_${seq}_${C2_TAG}.png"
done

# Aggregate this arm alongside A/B winners (if their npz files live in ${OUT})
"${PY}" diagnostics/aggregate_multiseq.py \
  --results-dir "${OUT}" \
  --output-dir "${OUT}" \
  --weights "${C2_TAG}=${C2_W}" \
  --seqs "${SEQS[@]}"

echo "Done. See ${OUT}/layer3_multiseq_${C2_TAG}.md"
