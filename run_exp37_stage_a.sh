#!/usr/bin/env bash
# Exp 37 Stage A — Ego-source ablation driver.
#
# A1 (baseline): ORB homography — reuses existing gmc_link_weights_v1train_stage1.pth
# A2 (challenger): cv2.recoverPose-projected plane-at-infinity (spec §10 kill switch #1,
#                  since SparseMFE upstream code is not public — see gatekeeper memo)
#
# Both weights eval on held-out {0005, 0011, 0013} via the same Layer 3 GT-cosine
# diagnostic used by the multiseq aggregator in prior experiments (Exp 34-36).
#
# Spec: docs/superpowers/specs/2026-04-22-ego-motion-systematic-design.md §§2, 4, 10
# Plan: docs/superpowers/plans/2026-04-22-exp37-ego-motion-systematic.md Task 8
#
# Usage:
#   bash run_exp37_stage_a.sh                 # train recoverpose + eval both
#   A2_EGO=orb bash run_exp37_stage_a.sh      # sanity-check path only (no-op re-train)
#   SKIP_TRAIN=1 bash run_exp37_stage_a.sh    # eval-only (A2 weights must exist)
#
# Output layout (mirrors diagnostics/results/multiseq):
#   diagnostics/results/exp37/layer3_{seq}_{tag}.npz
#   diagnostics/results/exp37/layer3_multiseq_{tag}.{md,json,png}
#   diagnostics/results/exp37/layer3_multiseq_comparison.md

set -euo pipefail

PY=~/miniconda/envs/RMOT/bin/python
DIAG=diagnostics/diag_gt_cosine_distributions.py
RESULTS_DIR=diagnostics/results
OUT="${RESULTS_DIR}/exp37"
mkdir -p "${OUT}"

# ─── A1: ORB baseline ──────────────────────────────────────────────
A1_TAG=v1train_stage1
A1_W=gmc_link_weights_v1train_stage1.pth

# ─── A2: recoverPose challenger ────────────────────────────────────
A2_EGO="${A2_EGO:-recoverpose}"
A2_TAG="exp37_stage_a2_${A2_EGO}"
A2_W="gmc_link_weights_${A2_TAG}.pth"

# ─── Pre-flight ────────────────────────────────────────────────────
if [[ ! -f "${A1_W}" ]]; then
  echo "ERROR: A1 baseline weight not found: ${A1_W}" >&2
  exit 1
fi

# ─── Train A2 ──────────────────────────────────────────────────────
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  echo "============================================================"
  echo "Training A2: --ego ${A2_EGO} → ${A2_W}"
  echo "============================================================"
  "${PY}" -m gmc_link.train \
    --split v1 \
    --ego "${A2_EGO}" \
    --epochs 100 \
    --lr 1e-3 \
    --batch-size 128 \
    --save-path "${A2_W}"
else
  echo "SKIP_TRAIN=1 set; skipping A2 training."
  if [[ ! -f "${A2_W}" ]]; then
    echo "ERROR: A2 weight not found: ${A2_W}" >&2
    exit 2
  fi
fi

# ─── Eval both on held-out seqs ────────────────────────────────────
SEQS=(0005 0011 0013)
declare -a WEIGHTS=("${A1_TAG}=${A1_W}" "${A2_TAG}=${A2_W}")

for seq in "${SEQS[@]}"; do
  echo "============================================================"
  echo "Sequence ${seq}"
  echo "============================================================"
  for entry in "${WEIGHTS[@]}"; do
    tag="${entry%%=*}"
    path="${entry#*=}"
    echo "--- ${seq} / ${tag} ---"
    "${PY}" "${DIAG}" --weights "${path}" --seq "${seq}"
    src="${RESULTS_DIR}/layer3_gt_cosine_${seq}.npz"
    dst="${OUT}/layer3_${seq}_${tag}.npz"
    if [[ ! -f "${src}" ]]; then
      echo "ERROR: expected ${src}, not produced" >&2
      exit 3
    fi
    mv "${src}" "${dst}"
    src_png="${RESULTS_DIR}/layer3_gt_cosine_${seq}.png"
    [[ -f "${src_png}" ]] && mv "${src_png}" "${OUT}/layer3_${seq}_${tag}.png"
  done
done

# ─── Aggregate ─────────────────────────────────────────────────────
AGG_WEIGHTS=()
for entry in "${WEIGHTS[@]}"; do
  AGG_WEIGHTS+=("${entry}")
done

# Carry legacy seq-0011 AUC for A1 so the comparison MD shows the reference.
LEGACY=("${A1_TAG}=0.779")

echo "============================================================"
echo "Aggregating Stage A results → ${OUT}"
echo "============================================================"
"${PY}" diagnostics/aggregate_multiseq.py \
  --results-dir "${OUT}" \
  --output-dir "${OUT}" \
  --weights "${AGG_WEIGHTS[@]}" \
  --seqs "${SEQS[@]}" \
  --legacy-seq-0011 "${LEGACY[@]}"

echo "Done. See ${OUT}/layer3_multiseq_comparison.md"
