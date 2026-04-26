#!/usr/bin/env bash
# Multi-sequence Layer 3 evaluation driver.
#
# For each (seq, weight), calls diag_gt_cosine_distributions.py, then
# relocates the output .npz into diagnostics/results/multiseq/ with a
# namespaced filename. Finally invokes the aggregator across all results.
#
# Spec: docs/superpowers/specs/2026-04-18-multi-sequence-eval-design.md
# Plan: docs/superpowers/plans/2026-04-18-multi-sequence-eval.md
#
# Usage: bash run_multiseq_eval.sh
#
# Loop order: seq-outer / weight-inner, to preserve OS page-cache warmth on
# each sequence's frames across the 11 weight invocations.

set -euo pipefail

PY=~/miniconda/envs/RMOT/bin/python
DIAG=diagnostics/diag_gt_cosine_distributions.py
RESULTS_DIR=diagnostics/results
MULTISEQ_DIR="${RESULTS_DIR}/multiseq"
mkdir -p "${MULTISEQ_DIR}"

SEQS=(0005 0011 0013)

# tag => weight path. Keep tags in sync with filename stems (strip
# gmc_link_weights_ prefix, .pth suffix) so aggregator can derive them.
declare -a WEIGHTS=(
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

# Pre-flight: every weight must exist on disk before we start.
for entry in "${WEIGHTS[@]}"; do
  path="${entry#*=}"
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: weight file not found: ${path}" >&2
    exit 1
  fi
done

# Main loop: seq-outer, weight-inner.
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
    dst="${MULTISEQ_DIR}/layer3_${seq}_${tag}.npz"
    if [[ ! -f "${src}" ]]; then
      echo "ERROR: expected ${src} but it was not produced" >&2
      exit 2
    fi
    mv "${src}" "${dst}"
    # Relocate the plot too so the legacy single-seq PNG in results/ stays
    # as a persistent reference from the original experiment rather than
    # being overwritten 11 times.
    src_png="${RESULTS_DIR}/layer3_gt_cosine_${seq}.png"
    if [[ -f "${src_png}" ]]; then
      mv "${src_png}" "${MULTISEQ_DIR}/layer3_${seq}_${tag}.png"
    fi
  done
done

# Build the weights argument for the aggregator.
AGG_WEIGHTS=()
for entry in "${WEIGHTS[@]}"; do
  AGG_WEIGHTS+=("${entry}")
done

# Pass through legacy seq-0011 AUCs so per-weight MDs carry them. These are
# the numbers reported in RESEARCH_NOTES for Exp 30-32; other ablations
# without a known legacy number are omitted (the MD will render "—").
LEGACY=(
  "v1train_stage1=0.779"
  "v1train_temporal=0.770"
)

echo "============================================================"
echo "Aggregating 33 result files → multiseq reports"
echo "============================================================"
"${PY}" diagnostics/aggregate_multiseq.py \
  --results-dir "${MULTISEQ_DIR}" \
  --output-dir "${MULTISEQ_DIR}" \
  --weights "${AGG_WEIGHTS[@]}" \
  --seqs "${SEQS[@]}" \
  --legacy-seq-0011 "${LEGACY[@]}"

echo "Done. See ${MULTISEQ_DIR}/layer3_multiseq_comparison.md"
