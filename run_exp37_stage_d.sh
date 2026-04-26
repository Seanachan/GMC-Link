#!/usr/bin/env bash
# Exp 37 Stage D — TempRMOT portability via env-var gated fusion.
#
# Arms:
#   bare          — EXP37_MODE=bare (skip GMC-Link fusion entirely; TempRMOT only)
#   alpha_beta05  — EXP37_MODE=alpha, EXP37_BETA=0.5
#   alpha_beta10  — EXP37_MODE=alpha, EXP37_BETA=1.0
#   alpha_beta20  — EXP37_MODE=alpha, EXP37_BETA=2.0
#
# For each arm: run inference.py → TrackEval HOTA → record summary row.
#
# Spec: §§2 H_D, 5, 10 (Stage D)
# Plan: Task 15, 16

set -euo pipefail

TEMPRMOT=/home/seanachan/TempRMOT
PY=/home/seanachan/miniconda/envs/tempRMOT/bin/python
GMC_LINK=/home/seanachan/GMC-Link
OUT="${GMC_LINK}/diagnostics/results/exp37/stage_d"
CHECKPOINT_EPOCH=50
GT_FOLDER="${TEMPRMOT}/datasets/refer-kitti/gt_template"
SEQMAP="${TEMPRMOT}/datasets/data_path/seqmap-v1.txt"

mkdir -p "${OUT}"

INFER_ARGS=(
  --meta_arch temp_rmot
  --dataset_file e2e_rmot
  --epoch 200
  --with_box_refine
  --lr_drop 100
  --lr 2e-4
  --lr_backbone 2e-5
  --batch_size 1
  --sample_mode random_interval
  --sample_interval 1
  --sampler_steps 50 90 150
  --sampler_lengths 2 3 4 5
  --update_query_pos
  --merger_dropout 0
  --dropout 0
  --random_drop 0.1
  --fp_ratio 0.3
  --query_interaction_layer QIM
  --extra_track_attn
  --resume exps/default_rk/checkpoint0050.pth
  --rmot_path "${TEMPRMOT}/datasets/refer-kitti"
  --hist_len 8
)

run_arm () {
  local MODE=$1
  local BETA=$2
  local TAG=$3
  local ARM_OUT="exps/exp37_stage_d_${TAG}"

  echo "============================================================"
  echo "Stage D arm ${TAG}  MODE=${MODE}  BETA=${BETA}"
  echo "============================================================"
  cd "${TEMPRMOT}"

  EXP37_MODE="${MODE}" EXP37_BETA="${BETA}" \
    "${PY}" inference.py "${INFER_ARGS[@]}" \
    --output_dir "${ARM_OUT}" 2>&1 | tee "${OUT}/infer_${TAG}.log" | tail -20

  local TRACKER_DIR="${TEMPRMOT}/${ARM_OUT}/results_epoch${CHECKPOINT_EPOCH}"

  echo "--- HOTA eval for ${TAG} ---"
  cd "${TEMPRMOT}/TrackEval/scripts"
  "${PY}" run_mot_challenge.py \
    --METRICS HOTA \
    --SEQMAP_FILE "${SEQMAP}" \
    --SKIP_SPLIT_FOL True \
    --GT_FOLDER "${GT_FOLDER}/" \
    --TRACKERS_FOLDER "${TRACKER_DIR}" \
    --TRACKERS_TO_EVAL "${TRACKER_DIR}" \
    --USE_PARALLEL True \
    --NUM_PARALLEL_CORES 2 \
    --GT_LOC_FORMAT "{gt_folder}/{video_id}/{expression_id}/gt.txt" \
    --PLOT_CURVES False 2>&1 | tee "${OUT}/hota_${TAG}.log" | tail -40

  cd "${GMC_LINK}"
}

ARM="${1:-all}"
case "${ARM}" in
  bare)          run_arm bare  1.0  bare ;;
  beta05)        run_arm alpha 0.5  alpha_beta05 ;;
  beta10)        run_arm alpha 1.0  alpha_beta10 ;;
  beta20)        run_arm alpha 2.0  alpha_beta20 ;;
  all)
    run_arm bare  1.0  bare
    run_arm alpha 0.5  alpha_beta05
    run_arm alpha 1.0  alpha_beta10
    run_arm alpha 2.0  alpha_beta20
    ;;
  *) echo "Unknown arm: ${ARM}"; exit 1 ;;
esac

echo "Done — artifacts in ${OUT}"
