#!/bin/bash
# Path B Phase-2 n=3 two-setting ablation (gated behind monocular reproduce).
#   A = lidar_oxts   : LiDAR depth + GT oxts ego-ΔZ   (full Path B)
#   B = lidar_cohort : LiDAR depth + stationary-cohort ego  (ego ABLATION)
#   + monocular control (surviving _sharedweight_depth_seed*_rawcos caches, no rebuild)
# A vs B isolates whether the ego swap is the lever (attribution). Single variable
# per setting; recipe/arch/eval all locked to the 13D-ship iKUN recipe.
# ~2-3h GPU (6 trains x 2m + 18 GMC caches x ~6m + fuses). Per-class HOTA per run.
set -e
cd /home/seanachan/GMC-Link
SEQS="0005 0011 0013"
RECIPE="--alpha 1.0 --gmc_scale 0.9 --thr 0.17 --alpha_appear 1.0 --gmc_scale_appear 0.30 --thr_appear 0.10"

run_setting () {   # $1=tag  $2=depth_source
  tag=$1; src=$2
  for s in 0 1 2; do
    w=gmc_link_weights_v1train_sw_${tag}_seed${s}.pth
    echo ">>> TRAIN $tag seed$s ($src)"
    python -m gmc_link.train --split v1 --stage 1 --architecture shared_weight --seed $s \
      --use-depth --depth-source $src --depth-cache-dir gmc_link/depth_cache_lidar \
      --save-path $w
    suf=_sw_${tag}_seed${s}_rawcos
    for seq in $SEQS; do
      rm -f gmc_link/gmc_scores_v1_${seq}${suf}_cache.json   # avoid skip-if-exists stale read
      GMC_WEIGHTS=$w GMC_SUFFIX=$suf GMC_RAW_COS=1 GMC_DEPTH_ARCH=ikun \
        python run_build_gmc_cache.py $seq >/dev/null
    done
    printf "RESULT %s seed%s  " "$tag" "$s"
    GMC_SUFFIX=$suf GMC_RAW_COS=1 python run_ikun_linear_additive.py $RECIPE 2>&1 | grep -oE 'pooled=.*STATIC=[0-9.]+'
  done
}

echo "===== Setting A: lidar_oxts (full Path B) ====="
run_setting lidaroxts lidar_oxts
echo "===== Setting B: lidar_cohort (ego ablation) ====="
run_setting lidarcohort lidar_cohort

echo "===== Monocular control (surviving caches, no rebuild) ====="
for s in 0 1 2; do
  printf "RESULT monocular seed%s  " "$s"
  GMC_SUFFIX=_sharedweight_depth_seed${s}_rawcos GMC_RAW_COS=1 \
    python run_ikun_linear_additive.py $RECIPE 2>&1 | grep -oE 'pooled=.*STATIC=[0-9.]+'
done
echo "===== DONE. Reference 13D ship seed0 = pooled 44.561 / STATIC 43.240 / MOVING 28.885 ====="
