# Stack: Path B (lidar_oxts 17D) + CLIP-L/14 appearance rerank — iKUN n=3

Same harness, V1 3-seq pooled HOTA. Rerank = CLIP-L/14 color subset (18 exprs), tau=0.216
(tuned seed0, applied all seeds). Path B = lidar_oxts GMC caches as fusion base.

| seed | ship | Path B | rerank | stack |
|------|--------|--------|--------|--------|
| 0    | 44.561 | 45.020 | 44.824 | 45.288 |
| 1    | 44.513 | 44.952 | 44.772 | 45.206 |
| 2    | 44.667 | 44.785 | 44.932 | 45.054 |
| mean | 44.580 | 44.919 | 44.843 | **45.183** |

Δ vs ship (n=3 mean): Path B +0.339 | rerank +0.263 | stack **+0.603** (sum-of-parts +0.602 → additive)
Per-seed stack Δ: [+0.727, +0.693, +0.387], 3/3 POS, paired t≈5.6, p≈0.015, std≈0.12.
stack 45.183 = +0.619 vs paper 44.564. Best iKUN of campaign.

Repro:
  ship:   GMC_SUFFIX=_sharedweight_seed{s}_rawcos GMC_RAW_COS=1 python run_ikun_linear_additive.py --alpha 1.0 --gmc_scale 0.9 --thr 0.17 --alpha_appear 1.0 --gmc_scale_appear 0.30 --thr_appear 0.10
  Path B: GMC_SUFFIX=_sw_lidaroxts_seed{s}_rawcos  (same args)
  rerank: GMC_SUFFIX=_sharedweight_seed{s}_rawcos GMC_RAW_COS=1 python -m diagnostics.appearance_rerank.step1_probe --signal clipL14 --target color --taus 0.216
  stack:  GMC_SUFFIX=_sw_lidaroxts_seed{s}_rawcos  (same rerank cmd)

Caveats: tau seed0-tuned (not held-out expr split); rerank cross-host FH V2 untested; Path B not merged.
