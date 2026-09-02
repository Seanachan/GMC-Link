# Pre-registration: TransRMOT as the third host architecture (A46)

Committed BEFORE any new inference or fusion data is generated. Date 2026-09-02.

## Motivation

The paper's central observation — the module's moving-class gain is inversely related to the host's
native motion understanding — currently rests on two host architectures (iKUN +9.44, FlexHook V1 +0.43 /
V2 +0.69 under A43/A44). Two points do not make a trend. TransRMOT (CVPR 2023) is the third architecture:
query-based one-stage, official checkpoint on disk (`~/RMOT/exps/default/checkpoint0099.pth`, the v1.0
release file), and a documented precedent for test-time plug-ins on this host class (C²RMOT). Our own
Exp 18 (2026-04) cascaded an early GMC variant onto TransRMOT with `min(vit, gmc)` fusion (+4.55 HOTA) —
that result is superseded (different fusion form, pre-A42 protocol, old GT) and is NOT comparable;
everything here is measured fresh.

## Configuration

- **Host**: TransRMOT, official checkpoint0099 (no retraining, no fine-tuning). Inference re-run only to
  dump raw per-box referring scores (`scores.txt` per expression: frame, id, box, referring score, det
  score) alongside the untouched thresholded `predict.txt` — the released code writes no scores.
- **Protocol**: official 150-expression seqmap (`seqmaps/refer_kitti_v1_test_official_150.txt`);
  GT = `gt_template/` (TransRMOT frame convention); classes = `gmc_link.moving_kw.classify` (A43);
  TrackEval invocation = the `te()` recipe used since A42.
- **Native disclosure** (A31/A42 style): the paper's 46.56 predates the community frame correction; the
  official repo reports 38.06 corrected. We reproduce the native under our protocol and compare gains
  against OUR reproduced native; both published values are disclosed next to it.
- **Fusion**: `s_final = s_host + α(expr)·s_gmc`; the admission gate is the host's native referring
  threshold exactly as invoked in `inference.py` (`filter_dt_by_ref_scores(dt_instances, 0.5)`,
  inference.py:575 — so the gate is **0.5**), so α = 0 reproduces the native output bit-for-bit — the same gate-freezing rule as iKUN/FlexHook.
- **Caches**: road chain, `GMC_GROUND_MODE=road GMC_MOTION_EMA=0`, weights
  `gmc_link_weights_v1train_sw12d_groad_seed{0,1,2}.pth`, warm-11 filter, raw cosine — the locked Option-B
  module, unchanged.
- **α selection**: LOSO over folds hold0005 / hold0011 / hold0013; single-α grid
  {0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.5, 2.0} (TransRMOT referring scores are sigmoid
  probabilities; if the observed native score spread makes this grid clearly mis-scaled, the revised grid
  is committed here BEFORE fold runs) and the two-α grid (α_mot × α_app, router = A43 classifier);
  selection rule identical to `run_two_alpha_sweep.py` (per-fold argmax on pooled HOTA, censor at grid
  edge, median over folds, off-grid median → lower grid point). Fold runs record pooled only.
- **Seeds**: 0/1/2 for full-test (n=3, mean ± std), matching the other hosts.

## Pre-registered hypotheses

- **H1 (curve)**: TransRMOT's native moving-class HOTA is low (spatially grounded, motion-blind referring
  head), and the module's MOVING gain at the LOSO α is large relative to FlexHook's — the third point
  falls on the host-deficit curve.
- **H2 (falsifier)**: TransRMOT's track-query propagation acts as temporal memory, and the module
  regresses pooled HOTA as on TempRMOT (−3.8 to −5.4). If H2 holds, the result is reported as the
  boundary of applicability, not suppressed.

## Integrity gates (halt conditions)

1. **Native reproduction**: TrackEval on the re-run thresholded outputs must match the existing
   `~/RMOT/exps/default_baseline/results_epoch99` evaluation up to deltas fully explained by (a) the
   official-150 list vs the old enumeration and (b) the GT template. Unexplained drift halts the campaign.
2. **α = 0 identity**: the fusion script at α = 0 must reproduce the native predict.txt tree bit-for-bit.
3. **Class counts**: MOVING/STATIC/APPEARANCE seqmap sizes = 21/12/117 (A43).
4. **LOSO hygiene**: fold outputs live in `_seqs`-suffixed dirs, never overwriting full-test results;
   no full-test number is read before the fold selection is fixed.
5. Warm-11 abstention and road-chain settings are byte-identical to the shipped caches' settings
   (verified via the cache builder's recorded config).

## Outputs

`~/RMOT/exps/off150_scores/` (host outputs, stays outside git), cache files
`gmc_link/gmc_scores_transrmot_{seq}_sw12d_groad_seed{N}_warm11_cache.json`, run trees
`hota_eval_transrmot_linear_additive_*`, summary `results/transrmot_host/transrmot_host.json`,
record RESEARCH_NOTES **A46**.
