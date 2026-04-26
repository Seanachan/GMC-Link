# Exp 37 — Systematic Ego-Motion Compensation Study (Design Spec)

**Date:** 2026-04-22
**Author:** Seanachan (with Claude Opus 4.7)
**Experiment ID:** Exp 37 (series)
**Branch target:** `exp/ego-motion-systematic` off `exp/hn-mining`
**Related prior work:**
- Exp 34 HN-InfoNCE (commit `f55713f`) — proved 0.779 AUC ceiling is representation-bound
- Exp 35 FlexHook-adjacent (specs `2026-04-19`, `2026-04-16`) — transformer arm (architecture lever falsified on 13D)
- Exp 36A–E (memory `project_exp36_series_negative`) — feature/arch/language levers all ≤ 0.779
- Aligner α=1.0 sweep (memory `project_aligner_sweep_at_alpha1_restores_auc`) — v1v2train = HOTA 35.848
**Reference papers:**
- Kashyap et al., *Sparse Representations for Object- and Ego-Motion Estimations in Dynamic Scenes* (IEEE TNNLS 2021) — **SparseMFE**
- EMAP: *Ego-Motion Aware Kalman* (arXiv 2404.03110) — ego-velocity structural decoupling
- Xue et al., *Refer-ASV* (J. Imaging 2026, 12, 145) — **RAMOT** baseline for portability test

---

## 0. One-Sentence Pitch

Systematically factor the GMC-Link ego-motion pipeline along three independent axes (ego source, motion feature, structural conditioning) via a greedy staged ablation, then verify the resulting ego-compensation signal **transfers** by plugging it into a second RMOT tracker (TempRMOT/RAMOT) besides iKUN.

## 1. Motivation & Strategic Reframe

Exp 34–36 exhausted the **loss**, **feature-dimension**, **architecture**, and **language-encoder** levers at the Layer-3 aligner. The 0.779 V1 micro-AUC ceiling is representation-bound relative to the *downstream aligner supervision signal*, but the **upstream ego-motion pipeline** (Stage 1 ORB homography + Stage 2 residual velocity) has **never been systematically ablated** since Exp 5.

Crucially, ego-motion compensation is GMC-Link's **principal differentiator** vs FlexHook (memory `project_flexhook_no_ego_compensation`). If we re-frame the project as an **ego-compensation service** that any spatially-ignorant RMOT tracker can consume — rather than as a scalar-match plug-in — then upstream Stage 1/2 quality directly determines external value.

This spec systematically tests: (a) whether SOTA ego-motion estimators (SparseMFE's 6DoF autoencoder) beat our 2D-planar ORB+RANSAC; (b) whether per-pixel Object-Motion-Field (OMF) features outperform hand-crafted 13D; (c) whether structural ego-velocity conditioning (EMAP-style) adds orthogonal signal; and (d) whether any gains **transfer** to a non-iKUN tracker.

## 2. Hypotheses & Falsification Criteria

### H_A — Ego source
**H₀:** SparseMFE 6DoF ego ≉ ORB+RANSAC 2D planar for Refer-KITTI V1 held-out AUC/HOTA.
**H₁:** SparseMFE raises V1 micro-AUC by ≥ 0.010 **or** lowers KITTI odometry ATE (seq 09/10) by ≥ 3× vs ORB homography.
**Decision band:** ΔAUC ≥ +0.010 positive · [−0.005, +0.010) inconclusive · < −0.005 negative.

### H_B — Motion feature
**H₀:** OMF per-bbox stats do not improve AUC over 13D hand-crafted.
**H₁:** `13D + OMF stats` (mean/std/max per scale → ≈28D) raises V1 micro-AUC by ≥ 0.010.
**Decision band:** same as H_A.

### H_C — Structural conditioning
**H₀:** Concatenating ego-velocity (2D) to aligner input adds nothing.
**H₁:** EMAP-style concat raises AUC by ≥ 0.005 **or** reduces per-seq-0011 HOTA variance materially.

### H_D — Portability (the headline claim)
**H₀:** The ego-compensation signal is entangled with iKUN's logit distribution; swapping tracker kills the delta.
**H₁:** `gmc_cos` + α-fusion on top of TempRMOT (or RAMOT if weights release) yields a HOTA delta of ≥ +1.0 vs that tracker's bare baseline, **regardless** of our iKUN fusion result.
**Importance:** H_D is the paper-worthy plug-in claim. H_A/B/C can all be negative and H_D alone, with the existing ORB pipeline, is still publishable.

## 3. Scope — What Changes, What Doesn't

### Does NOT change
- `gmc_link/losses.py` — InfoNCE + FNM unchanged.
- `gmc_link/fusion_head.py` — fusion stays; α=1.0 locked from prior sweep.
- `run_hota_eval_v1.py` — HOTA driver unchanged.
- `run_hota_eval_v1_logodds.py`, `run_hota_eval_v2.py` — untouched.
- V1 train split, V1 held-out {0005, 0011, 0013}.
- Cache format for existing 13D snapshots (Stage A/B/C write *new* cache keys).

### Does change
- **New:** `gmc_link/ego/sparsemfe_ego.py` — wrapper around upstream SparseMFE (submodule under `third_party/SparseMFE/`); exposes `estimate_ego(img_prev, img_curr) -> (H_6dof, omf_field)`.
- **New:** `gmc_link/ego/ego_router.py` — pluggable ego-source interface consumed by `GMCLinkManager`. Registers `{"orb", "sparsemfe"}` backends.
- `gmc_link/manager.py` — read ego source from config; residual-velocity math unchanged.
- **New:** `gmc_link/features/omf_stats.py` — per-bbox OMF summary (mean/std/max dx/dy over bbox pixels per scale).
- `gmc_link/dataset.py` — optional extra feature channels; new cache keys `cache_v1_orb_13d`, `cache_v1_sparsemfe_13d`, `cache_v1_orb_omf28d`, `cache_v1_sparsemfe_omf28d`.
- `gmc_link/alignment.py` — input dim becomes configurable (13 or 28 or +2 for EMAP concat).
- `gmc_link/train.py` — add flags `--ego {orb,sparsemfe}`, `--features {13d,omf28d}`, `--ego-cond {off,concat}`.
- **New drivers:**
  - `run_exp37_stage_a.sh` — ego-source A/B.
  - `run_exp37_stage_b.sh` — feature lever, uses winner(A).
  - `run_exp37_stage_c.sh` — structural conditioning, uses winner(B).
  - `run_exp37_stage_d.sh` — portability: feed ego-comp tracks into TempRMOT.

### Explicit non-goals
- We do **NOT** fine-tune SparseMFE. Frozen weights only.
- We do **NOT** extend to V2 yet — V1 systematic first, V2 replication only if H_D positive.
- We do **NOT** re-evaluate Exp 35 transformer aligner; `--arch mlp` locked for Exp 37.
- We do **NOT** touch fusion-head MLP vs additive debate — α=1.0 additive locked.

## 4. Experiment Matrix & Staged Execution

### Factors

| Axis | Symbol | Values |
|------|--------|--------|
| Ego source | **E** | ORB (current) / SparseMFE 6DoF |
| Motion features | **F** | 13D hand-crafted / 13D+OMF (≈28D) |
| Structural conditioning | **C** | off / EMAP ego-velocity concat (+2D) |

Full cross = 2 × 2 × 2 = 8 cells. Greedy staged ablation keeps 4 trainings + 1 portability run.

### Stage A — Ego source (F=13D, C=off)
| Cell | E | Prior |
|------|---|-------|
| A1 | ORB | = `stage1` (have cache) |
| A2 | SparseMFE | NEW |

**Deliverable:** winner(A) = arg max AUC.

### Stage B — Motion features (E=winner(A), C=off)
| Cell | F | Prior |
|------|---|-------|
| B1 | 13D | = winner(A) |
| B2 | 13D+OMF | NEW |

**Deliverable:** winner(B) = arg max AUC.

### Stage C — Structural conditioning (E,F=winner(B))
| Cell | C | Prior |
|------|---|-------|
| C1 | off | = winner(B) |
| C2 | EMAP concat | NEW |

**Deliverable:** winner(C) = final Exp 37 aligner.

### Stage D — Portability
Feed `winner(C)`'s ego-compensated cosine score into TempRMOT (primary — weights available) and RAMOT (secondary — pending code release).

### Total new trainings
4 (A2, B2, C2 + D fusion refit).

## 5. Metrics

| Metric | Tool | Stages | Proves |
|--------|------|--------|--------|
| V1 micro AUC (multi-seq held-out) | `gmc_link.eval.multiseq_eval` | A, B, C | representation quality |
| V1 macro AUC + per-seq | same | A, B, C | robustness to seq 0011 |
| HOTA @ α=1.0 on {0005,0011,0013} | `run_hota_eval_v1.py` | A, B, C, D | downstream |
| DetA / AssA / IDF1 | TrackEval | A, B, C, D | decomposition |
| KITTI odometry ATE (seq 09, 10) | new `eval/ego_ate.py` | A | ego-motion accuracy vs GT pose |
| ΔHOTA TempRMOT + GMC vs TempRMOT alone | new `run_exp37_stage_d.sh` | D | **plug-in portability** |

## 6. Gatekeepers & Risk

| Gate | Required before | Action if fails |
|------|----------------|-----------------|
| SparseMFE code/weights public? | Stage A | Fallback: implement closed-form 6DoF from `cv2.decomposeHomographyMat` ~~or~~ use Zhou et al. SfM-learner as A2 instead. |
| PWC-Net pretrained available | Stage A | Fallback: use RAFT (already in refer-KITTI tooling if cached). |
| KITTI odometry GT poses on disk | Stage A metric | Symlink from `/home/seanachan/data/Dataset/kitti_odometry/poses/` — check first; if missing, report ATE only qualitatively via scale-consistency. |
| TempRMOT code/weights | Stage D | Fallback: RAMOT (when released). If neither, port iKUN-2 as D. H_D is required for publication-grade claim. |
| GPU memory for OMF | Stage B | OMF is per-pixel; pool to 32×32 before per-bbox stats to keep RAM bounded. |
| Aligner retraining stability | All | Use existing `gmc_link/train.py` seed = 42; 3 seeds if ΔAUC < 0.010. |

## 7. Per-Stage Specs

### Stage A — Ego source

**A1 (already done).** Read `stage1.pth` aligner + existing 13D cache. Run multi-seq eval. Record baseline.

**A2.** Build SparseMFE submodule:
1. Clone upstream repo under `third_party/SparseMFE/` (pinned commit).
2. Load frozen weights.
3. Per-frame, per-seq: run SparseMFE on `(frame_t, frame_{t+1})` → `(ego_rot, ego_trans, omf_pixel_field)`. Cache OMF to `cache/sparsemfe/{seq}/{frame_id}.npz`.
4. Derive 2D-planar-projected homography from 6DoF for backwards-compat with `GMCLinkManager` centroid warping.
5. Build `cache_v1_sparsemfe_13d` by re-running dataset pipeline with SparseMFE as ego router.
6. Train aligner 100 epochs, seed 42. Save `sparsemfe_13d.pth`.
7. Multi-seq AUC eval. Compare to stage1.

**A2 decision:** AUC ≥ stage1 + 0.010 ⇒ pick SparseMFE for Stage B; inconclusive ⇒ pick whichever has better HOTA; negative ⇒ keep ORB.

### Stage B — Motion features

Holding E=winner(A):
1. Per-frame, per-bbox: extract OMF pixels inside bbox, compute `{mean_dx, mean_dy, std_dx, std_dy, max_|d|}` per scale in `FRAME_GAPS=[2,5,10]` → 5 × 3 = 15 features.
2. Concat with existing 13D → 28D feature vector.
3. Retrain aligner with input dim 28. Same loss, same hyperparams.
4. Evaluate.

Edge case: if E=ORB (no OMF available), Stage B uses **dense optical flow from PWC-Net** under ORB ego compensation to produce the OMF-equivalent. This decouples feature source from ego source.

### Stage C — Structural conditioning

Holding E, F = winner(B):
1. For each training sample: compute `(ego_vx, ego_vy) = ego_trans_xy / dt` averaged over FRAME_GAPS window.
2. Concat 2D ego-velocity to aligner input → dim +2.
3. Retrain, evaluate.

Structurally complements additive fusion at Layer 4 because the conditioning is at representation level, not score level.

### Stage D — Portability

Holding entire winner(C):
1. Download or build TempRMOT on V1 held-out; verify bare HOTA.
2. Replace TempRMOT's association cost term `c_app` with `c_app + β · gmc_cos`, where `gmc_cos` comes from winner(C) aligner.
3. Sweep β ∈ {0.5, 1.0, 2.0}; report best.
4. Compare ΔHOTA vs TempRMOT-alone.
5. If available, repeat on RAMOT.

**Publication frame:** If H_D positive on ≥1 non-iKUN tracker, GMC-Link is a genuine plug-in ego-compensation service.

## 8. Deliverables

- This spec (this file).
- Implementation plan (next step, via `superpowers:writing-plans`).
- One branch `exp/ego-motion-systematic` with commits per stage.
- `diagnostics/results/exp37/` directory with per-stage AUC/HOTA tables.
- Memory entries per stage result (`project_exp37_stage{a,b,c,d}_{positive,negative}.md`).
- Final retrospective `project_exp37_summary.md` marking which hypotheses landed.

## 9. Compute Estimate

| Stage | Wall-clock (GPU, bare minimum) |
|-------|--------------------------------|
| A setup + caching | ~6h (SparseMFE inference over V1 train + held-out) |
| A training | 2h |
| B setup + caching | ~2h (OMF stats only) |
| B training | 2h |
| C training | 2h |
| D TempRMOT build + eval | ~8h |
| HOTA evaluation (4 runs) | ~2h |
| **Total** | **~24 GPU-hours** over 3–4 wall-clock days |

## 10. Kill Switches

- If Stage A SparseMFE install blocked for > 1 day, pivot to **Stage A-alt**: use `cv2.recoverPose` on ORB correspondences as a 6DoF-without-autoencoder baseline. Smaller expected gain, zero install risk.
- If Stages A, B, C all negative (ego-motion not the lever either), Exp 37 conclusion is: **pipeline-bound at iKUN logit distribution**, and the next experiment is H_D *alone* on top of ORB + 13D baseline — a pure portability paper.
- If Stage D positive but A/B/C all flat: the portability win is the headline. Still publishable.

## 11. Open Questions

1. Does SparseMFE need re-training on KITTI odometry? (Paper reports KITTI results, so inference-only suffices.)
2. For the OMF-without-SparseMFE path: which dense flow backbone — PWC-Net, RAFT, or FlowFormer? Pick RAFT (shortest dependency chain), sensitivity test if time permits.
3. β sweep range in Stage D: mirror α=1.0 from iKUN fusion, or refit from scratch? Refit is safer — different tracker distributions.
4. For RAMOT portability: the paper's code release is contingent on "completion of open-source review." Watch `github.com/` for CASIA-IA repo. If not released within 2 weeks of Stage D start, skip.
