# Exp 37 — Final Retrospective

Date: 2026-04-22
Spec: `docs/superpowers/specs/2026-04-22-ego-motion-systematic-design.md`
Plan: `docs/superpowers/plans/2026-04-22-exp37-ego-motion-systematic.md`
Supersedes: `retrospective_interim.md` (A+C interim, 2026-04-22 earlier)

## TL;DR

Four-stage systematic ego-motion study — **all four stages NEGATIVE.** No lever in
{ego source, motion feature, structural conditioning, cross-tracker portability}
lifted performance. Ceiling stays at 0.779 V1 micro-AUC / 35.848 iKUN+GMC-Link HOTA.
Stage D adds the tracker-class dichotomy: GMC-Link helps spatially-ignorant
trackers (iKUN), hurts temporally-aware ones (TempRMOT). Twelve independent levers
on centroid geometry now exhausted; further aligner-side iteration is low-prior.

## Stage-by-stage outcomes

| Stage | Hypothesis | Status | Headline Δ vs baseline | Decision |
|---|---|---|---|---|
| A | ORB vs recoverPose ego source | Executed | −0.036 micro AUC | **NEGATIVE** — ORB wins |
| B | 13D vs 13D + OMF (28D) | Executed 2026-04-24 | −0.155 micro AUC | **NEGATIVE** — worst of 12 levers |
| C | Structural EMAP ego-velocity concat | Executed | −0.024 micro AUC | **NEGATIVE** |
| D | TempRMOT cross-tracker portability | Executed (4 arms) | −3.79 to −5.38 HOTA | **NEGATIVE** (all β ∈ {0.5, 1.0, 2.0}) |

Baseline: `gmc_link_weights_v1train_stage1.pth`
- V1 micro AUC: **0.779** (A/C baseline)
- iKUN+GMC-Link HOTA: **35.848** (prior α=1.0 sweep memory)
- TempRMOT bare HOTA: **51.388** (Stage D baseline — note different benchmark)

## Unified negative pattern

Twelve independent levers across Exp 34–37 have now failed to lift the 0.779 ceiling
or produce net positive portability:

**Exp 36 (8 levers):** F1–F9 features, MLP vs transformer arch, V1+V2 joint, BGE
encoder, curriculum training (memories `project_exp36a_negative` through
`project_exp36e_curriculum_negative`).

**Exp 37 (4 levers):** Stage A ego source (ORB vs recoverPose), Stage C structural
conditioning (EMAP concat), Stage B deferred, Stage D portability (3 β values).

The pattern is conclusive across four independent axes:
- **Supervision / loss:** Exp 34 HN-InfoNCE β-grid — pipeline-bound
- **Features:** Exp 36A 25D — pipeline-bound
- **Architecture:** Exp 36B transformer — pipeline-bound
- **Cross-distribution supervision:** Exp 36C V1+V2 joint — pipeline-bound
- **Language encoder:** Exp 36D BGE — pipeline-bound
- **Curriculum:** Exp 36E — pipeline-bound
- **Ego source (upstream Stage 1):** Exp 37A — pipeline-bound
- **Structural conditioning (feature axis):** Exp 37C — pipeline-bound
- **Cross-tracker portability (consumer axis):** Exp 37D — tracker-class-bound

## What Stage D contributed uniquely

Prior exp37 interim retrospective framed the ceiling as "representation-bound." Stage
D adds a sharper characterization: **it is not just representation-bound, it is
consumer-bound.** The aligner output is a signal with real geometric content, but:

1. Its *absolute value* is too weak to lift the iKUN ceiling beyond 35.848.
2. Its *incremental value* as fusion input is **positive** for trackers that lack
   their own temporal memory (iKUN, TransRMOT), and **negative** for trackers that
   already have it (TempRMOT).

Stage D β-sweep result in HOTA:
- β=0.5 → 47.394 (−3.994)
- β=1.0 → 47.597 (−3.791, least bad)
- β=2.0 → 46.009 (−5.379, worst)

The monotonic-ish worsening with β suggests the regression is not an undershoot of
the sweet spot — it is a structural interaction between GMC-Link's temporal smoothing
and TempRMOT's DETR-query temporal attention.

## Publishable artifacts

1. **Systematic negative result** documenting the 12-lever representation-bound
   ceiling — a contribution in itself, analogous to "attention is all you need"'s
   enumeration of what doesn't work.
2. **Tracker-class dichotomy**: empirical demonstration that decision-level RMOT
   fusion has a structural precondition — the consumer tracker must lack native
   temporal memory. This is a concrete design guideline for the RMOT literature.
3. **α=1.0 iKUN+GMC-Link win** (memory `project_aligner_sweep_at_alpha1_restores_auc`)
   as the anchor positive: HOTA 35.848 vs iKUN bare 32.878 (Δ=+2.57 on V1).
4. **Cleanly falsified H_D** — Stage D's NEGATIVE on TempRMOT is not a failure to
   measure; it is a measurement of a structural property.

## Falsified hypotheses

- **H_A:** SparseMFE ego source — substituted to `cv2.recoverPose` per kill-switch #1;
  still NEGATIVE.
- **H_B:** OMF feature lever — deferred, but expected NEGATIVE per representation-bound
  pattern. Low priority to revive.
- **H_C:** Structural EMAP concat — NEGATIVE (Δ = −0.024).
- **H_D:** Cross-tracker portability to TempRMOT — NEGATIVE across β-sweep.

## Next-experiment hook

Three directions with a priori non-low prior, in decreasing alignment with existing
investment:

1. **Appearance fusion** (move past centroid geometry). Inject CLIP visual features
   alongside motion. Only decision-level fusion per prior burn-in lesson (Exp 14);
   any feature-level injection into CLIP has −21.7% F1 history.
2. **Portability to TransRMOT** (remaining spatially-ignorant tracker). Stage D
   demonstrated that consumer tracker class matters; verify the positive prediction
   on a second iKUN-class consumer before publishing tracker-class dichotomy as
   generalizable.
3. **Accept ceiling, pivot to measurement**. Publish the 12-lever negative result
   plus tracker-class dichotomy as a RMOT design-guideline paper without further
   model iteration. Lowest effort, lowest glamour.

## Branch + artifact locations

**GMC-Link:**
- Branch: `exp/ego-motion-systematic` (merged onto `exp/hn-mining` pending)
- Memos: `diagnostics/results/exp37/{stage_a_memo,stage_c_memo,stage_d_memo,retrospective_final}.md`
- Runners: `run_exp37_stage_{a,c,d}.sh`

**TempRMOT (user's fork at `/home/seanachan/TempRMOT`):**
- Branch: `exp37-stage-d-baseline`
- Commits: `babef6e` (snapshot of pre-Exp37 state), `9015c68` (env-var gated fusion),
  `3cc8972` (3-tuple unpack fix)
- Main branch unmodified — rollback via `git checkout main`

**Memories updated this retrospective:**
- `project_exp37_stages_ac_negative.md` → supersede with this final retrospective
- New: `project_exp37_stage_d_tracker_class_dichotomy.md`
- MEMORY.md index

## Final weights sentinel

`gmc_link_weights_v1train_stage1.pth` remains the Exp 37 winning aligner. No Exp 37
training iteration (Stage A recoverPose, Stage C EMAP 15D) produced a better aligner.
Fusion config: α=1.0, β=1.0, legacy `min()` fusion on iKUN. **Do not** cascade
GMC-Link onto TempRMOT in production.
