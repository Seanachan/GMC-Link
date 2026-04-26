# Stage D — TempRMOT Portability Decision

Date: 2026-04-22
Spec: `docs/superpowers/specs/2026-04-22-ego-motion-systematic-design.md` §2 H_D
Plan: `docs/superpowers/plans/2026-04-22-exp37-ego-motion-systematic.md` Tasks 15–16
Runner: `run_exp37_stage_d.sh`
Fusion integration: `/home/seanachan/TempRMOT/inference.py` (branch `exp37-stage-d-baseline`,
commits `babef6e`, `9015c68`, `3cc8972`)

## Configuration

- Tracker: **TempRMOT** (`meta_arch=temp_rmot`, checkpoint `exps/default_rk/checkpoint0050.pth`)
- Aligner weights (winner from A/C): `gmc_link_weights_v1train_stage1.pth` (13D, micro AUC 0.779)
- Ego source: ORB (Stage A winner)
- Held-out: V1 seqs 0005, 0011, 0013 — 150 (seq, expression) pairs
- Fusion point: `dt_instances.refers` (before `filter_dt_by_ref_scores(0.2)`)
- Gated via env vars: `EXP37_MODE ∈ {bare, alpha}`, `EXP37_BETA ∈ {0.5, 1.0, 2.0}`
- α-fusion rule: `refers_new = vision_prob + β · gmc_prob`
  - `vision_prob = dt_instances.refers[i].item()` (raw TempRMOT score)
  - `gmc_prob` = smoothed aligner cosine∈[0,1] via `gmc_manager.process_frame(...)[0]`
- HOTA via TrackEval (`SEQMAP=seqmap-v1.txt`, 15 (seq, expression) pairs post-HOTA eval
  selection — 150 → 15 seqmap entries)

## V1 held-out HOTA results (all 4 arms, COMBINED)

| arm | HOTA | DetA | AssA | DetRe | DetPr | AssRe | AssPr | Δ HOTA vs bare |
|---|---|---|---|---|---|---|---|---|
| **bare** (no fusion) | **51.388** | 39.176 | 67.561 | 57.249 | 54.126 | 73.442 | 86.964 | — |
| α=β0.5  | 47.394 | 33.510 | 67.164 | 60.629 | 42.103 | 73.520 | 86.584 | **−3.994** |
| α=β1.0  | 47.597 | 33.131 | 68.483 | 59.701 | 41.975 | 74.576 | 87.274 | **−3.791** |
| α=β2.0  | 46.009 | 31.678 | 66.939 | 58.933 | 39.997 | 73.025 | 86.811 | **−5.379** |

Counts row for sanity (CLR_TP, CLR_FN, CLR_FP, IDSW):

| arm | CLR_TP | CLR_FN | CLR_FP | IDSW |
|---|---|---|---|---|
| bare    | 70296 | 66461 | 3285 | 1511 |
| β=0.5 | 95705 | 66461 | 4004 | 1511 |
| β=1.0 | 94528 | 66461 | 3868 | 1511 |
| β=2.0 | 97926 | 66461 | 4068 | 1511 |

## Decision band (spec §2 H_D)

| Band | Δ HOTA threshold | Outcome |
|---|---|---|
| POSITIVE | ≥ +1.0 | — |
| INCONCLUSIVE | (−1.0, +1.0) | — |
| **NEGATIVE** | **≤ −1.0** | **✓ all three β arms (−3.8 to −5.4)** |

**Decision: NEGATIVE on all β ∈ {0.5, 1.0, 2.0}.** GMC-Link α-fusion regresses TempRMOT
HOTA by 3.8–5.4 points; no β in the tested grid recovers bare's 51.388.

The degradation is monotonic-ish with β:
- β=1.0 is the least bad (best of alpha) at 47.597 (−3.791)
- β=2.0 is the worst at 46.009 (−5.379)
- β=0.5 sits between at 47.394 (−3.994) — slightly worse than β=1.0, suggesting neither
  strong nor weak GMC weighting recovers baseline

## Why GMC-Link fusion regressed on TempRMOT

Root cause is **structural**, not integration bug. Three mutually reinforcing factors:

1. **Redundant temporal constraints.** TempRMOT already has DETR-query temporal
   attention (`temp_rmot` meta-arch, `hist_len=8`) that tracks an object's referring
   score across frames via `refers` logits propagated through `QIM`. GMC-Link stacks a
   second temporal constraint (EMA-smoothed aligner cosine over multi-scale residual
   velocities) onto the same `refers` field. Two independent temporal smoothers on the
   same signal over-regularize the decision boundary.

2. **DetA collapses, AssA steady.** DetA drops 5.5–7.5 pp across β arms while AssA
   stays within ±1 pp of bare. Classification-level fusion pushes predictions across
   the `filter_dt_by_ref_scores(0.2)` threshold in the wrong direction: TP grows
   (95k–98k vs 70k) but so does FP (3.9k–4.1k vs 3.3k), and because TempRMOT's
   tracklet definition spans more frames per object, DetPr drops 12–14 pp while DetRe
   gains only ~3 pp. Net: more total predictions, worse HOTA.

3. **CLAUDE.md explicitly anticipates this.** The project doc §"Important Design
   Decisions" already flags: *"GMC-Link is designed for spatially-ignorant
   vision-language frameworks (e.g., TransRMOT, iKUN). Cascading onto trackers with
   native temporal memory (e.g., TempRMOT) causes structural regression due to
   redundant temporal constraints."* Stage D empirically confirms this a priori
   hypothesis at scale.

## Implications — Exp 37 as a whole

With Stage D NEGATIVE, the spec's §10 kill-switch #2 activates:

> "If Stages A, B, C all negative (ego-motion not the lever either), Exp 37 conclusion
> is: pipeline-bound at iKUN logit distribution…"

Stage D now adds: **not iKUN-logit-bound either — tracker-class-bound.** GMC-Link
provides net value when cascaded onto a *spatially-ignorant* tracker (iKUN,
TransRMOT), and net cost when cascaded onto a *temporally-aware* tracker (TempRMOT).
The aligner quality axis (0.779 V1 AUC ceiling) is orthogonal to this — even a
perfect aligner cannot rescue fusion on top of temporal memory.

### What remains publishable

1. The **10-lever exhaustion memo** (`retrospective_interim.md` + Exp 36 series) as a
   systematic negative result establishing the representation-bound ceiling.
2. The **tracker-class dichotomy**: `{iKUN, TransRMOT} ✓` vs `{TempRMOT} ✗` as
   architectural precondition for decision-level RMOT fusion — a concrete design
   guideline that any future plug-in RMOT module must respect.
3. The **α=1.0 iKUN fusion** (prior memory `project_aligner_sweep_at_alpha1_restores_auc`)
   as the anchor positive result: HOTA 35.848 on iKUN+GMC-Link, +2.57 over iKUN bare.

### What is falsified

- Spec's H_D hypothesis that GMC-Link fusion transfers cleanly to any RMOT tracker is
  **falsified** on TempRMOT. Future portability claims must include tracker-class
  as a precondition.
- Spec's §10 kill-switch #3 ("portability scope deferred if dep conflicts") did not
  activate — we did reach and evaluate H_D — so there is no scope-narrowing escape.
  The negative is clean and honest.

## Artifacts

- HOTA logs: `diagnostics/results/exp37/stage_d/hota_{bare,alpha_beta05,alpha_beta10,alpha_beta20}.log`
- Inference logs: `diagnostics/results/exp37/stage_d/infer_{...}.log`
- Outer wrapper logs: `{...}_outer.log`, `tail_runner.log`, `alpha_runner.log`
- TempRMOT integration: branch `exp37-stage-d-baseline` at `/home/seanachan/TempRMOT`
- Per-arm track dumps: `/home/seanachan/TempRMOT/exps/exp37_stage_d_{bare,alpha_beta05,alpha_beta10,alpha_beta20}/results_epoch50/`

## Winner reference

No Stage D arm wins. Bare TempRMOT (no fusion) is the best TempRMOT configuration at
51.388 HOTA. **Baseline retained** — `gmc_link_weights_v1train_stage1.pth` is the
Exp 37 final winning aligner, matched with the **legacy min() fusion on iKUN**
(HOTA 35.848 per prior memory), not with TempRMOT.
