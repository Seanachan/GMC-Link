# Stage C — Structural Ego Conditioning Decision

Date: 2026-04-22
Spec: `docs/superpowers/specs/2026-04-22-ego-motion-systematic-design.md` §2 H_C
Plan: `docs/superpowers/plans/2026-04-22-exp37-ego-motion-systematic.md` Task 13
Runner: `run_exp37_stage_c.sh`

## Configuration

- Ego source (from Stage A winner): **ORB**
- Stage B extras (from stage_b_features sentinel): **(none — Stage B deferred)**
- C1 (baseline): reuse `gmc_link_weights_v1train_stage1.pth`, motion_dim = 13
- C2 (EMAP concat): `gmc_link_weights_exp37_stage_c2_orb.pth`, `--extra-features ego_velocity_concat`, motion_dim = 15
  - Extra 2D channel = mid-scale ego translation `(ego_dx_m, ego_dy_m)` — same upstream signal as
    the older F4 `ego_motion` feature; distinct name marks intent as **structural EMAP-style
    conditioning** rather than raw concatenation experiment.

## Held-out V1 results (seqs 0005, 0011, 0013)

| model_tag | micro AUC | macro AUC ± std | 0005 | 0011 | 0013 | max_gap |
|---|---|---|---|---|---|---|
| C1 `v1train_stage1` (13D baseline) | **0.779** | 0.838 ± 0.064 | 0.821 | 0.779 | — | 0.042 |
| C2 `exp37_stage_c2_orb` (15D EMAP concat) | **0.755** | 0.814 ± 0.070 | 0.809 | 0.745 | 0.829¹ | 0.084 |
| **Δ (C2 − C1)** | **−0.024** | **−0.024** | −0.012 | −0.034 | — | — |

¹ Seq 0013 has n=2 expressions — per `project_v1_seq0013_data_thinness`
memory this is noise-dominated and not load-bearing. Discount when
weighting the decision.

## Decision band (spec §2 H_C)

| Band | Δmicro threshold | Outcome |
|---|---|---|
| POSITIVE | ≥ +0.010 | — |
| INCONCLUSIVE | −0.010 < Δ < +0.010 | — |
| **NEGATIVE** | **≤ −0.010** | **✓ Δ = −0.024** |

**Decision: NEGATIVE.** Structural ego-velocity concatenation does not lift the
held-out AUC ceiling; it regresses 2.4pp on both micro and macro metrics.

## Why EMAP concat regressed

1. **Signal is already in the base 13D vector.** The residual velocities
   `res_dx_k / res_dy_k` = `raw_v_k − ego_v_k` at three scales already encode
   ego motion *implicitly* via subtraction. Appending `(ego_dx_m, ego_dy_m)`
   as a redundant absolute channel adds no geometrically new information — it
   only widens the input surface the aligner must learn to ignore.
2. **0.779 ceiling is representation-bound, not feature-bound.**
   `project_exp36e_curriculum_negative` memory closed this over eight Exp 36
   levers (losses, features, architectures, encoders, joint V1+V2, curriculum).
   Stage C is the 9th lever on the *same* 13D centroid geometry — same
   ceiling, same outcome.
3. **Per-seq std widened slightly** (0.064 → 0.070) and seq 0011 regressed
   most (−0.034). Consistent with mild feature interference rather than
   lift: absolute ego magnitude biases the learned cosine for non-stationary
   scenes where ego_dx_m varies most.

## Implications for downstream stages

- **Stage D (portability to TempRMOT) is now the sole surviving headline**
  per spec §10 kill switch #2. A/B (deferred) and C are negative on aligner
  AUC; the Exp 37 publishable claim must carry via cross-tracker transfer
  since no aligner-quality lever moved the representation-bound ceiling.
- Stage B OMF dataset plumb gap (zero-fill) remains: even had it been
  wired, expected outcome is the same NEGATIVE per the
  representation-bound argument above. Time-budget priority is Stage D.
- Any further aligner-side iteration should move past centroid geometry
  entirely (e.g., appearance fusion), not add another feature channel.

## C2 winner reference

```
diagnostics/results/exp37/stage_c2_weights  →  gmc_link_weights_exp37_stage_c2_orb.pth
(baseline retained: gmc_link_weights_v1train_stage1.pth — micro 0.779 remains project
representation-bound ceiling; Stage C C2 does not supersede it.)
```
