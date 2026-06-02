# HOTA-Direct Learned Fusion — Gate C (cheap upper-bound probe)

**Date:** 2026-06-02
**Branch:** `exp/pathB-lidar-17d`
**Status:** design approved, spec for implementation
**Decision rule memory:** [[feedback_fusion_too_simple_2026_05_30]], [[project_signal_decomp_native_vetoes_gmc_2026_05_26]]

---

## Context

The shipped iKUN fusion is a hand-tuned additive recipe: per axis, `fused = native_logit + α·(sc·gmc + thr)`,
admit iff `fused > thr_axis`. iKUN uses **2 axes** — motion (incl. STATIC, since `is_motion` is True for both)
and appearance — i.e. **6 params** `(α, sc, thr) × {motion, appear}` (ship: motion 1.0/0.9/0.17, appear 1.0/0.30/0.10).

Four prior *learned* fusion attempts were NEG, but all optimized a **differentiable surrogate** (F1/MSE) that does
not match HOTA's trajectory/association structure ([[project_ikun_learned_residual_negative]],
[[project_flexhook_learned_fusion_negative]]). The open frontier (flagged in `feedback_fusion_too_simple`) is a
fusion optimized **directly against HOTA**. HOTA is non-differentiable (Hungarian matching + IoU thresholds +
trajectory association in TrackEval), so "HOTA-direct" must mean **black-box (derivative-free) optimization of the
fusion's parameters against measured HOTA**, not SGD.

**This spec is Gate C only** — the cheapest rung of a C→B→A ladder. C optimizes on the **test seqs** (same data the
hand recipe was tuned on) purely to answer one question: *is the additive FORM the ceiling, or does a richer fusion
find meaningfully more HOTA?* C's number is an **upper bound by design, not a shippable claim** (same status as the
oracle / if-else probes). Only if C shows real headroom do we pay for the honest held-out protocols (B = CV within
test; A = optimize on train seqs, report test).

## Goal & success criteria

Produce, for iKUN seed0, V1 3-seq pooled HOTA:
- **F0** = the 6-param additive recipe re-optimized by black-box search on test HOTA.
- **F1** = a richer ~10-param fusion (additive + per-axis `native×gmc` interaction + `gmc²` curvature) similarly optimized.

**Baseline = seed0 hand fusion 44.561** (C is seed0-only, so compare like-for-like to the seed0 hand number, NOT
the n=3 ship mean 44.634 — single-seed vs n=3 is not apples-to-apples). 44.634 quoted only as ship context.

**Gate decision** (vs seed0 hand 44.561):
- `max(F0, F1) ≈ 44.561` (within ~+0.1) → additive form is exhausted → **KILL** the direction (4-NEG history was right).
- `max(F0, F1) ≫ 44.561` (≥ ~+0.5) → form has headroom → **proceed to Gate B** (held-out CV) for an honest number.
- **Attribution**: F0≫hand ⇒ hand-tuning under-searched (free win + flag); F0≈hand but F1≫F0 ⇒ the *nonlinear form*
  is the lever.

## Components

### 1. Fusion forms (extend `run_ikun_linear_additive.py:gen_predicts`)
Add a `fusion_form` parameter + the F1 terms behind it; F0 = existing additive (no code-path change, just
optimizer-set params). Per axis `a ∈ {motion, appear}`, with `n = native_part = cs + b`, `g = gmc_term`:
- **F0**: `score = n + α_a·sc_a·g`; admit `score > thr_a`.  (params: α, sc, thr per axis = 6)
- **F1**: `score = n + α_a·sc_a·g + β_a·(n·g) + γ_a·g²`; admit `score > thr_a`.  (params: α, sc, thr, β, γ per axis = 10)

F1 is a strict superset of F0 (β=γ=0 ⇒ F0), so F1's optimum ≥ F0's optimum by construction; comparing them
attributes gain to the nonlinear terms. Forms kept low-dim (≤10) so the optimizer converges in ~hundreds of evals.

### 2. Black-box optimizer (`diagnostics/hota_fusion/optimize_fusion.py`)
`scipy.optimize.differential_evolution` (global, derivative-free; **scipy already a project dep**, no new install).
- Bounds per param: α∈[0,3], sc∈[0,20], thr∈[−1,3] (motion) / [−0.5,1] (appear), β,γ∈[−2,2] (centered 0).
- Objective = the eval engine's pooled HOTA, **STATIC<43.2 penalized −100** (same guardrail as the autoresearch
  eval). Maximize ⇒ minimize `−objective`.
- Budget cap (`maxiter`/`popsize`) sized to ~150–250 evals; log every eval (params, pooled, STATIC) to a CSV.
- Seed the initial population with the hand recipe point (so the optimizer starts at the known-good and can only
  improve). Determinism: pass a fixed `seed` to differential_evolution (no Math.random reliance).

### 3. Eval engine (in-process, reuse — like step0/step1 drivers)
Load text_feat + GMC caches + iKUN scores **once**; per param set call `gen_predicts(..., fusion_form, params)` +
`run_te(pooled)` + `run_te(STATIC)`. ~1 min/eval, **no retrain**. iKUN seed0 caches (`_sharedweight_seed0_rawcos`)
already on disk. Reuse `run_te`, `gen_predicts`, the seqmap plumbing.

## Data flow
```
hand recipe point ─seed→ differential_evolution
  └─ proposes params → gen_predicts(form, params) → predict.txt → run_te → pooled + STATIC
        └─ objective = pooled − (100 if STATIC<43.2 else 0) → DE updates population
  → best params per form (F0, F1) → report max vs hand 44.634 → GATE
```

## Testing / verification
1. **Sanity**: F0 with the hand params must reproduce 44.561 (seed0) exactly — proves the eval engine matches the
   shipped fusion before any optimization.
2. **Superset check**: F1 with β=γ=0 must equal F0 at the same α/sc/thr (proves F1 is a correct superset).
3. **Optimizer smoke**: a ~20-eval DE run completes, objective non-decreasing vs the seeded hand point, CSV logged.
4. **Gate run**: full F0 + F1 optimization (~150–250 evals each, background ~3 h each), report `max(F0,F1)` vs the
   seed0 hand baseline 44.561 + the F0-vs-F1 attribution.

## Scope / non-goals (YAGNI)
- iKUN seed0 only (gate is single-seed by design; n=3 + held-out come at Gate B if C passes).
- No train-seq infra (that's Gate A).
- No free-form MLP (kept low-dim parametric for DE efficiency + interpretability).
- Test-optimized number is an upper bound, never reported as a ship claim.

## Risks
- **Overfit (intended)**: C optimizes on test → its number is optimistic by construction. Mitigation: it's labeled a
  gate/upper-bound; the honest number is Gate B's job.
- **DE under-converges** in ~200 evals on 10 dims → mitigate by seeding the hand point + bounded ranges; report the
  best-found honestly (a non-improvement could be under-search, not form-exhaustion — F0 vs hand helps disambiguate,
  since F0=hand-form so DE should at least match it).
- **Eval cost** ~1 min × ~400 total evals ≈ 6–7 h → run in background; cap maxiter.
