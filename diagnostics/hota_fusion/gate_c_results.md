# Gate C results — HOTA-direct fusion optimization (iKUN seed0, V1 3-seq pooled)

**Date:** 2026-06-04
**Status:** Gate C only — TEST-optimized upper bound, NOT a ship claim.
**Baseline:** seed0 hand recipe `B_seed0 = 44.561` (n=3 ship mean 44.634 is context only).
**Spec/plan:** `docs/superpowers/specs/2026-06-02-hota-direct-fusion-gate-c-design.md`,
`docs/superpowers/plans/2026-06-04-hota-direct-fusion-gate-c.md`.

## Method

Black-box (derivative-free) `scipy.optimize.differential_evolution` maximizing measured
3-seq pooled HOTA (non-differentiable), STATIC<43.2 penalized −100, DE seeded at the hand
recipe so it can only improve. Effective coefficient `coef = α·sc` per axis (α/sc are
multiplicatively degenerate). Bounds are a **tight basin around the hand recipe** — the
first broad `[0,20]` run was killed mid-initial-population (eval 24 of a 32/40-member
pop, before evolution) and under-resolved the basin; re-run with basin bounds
(`coef_m[0.3,3] thr_m[-0.2,0.6] coef_a[0.1,1.5] thr_a[-0.2,0.5]`, F1 `β,γ[-1,1]`).
Sanity verified pre-run: hand recipe reproduces 44.561 and F1(β=γ=0)==F0 exactly.

## Results

| form | best params | best pooled | Δ vs B_seed0 | best STATIC | evals |
|---|---|---|---|---|---|
| hand (seed0) | coef_m=0.9 thr_m=0.17 coef_a=0.30 thr_a=0.10 | 44.561 | — | 43.240 | — |
| F0 (additive, re-opt) | [0.9, 0.17, 0.30, 0.10] (= hand) | 44.561 | +0.000 | 43.240 | 224 |
| F1 (+native·gmc +gmc²) | [0.9, 0.17, 0,0, 0.30, 0.10, 0,0] (β=γ=0 = hand) | 44.561 | +0.000 | 43.240 | 240 |

**max(F0, F1) = 44.561  →  Δ = +0.000 vs B_seed0.**

Best *non-seed* PASS configs (nothing beat the hand point):
- F0: 44.549 / 44.531 / 44.508 / 44.497 — all below 44.561.
- F1 (with nonzero β/γ): 44.518 / 44.468 / 44.463 / 44.392 — all below 44.561; the nonlinear terms only subtract.

## Verdict: KILL

`max(F0,F1) = B_seed0` (Δ = +0.000, well within the ~+0.1 flat band) → **the additive fusion
FORM is the ceiling.** A properly-powered HOTA-direct black-box search of both the additive
form (F0) and a strict nonlinear superset (F1 = + native·gmc interaction + gmc² curvature),
seeded at and searched around the hand recipe's basin, finds **no configuration that beats
the hand recipe.** Do not pursue further fusion-form levers.

This is **stronger** than the 4 prior learned-fusion NEGs
([[project_ikun_learned_residual_negative]], [[project_flexhook_learned_fusion_negative]]):
those optimized a differentiable F1/MSE surrogate (objective-bug suspicion). Gate C optimized
HOTA directly and still flat → the form itself is the wall, not the objective.

## Attribution

- **F0 ≈ hand** ⇒ the hand recipe was NOT under-searched; it sits at the additive optimum
  (no free re-search win).
- **F1 ≈ F0** (F1's optimum is β=γ=0 = F0) ⇒ the nonlinear form adds nothing; interaction
  and curvature terms only degrade pooled HOTA.
- Both ≈ hand ⇒ neither search-quality nor nonlinear form is a lever.

## Caveats

- Test-optimized (optimizes on the eval seqs) → its number is an optimistic upper bound by
  construction. Because the verdict is KILL (no improvement even with that optimism), the
  honest Gate B (held-out CV) is moot — there is nothing to validate. Gate A also skipped.
- DE seeded at hand can only improve; best = seed means DE genuinely found no better point in
  the basin across 224 (F0) / 240 (F1) evals.

## Consistency with the campaign

Corroborates the oracle decomposition ([[project_signal_decomp_native_vetoes_gmc_2026_05_26]]):
the fusion LAYER is exhausted (additive ship is at its optimum; un-veto NEG; 18-param recipe
irreducible; and now HOTA-direct learned fusion NEG). Remaining reachable headroom is the
upstream MOTION CLASSIFIER (ship→oracle_motion +6.13), not the fusion form.
