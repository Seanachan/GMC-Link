# HOTA-Direct Learned Fusion — Gate C Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Black-box optimize the iKUN GMC fusion parameters directly against measured 3-seq pooled HOTA, for two forms (F0 additive, F1 +nonlinear), to answer whether the additive fusion FORM is the ceiling — a cheap test-optimized upper-bound gate.

**Architecture:** Reuse the existing in-process eval engine (`run_ikun_linear_additive.py` `gen_predicts` + `run_te`, same as the step0/step1 rerank drivers). Add a tiny pure helper `f1_extra` + a `fusion_form` switch to `gen_predicts` for the F1 nonlinear terms. A new optimizer module (`diagnostics/hota_fusion/optimize_fusion.py`) drives `scipy.optimize.differential_evolution` (derivative-free, already a project dep) against pooled HOTA with the STATIC<43.2 guardrail baked into the objective, seeded at the hand recipe so it can only improve.

**Tech Stack:** Python, numpy, scipy 1.17.1 (`differential_evolution`, `x0=` supported), TrackEval (subprocess via `run_te`). iKUN seed0 ship caches (`_sharedweight_seed0_rawcos`) already on disk. No GPU, no retrain (~1–1.2 min/eval, pure CPU + TrackEval).

**Spec:** `docs/superpowers/specs/2026-06-02-hota-direct-fusion-gate-c-design.md`

---

## Background the engineer needs

- **Shipped iKUN fusion** (per box, per axis): `fused = native_part + α·sc·g`; admit iff `fused > thr`. `native_part = cs + b` (iKUN cascade score + simcalib bias), `g = gmc_term` (raw cosine in [−1,+1] because `GMC_RAW_COS=1`). Two axes: **motion** (incl. STATIC) and **appearance**. Hand recipe seed0: motion `α=1.0, sc=0.9, thr=+0.17`; appear `α=1.0, sc=0.30, thr=+0.10`.
- **α and sc are multiplicatively degenerate** — only the product `coef = α·sc` affects the score. We therefore optimize one effective coefficient per axis (`coef_m`, `coef_a`), not α and sc separately. This removes 2 redundant search dims vs the spec's nominal "6 params" while reaching the **identical** F0 score set. Hand point maps to `coef_m=0.9, coef_a=0.30`.
- **F0** (4 effective params): `coef_m, thr_m, coef_a, thr_a`.
- **F1** (8 effective params, strict superset of F0): adds per axis `+ β·(native_part·g) + γ·g²`. Params: `coef_m, thr_m, β_m, γ_m, coef_a, thr_a, β_a, γ_a`. `β=γ=0 ⇒ F0`.
- **Objective** = 3-seq pooled HOTA, **STATIC<43.2 penalized −100** (same guardrail as `autoresearch/eval_ikun_motionrep.py`). Maximize ⇒ DE minimizes `−objective`.
- **Baseline** = seed0 hand recipe pooled HOTA measured on the ship seed0 caches (expected ≈ **44.561**; Task 2 measures the exact value and records it as `B_seed0`). Compare like-for-like to seed0, NOT the n=3 ship mean 44.634.
- **Gate decision** (vs `B_seed0`): `max(F0,F1) ≈ B_seed0` (within ~+0.1) → additive form exhausted → **KILL**. `≥ ~+0.5` → headroom → **proceed to Gate B** (held-out CV). Attribution: F0≫hand ⇒ hand under-searched; F0≈hand but F1≫F0 ⇒ the nonlinear form is the lever.
- **Env-at-import:** `run_ikun_linear_additive.py` reads `GMC_SUFFIX` / `GMC_RAW_COS` at module-import time (lines 48–52). The optimizer MUST set them via `os.environ.setdefault(...)` BEFORE `import run_ikun_linear_additive`.
- **Memory constraint:** commits must NOT include a `Co-Authored-By: Claude` trailer (project feedback).

---

## File Structure

- **Modify** `run_ikun_linear_additive.py`
  - Add module-level pure helper `f1_extra(native_part, gmc_term, beta, gamma)` (the F1 nonlinear terms; returns 0 when β=γ=0 → keeps F1 a strict superset of F0).
  - Extend `gen_predicts(...)` signature with `fusion_form="F0", beta_m=0.0, gamma_m=0.0, beta_a=0.0, gamma_a=0.0`; wire `f1_extra` into the motion and appearance branches. Default `fusion_form="F0"` ⇒ all existing callers unchanged.
- **Create** `diagnostics/hota_fusion/test_fuse_form.py` — self-running unit tests for `f1_extra` (superset + interaction + curvature). No pytest dependency.
- **Create** `diagnostics/hota_fusion/optimize_fusion.py` — load-once eval engine, `evaluate()`, `objective()`, DE driver, CSV per-eval log, `--sanity` / `--smoke` / gate modes.
- **Outputs** (artifacts): `diagnostics/hota_fusion/optimize_F0_log.csv`, `optimize_F1_log.csv`, `optimize_F0_best.json`, `optimize_F1_best.json`, `gate_c_results.md`.

---

## Task 1: F1 fusion term in `gen_predicts`

**Files:**
- Create: `diagnostics/hota_fusion/test_fuse_form.py`
- Modify: `run_ikun_linear_additive.py` (add `f1_extra` after `_iou_xywh` ~line 113; extend `gen_predicts` signature line 128–131; motion branch lines 234–239; appearance branch lines 240–250)

- [ ] **Step 1: Write the failing test**

Create `diagnostics/hota_fusion/test_fuse_form.py`:

```python
"""Unit tests for the F1 nonlinear fusion term. Self-running (no pytest dep):
    python diagnostics/hota_fusion/test_fuse_form.py
"""
import sys
sys.path.insert(0, "/home/seanachan/GMC-Link")
from run_ikun_linear_additive import f1_extra


def test_f1_superset_zero():
    # beta=gamma=0 => F1 adds nothing => strict superset of F0
    assert f1_extra(2.0, 0.5, 0.0, 0.0) == 0.0
    assert f1_extra(-3.0, 0.8, 0.0, 0.0) == 0.0


def test_f1_interaction_term():
    # beta*(native*gmc) = 1.0*(2.0*0.5) = 1.0 ; gamma=0
    assert f1_extra(2.0, 0.5, 1.0, 0.0) == 1.0


def test_f1_curvature_term():
    # gamma*(gmc^2) = 2.0*(0.5*0.5) = 0.5 ; beta=0
    assert f1_extra(10.0, 0.5, 0.0, 2.0) == 0.5


def test_f1_both_terms():
    # beta*(native*gmc) + gamma*(gmc^2) = 1.0*(2.0*0.5) + 1.0*(0.25) = 1.25
    assert f1_extra(2.0, 0.5, 1.0, 1.0) == 1.25


if __name__ == "__main__":
    test_f1_superset_zero()
    test_f1_interaction_term()
    test_f1_curvature_term()
    test_f1_both_terms()
    print("OK: all f1_extra tests passed")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python diagnostics/hota_fusion/test_fuse_form.py`
Expected: FAIL with `ImportError: cannot import name 'f1_extra' from 'run_ikun_linear_additive'`

- [ ] **Step 3: Add the `f1_extra` helper**

In `run_ikun_linear_additive.py`, immediately AFTER the `_iou_xywh` function (ends ~line 112, before `_load_gt_boxes` at line 115), insert:

```python
def f1_extra(native_part, gmc_term, beta, gamma):
    """F1 nonlinear GMC terms: native*gmc interaction + gmc^2 curvature.
    Returns 0 when beta=gamma=0, so F1 is a strict superset of F0 (additive)."""
    return beta * (native_part * gmc_term) + gamma * (gmc_term * gmc_term)
```

- [ ] **Step 4: Extend `gen_predicts` signature**

In `run_ikun_linear_additive.py`, change the `gen_predicts` signature (lines 128–131) from:

```python
def gen_predicts(text_feat, gmc_caches, alpha, gmc_scale, thr_motion, run_dir,
                 alpha_a=0.0, scale_a=0.0, thr_a=0.0, mode="ship", dump_path=None,
                 motion_fuse="add", gmc_gate=0.35, rerank_set=None,
                 clip_caches=None, rerank_tau=0.0):
```

to:

```python
def gen_predicts(text_feat, gmc_caches, alpha, gmc_scale, thr_motion, run_dir,
                 alpha_a=0.0, scale_a=0.0, thr_a=0.0, mode="ship", dump_path=None,
                 motion_fuse="add", gmc_gate=0.35, rerank_set=None,
                 clip_caches=None, rerank_tau=0.0,
                 fusion_form="F0", beta_m=0.0, gamma_m=0.0, beta_a=0.0, gamma_a=0.0):
```

- [ ] **Step 5: Wire `f1_extra` into the motion branch**

In `gen_predicts`, replace the motion branch (lines 234–239):

```python
                    if motion:
                        default = 0.0 if RAW_COS else 0.5
                        gmc = float(per_expr_gmc.get(str(fid), {}).get(str(oid), default))
                        gmc_term = gmc if RAW_COS else (gmc - 0.5)
                        gmc_part = alpha * gmc_term * gmc_scale
                        thr = thr_motion
```

with:

```python
                    if motion:
                        default = 0.0 if RAW_COS else 0.5
                        gmc = float(per_expr_gmc.get(str(fid), {}).get(str(oid), default))
                        gmc_term = gmc if RAW_COS else (gmc - 0.5)
                        gmc_part = alpha * gmc_term * gmc_scale
                        if fusion_form == "F1":
                            gmc_part += f1_extra(native_part, gmc_term, beta_m, gamma_m)
                        thr = thr_motion
```

- [ ] **Step 6: Wire `f1_extra` into the appearance branch**

In `gen_predicts`, replace the appearance branch (lines 240–250):

```python
                    else:
                        if scale_a != 0.0:
                            default = 0.0 if RAW_COS else 0.5
                            gmc = float(per_expr_gmc.get(str(fid), {}).get(str(oid), default))
                            gmc_term = gmc if RAW_COS else (gmc - 0.5)
                            gmc_part = alpha_a * gmc_term * scale_a
                            thr = thr_a
                        else:
                            gmc = float("nan")
                            gmc_part = 0.0
                            thr = 0.0
```

with (note the guard now also fires for F1 so the appearance GMC term stays active even if DE drives `coef_a`→0):

```python
                    else:
                        if scale_a != 0.0 or fusion_form == "F1":
                            default = 0.0 if RAW_COS else 0.5
                            gmc = float(per_expr_gmc.get(str(fid), {}).get(str(oid), default))
                            gmc_term = gmc if RAW_COS else (gmc - 0.5)
                            gmc_part = alpha_a * gmc_term * scale_a
                            if fusion_form == "F1":
                                gmc_part += f1_extra(native_part, gmc_term, beta_a, gamma_a)
                            thr = thr_a
                        else:
                            gmc = float("nan")
                            gmc_part = 0.0
                            thr = 0.0
```

- [ ] **Step 7: Run test to verify it passes**

Run: `python diagnostics/hota_fusion/test_fuse_form.py`
Expected: `OK: all f1_extra tests passed`

- [ ] **Step 8: Commit**

```bash
git add run_ikun_linear_additive.py diagnostics/hota_fusion/test_fuse_form.py
git commit -m "feat(gateC): F1 nonlinear fusion term (native*gmc + gmc^2) in gen_predicts"
```

---

## Task 2: Optimizer module + sanity (baseline + superset verification)

**Files:**
- Create: `diagnostics/hota_fusion/optimize_fusion.py`

- [ ] **Step 1: Write the optimizer module**

Create `diagnostics/hota_fusion/optimize_fusion.py`:

```python
"""Gate C — black-box (derivative-free) optimization of the iKUN GMC fusion
parameters DIRECTLY against measured 3-seq pooled HOTA (non-differentiable:
Hungarian matching + IoU + trajectory association in TrackEval).

F0 = additive recipe re-optimized. 4 effective params: coef_m, thr_m, coef_a,
     thr_a (alpha/scale are multiplicatively degenerate => one coef per axis).
F1 = + per-axis native*gmc interaction + gmc^2 curvature. 8 params; strict
     superset of F0 (beta=gamma=0 => F0).

Objective = pooled HOTA, STATIC<floor penalized -100 (same guardrail as
autoresearch/eval_ikun_motionrep.py). Seeded at the hand recipe so DE can only
improve. iKUN seed0 only; TEST-optimized => UPPER BOUND, not a ship claim
(Gate C of the C->B->A ladder).

Run (background, ~1-1.2 min/eval):
  python diagnostics/hota_fusion/optimize_fusion.py --form F0 --maxiter 6 --popsize 8
  python diagnostics/hota_fusion/optimize_fusion.py --form F1 --maxiter 5 --popsize 5
Sanity (hand recipe reproduces baseline + F0==F1 superset):
  python diagnostics/hota_fusion/optimize_fusion.py --sanity
Smoke (tiny DE, verify plumbing):
  python diagnostics/hota_fusion/optimize_fusion.py --form F0 --smoke
"""
import argparse
import csv
import json
import os
import sys

# Must set BEFORE importing run_ikun_linear_additive (env read at import time).
os.environ.setdefault("GMC_SUFFIX", "_sharedweight_seed0_rawcos")
os.environ.setdefault("GMC_RAW_COS", "1")
sys.path.insert(0, "/home/seanachan/GMC-Link")

from scipy.optimize import differential_evolution
import run_ikun_linear_additive as R

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_FLOOR = 43.2
HAND = dict(coef_m=0.9, thr_m=0.17, coef_a=0.30, thr_a=0.10)  # seed0 ship (alpha=1)

# Param layout (effective coef = alpha*scale collapsed):
#   F0: [coef_m, thr_m, coef_a, thr_a]
#   F1: [coef_m, thr_m, beta_m, gamma_m, coef_a, thr_a, beta_a, gamma_a]
BOUNDS = {
    "F0": [(0.0, 20.0), (-1.0, 3.0), (0.0, 20.0), (-0.5, 1.0)],
    "F1": [(0.0, 20.0), (-1.0, 3.0), (-2.0, 2.0), (-2.0, 2.0),
           (0.0, 20.0), (-0.5, 1.0), (-2.0, 2.0), (-2.0, 2.0)],
}
X0 = {
    "F0": [HAND["coef_m"], HAND["thr_m"], HAND["coef_a"], HAND["thr_a"]],
    "F1": [HAND["coef_m"], HAND["thr_m"], 0.0, 0.0,
           HAND["coef_a"], HAND["thr_a"], 0.0, 0.0],
}

_TF = None
_GMC = None


def _load():
    global _TF, _GMC
    if _TF is None:
        _TF = json.load(open(R.TEXT_FEAT_JSON))
        _GMC = {s: json.load(open(R.GMC_CACHE_TPL.format(seq=s))) for s in R.TEST_SEQS}


def _unpack(x, form):
    if form == "F0":
        coef_m, thr_m, coef_a, thr_a = x
        return dict(alpha=1.0, gmc_scale=coef_m, thr=thr_m,
                    alpha_a=1.0, scale_a=coef_a, thr_a=thr_a, fusion_form="F0",
                    beta_m=0.0, gamma_m=0.0, beta_a=0.0, gamma_a=0.0)
    coef_m, thr_m, beta_m, gamma_m, coef_a, thr_a, beta_a, gamma_a = x
    return dict(alpha=1.0, gmc_scale=coef_m, thr=thr_m,
                alpha_a=1.0, scale_a=coef_a, thr_a=thr_a, fusion_form="F1",
                beta_m=beta_m, gamma_m=gamma_m, beta_a=beta_a, gamma_a=gamma_a)


def evaluate(x, form, run_dir):
    _load()
    p = _unpack(x, form)
    os.makedirs(run_dir, exist_ok=True)
    res_dir, sm = R.gen_predicts(
        _TF, _GMC, p["alpha"], p["gmc_scale"], p["thr"], run_dir,
        alpha_a=p["alpha_a"], scale_a=p["scale_a"], thr_a=p["thr_a"],
        fusion_form=p["fusion_form"], beta_m=p["beta_m"], gamma_m=p["gamma_m"],
        beta_a=p["beta_a"], gamma_a=p["gamma_a"])
    pooled = R.run_te(sm, res_dir)
    static = R.run_te(sm, res_dir, class_filter="STATIC")
    return pooled, static


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", choices=["F0", "F1"], default="F0")
    ap.add_argument("--maxiter", type=int, default=6)
    ap.add_argument("--popsize", type=int, default=8)
    ap.add_argument("--sanity", action="store_true",
                    help="hand recipe reproduces baseline + F0==F1 superset check")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny DE (maxiter=1, popsize=2) to verify plumbing")
    args = ap.parse_args()

    if args.sanity:
        f0p, f0s = evaluate(X0["F0"], "F0", os.path.join(R.OUT_ROOT, "gateC_sanity_F0"))
        f1p, f1s = evaluate(X0["F1"], "F1", os.path.join(R.OUT_ROOT, "gateC_sanity_F1"))
        print(f"SANITY F0 hand:       pooled={f0p:.3f} STATIC={f0s:.3f}")
        print(f"SANITY F1(b=g=0) hand: pooled={f1p:.3f} STATIC={f1s:.3f}")
        print(f"superset match (|F0-F1|<1e-6): {abs(f0p - f1p) < 1e-6}")
        print(f"baseline B_seed0 = {f0p:.3f}  (expect ~44.561)")
        json.dump({"B_seed0": f0p, "F0_static": f0s, "F1_pooled": f1p,
                   "superset_match": abs(f0p - f1p) < 1e-6},
                  open(os.path.join(HERE, "sanity.json"), "w"), indent=2)
        return

    if args.smoke:
        args.maxiter, args.popsize = 1, 2

    run_dir = os.path.join(R.OUT_ROOT, f"gateC_{args.form}")
    log_path = os.path.join(HERE, f"optimize_{args.form}_log.csv")
    log_f = open(log_path, "w", newline="")
    writer = csv.writer(log_f)
    writer.writerow(["eval", "pooled", "static", "passed", "obj"] +
                    [f"x{i}" for i in range(len(BOUNDS[args.form]))])
    counter = {"n": 0}
    best = {"obj": -1e9, "x": None, "pooled": None, "static": None}

    def objective(x):
        pooled, static = evaluate(x, args.form, run_dir)
        passed = static >= STATIC_FLOOR
        obj = pooled - (0.0 if passed else 100.0)
        counter["n"] += 1
        writer.writerow([counter["n"], f"{pooled:.4f}", f"{static:.4f}",
                         int(passed), f"{obj:.4f}"] + [f"{v:.5f}" for v in x])
        log_f.flush()
        if obj > best["obj"]:
            best.update(obj=obj, x=list(x), pooled=pooled, static=static)
        print(f"[{counter['n']:>4}] pooled={pooled:.3f} STATIC={static:.3f} "
              f"{'PASS' if passed else 'FAIL'} obj={obj:.3f} | best={best['obj']:.3f}",
              flush=True)
        return -obj  # DE minimizes

    result = differential_evolution(
        objective, BOUNDS[args.form], x0=X0[args.form], seed=0,
        maxiter=args.maxiter, popsize=args.popsize, polish=False,
        mutation=(0.5, 1.0), recombination=0.7, workers=1, updating="immediate")
    log_f.close()

    print(f"\n=== Gate C {args.form} DONE ({counter['n']} evals) ===")
    print(f"best pooled={best['pooled']:.3f} STATIC={best['static']:.3f}")
    print(f"best params={best['x']}")
    json.dump({"form": args.form, "evals": counter["n"],
               "best_pooled": best["pooled"], "best_static": best["static"],
               "best_x": best["x"], "de_fun": float(-result.fun)},
              open(os.path.join(HERE, f"optimize_{args.form}_best.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the sanity check (measures baseline + verifies superset)**

Run: `python diagnostics/hota_fusion/optimize_fusion.py --sanity`
Expected (≈2 evals, ~2–3 min):
- `SANITY F0 hand: pooled=44.561 STATIC=43.240` (pooled within ±0.05 of 44.561; if it differs, that measured value IS the gate baseline — record it).
- `superset match (|F0-F1|<1e-6): True` (F1 with β=γ=0 must equal F0 exactly).
- `baseline B_seed0 = 44.561 (expect ~44.561)`.

If `superset match` is `False`, the F1 wiring is wrong — revisit Task 1 Steps 5–6 before continuing.

- [ ] **Step 3: Commit**

```bash
git add diagnostics/hota_fusion/optimize_fusion.py diagnostics/hota_fusion/sanity.json
git commit -m "feat(gateC): black-box fusion optimizer + sanity (baseline + F0/F1 superset)"
```

---

## Task 3: Optimizer smoke test (verify DE plumbing end-to-end)

**Files:**
- Uses: `diagnostics/hota_fusion/optimize_fusion.py` (no edits)

- [ ] **Step 1: Run a tiny DE in the background**

Run (background — ~16 evals, ~20 min):
```bash
python diagnostics/hota_fusion/optimize_fusion.py --form F0 --smoke
```
(Launch with `run_in_background: true`.)

- [ ] **Step 2: Verify it completes and improves-or-matches the seeded hand point**

When the background run finishes, check:
- `diagnostics/hota_fusion/optimize_F0_log.csv` exists with one row per eval (header `eval,pooled,static,passed,obj,x0,x1,x2,x3`).
- Final stdout line `=== Gate C F0 DONE (N evals) ===` with `best pooled` ≥ ~44.55 (DE includes the seeded hand point, so best can never fall below the seed minus eval noise).
- `diagnostics/hota_fusion/optimize_F0_best.json` exists with `best_pooled`, `best_x`.

Expected: completes without error; `best_pooled ≥ 44.55`. If `best_pooled` is far below 44.5, something is mis-wired (likely env/cache) — stop and inspect the CSV's first row (should be the hand point ≈44.561).

- [ ] **Step 3: Commit the verified script (no result artifacts yet)**

The smoke artifacts are throwaway; only commit if the script needed a fix. If Steps 1–2 passed with no edits, skip the commit and proceed to Task 4.

---

## Task 4: Gate run F0 (full optimization)

**Files:**
- Produces: `diagnostics/hota_fusion/optimize_F0_log.csv`, `optimize_F0_best.json`

- [ ] **Step 1: Launch the F0 optimization in the background**

Run (background — ~224 evals, ~4–5 h):
```bash
python diagnostics/hota_fusion/optimize_fusion.py --form F0 --maxiter 6 --popsize 8
```
(Launch with `run_in_background: true`. This overwrites the smoke CSV/JSON.)

- [ ] **Step 2: Record the F0 result**

When complete, read `diagnostics/hota_fusion/optimize_F0_best.json`. Record `best_pooled` (= F0 optimum) and `best_x` (= `[coef_m, thr_m, coef_a, thr_a]`). Sanity: `best_static ≥ 43.2` (guardrail held) and `best_pooled ≥ B_seed0` (DE seeded at hand, can only improve modulo eval noise).

---

## Task 5: Gate run F1 (full optimization)

**Files:**
- Produces: `diagnostics/hota_fusion/optimize_F1_log.csv`, `optimize_F1_best.json`

- [ ] **Step 1: Launch the F1 optimization in the background**

Run (background — ~240 evals, ~4–5 h):
```bash
python diagnostics/hota_fusion/optimize_fusion.py --form F1 --maxiter 5 --popsize 5
```
(Launch with `run_in_background: true`. Can run concurrently with Task 4 — distinct `run_dir` (`gateC_F1`) and distinct log files — but sequential keeps the machine responsive; the engineer's call.)

- [ ] **Step 2: Record the F1 result**

When complete, read `diagnostics/hota_fusion/optimize_F1_best.json`. Record `best_pooled` (= F1 optimum) and `best_x` (= `[coef_m, thr_m, β_m, γ_m, coef_a, thr_a, β_a, γ_a]`). Sanity: `best_static ≥ 43.2`; `best_pooled ≥ F0 best_pooled` (F1 superset of F0, modulo DE under-search — if F1 < F0 meaningfully, that's DE under-convergence on 8 dims, note it).

---

## Task 6: Gate verdict + report

**Files:**
- Create: `diagnostics/hota_fusion/gate_c_results.md`

- [ ] **Step 1: Write the results report**

Read `sanity.json` (B_seed0), `optimize_F0_best.json`, `optimize_F1_best.json`. Create `diagnostics/hota_fusion/gate_c_results.md` filling in the measured numbers:

```markdown
# Gate C results — HOTA-direct fusion optimization (iKUN seed0, V1 3-seq pooled)

**Status:** Gate C only — TEST-optimized upper bound, NOT a ship claim.
**Baseline:** seed0 hand recipe `B_seed0 = <sanity B_seed0>` (n=3 ship mean 44.634 is context only).

| form | params | best pooled | Δ vs B_seed0 | best STATIC | evals |
|---|---|---|---|---|---|
| hand (seed0) | coef_m=0.9 thr_m=0.17 coef_a=0.30 thr_a=0.10 | <B_seed0> | — | <F0 static at hand> | — |
| F0 (additive, re-opt) | <best_x F0> | <F0 best_pooled> | <Δ> | <F0 best_static> | <F0 evals> |
| F1 (+native·gmc +gmc²) | <best_x F1> | <F1 best_pooled> | <Δ> | <F1 best_static> | <F1 evals> |

**max(F0, F1) = <max>  →  Δ = <max − B_seed0>**

## Verdict
- [ ] `max(F0,F1) ≈ B_seed0` (within ~+0.1) → **KILL**: additive form is the ceiling; the 4-NEG learned-fusion history was right (the prior NEGs used differentiable F1/MSE surrogates; this is HOTA-direct and still flat). Record as the definitive close of the fusion-form lever.
- [ ] `max(F0,F1) ≥ ~+0.5` → **PROCEED to Gate B** (held-out CV within test) for an honest, non-test-optimized number.

## Attribution
- F0 ≫ hand ⇒ hand recipe was under-searched (a free additive win + flag).
- F0 ≈ hand but F1 ≫ F0 ⇒ the **nonlinear form** (native·gmc interaction / gmc² curvature) is the lever.
- Both ≈ hand ⇒ neither search-quality nor nonlinear form helps → form exhausted.

## Caveats
- Test-optimized (optimizes on the eval seqs) → optimistic by construction; the honest number is Gate B's job.
- DE on 8 dims (F1) in ~240 evals may under-converge; if F1 < F0, treat as under-search, not form-exhaustion.
```

- [ ] **Step 2: Update project memory with the verdict**

Per project feedback ("always document observations and metrics when changing the model"), write a memory file recording the Gate C outcome. Create `/home/seanachan/.claude/projects/-home-seanachan-GMC-Link/memory/project_hota_direct_fusion_gate_c_<date>.md` with frontmatter (`type: project`) summarizing: B_seed0, F0/F1 best pooled + Δ, the verdict (KILL or PROCEED), and the attribution. Add a one-line pointer under `## Project` in `MEMORY.md`. (MEMORY.md is already over its size warning — keep the new index line under ~150 chars.)

- [ ] **Step 3: Commit**

```bash
git add diagnostics/hota_fusion/gate_c_results.md \
        diagnostics/hota_fusion/optimize_F0_best.json \
        diagnostics/hota_fusion/optimize_F1_best.json \
        diagnostics/hota_fusion/optimize_F0_log.csv \
        diagnostics/hota_fusion/optimize_F1_log.csv
git commit -m "results(gateC): HOTA-direct fusion F0/F1 optimization + gate verdict"
```

---

## Self-Review (completed by plan author)

**Spec coverage:**
- Spec §1 fusion forms F0/F1 → Task 1 (f1_extra + gen_predicts wiring) + Task 2 (`_unpack` param layout). ✓
- Spec §2 black-box optimizer (differential_evolution, bounds, STATIC penalty, hand-seed, determinism, CSV) → Task 2 `optimize_fusion.py`. ✓ (seed=0, x0=hand, polish=False, per-eval CSV.)
- Spec §3 in-process eval engine (load once, reuse gen_predicts/run_te) → Task 2 `_load`/`evaluate`. ✓
- Spec Testing 1 (sanity reproduces 44.561) → Task 2 Step 2. ✓
- Spec Testing 2 (F1 β=γ=0 == F0) → Task 1 unit test + Task 2 sanity superset check. ✓
- Spec Testing 3 (optimizer smoke, non-decreasing vs hand, CSV logged) → Task 3. ✓
- Spec Testing 4 (full F0+F1, report max vs baseline + attribution) → Tasks 4–6. ✓
- Spec gate decision / attribution / caveats → Task 6 report template. ✓

**Placeholder scan:** No TBD/TODO. The Task 6 `<...>` are intentional fill-in slots for measured numbers (cannot be known until Tasks 4–5 run), not unspecified logic. ✓

**Type/name consistency:** `f1_extra(native_part, gmc_term, beta, gamma)` signature identical across the unit test, the motion-branch call (`beta_m, gamma_m`), and the appearance-branch call (`beta_a, gamma_a`). `gen_predicts` new kwargs (`fusion_form, beta_m, gamma_m, beta_a, gamma_a`) match `_unpack`'s returned dict keys and the `evaluate` call site. F0 param vector `[coef_m, thr_m, coef_a, thr_a]` and F1 `[coef_m, thr_m, β_m, γ_m, coef_a, thr_a, β_a, γ_a]` are consistent between `BOUNDS`, `X0`, and `_unpack`. ✓

**Deviation from spec (intentional, documented):** spec nominally says "6 params" (F0) / "~10" (F1) counting α and sc separately; this plan collapses the multiplicatively-degenerate α·sc into one `coef` per axis (F0=4, F1=8 effective params). Same reachable score set, fewer redundant search dims → strictly better DE convergence; hand recipe still reproduced exactly. ✓
```
