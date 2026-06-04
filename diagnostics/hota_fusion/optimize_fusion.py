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
