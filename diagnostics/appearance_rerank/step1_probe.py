"""Step 1 — rerank probe, signal-selectable (clipL14 | hsv | b32).

Same engine as step05 but the cache template is chosen by --signal. tau swept over
percentiles of the observed track-median-score distribution (auto-adapts to each
signal's range). Reports subset HOTA + DetRe/DetPr + pooled per tau.

Run: GMC_SUFFIX=_sharedweight_seed0_rawcos GMC_RAW_COS=1 \
     python -m diagnostics.appearance_rerank.step1_probe --signal clipL14 --target color
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, "/home/seanachan/GMC-Link")
import run_ikun_linear_additive as R

SHIP = dict(alpha=1.0, gmc_scale=0.9, thr=0.17, alpha_a=1.0, scale_a=0.30, thr_a=0.10)
TPL = {
    "b32":     "/home/seanachan/GMC-Link/gmc_link/clip_logit_neuralsort_{seq}_cache.json",
    "clipL14": "/home/seanachan/GMC-Link/gmc_link/rerank_clipL14_neuralsort_{seq}_cache.json",
    "hsv":     "/home/seanachan/GMC-Link/gmc_link/rerank_hsv_neuralsort_{seq}_cache.json",
}


def load_set(p):
    return set(l.strip() for l in open(p) if l.strip())


def parse_summary(res_dir):
    sp = os.path.join(res_dir, "pedestrian_summary.txt")
    if not os.path.exists(sp):
        return {}
    lines = open(sp).read().splitlines()
    return {h: float(v) for h, v in zip(lines[0].split(), lines[1].split())}


def track_medians(caches, keys):
    vals = []
    for key in keys:
        seq, expr = key.split("+", 1)
        pe = caches.get(seq, {}).get(expr, {})
        tr = {}
        for fid, objs in pe.items():
            for oid, c in objs.items():
                tr.setdefault(oid, []).append(float(c))
        vals.extend(float(np.median(cs)) for cs in tr.values() if cs)
    return np.array(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", choices=["b32", "clipL14", "hsv"], required=True)
    ap.add_argument("--target", choices=["color", "catastrophic"], default="color")
    ap.add_argument("--taus", default=None,
                    help="comma-separated explicit tau list (overrides percentile grid)")
    args = ap.parse_args()

    sm_file = os.path.join(HERE, f"seqmap_{args.target}.txt")
    rerank_set = load_set(sm_file)
    text_feat = json.load(open(R.TEXT_FEAT_JSON))
    gmc_caches = {s: json.load(open(R.GMC_CACHE_TPL.format(seq=s))) for s in R.TEST_SEQS}
    caches = {s: json.load(open(TPL[args.signal].format(seq=s))) for s in R.TEST_SEQS}
    print(f"signal={args.signal} target={args.target}: {len(rerank_set)} exprs", flush=True)

    sc = track_medians(caches, rerank_set)
    pcts = {p: float(np.percentile(sc, p)) for p in (10, 30, 50, 70, 90)}
    print(f"track-median {args.signal} dist (n={len(sc)}): min={sc.min():.3f} "
          f"p10={pcts[10]:.3f} p30={pcts[30]:.3f} p50={pcts[50]:.3f} "
          f"p70={pcts[70]:.3f} p90={pcts[90]:.3f} max={sc.max():.3f}", flush=True)

    if args.taus:
        taus = sorted(float(t) for t in args.taus.split(","))
    else:
        taus = sorted({round(v, 4) for v in pcts.values()})
    res = []
    for tau in taus:
        run_dir = os.path.join(R.OUT_ROOT, f"step1_{args.signal}_{args.target}_tau{tau}")
        os.makedirs(run_dir, exist_ok=True)
        res_dir, sm = R.gen_predicts(
            text_feat, gmc_caches, SHIP["alpha"], SHIP["gmc_scale"], SHIP["thr"], run_dir,
            alpha_a=SHIP["alpha_a"], scale_a=SHIP["scale_a"], thr_a=SHIP["thr_a"],
            mode="rerank_clipb32", rerank_set=rerank_set, clip_caches=caches, rerank_tau=tau)
        pooled = R.run_te(sm, res_dir)
        sub = R.run_te(sm_file, res_dir); m = parse_summary(res_dir)
        res.append((tau, pooled, sub, m.get("DetRe"), m.get("DetPr")))
        print(f"  tau={tau:.4f}: pooled={pooled:.3f}  subset_HOTA={sub:.3f}  "
              f"DetRe={m.get('DetRe')}  DetPr={m.get('DetPr')}", flush=True)

    ceil = {"color": 63.5, "catastrophic": 66.4}[args.target]
    base = {"color": 2.1, "catastrophic": 1.3}[args.target]
    print(f"\n=== STEP 1 ({args.signal}, {args.target}) | ship subset~{base} oracle~{ceil} | ship pooled 44.561 ===")
    print(f"{'tau':>9}{'pooled':>10}{'subsetHOTA':>12}{'DetRe':>9}{'DetPr':>9}")
    for tau, p, s, dre, dpr in res:
        print(f"{tau:>9.4f}{p:>10.3f}{s:>12.3f}{(dre or 0):>9.2f}{(dpr or 0):>9.2f}")
    json.dump([{"tau": t, "pooled": p, "subset_hota": s, "DetRe": dre, "DetPr": dpr}
               for t, p, s, dre, dpr in res],
              open(os.path.join(HERE, f"step1_{args.signal}_{args.target}_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
