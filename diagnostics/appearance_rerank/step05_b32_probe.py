"""Step 0.5 — cheapest mechanism probe: re-rank catastrophic exprs by the EXISTING
CLIP-B/32 cosine cache (clip_logit_neuralsort), track-level, admit by tau.

Isolates "does rerank-not-fusion beat catastrophic?" with ZERO new encoder. The
B/32 cache is a DIFFERENT CLIP than iKUN-native (DataComp-XL vs MyCLIP) so it may
carry signal; Exp43 used it ADDITIVELY (vetoed, NEG) — here it REPLACES admit.

Sweeps tau over the observed track-median-cosine distribution. Reports, per tau,
subset HOTA + DetRe/DetPr + pooled, vs ship (~2) and oracle ceiling (~63).

Run: GMC_SUFFIX=_sharedweight_seed0_rawcos GMC_RAW_COS=1 \
     python -m diagnostics.appearance_rerank.step05_b32_probe [color|catastrophic]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, "/home/seanachan/GMC-Link")
import run_ikun_linear_additive as R

SHIP = dict(alpha=1.0, gmc_scale=0.9, thr=0.17, alpha_a=1.0, scale_a=0.30, thr_a=0.10)
CLIP_TPL = "/home/seanachan/GMC-Link/gmc_link/clip_logit_neuralsort_{seq}_cache.json"


def load_set(path):
    return set(l.strip() for l in open(path) if l.strip())


def parse_summary(res_dir):
    """Return {metric: value} from pedestrian_summary.txt (HOTA DetA AssA DetRe DetPr ...)."""
    sp = os.path.join(res_dir, "pedestrian_summary.txt")
    if not os.path.exists(sp):
        return {}
    lines = open(sp).read().splitlines()
    hdr, val = lines[0].split(), lines[1].split()
    return {h: float(v) for h, v in zip(hdr, val)}


def track_median_cosines(clip_caches, target_keys):
    """All per-track median B/32 cosines over target exprs (for tau grid)."""
    vals = []
    for key in target_keys:
        seq, expr = key.split("+", 1)
        per_expr = clip_caches.get(seq, {}).get(expr, {})
        track = {}
        for fid, objs in per_expr.items():
            for oid, c in objs.items():
                track.setdefault(oid, []).append(float(c))
        vals.extend(float(np.median(cs)) for cs in track.values() if cs)
    return np.array(vals)


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "catastrophic"
    sm_file = os.path.join(HERE, f"seqmap_{target}.txt")
    rerank_set = load_set(sm_file)
    print(f"target={target}: {len(rerank_set)} exprs", flush=True)

    text_feat = json.load(open(R.TEXT_FEAT_JSON))
    gmc_caches = {s: json.load(open(R.GMC_CACHE_TPL.format(seq=s))) for s in R.TEST_SEQS}
    clip_caches = {s: json.load(open(CLIP_TPL.format(seq=s))) for s in R.TEST_SEQS}

    cos = track_median_cosines(clip_caches, rerank_set)
    pcts = {p: float(np.percentile(cos, p)) for p in (10, 30, 50, 70, 90)}
    print(f"track-median B/32 cosine dist (n={len(cos)}): "
          f"min={cos.min():.3f} p10={pcts[10]:.3f} p30={pcts[30]:.3f} "
          f"p50={pcts[50]:.3f} p70={pcts[70]:.3f} p90={pcts[90]:.3f} max={cos.max():.3f}", flush=True)

    taus = sorted({round(v, 4) for v in (pcts[10], pcts[30], pcts[50], pcts[70], pcts[90])})
    results = []
    for tau in taus:
        run_dir = os.path.join(R.OUT_ROOT, f"step05_{target}_tau{tau}")
        os.makedirs(run_dir, exist_ok=True)
        res_dir, sm = R.gen_predicts(
            text_feat, gmc_caches, SHIP["alpha"], SHIP["gmc_scale"], SHIP["thr"], run_dir,
            alpha_a=SHIP["alpha_a"], scale_a=SHIP["scale_a"], thr_a=SHIP["thr_a"],
            mode="rerank_clipb32", rerank_set=rerank_set,
            clip_caches=clip_caches, rerank_tau=tau)
        pooled = R.run_te(sm, res_dir)
        sub = R.run_te(sm_file, res_dir); sub_m = parse_summary(res_dir)
        results.append((tau, pooled, sub, sub_m.get("DetRe"), sub_m.get("DetPr")))
        print(f"  tau={tau:.4f}: pooled={pooled:.3f}  subset_HOTA={sub:.3f}  "
              f"DetRe={sub_m.get('DetRe')}  DetPr={sub_m.get('DetPr')}", flush=True)

    print(f"\n=== STEP 0.5 SUMMARY (target={target}) ===")
    print(f"  ship baseline: subset_HOTA~={'2.1' if target=='color' else '1.3'}  oracle ceiling~={'63.5' if target=='color' else '66.4'}")
    print(f"{'tau':>8}{'pooled':>10}{'subsetHOTA':>12}{'DetRe':>9}{'DetPr':>9}")
    for tau, pooled, sub, dre, dpr in results:
        print(f"{tau:>8.4f}{pooled:>10.3f}{sub:>12.3f}{(dre or 0):>9.2f}{(dpr or 0):>9.2f}")
    with open(os.path.join(HERE, f"step05_{target}_results.json"), "w") as f:
        json.dump([{"tau": t, "pooled": p, "subset_hota": s, "DetRe": dre, "DetPr": dpr}
                   for t, p, s, dre, dpr in results], f, indent=2)


if __name__ == "__main__":
    main()
