"""autoresearch eval — APPEARANCE re-rank (phase 3).

Single-config full-pooled HOTA for the CLIP-L/14 + spatial-gate appearance
re-ranker stacked on the iKUN ship GMC caches. The re-ranked expression set is
read from `seqmap_active.txt` — THE optimization target the loop edits
(subset-expansion: which appearance exprs benefit from CLIP-L/14 rerank).

Mechanism (run_ikun_linear_additive.gen_predicts mode="rerank_clipb32"):
track-median CLIP-L/14 cosine > tau REPLACES native admit on listed exprs
(non-additive => dodges iKUN native-veto). RERANK_SPATIAL gates admits by
centroid side (cx) for left/right exprs.

Uses PRE-BUILT ship GMC caches (no retrain/cache-build) => ~1-2 min/iter.
Prints `pooled_hota: <full 3-seq pooled HOTA>`. Higher is better; FP-flood
regressions show directly in pooled (no separate guardrail).
"""
import os
import sys

# run_ikun reads these at import — must set BEFORE importing it.
os.environ.setdefault("GMC_SUFFIX", "_sharedweight_seed0_rawcos")
os.environ.setdefault("GMC_RAW_COS", "1")
os.environ["RERANK_SPATIAL"] = "1"  # cx-side spatial gate (the +0.690 lever)

import json

sys.path.insert(0, "/home/seanachan/GMC-Link")
import run_ikun_linear_additive as R

ACTIVE = "/home/seanachan/GMC-Link/diagnostics/appearance_rerank/seqmap_active.txt"
TPL_L14 = "/home/seanachan/GMC-Link/gmc_link/rerank_clipL14_neuralsort_{seq}_cache.json"
# iKUN ship recipe (locked) + posqual-era rerank threshold.
SHIP = dict(alpha=1.0, gmc_scale=0.9, thr=0.17, alpha_a=1.0, scale_a=0.30, thr_a=0.10)
TAU = float(os.environ.get("RERANK_TAU", "0.205"))


def main():
    rerank_set = set(l.strip() for l in open(ACTIVE) if l.strip())
    text_feat = json.load(open(R.TEXT_FEAT_JSON))
    gmc_caches = {s: json.load(open(R.GMC_CACHE_TPL.format(seq=s))) for s in R.TEST_SEQS}
    clip_caches = {s: json.load(open(TPL_L14.format(seq=s))) for s in R.TEST_SEQS}
    print(f"rerank_set={len(rerank_set)} exprs  tau={TAU}  spatial={os.environ['RERANK_SPATIAL']}",
          flush=True)

    run_dir = os.path.join(R.OUT_ROOT, "ar_appearance_rerank")
    os.makedirs(run_dir, exist_ok=True)
    res_dir, sm = R.gen_predicts(
        text_feat, gmc_caches, SHIP["alpha"], SHIP["gmc_scale"], SHIP["thr"], run_dir,
        alpha_a=SHIP["alpha_a"], scale_a=SHIP["scale_a"], thr_a=SHIP["thr_a"],
        mode="rerank_clipb32", rerank_set=rerank_set, clip_caches=clip_caches, rerank_tau=TAU)
    pooled = R.run_te(sm, res_dir)
    if pooled is None:
        print("pooled_hota: -100.0  (run_te returned None — eval failed)")
        return
    print(f"pooled_hota: {pooled:.4f}")


if __name__ == "__main__":
    main()
