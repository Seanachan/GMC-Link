"""Step 0 — size the POOLED prize + GT sanity for the APPEARANCE re-ranker.

Reuses run_ikun_linear_additive.gen_predicts/run_te. For each mode we generate a
full predict set, then eval HOTA on three seqmaps: full (pooled), the color subset,
and the catastrophic subset.

Modes:
  ship                         — baseline (current ship recipe), no override.
  oracle_appear + rerank_set   — admit IoU>=0.5 GT on ONLY the listed exprs (ship
                                 elsewhere). Restricting oracle to color / catastrophic
                                 isolates "fix ONLY these exprs" pooled headroom.

Two reads:
  - POOLED Δ (oracle_subset − ship): the benchmark prize of perfectly fixing the subset.
  - subset HOTA under oracle: GT sanity. High => GT clean+separable, worth chasing.
    Low => GT itself sparse/noisy => no reranker recovers it (kill).

Ship recipe locked (CLAUDE.md). Run with:
  GMC_SUFFIX=_sharedweight_seed0_rawcos GMC_RAW_COS=1 python -m diagnostics.appearance_rerank.step0_size_prize
"""
import json
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, "/home/seanachan/GMC-Link")
import run_ikun_linear_additive as R

SHIP = dict(alpha=1.0, gmc_scale=0.9, thr=0.17,
            alpha_a=1.0, scale_a=0.30, thr_a=0.10)
COLOR_SM = os.path.join(HERE, "seqmap_color.txt")
CAT_SM = os.path.join(HERE, "seqmap_catastrophic.txt")


def load_set(path):
    return set(l.strip() for l in open(path) if l.strip())


def evaluate(tag, mode, rerank_set, text_feat, gmc_caches):
    """Generate predicts then eval pooled + color-subset + catastrophic-subset HOTA."""
    run_dir = os.path.join(R.OUT_ROOT, f"step0_{tag}")
    os.makedirs(run_dir, exist_ok=True)
    res_dir, sm = R.gen_predicts(
        text_feat, gmc_caches, SHIP["alpha"], SHIP["gmc_scale"], SHIP["thr"], run_dir,
        alpha_a=SHIP["alpha_a"], scale_a=SHIP["scale_a"], thr_a=SHIP["thr_a"],
        mode=mode, rerank_set=rerank_set)
    pooled = R.run_te(sm, res_dir)
    color = R.run_te(COLOR_SM, res_dir)
    cat = R.run_te(CAT_SM, res_dir)
    return {"pooled": pooled, "color_subset": color, "catastrophic_subset": cat}


def main():
    print("Loading text_feat + GMC caches...", flush=True)
    text_feat = json.load(open(R.TEXT_FEAT_JSON))
    gmc_caches = {s: json.load(open(R.GMC_CACHE_TPL.format(seq=s))) for s in R.TEST_SEQS}
    color_set = load_set(COLOR_SM)
    cat_set = load_set(CAT_SM)
    print(f"color subset: {len(color_set)} exprs | catastrophic: {len(cat_set)} exprs", flush=True)

    runs = [
        ("ship", "ship", None),
        ("oracle_color", "oracle_appear", color_set),
        ("oracle_catastrophic", "oracle_appear", cat_set),
    ]
    out = {}
    for tag, mode, rset in runs:
        print(f"\n=== {tag} (mode={mode}, restrict={len(rset) if rset else 'none'}) ===", flush=True)
        out[tag] = evaluate(tag, mode, rset, text_feat, gmc_caches)
        print(f"  {out[tag]}", flush=True)

    ship = out["ship"]
    print("\n=== STEP 0 SUMMARY ===")
    print(f"{'run':<22}{'pooled':>10}{'color_sub':>12}{'catas_sub':>12}")
    for tag in ("ship", "oracle_color", "oracle_catastrophic"):
        r = out[tag]
        print(f"{tag:<22}{r['pooled']:>10.3f}{r['color_subset']:>12.3f}{r['catastrophic_subset']:>12.3f}")
    print("\n--- prize (pooled Δ vs ship) ---")
    print(f"  fix color-only:        {out['oracle_color']['pooled'] - ship['pooled']:+.3f}")
    print(f"  fix catastrophic-only: {out['oracle_catastrophic']['pooled'] - ship['pooled']:+.3f}")
    print("--- GT sanity (subset HOTA: ship -> oracle) ---")
    print(f"  color subset:        {ship['color_subset']:.3f} -> {out['oracle_color']['color_subset']:.3f}")
    print(f"  catastrophic subset: {ship['catastrophic_subset']:.3f} -> {out['oracle_catastrophic']['catastrophic_subset']:.3f}")
    print("\nGATE: proceed iff pooled Δ(catastrophic) >= ~+1.0 AND oracle subset HOTA high (GT clean).")
    with open(os.path.join(HERE, "step0_results.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
