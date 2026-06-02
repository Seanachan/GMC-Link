"""Step 2 cross-host — does CLIP-L/14 appearance rerank transfer to FlexHook-V2?

iKUN showed rerank-admit (CLIP-L/14, REPLACES native) recovers catastrophic color
exprs (+0.26 color subset). Rerank is host-DECOUPLED (operates on tracker crops, not
host logits) → should transfer even though Path B GMC did NOT. This tests it.

Approach (no edit to the FH V2 sweep): run FH V2 ship via gen_predicts, then OVERWRITE
only the target color-appearance exprs' predict.txt with track-level CLIP-L admits, re-eval.

Run: GMC_SUFFIX=_sharedweight_seed0_rawcos GMC_RAW_COS=1 \
     python -m diagnostics.appearance_rerank.step2_fhv2_rerank
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, "/home/seanachan/GMC-Link")
import run_flexhook_v2_raw_sweep as F

# FH V2 ship recipe (CLAUDE.md, locked)
SHIP = dict(alpha=0.4, gmc_scale=10.0, thr=1.3,
            alpha_a=1.0, scale_a=3.5, thr_a=1.2)
CLIP_TPL = "/home/seanachan/GMC-Link/gmc_link/rerank_clipL14_flexhook_v2_{seq}_cache.json"
COLOR = re.compile(r'silver|light.?color|light.?colou?red|light.?hue|light.?shade', re.I)
TAUS = [0.18, 0.19, 0.20, 0.21, 0.216, 0.225, 0.235]


def target_exprs():
    """seq+expr_id for V2 color exprs classified APPEARANCE (exclude motion-mixed)."""
    out = set()
    for seq in F.TEST_SEQS:
        d = os.path.join(F.DATA_ROOT, "expression", seq)
        for ef in os.listdir(d):
            if not ef.endswith(".json"):
                continue
            eid = ef[:-5]
            if COLOR.search(eid) and F.classify(eid) == "APPEARANCE":
                out.add(f"{seq}+{eid}")
    return out


def write_seqmap(path, keys):
    open(path, "w").write("\n".join(sorted(keys)) + "\n")


def overwrite_targets(res_dir, tracks_by_seq, clip_caches, targets, tau):
    """For each target seq+expr, admit whole tracks with median CLIP-L cosine > tau."""
    for key in targets:
        seq, eid = key.split("+", 1)
        min_f, max_f = F.FRAMES[seq]
        per_expr = clip_caches.get(seq, {}).get(eid, {})
        track_boxes, track_cos = {}, {}
        for row in tracks_by_seq[seq]:
            if not (min_f <= row[0] <= max_f):
                continue
            oid = int(row[1]); fid_pred = int(row[0]) + 1
            bbox = row.copy(); bbox[0] += 1
            track_boxes.setdefault(oid, []).append(",".join(map(str, bbox.tolist())))
            c = per_expr.get(str(fid_pred), {}).get(str(oid))
            if c is not None:
                track_cos.setdefault(oid, []).append(float(c))
        admitted = []
        for oid, boxes in track_boxes.items():
            cl = track_cos.get(oid, [])
            if cl and float(np.median(cl)) > tau:
                admitted.extend(boxes)
        outd = os.path.join(res_dir, seq, eid)
        with open(os.path.join(outd, "predict.txt"), "w") as f:
            if admitted:
                f.write("\n".join(admitted) + "\n")


def parse_summary(res_dir):
    sp = os.path.join(res_dir, "pedestrian_summary.txt")
    if not os.path.exists(sp):
        return {}
    L = open(sp).read().splitlines()
    return {h: float(v) for h, v in zip(L[0].split(), L[1].split())}


def main():
    cls_dict = json.load(open(F.RESULT_JSON))
    gmc_caches = {}
    for s in F.TEST_SEQS:
        cp = F.GMC_CACHE_TPL.format(seq=s)
        if os.path.exists(cp):
            gmc_caches[s] = json.load(open(cp))
    tracks_by_seq = {s: F.load_tracks(s) for s in F.TEST_SEQS}
    clip_caches = {s: json.load(open(CLIP_TPL.format(seq=s))) for s in F.TEST_SEQS}
    targets = target_exprs()
    print(f"FH V2 color-APPEARANCE target exprs: {len(targets)}", flush=True)

    out_root = os.path.join(F.OUT_ROOT, "step2_fhv2")
    os.makedirs(out_root, exist_ok=True)
    sub_sm = os.path.join(out_root, "seqmap_color.txt")

    # Ship baseline
    run_dir = os.path.join(out_root, "ship")
    os.makedirs(run_dir, exist_ok=True)
    res_dir, full_sm = F.gen_predicts(
        cls_dict, tracks_by_seq, gmc_caches, SHIP["alpha"], SHIP["gmc_scale"], SHIP["thr"],
        run_dir, alpha_a=SHIP["alpha_a"], scale_a=SHIP["scale_a"], thr_a=SHIP["thr_a"], mode="ship")
    write_seqmap(sub_sm, targets)
    ship_pool = F.run_te(full_sm, res_dir)
    ship_sub = F.run_te(sub_sm, res_dir)
    print(f"SHIP: pooled={ship_pool:.3f}  color_subset={ship_sub:.3f}", flush=True)

    print(f"{'tau':>8}{'pooled':>10}{'Δpool':>9}{'subset':>9}{'DetRe':>8}{'DetPr':>8}")
    res = []
    for tau in TAUS:
        overwrite_targets(res_dir, tracks_by_seq, clip_caches, targets, tau)
        pool = F.run_te(full_sm, res_dir)
        sub = F.run_te(sub_sm, res_dir); m = parse_summary(res_dir)
        res.append((tau, pool, sub, m.get("DetRe"), m.get("DetPr")))
        print(f"{tau:>8.3f}{pool:>10.3f}{pool-ship_pool:>+9.3f}{sub:>9.3f}"
              f"{(m.get('DetRe') or 0):>8.2f}{(m.get('DetPr') or 0):>8.2f}", flush=True)

    print(f"\nFH V2 ship pooled={ship_pool:.3f} / paper 42.526 / ship-recipe 42.807")
    json.dump({"ship_pool": ship_pool, "ship_sub": ship_sub, "n_targets": len(targets),
               "sweep": [{"tau": t, "pooled": p, "subset": s, "DetRe": dr, "DetPr": dp}
                         for t, p, s, dr, dp in res]},
              open(os.path.join(os.path.dirname(__file__), "step2_fhv2_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
