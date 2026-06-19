"""Direction-relabel of the ego-motion ladder: SAME vs OPPOSITE ego-relative direction.

Reuses ladder_lib's geometry/depth/ego/level machinery VERBATIM. The ONLY change
vs run_ladder.py is the per-(frame,track) label: instead of MOVING/STATIC from
velocity-asserting vs rest-asserting keywords, each instance is labelled by the
ego-relative DIRECTION of its matched (label=1) expression:

  SAME = expression sentence contains 'same direction' / 'same way' / 'same path'
         / 'same-as-ours'
  OPP  = 'counter direction' / 'opposite' / 'opposite-direction' / 'other-direction'
         / 'contrary direction' / 'reverse direction'

Only instances whose matched expression is SAME or OPP are kept. Conflicts (a
(frame,track) referenced by both a SAME and an OPP expression) are excluded,
mirroring run_ladder.py's MOVING/STATIC conflict handling.

KEY contrast = scale-honest L1->L3 delta (both pure pixel, same units). 'opposite
direction' (oncoming) objects have HIGH world-relative speed; 'same direction'
(following) LOW. Raw image-plane velocity (L1) is ego-confounded (a parked car
under ego-motion looks moving); ego-compensated pixel velocity (L3) should recover
the same/opp separation if the ego confound is real.

Read-only: GT bboxes, GT depth cache (DAv2), V1 expressions. No training/model/GPU.

Usage:  python diagnostics/ego_motion_ladder/run_ladder_direction.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import ladder_lib as L

SEQS = ["0005", "0011", "0013"]          # V1 test split (same as run_ladder.py)
PRIMARY_GAP = 5
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "ego_motion_ladder")

FX, FY, CX, CY = L.INTRINSICS


# ── direction label from expression text ──────────────────────────────────

SAME_KEYS = ["same direction", "same-direction", "same way", "same path",
             "same-as-ours", "same as ours"]
OPP_KEYS = ["counter direction", "counter-direction", "opposite direction",
            "opposite-direction", "opposite", "other direction",
            "other-direction", "contrary direction", "reverse direction"]


def direction_label(sentence: str) -> Optional[str]:
    """'SAME' / 'OPP' / None. None if neither or (defensively) both present."""
    s = sentence.lower()
    has_same = any(k in s for k in SAME_KEYS)
    has_opp = any(k in s for k in OPP_KEYS)
    if has_same and not has_opp:
        return "SAME"
    if has_opp and not has_same:
        return "OPP"
    return None  # neither (e.g. 'horizon direction'), or ambiguous both


def load_direction_referents(seq: str):
    """List of (sentence, dir_label, {frame_id: set(track_ids)}) for SAME/OPP exprs.

    Mirrors ladder_lib.load_motion_referents but keyed on direction_label, and
    reads ALL expression JSONs (not only motion-state ones).
    """
    import glob
    out = []
    for jf in sorted(glob.glob(os.path.join(L.V1_EXPR_DIR, seq, "*.json"))):
        j = json.load(open(jf))
        sent = j["sentence"]
        dlab = direction_label(sent)
        if dlab is None:
            continue
        refs: Dict[int, set] = {}
        for fr, objs in j["label"].items():
            refs.setdefault(int(fr), set()).update(int(o) for o in objs)
        out.append((sent, dlab, refs))
    return out


# ── instance construction (geometry/depth/ego identical to run_ladder.py) ──

def build_instances(seq: str, gap: int):
    geom = L.load_geometry(seq)
    depth = L.load_depth(seq)
    referents = load_direction_referents(seq)

    # Aggregate direction label per (frame, track); drop SAME/OPP conflicts.
    states: Dict[tuple, set] = defaultdict(set)
    sents: Dict[tuple, set] = defaultdict(set)
    for _sent, dlab, refs in referents:
        for fid, tids in refs.items():
            for tid in tids:
                states[(fid, tid)].add(dlab)
                sents[(fid, tid)].add(_sent)

    # Per-frame ego over ALL tracks present at both t and t-gap with depth
    # (identical estimator to run_ladder.py — NOT restricted to direction tracks).
    ego_by_frame: Dict[int, tuple] = {}
    n_frames_unreliable = 0
    for fid, frame in geom.items():
        prev = geom.get(fid - gap)
        if prev is None:
            continue
        vels, dzs = {}, {}
        for tid, (cx, cy, _w, _h) in frame.items():
            if tid not in prev:
                continue
            if tid not in depth or fid not in depth[tid] or (fid - gap) not in depth[tid]:
                continue
            pcx, pcy, _pw, _ph = prev[tid]
            vels[tid] = ((cx - pcx) / gap, (cy - pcy) / gap)
            dzs[tid] = (depth[tid][fid] - depth[tid][fid - gap]) / gap
        if len(vels) >= 3:
            ego_by_frame[fid] = L.estimate_ego(vels, dzs)
        else:
            n_frames_unreliable += 1

    out = []
    for (fid, tid), st in states.items():
        if len(st) != 1:                       # SAME+OPP conflict -> exclude
            continue
        label = next(iter(st))                 # 'SAME' or 'OPP'
        if fid not in ego_by_frame:
            continue
        prev = geom.get(fid - gap)
        if prev is None or tid not in geom.get(fid, {}) or tid not in prev:
            continue
        if tid not in depth or fid not in depth[tid] or (fid - gap) not in depth[tid]:
            continue
        cx, cy, w, h = geom[fid][tid]
        pcx, pcy, _pw, _ph = prev[tid]
        v_pix = ((cx - pcx) / gap, (cy - pcy) / gap)
        dz = (depth[tid][fid] - depth[tid][fid - gap]) / gap
        lv = L.compute_levels(v_pix, depth[tid][fid], dz, ego_by_frame[fid], FX, FY)
        out.append({
            "seq": seq, "frame": fid, "track": tid, "label": label,
            "bbox_h": h,
            "L1": lv["L1"], "L2": lv["L2"], "L3": lv["L3"], "L4": lv["L4"],
            "v_metric_x": lv["v_metric_x"], "v_metric_z": lv["v_metric_z"],
            "dz_res": lv["dz_res"],
            # static control features (depth/scale/layout confound exposure)
            "z_t": depth[tid][fid], "cx_abs": abs(cx - CX), "bbox_area": w * h,
            "cx": cx,
        })
    return out, n_frames_unreliable


# ── AUC: SAME=positive (HIGH world speed expected), OPP would be flipped ────
# We want a direction-discrimination AUC. Convention: label OPP=1 (oncoming, HIGH
# world-relative speed) vs SAME=0 (following, LOW). Then a feature that ranks
# oncoming higher gives AUC>0.5. AUC is symmetric: AUC(SAME=1) = 1-AUC(OPP=1), so
# the |0.5-AUC| separability is convention-free; we report the OPP=1 orientation
# (physically: oncoming should have larger residual/metric velocity).

def auc_dir(rows: List[dict], feat: str) -> Optional[float]:
    labels = [1 if r["label"] == "OPP" else 0 for r in rows]
    return L.safe_auc(labels, [r[feat] for r in rows])


def auc_ladder(rows: List[dict]) -> dict:
    labels = [1 if r["label"] == "OPP" else 0 for r in rows]
    out = {"n": len(rows), "n_opp": sum(labels), "n_same": len(labels) - sum(labels)}
    for lvl in ("L1", "L2", "L3", "L4"):
        out[lvl] = auc_dir(rows, lvl)
    return out


def control_auc(rows: List[dict]) -> dict:
    return {
        "z_t": auc_dir(rows, "z_t"),
        "bbox_area": auc_dir(rows, "bbox_area"),
        "cx_abs": auc_dir(rows, "cx_abs"),
        "cx": auc_dir(rows, "cx"),
    }


def mag_stats(rows: List[dict]) -> dict:
    out = {}
    for lvl in ("L1", "L2", "L3", "L4"):
        v = np.array([r[lvl] for r in rows], dtype=float)
        out[lvl] = {
            "median": float(np.median(v)) if len(v) else None,
            "p25": float(np.percentile(v, 25)) if len(v) else None,
            "p75": float(np.percentile(v, 75)) if len(v) else None,
        }
    return out


def main():
    rows: List[dict] = []
    unreliable = 0
    for seq in SEQS:
        r, nu = build_instances(seq, PRIMARY_GAP)
        rows.extend(r)
        unreliable += nu

    n_same = sum(1 for r in rows if r["label"] == "SAME")
    n_opp = sum(1 for r in rows if r["label"] == "OPP")

    a = auc_ladder(rows)
    ca = control_auc(rows)

    report = {
        "config": {"seqs": SEQS, "primary_gap": PRIMARY_GAP,
                   "n_instances": len(rows), "n_same": n_same, "n_opp": n_opp,
                   "n_frames_skipped_ego": unreliable,
                   "auc_orientation": "OPP=positive (oncoming, HIGH world speed)"},
        "auc_ladder_same_vs_opp": a,
        "control_auc_same_vs_opp": ca,
        "mag": mag_stats(rows),
        "L1_to_L3_delta": (a["L3"] - a["L1"]) if (a["L1"] is not None and a["L3"] is not None) else None,
        "per_seq_counts": {seq: {
            "n_same": sum(1 for r in rows if r["seq"] == seq and r["label"] == "SAME"),
            "n_opp": sum(1 for r in rows if r["seq"] == seq and r["label"] == "OPP"),
        } for seq in SEQS},
        "published_moving_static": {"L1": 0.5121311157166856, "L3": 0.5147016878309805,
                                    "L1_to_L3_delta": 0.5147016878309805 - 0.5121311157166856},
    }

    print(json.dumps(report, indent=2))

    with open(os.path.join(OUT_DIR, "ladder_direction_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(OUT_DIR, "ladder_direction_per_instance.json"), "w") as f:
        json.dump(rows, f)
    print(f"\n[ladder-dir] wrote artifacts to {OUT_DIR}")
    return report


if __name__ == "__main__":
    main()
