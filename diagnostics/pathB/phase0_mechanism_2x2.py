"""Path B Phase-0 mechanism test (2x2): does cohort dz_ego inject STATIC residual?

For world-static (parked) vehicles, dz_residual SHOULD be ~0. We compute it under
2 depth sources x 2 ego sources:

  dz_track in {GT (label_02 3D Z), DAv2 (z_track_gt cache, IoU-matched)}
  dz_ego   in {cohort (median DAv2 dZ, the ship), oxts (GT ego, the swap)}

Predictions (bias-cancellation aware):
  GT x oxts    -> ~0   (both GT, consistent)            <- Path B preview
  GT x cohort  -> large (GT depth vs DAv2 ego mismatch) <- the STATIC -0.47 mechanism
  DAv2 x cohort-> ~0   (DAv2 self-cancel)               <- negative-control artifact
  DAv2 x oxts  -> large (DAv2 depth vs GT ego mismatch)

Gate (user): GT-row cohort |dz_res| median > 2x GT-row oxts -> mechanism confirmed.
Velodyne NOT required (uses label_02 GT depth). No training.
"""
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "gmc_link"))
import kitti_tracking_gt as G

TRK = "/home/seanachan/data/Dataset/kitti_tracking/training"
LWIDS = "/home/seanachan/data/Dataset/refer-kitti/KITTI/labels_with_ids/image_02"
DEPTH = os.path.join(os.path.dirname(__file__), "..", "..", "gmc_link", "depth_cache")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "pathB")
SEQS = ["0005", "0011", "0013"]
GAP, DT = 5, 0.1
W, H = 1242, 375
STATIC_WORLD_SPEED = 0.5   # m/s; below = parked


def load_label02(seq):
    """tid -> frame -> dict(box=(x1,y1,x2,y2), Z=cam-Z, xyz=(X,Y,Z))."""
    out = collections.defaultdict(dict)
    for ln in open(os.path.join(TRK, "label_02", f"{seq}.txt")):
        p = ln.split()
        fr, tid, cls = int(p[0]), int(p[1]), p[2]
        if tid < 0 or cls not in ("Car", "Van", "Truck"):
            continue
        x1, y1, x2, y2 = map(float, p[6:10])
        X, Y, Z = float(p[13]), float(p[14]), float(p[15])
        out[tid][fr] = {"box": (x1, y1, x2, y2), "Z": Z, "xyz": (X, Y, Z)}
    return out


def load_lwids_boxes(seq):
    """frame -> list of (tid, (x1,y1,x2,y2) px) from refer-kitti labels_with_ids."""
    out = collections.defaultdict(list)
    for f in glob.glob(os.path.join(LWIDS, seq, "*.txt")):
        fr = int(os.path.basename(f)[:-4])
        for ln in open(f):
            p = ln.split()
            if len(p) < 6:
                continue
            tid = int(p[1]); x1n, y1n, wn, hn = map(float, p[2:6])
            out[fr].append((tid, (x1n * W, y1n * H, (x1n + wn) * W, (y1n + hn) * H)))
    return out


def load_dav2(seq):
    raw = json.load(open(os.path.join(DEPTH, f"z_track_gt_{seq}.json")))
    return {int(t): {int(f): float(z) for f, z in fr.items()} for t, fr in raw.items()}


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match_dav2_id(box, lw_frame):
    """Best IoU>0.5 labels_with_ids tid for a label_02 box in this frame, else None."""
    best, best_iou = None, 0.5
    for tid, lwbox in lw_frame:
        v = iou(box, lwbox)
        if v > best_iou:
            best, best_iou = tid, v
    return best


def zdav2(dav2, tid, img_i):
    """DAv2 Z at 0-based IMAGE index img_i. Cache is keyed 1-based (f1=f0+1)."""
    return dav2.get(tid, {}).get(img_i + 1)


def cohort_dz(dav2, gap):
    """IMAGE-index (0-based) -> median DAv2 dZ over all cache tracks (ship cohort
    proxy). Keyed in image index to match label_02/poses (cache is 1-based)."""
    img_frames = sorted({k - 1 for tr in dav2.values() for k in tr})
    out = {}
    for t in img_frames:
        dz = [zdav2(dav2, tid, t) - zdav2(dav2, tid, t - gap)
              for tid in dav2
              if zdav2(dav2, tid, t) is not None and zdav2(dav2, tid, t - gap) is not None]
        if len(dz) >= 3:
            out[t] = float(np.median(dz))
    return out


def world_static_ids(l2, poses, calib):
    """label_02 tids whose median world-frame speed < STATIC_WORLD_SPEED (parked)."""
    static = set()
    for tid, frames in l2.items():
        fs = sorted(frames)
        if len(fs) < 6:
            continue
        wpos = {}
        for f in fs:
            if f >= len(poses):
                continue
            Xc = np.array([*frames[f]["xyz"], 1.0])
            wpos[f] = (G.cam_to_world(calib, poses[f]) @ Xc)[:3]
        speeds = [np.linalg.norm(wpos[b] - wpos[a]) / (DT * (b - a))
                  for a, b in zip(fs, fs[1:]) if a in wpos and b in wpos and b > a]
        if speeds and np.median(speeds) < STATIC_WORLD_SPEED:
            static.add(tid)
    return static


def stats(vals):
    if len(vals) == 0:
        return None
    a = np.abs(vals)
    return {"n": len(a), "median": float(np.median(a)),
            "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75))}


def main():
    os.makedirs(OUT, exist_ok=True)
    cells = {k: [] for k in ("GT_oxts", "GT_cohort", "DAv2_oxts", "DAv2_cohort")}
    dav2_vs_gt_Z = []        # depth accuracy after IoU match
    n_static_total = 0

    for seq in SEQS:
        l2 = load_label02(seq)
        lw = load_lwids_boxes(seq)
        dav2 = load_dav2(seq)
        oxts = G.load_oxts(seq); poses = G.oxts_to_poses(oxts); calib = G.load_calib(seq)
        cdz = cohort_dz(dav2, GAP)
        static = world_static_ids(l2, poses, calib)
        n_static_total += len(static)

        for tid in static:
            for t in sorted(l2[tid]):
                if (t - GAP) not in l2[tid] or t not in cdz or t - GAP < 0 or t >= len(poses):
                    continue
                dz_gt = l2[tid][t]["Z"] - l2[tid][t - GAP]["Z"]
                ego_cohort = cdz[t]
                ego_oxts = G.ego_dz_camera(poses, calib, t, GAP)
                cells["GT_oxts"].append(dz_gt - ego_oxts)
                cells["GT_cohort"].append(dz_gt - ego_cohort)
                # DAv2 arm: IoU-match this box at t and t-GAP to a cache track
                m_t = match_dav2_id(l2[tid][t]["box"], lw.get(t, []))
                m_p = match_dav2_id(l2[tid][t - GAP]["box"], lw.get(t - GAP, []))
                if (m_t is not None and m_t == m_p
                        and zdav2(dav2, m_t, t) is not None
                        and zdav2(dav2, m_t, t - GAP) is not None):
                    dz_dav2 = zdav2(dav2, m_t, t) - zdav2(dav2, m_t, t - GAP)
                    cells["DAv2_oxts"].append(dz_dav2 - ego_oxts)
                    cells["DAv2_cohort"].append(dz_dav2 - ego_cohort)
                    dav2_vs_gt_Z.append((zdav2(dav2, m_t, t), l2[tid][t]["Z"]))

    report = {"config": {"seqs": SEQS, "gap": GAP, "static_world_speed_mps": STATIC_WORLD_SPEED,
                         "n_static_tracks": n_static_total},
              "cells": {k: stats(np.array(v)) for k, v in cells.items()}}

    # depth accuracy
    if dav2_vs_gt_Z:
        arr = np.array(dav2_vs_gt_Z)
        ratio = arr[:, 0] / np.clip(arr[:, 1], 1e-3, None)
        report["dav2_vs_gt_depth"] = {"n": len(arr), "ratio_median": float(np.median(ratio)),
                                      "ratio_p25": float(np.percentile(ratio, 25)),
                                      "ratio_p75": float(np.percentile(ratio, 75))}

    # gate
    g = report["cells"]
    if g["GT_oxts"] and g["GT_cohort"]:
        r = g["GT_cohort"]["median"] / max(g["GT_oxts"]["median"], 1e-6)
        report["gate"] = {
            "GT_cohort_over_GT_oxts": r,
            "verdict": ("MECHANISM CONFIRMED (>2x): cohort ego injects STATIC residual; "
                        "oxts fixes it -> Path B premise supported" if r > 2 else
                        "MECHANISM WEAK (<2x): flag to user before Option 1")}

    with open(os.path.join(OUT, "mechanism_2x2.json"), "w") as f:
        json.dump(report, f, indent=2)

    # histogram of |dz_res| per cell
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        bins = np.linspace(0, 2.0, 41)
        order = [("GT_oxts", "GT depth + oxts ego  (Path B target)"),
                 ("GT_cohort", "GT depth + cohort ego  (STATIC mechanism)"),
                 ("DAv2_cohort", "DAv2 + cohort ego  (current ship)"),
                 ("DAv2_oxts", "DAv2 + oxts ego  (mismatch)")]
        for k, lab in order:
            if cells[k]:
                a = np.abs(np.array(cells[k]))
                ax.hist(np.clip(a, 0, 2), bins=bins, histtype="step", lw=2,
                        label=f"{lab}  (med {np.median(a):.3f})")
        ax.set_xlabel("|dz_residual| on parked vehicles  (m / 5-frame gap)")
        ax.set_ylabel("count")
        ax.set_title("Path B mechanism: cohort ego injects STATIC depth-residual; oxts removes it")
        ax.legend(fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "mechanism_2x2.png"), dpi=130)
        plt.close(fig)
    except Exception as e:
        print("plot skipped:", e)

    # print
    print(f"static parked vehicle tracks: {n_static_total}  (instances per cell below)")
    print(f"{'cell':<14}{'n':>6}{'|dz_res| med':>14}{'p25':>9}{'p75':>9}")
    for k in ("GT_oxts", "GT_cohort", "DAv2_cohort", "DAv2_oxts"):
        s = report["cells"][k]
        if s:
            print(f"{k:<14}{s['n']:>6}{s['median']:>14.3f}{s['p25']:>9.3f}{s['p75']:>9.3f}")
    if "dav2_vs_gt_depth" in report:
        d = report["dav2_vs_gt_depth"]
        print(f"\nDAv2/GT depth ratio (IoU-matched): median {d['ratio_median']:.2f} "
              f"[{d['ratio_p25']:.2f}, {d['ratio_p75']:.2f}]  (1.0 = accurate)")
    if "gate" in report:
        print(f"\nGATE GT_cohort/GT_oxts = {report['gate']['GT_cohort_over_GT_oxts']:.2f}")
        print(report["gate"]["verdict"])
    print(f"\nwrote {OUT}/mechanism_2x2.json")


if __name__ == "__main__":
    main()
