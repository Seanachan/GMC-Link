"""Path B Phase-0 depth-half sanity: LiDAR per-bbox depth pipeline checks.

Two DISTINCT questions (do not conflate):
  - PIPELINE CORRECT? (this script) -> static-vehicle consistency, scale, hole rate.
  - LiDAR vs DAv2 QUALITY? -> already answered by the 2x2 mechanism test; not re-litigated.

Frame convention (matches run_build_depth_cache_lidar.py == DAv2 builder):
  cache key f1 = f0+1; box from labels file f1; sensor (velodyne/DAv2) from image f0 = f1-1.
  labels/label_02/poses are 0-based image index; DAv2 cache is 1-based (z = dav2[tid][i+1] for image i).

Measurements (primary):
  1. LiDAR median Z vs DAv2 Z per gt-bbox: ratio + abs-diff distribution (0005, 0011).
  2. Hole rate: overall + per bbox-size (small<50px / med / large) + per distance bucket.
  3. STATIC consistency: world-static vehicles, |dZ_lidar - ego_dz_oxts| (~0 if correct).
     Computed ALIGNED (velo f, box f) and STALE (velo f, box f+1 = builder quirk) to test if
     the box/sensor staleness harms LiDAR (sparse) — resolves the alignment fork empirically.
Secondary: LiDAR median vs label_02 3D-center Z ratio (~10% differential acceptable).

PRE-COMMITTED hole-rate policy applied at end. Untested on velodyne; exits cleanly if absent.
"""
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "gmc_link"))
sys.path.insert(0, HERE)
import kitti_tracking_gt as K
from phase0_mechanism_2x2 import load_label02, load_lwids_boxes, load_dav2, world_static_ids

OUT = os.path.join(HERE, "..", "results", "pathB")
SEQS = ["0005", "0011"]
MIN_PTS = 1
SMALL_H, MED_H = 50.0, 100.0          # bbox height px buckets
DIST = [(0, 15), (15, 35), (35, 1e9)]  # near / mid / far (m)


def size_bucket(h):
    return "small" if h < SMALL_H else ("medium" if h < MED_H else "large")


def dist_bucket(z):
    for lo, hi in DIST:
        if lo <= z < hi:
            return f"{lo}-{hi if hi < 1e9 else 'inf'}m"
    return "na"


def hole_policy(overall, small):
    if overall < 0.10 and small < 0.20:
        return "(a) leave holes, SINGLE training run"
    if overall > 0.30 or small > 0.50:
        return "PAUSE: reconsider Path B feasibility (do NOT full-build; flag user)"
    return "(a)+(c): TWO runs (leave-holes + LiDAR-covered-only), paper reports both"


def main():
    os.makedirs(OUT, exist_ok=True)
    # velodyne presence guard
    missing = [s for s in SEQS if not os.path.isdir(os.path.join(K.TRACKING_ROOT, "velodyne", s))]
    if missing:
        print(f"[sanity] velodyne not extracted for {missing} -> "
              f"expected at {K.TRACKING_ROOT}/velodyne/<seq>/. Exiting (run after extraction).")
        return

    import cv2
    ratios, absdiffs = [], []                       # LiDAR vs DAv2
    lidar_vs_l2 = []                                # LiDAR vs label_02 3D Z
    consist = {"aligned": [], "stale": []}          # |dZ_lidar - ego|
    raw_dz = {"aligned": [], "stale": []}           # raw |dZ_lidar| (tracks ego, NOT ~0)
    holes = collections.Counter(); boxes = collections.Counter()       # by size
    holes_d = collections.Counter(); boxes_d = collections.Counter()   # by distance
    n_box_total = n_hole_total = 0

    for seq in SEQS:
        lw = load_lwids_boxes(seq)                  # image-idx(0-based) -> [(tid, box)]
        dav2 = load_dav2(seq)
        l2 = load_label02(seq)
        poses = K.oxts_to_poses(K.load_oxts(seq)); calib = K.load_calib(seq)
        static = world_static_ids(l2, poses, calib)
        frame_dir = f"/home/seanachan/data/Dataset/refer-kitti/KITTI/training/image_02/{seq}"
        total = len([f for f in os.listdir(frame_dir) if f.endswith((".png", ".jpg"))])
        slz = collections.defaultdict(dict)         # static aligned lidar Z by image idx
        slz_stale = collections.defaultdict(dict)

        for f0 in range(total):
            f1 = f0 + 1
            try:
                pts = K.load_velodyne(seq, f0)[:, :3]
            except FileNotFoundError:
                continue
            uv, z = K.project_velo_to_image(pts, calib)

            # (1)+(2) gt-bbox LiDAR vs DAv2 + hole stats — BUILDER convention (box=labels[f1])
            for tid, box in lw.get(f1, []):
                h = box[3] - box[1]
                res = K.bbox_lidar_depth(uv, z, box, MIN_PTS)
                dav2Z = dav2.get(tid, {}).get(f1)
                sb = size_bucket(h)
                boxes[sb] += 1; n_box_total += 1
                zb = dist_bucket(dav2Z if dav2Z else (res[0] if res else 1e9))
                boxes_d[zb] += 1
                if res is None:
                    holes[sb] += 1; holes_d[zb] += 1; n_hole_total += 1
                elif dav2Z:
                    ratios.append(res[0] / dav2Z); absdiffs.append(res[0] - dav2Z)

            # (3)+secondary: static vehicles, aligned + stale sampling
            for tid in static:
                if f0 in l2[tid]:
                    r = K.bbox_lidar_depth(uv, z, l2[tid][f0]["box"], MIN_PTS)
                    if r:
                        slz[tid][f0] = r[0]
                        lidar_vs_l2.append(r[0] / max(l2[tid][f0]["Z"], 1e-3))
                if (f0 + 1) in l2[tid]:              # stale: next-frame box on this cloud
                    r2 = K.bbox_lidar_depth(uv, z, l2[tid][f0 + 1]["box"], MIN_PTS)
                    if r2:
                        slz_stale[tid][f0] = r2[0]

        # consistency: consecutive aligned/stale dZ vs oxts ego (gap=1)
        for store, key in ((slz, "aligned"), (slz_stale, "stale")):
            for tid, zmap in store.items():
                fs = sorted(zmap)
                for a, b in zip(fs, fs[1:]):
                    if b == a + 1:
                        dz = zmap[b] - zmap[a]
                        ego = K.ego_dz_camera(poses, calib, b, 1)
                        consist[key].append(abs(dz - ego)); raw_dz[key].append(abs(dz))

    def st(x):
        a = np.abs(np.array(x)) if x else np.array([])
        return None if a.size == 0 else {"n": int(a.size), "median": float(np.median(a)),
                                         "p25": float(np.percentile(a, 25)),
                                         "p75": float(np.percentile(a, 75))}

    overall_hole = n_hole_total / max(1, n_box_total)
    small_hole = holes["small"] / max(1, boxes["small"])
    report = {
        "config": {"seqs": SEQS, "min_pts": MIN_PTS},
        "lidar_vs_dav2_ratio": st(ratios), "lidar_vs_dav2_absdiff_m": st(absdiffs),
        "lidar_vs_label02_3dZ_ratio": st(lidar_vs_l2),
        "static_consistency_abs_dz_minus_ego": {k: st(v) for k, v in consist.items()},
        "static_raw_abs_dz": {k: st(v) for k, v in raw_dz.items()},
        "hole_rate": {
            "overall": overall_hole,
            "by_size": {k: holes[k] / max(1, boxes[k]) for k in boxes},
            "by_distance": {k: holes_d[k] / max(1, boxes_d[k]) for k in boxes_d},
            "n_box": n_box_total, "n_hole": n_hole_total},
        "policy_decision": {"overall_hole": overall_hole, "small_bbox_hole": small_hole,
                            "policy": hole_policy(overall_hole, small_hole)},
    }
    with open(os.path.join(OUT, "depth_half_sanity.json"), "w") as f:
        json.dump(report, f, indent=2)

    # print summary
    print(f"hole rate overall {overall_hole:.3f}  small-bbox {small_hole:.3f}")
    print("by size:", {k: round(v, 3) for k, v in report["hole_rate"]["by_size"].items()})
    print("by dist:", {k: round(v, 3) for k, v in report["hole_rate"]["by_distance"].items()})
    r = report["lidar_vs_dav2_ratio"]
    if r: print(f"LiDAR/DAv2 Z ratio: med {r['median']:.2f} [{r['p25']:.2f},{r['p75']:.2f}]")
    l = report["lidar_vs_label02_3dZ_ratio"]
    if l: print(f"LiDAR/label02 Z ratio: med {l['median']:.2f} [{l['p25']:.2f},{l['p75']:.2f}]")
    for k in ("aligned", "stale"):
        c = report["static_consistency_abs_dz_minus_ego"][k]
        if c: print(f"static |dZ_lidar-ego| ({k}): med {c['median']:.3f}  (lower=correct; stale>aligned => staleness harms)")
    print(f"\nPOLICY: {report['policy_decision']['policy']}")
    print(f"wrote {OUT}/depth_half_sanity.json")


if __name__ == "__main__":
    main()
