"""Pin the Path B oxts-ego frame convention BEFORE wiring it into the model.

Guards the known off-by-one bug class (commit c82f202: 1-based cache key vs
0-based label/pose index). World-static (parked) vehicles must have
|dz_lidar - ego_oxts| ~ 0 under the EXACT convention dataset.py will use:

  cache key curr_fid is 1-based (f1 = f0+1)  ->  0-based pose/image idx = curr_fid-1
  forward dz_track = z[curr_fid+gap] - z[curr_fid]   (interval [curr_fid-1, curr_fid-1+gap] in 0-based img)
  ego_oxts = ego_dz_camera(poses, calib, t=(curr_fid-1)+gap, gap)   (camera-frame ego ΔZ of a static pt)

Reference numbers (depth_half_sanity / mechanism_2x2): LiDAR+oxts gap=1 median
|dz-ego| = 0.099m; cohort-ego mechanism ~0.31m. An off-by-one in the pose index
multiplies ego error by the per-frame ego step (~0.5-2m), so a wrong convention
blows the static residual well past 0.20m and erases the oxts<<cohort gap.

Run: python tests/test_pathB_oxts_frame_alignment.py   (CPU only, no GPU/training).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "gmc_link"))
sys.path.insert(0, os.path.join(HERE, "..", "diagnostics", "pathB"))
import kitti_tracking_gt as K
from phase0_mechanism_2x2 import (load_label02, load_lwids_boxes,
                                  world_static_ids, match_dav2_id)

LIDAR_DIR = os.path.join(HERE, "..", "gmc_link", "depth_cache_lidar")
SEQS = ["0005", "0011"]
GAP = 5  # a real FRAME_GAPS member; the off-by-one error scales with gap


def load_lidar_gt(seq):
    """LiDAR gt cache keyed by refer-kitti (labels_with_ids) tid, 1-based frame."""
    raw = json.load(open(os.path.join(LIDAR_DIR, f"z_track_lidar_gt_{seq}.json")))
    return {int(t): {int(f): float(z) for f, z in fr.items()} for t, fr in raw.items()}


def main():
    oxts_res, cohort_res = [], []
    for seq in SEQS:
        l2 = load_label02(seq)            # label_02 tid -> f0(0-based) -> box/Z
        lw = load_lwids_boxes(seq)        # f0(0-based) -> [(lwids_tid, box)]
        lidar = load_lidar_gt(seq)        # lwids_tid -> f1(1-based) -> Z
        poses, calib = K.seq_poses_calib(seq)
        static = world_static_ids(l2, poses, calib)

        # cohort forward-dz median per 1-based key (mirror _frame_cohort_dz_ego direction)
        keys = sorted({k for tr in lidar.values() for k in tr})
        cohort = {}
        for key in keys:
            dz = [lidar[t][key + GAP] - lidar[t][key] for t in lidar
                  if key in lidar[t] and (key + GAP) in lidar[t]]
            if len(dz) >= 3:
                cohort[key] = float(np.median(dz))

        # offset sweep: correct 0-based img idx for cache key is (key-1).
        # ego over forward interval [a, a+GAP] = ego_dz_camera(t=a+GAP, GAP).
        # off=0 is the dataset.py convention (a=key-1); off=+1/-1 are the
        # neighbouring (wrong) alignments that the off-by-one bug would pick.
        raw, off = [], {-1: [], 0: [], 1: []}
        for tid_l2 in static:
            for f0 in sorted(l2[tid_l2]):                      # f0 = 0-based image idx
                m = match_dav2_id(l2[tid_l2][f0]["box"], lw.get(f0, []))  # -> lwids tid
                if m is None:
                    continue
                key = f0 + 1                                   # 1-based cache key
                tr = lidar.get(m, {})
                if key not in tr or (key + GAP) not in tr:
                    continue
                dz_track = tr[key + GAP] - tr[key]             # forward, dataset.py direction
                raw.append(abs(dz_track))                      # no ego comp (static tracks ego -> large)
                for d in (-1, 0, 1):
                    a = (key - 1) + d                          # candidate 0-based img idx
                    off[d].append(abs(dz_track - K.ego_dz_camera(poses, calib, a + GAP, GAP)))
                oxts_res.append(off[0][-1])
                if key in cohort:
                    cohort_res.append(abs(dz_track - cohort[key]))
        for d in (-1, 0, 1):
            globals().setdefault("_OFF", {}).setdefault(d, []).extend(off[d])
        globals().setdefault("_RAW", []).extend(raw)

    o = float(np.median(oxts_res))
    c = float(np.median(cohort_res))
    raw_med = float(np.median(globals()["_RAW"]))
    off_med = {d: float(np.median(globals()["_OFF"][d])) for d in (-1, 0, 1)}
    print(f"n_static_samples={len(oxts_res)}")
    print(f"raw |dz_lidar| (no ego comp)                 = {raw_med:.3f} m  (static tracks follow ego -> large)")
    print(f"offset sweep |dz - ego|  off-1={off_med[-1]:.3f}  off0={off_med[0]:.3f}  off+1={off_med[1]:.3f} m")
    print(f"  off0 = dataset.py convention a=(curr_fid-1); correct offset MINIMIZES residual")
    print(f"median |dz_lidar - ego_oxts| (off0, gap={GAP}) = {o:.3f} m")
    print(f"median |dz_lidar - cohort_LiDAR| (gap={GAP})   = {c:.3f} m  "
          f"(LiDAR-cohort self-cancels; oxts edge shrinks vs Phase-0 DAv2-cohort 6.47x)")

    assert len(oxts_res) > 30, f"too few static samples ({len(oxts_res)})"
    # (1) ego compensation works: comped residual << raw (static cars track ego)
    assert o < 0.5 * raw_med, f"ego comp ineffective: off0 {o:.3f} vs raw {raw_med:.3f}"
    # (2) frame ALIGNED: dataset.py offset (0) beats both off-by-one neighbours
    assert off_med[0] <= off_med[1] and off_med[0] <= off_med[-1], (
        f"off-by-one: off0 {off_med[0]:.3f} not minimal vs "
        f"off-1 {off_med[-1]:.3f}/off+1 {off_med[1]:.3f}")
    # (3) residual in the depth_half_sanity scale (0.099 @ gap1 -> ~0.2 @ gap5)
    assert o < 0.30, f"static residual {o:.3f}m too large for aligned LiDAR+oxts"
    print(f"PASS: oxts frame-aligned (off0 minimal, residual {o:.3f}m << raw {raw_med:.3f}m)")


if __name__ == "__main__":
    main()
