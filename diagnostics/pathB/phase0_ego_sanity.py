"""Path B Phase-0 ego sanity: oxts GT ego-ΔZ vs stationary-cohort proxy.

Compares, per frame (gap=5), the ego-induced camera-frame ΔZ from:
  (A) oxts GT  -> kitti_tracking_gt.ego_dz_camera  (the Path B replacement)
  (B) cohort proxy -> median DAv2 ΔZ over tracks    (what the ship currently uses)
Quantifies whether the monocular proxy is noisy vs GT (the Path B premise).
Depth-half (LiDAR vs DAv2 per-bbox Z) is deferred until velodyne is downloaded.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "gmc_link"))
import kitti_tracking_gt as G

DEPTH_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "gmc_link", "depth_cache")
GAP = 5
SEQS = ["0005", "0011", "0013"]


def cohort_dz_dav2(seq, gap):
    """{frame: median ΔZ over all tracks present at t and t-gap} from DAv2 cache."""
    raw = json.load(open(os.path.join(DEPTH_DIR, f"z_track_gt_{seq}.json")))
    z = {int(t): {int(f): float(v) for f, v in fr.items()} for t, fr in raw.items()}
    frames = sorted({f for tr in z.values() for f in tr})
    out = {}
    for t in frames:
        dzs = [z[tid][t] - z[tid][t - gap]
               for tid in z if t in z[tid] and (t - gap) in z[tid]]
        if len(dzs) >= 3:
            out[t] = float(np.median(dzs))
    return out


def main():
    print(f"{'seq':<6}{'n':>5}{'oxts_dz med':>13}{'cohort_dz med':>15}"
          f"{'MAE':>9}{'corr':>8}{'vf m/s':>9}")
    print("-" * 65)
    for seq in SEQS:
        oxts = G.load_oxts(seq)
        poses = G.oxts_to_poses(oxts)
        calib = G.load_calib(seq)
        cohort = cohort_dz_dav2(seq, GAP)
        ox, co = [], []
        for t, c_dz in cohort.items():
            if t - GAP < 0 or t >= len(poses):
                continue
            ox.append(G.ego_dz_camera(poses, calib, t, GAP))
            co.append(c_dz)
        ox, co = np.array(ox), np.array(co)
        mae = float(np.mean(np.abs(ox - co)))
        corr = float(np.corrcoef(ox, co)[0, 1]) if len(ox) > 2 else float("nan")
        print(f"{seq:<6}{len(ox):>5}{np.median(ox):>13.3f}{np.median(co):>15.3f}"
              f"{mae:>9.3f}{corr:>8.3f}{oxts[:, 8].mean():>9.2f}")
    print("\noxts_dz = GT ego ΔZ (m/gap, camera frame, negative=closing).")
    print("cohort_dz = median DAv2 ΔZ over tracks (the monocular proxy being replaced).")
    print("Large MAE / low corr => proxy is noisy vs GT => Path B premise supported.")


if __name__ == "__main__":
    main()
