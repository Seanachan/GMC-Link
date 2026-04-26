# KITTI Odometry Poses Gatekeeper

**Decision date:** 2026-04-22
**Status:** MISSING

## Probe results

| Path | Result |
|------|--------|
| `/home/seanachan/data/Dataset/kitti_odometry/` | does not exist |
| `~/Downloads/data_odometry_calib.zip` | present (~1MB) — contains only `calib.txt` + `times.txt` per sequence, no poses |
| `~/Downloads/data_odometry_poses.zip` | not present |

KITTI distributes ground-truth poses in a separate ~4MB zip (`data_odometry_poses.zip`) available from https://www.cvlibs.net/datasets/kitti/eval_odometry.php (requires KITTI account).

## Decision

**DEFER DOWNLOAD — continue with code path, report ATE qualitatively.**

Per spec §6 gatekeeper row 3 fallback: "if missing, report ATE only qualitatively via scale-consistency."

## Rationale

1. Task 7 (`eval/ego_ate.py`) can still be implemented and unit-tested with synthetic trajectories; only the seq 09/10 end-to-end ATE numbers require real poses.
2. Stage A decision (§7 A2) can be made on V1 micro-AUC alone; ATE is supporting evidence, not gating.
3. Quantitative ATE is desirable for paper framing (§1 ego-compensation service), but not on the Stage A critical path.
4. If Stage A is positive and we want to strengthen the causal story, download poses at that point (4MB, ~1 minute) and rerun ATE only.

## Action item (deferred)

If Stage A yields ΔAUC ≥ +0.010, download `data_odometry_poses.zip`, extract to `/home/seanachan/data/Dataset/kitti_odometry/poses/`, rerun ATE side-experiment, append numbers to Stage A memo.
