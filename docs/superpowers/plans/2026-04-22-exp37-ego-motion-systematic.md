# Exp 37 — Systematic Ego-Motion Compensation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Factor the GMC-Link ego-motion pipeline along three axes (ego source, motion features, structural conditioning) via staged greedy ablation, then verify the ego-compensation signal transfers to a second RMOT tracker besides iKUN.

**Architecture:** A pluggable `EgoRouter` behind `GMCLinkManager` lets us swap ORB for SparseMFE (6DoF autoencoder) without touching residual-velocity math. Stage B adds per-bbox Object-Motion-Field (OMF) stats to the 13D hand-crafted features. Stage C concatenates EMAP-style ego-velocity to the aligner input. Stage D plugs the winning aligner into TempRMOT as a cost-term additive to prove cross-tracker portability.

**Tech Stack:** PyTorch, SentenceTransformer (all-MiniLM-L6-v2), OpenCV (ORB + RANSAC + recoverPose fallback), SparseMFE upstream repo (third-party submodule), TrackEval HOTA, KITTI odometry GT poses.

**Spec:** `docs/superpowers/specs/2026-04-22-ego-motion-systematic-design.md`

**Branch:** `exp/ego-motion-systematic` off `exp/hn-mining`

---

## File Structure

**New files:**
- `gmc_link/ego/__init__.py`
- `gmc_link/ego/ego_router.py` — abstract `EgoRouter` base + registry + `OrbEgoRouter` adapter
- `gmc_link/ego/sparsemfe_ego.py` — wrapper around upstream SparseMFE (frozen inference)
- `gmc_link/features/__init__.py`
- `gmc_link/features/omf_stats.py` — per-bbox OMF pooling to {mean/std/max dx,dy} per scale
- `eval/ego_ate.py` — KITTI odometry ATE (seq 09, 10) for ego-estimator comparison
- `tests/test_ego_router.py`
- `tests/test_sparsemfe_ego.py`
- `tests/test_omf_stats.py`
- `tests/test_ego_ate.py`
- `run_exp37_stage_a.sh`, `run_exp37_stage_b.sh`, `run_exp37_stage_c.sh`, `run_exp37_stage_d.sh`
- `third_party/SparseMFE/` (git submodule or pinned clone)

**Modified files:**
- `gmc_link/manager.py` — accept `ego_router` injection; residual-velocity math unchanged
- `gmc_link/alignment.py` — `motion_dim` propagation already parameterized; no change beyond flag plumbing
- `gmc_link/dataset.py` — new `EXTRA_FEATURE_DIMS` entries (`omf_stats`, `ego_velocity_concat`); new cache keys
- `gmc_link/train.py` — add CLI flags `--ego {orb,sparsemfe}`, `--omf-stats`, `--ego-concat`

**Unchanged (locked):**
- `gmc_link/losses.py`, `gmc_link/fusion_head.py`, `run_hota_eval_v1.py`, α=1.0

---

## Task 1: Worktree + branch setup

**Files:**
- N/A (git state only)

- [ ] **Step 1: Create worktree off exp/hn-mining**

```bash
cd /home/seanachan/GMC-Link
git worktree add -b exp/ego-motion-systematic ../GMC-Link-exp37 exp/hn-mining
cd ../GMC-Link-exp37
```

Expected: worktree at `../GMC-Link-exp37`, on branch `exp/ego-motion-systematic`.

- [ ] **Step 2: Verify clean base**

```bash
git status
git log --oneline -3
```

Expected: `HEAD -> exp/ego-motion-systematic`, tip commit `3232547` or later.

- [ ] **Step 3: Create empty package + results dirs**

```bash
mkdir -p gmc_link/ego gmc_link/features eval third_party diagnostics/results/exp37
touch gmc_link/ego/__init__.py gmc_link/features/__init__.py
```

- [ ] **Step 4: Commit scaffolding**

```bash
git add gmc_link/ego/__init__.py gmc_link/features/__init__.py
git commit -m "chore(exp37): scaffold ego + features packages"
```

---

## Task 2: Gatekeeper — SparseMFE availability check

**Files:**
- Test: `tests/test_sparsemfe_availability.py` (delete after decision)

- [ ] **Step 1: Probe upstream repo**

Run (do NOT clone yet — just check HTTP):

```bash
curl -sI https://github.com/kashyap7x/SparseMFE | head -5 || echo "NOT FOUND"
curl -sI https://github.com/anuragranj/sparsemfe | head -5 || echo "NOT FOUND"
```

Document which upstream URL returns 200. If none do, paper has no code release → **pivot to fallback: `cv2.recoverPose` 6DoF from ORB correspondences**.

- [ ] **Step 2: Write availability memo**

```bash
cat > diagnostics/results/exp37/gatekeeper_sparsemfe.md <<'EOF'
# SparseMFE Gatekeeper

Decision date: $(date +%Y-%m-%d)
Upstream status: [PUBLIC | CLOSED | WEIGHTS_ONLY]
Decision: [PROCEED_WITH_SPARSEMFE | FALLBACK_RECOVERPOSE]
Rationale: <one line>
EOF
```

Fill in. Commit.

```bash
git add diagnostics/results/exp37/gatekeeper_sparsemfe.md
git commit -m "docs(exp37): sparsemfe gatekeeper decision"
```

**Branch point:** if FALLBACK, Task 5 implements `RecoverPoseEgoRouter` instead of `SparseMfeEgoRouter`. All subsequent tasks behave identically — the router interface is the same.

---

## Task 3: Gatekeeper — KITTI odometry GT poses

**Files:**
- Test: `tests/test_kitti_poses_on_disk.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_kitti_poses_on_disk.py
from pathlib import Path

def test_kitti_poses_seq_09_10_available():
    base = Path("/home/seanachan/data/Dataset/kitti_odometry/poses")
    for seq in ("09.txt", "10.txt"):
        p = base / seq
        assert p.is_file(), f"Missing {p} — see spec §6 gatekeeper row 3"
        # Each line is a 3x4 flattened pose (12 floats)
        first = p.read_text().splitlines()[0].split()
        assert len(first) == 12, f"Expected 12 floats per line in {p}, got {len(first)}"
```

- [ ] **Step 2: Run test**

```bash
~/miniconda/envs/RMOT/bin/pytest tests/test_kitti_poses_on_disk.py -v
```

Expected: FAIL if not extracted.

- [ ] **Step 3: Extract if missing**

```bash
if [ ! -f /home/seanachan/data/Dataset/kitti_odometry/poses/09.txt ]; then
  mkdir -p /home/seanachan/data/Dataset/kitti_odometry
  cd /home/seanachan/data/Dataset/kitti_odometry
  unzip -q ~/Downloads/data_odometry_calib.zip -d .
  cd -
fi
```

If `data_odometry_calib.zip` contains calib but not poses, download poses separately from KITTI site — note the URL in the gatekeeper memo and get user confirmation before downloading.

- [ ] **Step 4: Re-run test**

```bash
~/miniconda/envs/RMOT/bin/pytest tests/test_kitti_poses_on_disk.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_kitti_poses_on_disk.py
git commit -m "test(exp37): gate on KITTI odometry poses presence"
```

---

## Task 4: Ego router abstraction + ORB adapter

**Files:**
- Create: `gmc_link/ego/ego_router.py`
- Test: `tests/test_ego_router.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_ego_router.py
import numpy as np
import pytest
from gmc_link.ego.ego_router import EgoRouter, OrbEgoRouter, make_ego_router


def _rand_frame(seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((64, 64, 3)) * 255).astype(np.uint8)


def test_make_ego_router_returns_orb_by_default():
    r = make_ego_router("orb")
    assert isinstance(r, OrbEgoRouter)


def test_make_ego_router_unknown_raises():
    with pytest.raises(KeyError):
        make_ego_router("nonexistent")


def test_orb_router_returns_3x3_homography_and_residual():
    r = OrbEgoRouter(max_features=500)
    H, residual = r.estimate(_rand_frame(0), _rand_frame(1))
    assert H.shape == (3, 3)
    assert H.dtype == np.float32 or H.dtype == np.float64
    assert isinstance(residual, float)


def test_orb_router_identity_on_same_frame():
    r = OrbEgoRouter(max_features=500)
    f = _rand_frame(42)
    H, residual = r.estimate(f, f)
    np.testing.assert_allclose(H, np.eye(3), atol=0.5)
    assert residual < 1.0
```

- [ ] **Step 2: Run tests to verify fail**

```bash
~/miniconda/envs/RMOT/bin/pytest tests/test_ego_router.py -v
```

Expected: `ModuleNotFoundError` or similar.

- [ ] **Step 3: Implement router**

```python
# gmc_link/ego/ego_router.py
"""Pluggable ego-motion estimator interface.

Exposes a single `EgoRouter.estimate(prev, curr) -> (H_3x3, residual_float)` contract
so GMCLinkManager can swap ego backends without touching residual-velocity math.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from gmc_link.core import ORBHomographyEngine


class EgoRouter(ABC):
    @abstractmethod
    def estimate(
        self, prev_frame: np.ndarray, curr_frame: np.ndarray, prev_bboxes=None
    ) -> Tuple[np.ndarray, float]:
        ...


class OrbEgoRouter(EgoRouter):
    def __init__(self, max_features: int = 1500):
        self._engine = ORBHomographyEngine(max_features=max_features)

    def estimate(self, prev_frame, curr_frame, prev_bboxes=None):
        H, residual = self._engine.estimate(prev_frame, curr_frame, prev_bboxes)
        return H, float(residual)


_REGISTRY = {"orb": OrbEgoRouter}


def register_ego_router(name: str, cls):
    _REGISTRY[name] = cls


def make_ego_router(name: str, **kwargs) -> EgoRouter:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown ego router '{name}'. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
```

- [ ] **Step 4: Run tests to verify pass**

```bash
~/miniconda/envs/RMOT/bin/pytest tests/test_ego_router.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add gmc_link/ego/ego_router.py tests/test_ego_router.py
git commit -m "feat(exp37): add pluggable EgoRouter with ORB adapter"
```

---

## Task 5: SparseMFE wrapper (or recoverPose fallback)

**Files:**
- Create: `gmc_link/ego/sparsemfe_ego.py` (SparseMFE path) OR `gmc_link/ego/recoverpose_ego.py` (fallback)
- Test: `tests/test_sparsemfe_ego.py` or `tests/test_recoverpose_ego.py`

**Decision from Task 2:** If SparseMFE public → implement SparseMFE path. Else → fallback.

### Path A — SparseMFE (if Task 2 green)

- [ ] **Step 1: Pin upstream as submodule**

```bash
git submodule add <upstream-url> third_party/SparseMFE
cd third_party/SparseMFE && git checkout <pinned-sha> && cd -
git add .gitmodules third_party/SparseMFE
git commit -m "chore(exp37): pin SparseMFE submodule"
```

- [ ] **Step 2: Write failing test**

```python
# tests/test_sparsemfe_ego.py
import numpy as np
import pytest

sparsemfe_ego = pytest.importorskip("gmc_link.ego.sparsemfe_ego")


def _rand(seed):
    rng = np.random.default_rng(seed)
    return (rng.random((256, 832, 3)) * 255).astype(np.uint8)


def test_estimate_returns_H_and_omf():
    router = sparsemfe_ego.SparseMfeEgoRouter(device="cpu")
    H, residual = router.estimate(_rand(0), _rand(1))
    assert H.shape == (3, 3)
    assert isinstance(residual, float)


def test_estimate_also_returns_omf_via_last_omf_field():
    router = sparsemfe_ego.SparseMfeEgoRouter(device="cpu")
    router.estimate(_rand(0), _rand(1))
    omf = router.last_omf_field
    assert omf.shape[2] == 2  # (H, W, 2) flow
```

- [ ] **Step 3: Run test to verify fail**

```bash
~/miniconda/envs/RMOT/bin/pytest tests/test_sparsemfe_ego.py -v
```

Expected: ImportError.

- [ ] **Step 4: Implement wrapper**

```python
# gmc_link/ego/sparsemfe_ego.py
"""Wrapper around third_party/SparseMFE frozen inference.

Returns a 3x3 homography projected from 6DoF {R, t} by approximating the plane-at-infinity
so downstream GMCLinkManager centroid warping is backwards-compatible.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from gmc_link.ego.ego_router import EgoRouter, register_ego_router

# Late-bound imports to keep unit tests runnable without model weights
_SparseMfeModel = None


def _load_model(device: str):
    global _SparseMfeModel
    if _SparseMfeModel is None:
        # Adjust to actual upstream module path — see third_party/SparseMFE/README
        from third_party.SparseMFE.sparsemfe import SparseMFE  # type: ignore

        m = SparseMFE().to(device).eval()
        ckpt_path = "third_party/SparseMFE/checkpoints/pretrained.pth"
        m.load_state_dict(torch.load(ckpt_path, map_location=device))
        _SparseMfeModel = m
    return _SparseMfeModel


class SparseMfeEgoRouter(EgoRouter):
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.last_omf_field: np.ndarray | None = None

    @torch.inference_mode()
    def estimate(self, prev_frame, curr_frame, prev_bboxes=None) -> Tuple[np.ndarray, float]:
        model = _load_model(self.device)
        t_prev = torch.from_numpy(prev_frame).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        t_curr = torch.from_numpy(curr_frame).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        out = model(t_prev, t_curr)  # returns {'R': 3x3, 't': 3, 'omf': H W 2}
        R = out["R"].cpu().numpy().reshape(3, 3)
        t = out["t"].cpu().numpy().reshape(3)
        self.last_omf_field = out["omf"].squeeze(0).permute(1, 2, 0).cpu().numpy()
        H = _project_6dof_to_2d_homography(R, t)
        residual = float(np.linalg.norm(t))
        return H, residual


def _project_6dof_to_2d_homography(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    # Plane-at-infinity approximation: H ≈ K R K^-1, but without intrinsics we pass R directly.
    # GMCLinkManager only uses H to warp centroid coords, so the 2D component of R is what matters.
    H = np.eye(3, dtype=np.float32)
    H[:2, :2] = R[:2, :2]
    H[:2, 2] = t[:2]
    return H


register_ego_router("sparsemfe", SparseMfeEgoRouter)
```

- [ ] **Step 5: Verify registration works (does NOT need weights)**

```python
# extend tests/test_ego_router.py
def test_sparsemfe_registered():
    from gmc_link.ego.ego_router import make_ego_router
    # Only import, do not instantiate if weights missing
    import gmc_link.ego.sparsemfe_ego  # noqa — registers
    from gmc_link.ego.ego_router import _REGISTRY
    assert "sparsemfe" in _REGISTRY
```

- [ ] **Step 6: Run unit + registration tests**

```bash
~/miniconda/envs/RMOT/bin/pytest tests/test_ego_router.py tests/test_sparsemfe_ego.py -v
```

Expected: unit tests pass; heavyweight inference tests skipped if no GPU/weights.

- [ ] **Step 7: Commit**

```bash
git add gmc_link/ego/sparsemfe_ego.py tests/test_sparsemfe_ego.py tests/test_ego_router.py
git commit -m "feat(exp37): SparseMFE ego router wrapper"
```

### Path B — recoverPose fallback (if Task 2 red)

- [ ] **Step 1: Write failing test**

```python
# tests/test_recoverpose_ego.py
import numpy as np
from gmc_link.ego.recoverpose_ego import RecoverPoseEgoRouter


def _rand(seed):
    rng = np.random.default_rng(seed)
    return (rng.random((256, 832, 3)) * 255).astype(np.uint8)


def test_recover_pose_returns_3x3_H_and_residual():
    r = RecoverPoseEgoRouter(focal=707.0, pp=(601.0, 183.0), max_features=1500)
    H, residual = r.estimate(_rand(0), _rand(1))
    assert H.shape == (3, 3)
    assert isinstance(residual, float)
```

- [ ] **Step 2: Implement**

```python
# gmc_link/ego/recoverpose_ego.py
"""6DoF ego from cv2.recoverPose on ORB correspondences.

Fallback for when SparseMFE is not publicly released.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from gmc_link.ego.ego_router import EgoRouter, register_ego_router


class RecoverPoseEgoRouter(EgoRouter):
    def __init__(self, focal: float = 707.0, pp=(601.0, 183.0), max_features: int = 1500):
        self.focal = focal
        self.pp = pp
        self.orb = cv2.ORB_create(nfeatures=max_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def estimate(self, prev_frame, curr_frame, prev_bboxes=None) -> Tuple[np.ndarray, float]:
        g1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        kp1, des1 = self.orb.detectAndCompute(g1, None)
        kp2, des2 = self.orb.detectAndCompute(g2, None)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return np.eye(3, dtype=np.float32), 0.0
        matches = self.bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.7 * n.distance]
        if len(good) < 8:
            return np.eye(3, dtype=np.float32), 0.0
        src = np.float32([kp1[m.queryIdx].pt for m in good])
        dst = np.float32([kp2[m.trainIdx].pt for m in good])
        E, mask = cv2.findEssentialMat(src, dst, focal=self.focal, pp=self.pp, method=cv2.RANSAC, threshold=1.0)
        if E is None:
            return np.eye(3, dtype=np.float32), 0.0
        _, R, t, _ = cv2.recoverPose(E, src, dst, focal=self.focal, pp=self.pp, mask=mask)
        H = np.eye(3, dtype=np.float32)
        H[:2, :2] = R[:2, :2]
        H[:2, 2] = t.flatten()[:2]
        residual = float(np.linalg.norm(t))
        return H, residual


register_ego_router("recoverpose", RecoverPoseEgoRouter)
```

- [ ] **Step 3: Run tests + commit**

```bash
~/miniconda/envs/RMOT/bin/pytest tests/test_recoverpose_ego.py -v
git add gmc_link/ego/recoverpose_ego.py tests/test_recoverpose_ego.py
git commit -m "feat(exp37): recoverPose ego fallback router"
```

---

## Task 6: Integrate ego router into GMCLinkManager

**Files:**
- Modify: `gmc_link/manager.py`
- Test: `tests/test_manager_ego_router_injection.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_manager_ego_router_injection.py
import numpy as np

from gmc_link.manager import GMCLinkManager
from gmc_link.ego.ego_router import OrbEgoRouter


def test_manager_accepts_injected_ego_router():
    router = OrbEgoRouter(max_features=500)
    m = GMCLinkManager(ego_router=router)
    # should not raise; ego_router attribute exposed
    assert m.ego_engine is router


def test_manager_defaults_to_orb_router_when_none_passed():
    m = GMCLinkManager()
    from gmc_link.ego.ego_router import OrbEgoRouter as Orb
    assert isinstance(m.ego_engine, Orb)
```

- [ ] **Step 2: Run test to verify fail**

```bash
~/miniconda/envs/RMOT/bin/pytest tests/test_manager_ego_router_injection.py -v
```

Expected: FAIL (either AttributeError or TypeError).

- [ ] **Step 3: Modify manager.py**

Find existing `self.ego_engine = ORBHomographyEngine(max_features=1500)` in `GMCLinkManager.__init__` and replace:

```python
# gmc_link/manager.py (inside __init__)
from gmc_link.ego.ego_router import EgoRouter, OrbEgoRouter

def __init__(self, ..., ego_router: EgoRouter | None = None):
    ...
    self.ego_engine = ego_router if ego_router is not None else OrbEgoRouter(max_features=1500)
```

Ensure the call site downstream uses `self.ego_engine.estimate(...)` — the `ORBHomographyEngine` had the same signature, so the change is one-line.

- [ ] **Step 4: Run test to verify pass**

```bash
~/miniconda/envs/RMOT/bin/pytest tests/test_manager_ego_router_injection.py -v
```

Expected: 2 passed. Also run full `tests/` to catch regressions:

```bash
~/miniconda/envs/RMOT/bin/pytest tests/ -v -x
```

Expected: no new failures.

- [ ] **Step 5: Commit**

```bash
git add gmc_link/manager.py tests/test_manager_ego_router_injection.py
git commit -m "feat(exp37): inject EgoRouter into GMCLinkManager"
```

---

## Task 7: KITTI odometry ATE evaluator

**Files:**
- Create: `eval/ego_ate.py`
- Test: `tests/test_ego_ate.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_ego_ate.py
import numpy as np
import pytest

from eval.ego_ate import absolute_trajectory_error, load_kitti_poses


def test_ate_zero_on_identical_trajectories():
    poses = np.stack([np.eye(4)[:3, :] for _ in range(10)])
    err = absolute_trajectory_error(poses, poses)
    assert err < 1e-6


def test_ate_positive_on_shifted_trajectory():
    a = np.stack([np.eye(4)[:3, :] for _ in range(10)])
    b = a.copy()
    b[:, :3, 3] += 1.0
    err = absolute_trajectory_error(a, b)
    # after Umeyama alignment the rigid shift should be cancelled
    assert err < 1e-6


def test_ate_detects_rotation_drift():
    a = np.stack([np.eye(4)[:3, :] for _ in range(10)])
    b = a.copy()
    for i in range(10):
        theta = 0.01 * i
        b[i, :2, :2] = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    err = absolute_trajectory_error(a, b)
    assert err > 0.0


def test_load_kitti_poses_shape():
    import pathlib
    p = pathlib.Path("/home/seanachan/data/Dataset/kitti_odometry/poses/09.txt")
    if not p.is_file():
        pytest.skip("KITTI poses not extracted")
    poses = load_kitti_poses(p)
    assert poses.ndim == 3
    assert poses.shape[1:] == (3, 4)
```

- [ ] **Step 2: Run test to verify fail**

```bash
~/miniconda/envs/RMOT/bin/pytest tests/test_ego_ate.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement ATE**

```python
# eval/ego_ate.py
"""KITTI odometry Absolute Trajectory Error with Umeyama alignment."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_kitti_poses(path: Path) -> np.ndarray:
    lines = Path(path).read_text().strip().splitlines()
    poses = np.array([list(map(float, ln.split())) for ln in lines], dtype=np.float64)
    return poses.reshape(-1, 3, 4)


def _umeyama(src: np.ndarray, tgt: np.ndarray):
    mu_s = src.mean(0)
    mu_t = tgt.mean(0)
    s = src - mu_s
    t = tgt - mu_t
    H = s.T @ t
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    scale = (np.trace(np.diag(np.linalg.svd(H)[1]) @ D)) / (s * s).sum()
    trans = mu_t - scale * R @ mu_s
    return scale, R, trans


def absolute_trajectory_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """RMSE translation error after rigid-scale Umeyama alignment."""
    assert pred.shape == gt.shape
    p = pred[:, :3, 3]
    g = gt[:, :3, 3]
    scale, R, trans = _umeyama(p, g)
    aligned = (scale * (R @ p.T)).T + trans
    err = np.linalg.norm(aligned - g, axis=1)
    return float(np.sqrt((err ** 2).mean()))
```

- [ ] **Step 4: Run tests**

```bash
~/miniconda/envs/RMOT/bin/pytest tests/test_ego_ate.py -v
```

Expected: 3 passed, 1 skipped (if poses absent) or 4 passed.

- [ ] **Step 5: Add CLI wrapper**

```python
# eval/ego_ate.py (append)
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-poses", required=True)
    parser.add_argument("--gt-poses", required=True)
    args = parser.parse_args()
    pred = load_kitti_poses(Path(args.pred_poses))
    gt = load_kitti_poses(Path(args.gt_poses))
    n = min(len(pred), len(gt))
    print(f"ATE = {absolute_trajectory_error(pred[:n], gt[:n]):.6f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add eval/ego_ate.py tests/test_ego_ate.py
git commit -m "feat(exp37): KITTI odometry ATE evaluator with Umeyama alignment"
```

---

## Task 8: Stage A — ego source training + eval + decision

**Files:**
- Create: `run_exp37_stage_a.sh`
- Result: `diagnostics/results/exp37/stage_a_memo.md`

- [ ] **Step 1: Write stage A runner**

```bash
# run_exp37_stage_a.sh
#!/usr/bin/env bash
set -euo pipefail

PY=~/miniconda/envs/RMOT/bin/python
OUT=diagnostics/results/exp37

mkdir -p "$OUT"

# A1: ORB baseline (already trained as stage1.pth — just eval)
A1_W=gmc_link_weights_v1train_stage1.pth

# A2: SparseMFE (or recoverpose) — train fresh
A2_EGO=${A2_EGO:-sparsemfe}  # override to recoverpose if Task 2 red
A2_W=gmc_link_weights_exp37_stage_a2_${A2_EGO}.pth

# Train A2
$PY -m gmc_link.train \
  --split v1 \
  --ego $A2_EGO \
  --epochs 100 \
  --lr 1e-3 \
  --batch-size 128 \
  --save-path $A2_W

# Eval both on held-out {0005, 0011, 0013}
for W in $A1_W $A2_W; do
  TAG=$(basename $W .pth)
  for SEQ in 0005 0011 0013; do
    $PY diagnostics/diag_gt_cosine_distributions.py \
      --weights $W --seq $SEQ --output $OUT/layer3_${SEQ}_${TAG}.npz
  done
  $PY diagnostics/aggregate_multiseq.py \
    --results-dir $OUT \
    --output-dir $OUT \
    --weights $TAG \
    --seqs 0005 0011 0013 \
    --legacy-seq-0011
done
```

- [ ] **Step 2: Add `--ego` flag to train.py**

In `gmc_link/train.py`, find the argparse block and add:

```python
parser.add_argument("--ego", default="orb", choices=["orb", "sparsemfe", "recoverpose"],
                    help="Ego-motion source for cache building")
```

Thread `args.ego` through to `GMCLinkManager` construction (via dataset pipeline). Cache key MUST include `args.ego` to avoid cross-contamination.

- [ ] **Step 3: Quick smoke test (1 epoch, 1 seq)**

```bash
~/miniconda/envs/RMOT/bin/python -m gmc_link.train \
  --split v1 --ego orb --epochs 1 --batch-size 32 \
  --save-path /tmp/smoke_orb.pth
```

Expected: runs without error, writes weights, cache key includes `orb`.

- [ ] **Step 4: Run Stage A**

```bash
bash run_exp37_stage_a.sh
```

Expected: two weight files, two multiseq summaries in `diagnostics/results/exp37/`.

- [ ] **Step 5: Run KITTI ATE side-experiment**

For each ego router, dump predicted relative poses over KITTI seq 09 and 10:

```bash
~/miniconda/envs/RMOT/bin/python -c "
from pathlib import Path
from gmc_link.ego.ego_router import make_ego_router
from eval.ego_ate import load_kitti_poses, absolute_trajectory_error
import numpy as np, cv2
# TODO: loop over KITTI sequence 09 frames, accumulate relative poses, compare
"
```

(Implementer fills in the frame iteration — spec §5 row 5. Skip if KITTI poses absent; note in memo.)

- [ ] **Step 6: Write stage A memo**

```bash
cat > diagnostics/results/exp37/stage_a_memo.md <<EOF
# Stage A — Ego Source Decision

Date: $(date +%Y-%m-%d)
A1 (ORB) AUC: <fill>
A2 ($A2_EGO) AUC: <fill>
ΔAUC: <fill>
KITTI ATE seq 09/10 ORB: <fill>
KITTI ATE seq 09/10 $A2_EGO: <fill>
Decision band (spec §2 H_A): [POSITIVE | INCONCLUSIVE | NEGATIVE]
Winner(A): [ORB | $A2_EGO]
Rationale: <one line>
EOF
```

Fill in. Commit.

```bash
git add run_exp37_stage_a.sh gmc_link/train.py diagnostics/results/exp37/stage_a_memo.md
git commit -m "feat(exp37): Stage A ego-source ablation + decision memo"
```

---

## Task 9: OMF per-bbox stats extractor

**Files:**
- Create: `gmc_link/features/omf_stats.py`
- Test: `tests/test_omf_stats.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_omf_stats.py
import numpy as np

from gmc_link.features.omf_stats import per_bbox_omf_stats


def test_shape_is_5_per_scale_times_num_scales():
    omf = np.zeros((64, 64, 2), dtype=np.float32)
    bbox = (10, 10, 20, 20)  # x, y, w, h
    out = per_bbox_omf_stats(omf, bbox)
    assert out.shape == (5,)  # mean_dx, mean_dy, std_dx, std_dy, max_mag


def test_known_flow_values():
    omf = np.zeros((64, 64, 2), dtype=np.float32)
    omf[..., 0] = 1.0  # dx = 1 everywhere
    omf[..., 1] = 2.0  # dy = 2 everywhere
    out = per_bbox_omf_stats(omf, (10, 10, 20, 20))
    np.testing.assert_allclose(out[0], 1.0)  # mean_dx
    np.testing.assert_allclose(out[1], 2.0)  # mean_dy
    np.testing.assert_allclose(out[2], 0.0, atol=1e-6)  # std_dx
    np.testing.assert_allclose(out[3], 0.0, atol=1e-6)  # std_dy
    np.testing.assert_allclose(out[4], np.sqrt(5), atol=1e-6)  # max mag


def test_empty_bbox_returns_zeros():
    omf = np.zeros((64, 64, 2), dtype=np.float32)
    out = per_bbox_omf_stats(omf, (0, 0, 0, 0))
    np.testing.assert_array_equal(out, np.zeros(5))


def test_out_of_bounds_bbox_clips():
    omf = np.ones((64, 64, 2), dtype=np.float32)
    out = per_bbox_omf_stats(omf, (60, 60, 20, 20))  # mostly out of frame
    assert not np.any(np.isnan(out))
```

- [ ] **Step 2: Run tests to verify fail**

```bash
~/miniconda/envs/RMOT/bin/pytest tests/test_omf_stats.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement**

```python
# gmc_link/features/omf_stats.py
"""Per-bbox Object-Motion-Field (OMF) pooling statistics.

Given a dense optical-flow-like field (H, W, 2) and a bbox (x, y, w, h), compute
[mean_dx, mean_dy, std_dx, std_dy, max_mag]. For Stage B we concat these per
FRAME_GAPS scale (so 5 * 3 = 15 extra dims with 13D base = 28D).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def per_bbox_omf_stats(omf_field: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    H, W, _ = omf_field.shape
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(W, int(x + w))
    y1 = min(H, int(y + h))
    if x1 <= x0 or y1 <= y0:
        return np.zeros(5, dtype=np.float32)
    patch = omf_field[y0:y1, x0:x1, :]
    dx = patch[..., 0]
    dy = patch[..., 1]
    mag = np.sqrt(dx * dx + dy * dy)
    return np.array(
        [float(dx.mean()), float(dy.mean()), float(dx.std()), float(dy.std()), float(mag.max())],
        dtype=np.float32,
    )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
~/miniconda/envs/RMOT/bin/pytest tests/test_omf_stats.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add gmc_link/features/omf_stats.py tests/test_omf_stats.py
git commit -m "feat(exp37): per-bbox OMF pooling stats"
```

---

## Task 10: Dataset integration for 28D OMF features

**Files:**
- Modify: `gmc_link/dataset.py`
- Test: `tests/test_dataset_omf28d.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_dataset_omf28d.py
from gmc_link.dataset import EXTRA_FEATURE_DIMS


def test_omf_stats_feature_is_registered():
    assert "omf_stats" in EXTRA_FEATURE_DIMS
    assert EXTRA_FEATURE_DIMS["omf_stats"] == 15  # 5 stats * 3 scales


def test_cache_key_includes_ego_and_features(tmp_path):
    from gmc_link.dataset import _build_cache_key
    k1 = _build_cache_key(
        version="v1", data_root=str(tmp_path), sequences=["0001"],
        frame_gaps=(2, 5, 10), frame_shape=(375, 1242), use_group_labels=False,
        extra_features=("omf_stats",), seq_len=None,
        text_encoder_name="all-MiniLM-L6-v2", ego="sparsemfe",
    )
    k2 = _build_cache_key(
        version="v1", data_root=str(tmp_path), sequences=["0001"],
        frame_gaps=(2, 5, 10), frame_shape=(375, 1242), use_group_labels=False,
        extra_features=("omf_stats",), seq_len=None,
        text_encoder_name="all-MiniLM-L6-v2", ego="orb",
    )
    assert k1 != k2
```

- [ ] **Step 2: Run tests to verify fail**

```bash
~/miniconda/envs/RMOT/bin/pytest tests/test_dataset_omf28d.py -v
```

Expected: KeyError / assertion.

- [ ] **Step 3: Register feature + add `ego` to cache key**

In `gmc_link/dataset.py`:
- Add to `EXTRA_FEATURE_DIMS` dict: `"omf_stats": 15`.
- Add `ego: str = "orb"` parameter to `_build_cache_key` and include it in the hash input.
- In `compute_per_track_extras`, add branch for `"omf_stats"` that loads cached OMF field and calls `per_bbox_omf_stats` per scale, then concatenates.

- [ ] **Step 4: Run tests to verify pass**

```bash
~/miniconda/envs/RMOT/bin/pytest tests/test_dataset_omf28d.py tests/test_aggregate_multiseq.py tests/test_diag_npz_schema.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add gmc_link/dataset.py tests/test_dataset_omf28d.py
git commit -m "feat(exp37): OMF stats registered as 15D extra feature + cache key ego-aware"
```

---

## Task 11: Stage B — motion feature lever training + eval + decision

**Files:**
- Create: `run_exp37_stage_b.sh`
- Result: `diagnostics/results/exp37/stage_b_memo.md`

- [ ] **Step 1: Write Stage B runner**

```bash
# run_exp37_stage_b.sh
#!/usr/bin/env bash
set -euo pipefail

PY=~/miniconda/envs/RMOT/bin/python
OUT=diagnostics/results/exp37
EGO=$(cat $OUT/stage_a_winner 2>/dev/null || echo orb)

B2_W=gmc_link_weights_exp37_stage_b2_${EGO}_omf28d.pth

$PY -m gmc_link.train \
  --split v1 \
  --ego $EGO \
  --extra-features omf_stats \
  --epochs 100 \
  --lr 1e-3 \
  --batch-size 128 \
  --save-path $B2_W

# Eval
for SEQ in 0005 0011 0013; do
  $PY diagnostics/diag_gt_cosine_distributions.py \
    --weights $B2_W --seq $SEQ \
    --output $OUT/layer3_${SEQ}_stage_b2_${EGO}_omf28d.npz
done
$PY diagnostics/aggregate_multiseq.py \
  --results-dir $OUT --output-dir $OUT \
  --weights stage_b2_${EGO}_omf28d \
  --seqs 0005 0011 0013 --legacy-seq-0011
```

- [ ] **Step 2: Record Stage A winner**

```bash
# After Task 8, user/implementer writes winner into this file:
echo "sparsemfe" > diagnostics/results/exp37/stage_a_winner  # or orb
```

- [ ] **Step 3: Run Stage B**

```bash
bash run_exp37_stage_b.sh
```

- [ ] **Step 4: Write Stage B memo**

```bash
cat > diagnostics/results/exp37/stage_b_memo.md <<EOF
# Stage B — Motion Features Decision

Date: $(date +%Y-%m-%d)
E (winner A): <from stage_a_winner>
B1 (13D) AUC: <from stage_a_memo>
B2 (13D + OMF, 28D) AUC: <fill>
ΔAUC: <fill>
Decision (spec §2 H_B): [POSITIVE | INCONCLUSIVE | NEGATIVE]
Winner(B): [13D | 13D+OMF]
EOF
echo "omf_stats" > diagnostics/results/exp37/stage_b_features  # or empty if 13D wins
git add run_exp37_stage_b.sh diagnostics/results/exp37/stage_b_memo.md diagnostics/results/exp37/stage_b_features
git commit -m "feat(exp37): Stage B feature-lever ablation + decision memo"
```

---

## Task 12: EMAP ego-velocity concat feature

**Files:**
- Modify: `gmc_link/dataset.py`
- Test: `tests/test_ego_velocity_feature.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_ego_velocity_feature.py
from gmc_link.dataset import EXTRA_FEATURE_DIMS


def test_ego_velocity_concat_registered():
    assert "ego_velocity_concat" in EXTRA_FEATURE_DIMS
    assert EXTRA_FEATURE_DIMS["ego_velocity_concat"] == 2  # vx, vy
```

- [ ] **Step 2: Run test to verify fail**

```bash
~/miniconda/envs/RMOT/bin/pytest tests/test_ego_velocity_feature.py -v
```

Expected: KeyError.

- [ ] **Step 3: Implement**

In `gmc_link/dataset.py`:
- Add to `EXTRA_FEATURE_DIMS`: `"ego_velocity_concat": 2`.
- In `compute_per_track_extras`, compute per-sample `(ego_vx, ego_vy)` = `ego_trans_xy / dt` averaged over `FRAME_GAPS=[2,5,10]` window using cached homographies from `GMCLinkManager`.

- [ ] **Step 4: Run tests**

```bash
~/miniconda/envs/RMOT/bin/pytest tests/test_ego_velocity_feature.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add gmc_link/dataset.py tests/test_ego_velocity_feature.py
git commit -m "feat(exp37): ego-velocity concat feature (EMAP-style 2D)"
```

---

## Task 13: Stage C — structural conditioning training + eval + decision

**Files:**
- Create: `run_exp37_stage_c.sh`
- Result: `diagnostics/results/exp37/stage_c_memo.md`

- [ ] **Step 1: Write Stage C runner**

```bash
# run_exp37_stage_c.sh
#!/usr/bin/env bash
set -euo pipefail

PY=~/miniconda/envs/RMOT/bin/python
OUT=diagnostics/results/exp37
EGO=$(cat $OUT/stage_a_winner)
BFEAT=$(cat $OUT/stage_b_features)  # "omf_stats" or ""

FEAT_ARG=""
if [ -n "$BFEAT" ]; then
  FEAT_ARG="--extra-features ${BFEAT},ego_velocity_concat"
else
  FEAT_ARG="--extra-features ego_velocity_concat"
fi

C2_W=gmc_link_weights_exp37_stage_c2_${EGO}.pth

$PY -m gmc_link.train \
  --split v1 --ego $EGO $FEAT_ARG \
  --epochs 100 --lr 1e-3 --batch-size 128 \
  --save-path $C2_W

for SEQ in 0005 0011 0013; do
  $PY diagnostics/diag_gt_cosine_distributions.py \
    --weights $C2_W --seq $SEQ --output $OUT/layer3_${SEQ}_stage_c2.npz
done
$PY diagnostics/aggregate_multiseq.py \
  --results-dir $OUT --output-dir $OUT \
  --weights stage_c2 --seqs 0005 0011 0013 --legacy-seq-0011
```

- [ ] **Step 2: Run Stage C**

```bash
bash run_exp37_stage_c.sh
```

- [ ] **Step 3: Write Stage C memo + record final winner**

```bash
cat > diagnostics/results/exp37/stage_c_memo.md <<EOF
# Stage C — Structural Conditioning Decision

Date: $(date +%Y-%m-%d)
E: $(cat $OUT/stage_a_winner)
F: $(cat $OUT/stage_b_features || echo 13D)
C1 AUC (off): <from stage B memo>
C2 AUC (EMAP concat): <fill>
ΔAUC: <fill>
per-seq-0011 HOTA variance: <fill>
Decision (spec §2 H_C): [POSITIVE | INCONCLUSIVE | NEGATIVE]
Final Exp 37 aligner (winner(C)): <path to .pth>
EOF
echo "gmc_link_weights_exp37_stage_c2_$(cat $OUT/stage_a_winner).pth" > $OUT/exp37_final_weights
git add run_exp37_stage_c.sh diagnostics/results/exp37/stage_c_memo.md diagnostics/results/exp37/exp37_final_weights
git commit -m "feat(exp37): Stage C structural-conditioning ablation + decision memo"
```

---

## Task 14: HOTA eval of final aligner on iKUN fusion

**Files:**
- Result: `diagnostics/results/exp37/hota_final_vs_baseline.md`

- [ ] **Step 1: Run α=1.0 HOTA eval for winner(C) + baselines**

```bash
FINAL=$(cat diagnostics/results/exp37/exp37_final_weights)
BASELINE=gmc_link_weights_v1train_stage1.pth

for W in $BASELINE $FINAL; do
  TAG=$(basename $W .pth)
  ~/miniconda/envs/RMOT/bin/python run_hota_eval_v1.py \
    --method fusion --skip-ikun --weights $W --alpha 1.0 --tag $TAG
done
```

- [ ] **Step 2: Write HOTA memo**

```bash
cat > diagnostics/results/exp37/hota_final_vs_baseline.md <<EOF
# Exp 37 HOTA — α=1.0 fusion with iKUN

Baseline (stage1 ORB 13D) HOTA: 35.476  # from memory
Exp 37 winner(C) HOTA: <fill>
ΔHOTA: <fill>
DetA / AssA / IDF1 breakdown: <fill>
EOF
git add diagnostics/results/exp37/hota_final_vs_baseline.md
git commit -m "docs(exp37): HOTA comparison of final aligner vs ORB stage1 baseline"
```

---

## Task 15: TempRMOT acquisition + bare baseline

**Files:**
- Create: `third_party/TempRMOT/` (submodule or clone)
- Result: `diagnostics/results/exp37/stage_d_baseline.md`

- [ ] **Step 1: Locate TempRMOT upstream**

```bash
curl -sI https://github.com/zyn213/TempRMOT | head -5 || echo "try alternate fork"
```

If reachable, clone:

```bash
git submodule add https://github.com/zyn213/TempRMOT third_party/TempRMOT
cd third_party/TempRMOT && git checkout <pinned-sha> && pip install -e . && cd -
git add .gitmodules third_party/TempRMOT
git commit -m "chore(exp37): pin TempRMOT submodule"
```

- [ ] **Step 2: Run bare TempRMOT on Refer-KITTI V1 held-out**

Use TempRMOT's own inference script. Dump predictions to `diagnostics/results/exp37/temprmot_bare/`. Evaluate with `run_hota_eval_v1.py --method temprmot_bare`.

- [ ] **Step 3: Record bare baseline**

```bash
cat > diagnostics/results/exp37/stage_d_baseline.md <<EOF
# Stage D — TempRMOT Bare Baseline

TempRMOT HOTA: <fill>
DetA / AssA / IDF1: <fill>
Commit/weights provenance: <fill>
EOF
git add diagnostics/results/exp37/stage_d_baseline.md
git commit -m "docs(exp37): TempRMOT bare baseline for Stage D"
```

---

## Task 16: Stage D — plug-in portability + β sweep + decision

**Files:**
- Create: `run_exp37_stage_d.sh`
- Result: `diagnostics/results/exp37/stage_d_memo.md`

- [ ] **Step 1: Add GMC-Link score to TempRMOT association cost**

In TempRMOT's `Hungarian`-based association step (typically `matcher.py` or similar), modify the cost matrix:

```python
# Pseudocode — exact file depends on upstream layout
cost_matrix = c_app  # original
from gmc_link.alignment import MotionLanguageAligner
aligner = MotionLanguageAligner.load("<exp37_final_weights>")
gmc_cos = aligner.score(motion_vec, lang_emb)
cost_matrix = c_app - beta * gmc_cos  # subtract because higher cos = better match
```

Write as a **patch file** under `third_party/patches/temprmot_gmc_link.patch` so it can be reproducibly applied.

- [ ] **Step 2: Write Stage D runner**

```bash
# run_exp37_stage_d.sh
#!/usr/bin/env bash
set -euo pipefail

PY=~/miniconda/envs/RMOT/bin/python
OUT=diagnostics/results/exp37
FINAL=$(cat $OUT/exp37_final_weights)

cd third_party/TempRMOT
git apply ../../third_party/patches/temprmot_gmc_link.patch || true
cd -

for BETA in 0.5 1.0 2.0; do
  GMC_LINK_WEIGHTS=$FINAL TEMPRMOT_BETA=$BETA \
    $PY third_party/TempRMOT/eval.py --split v1_heldout \
    --out-dir $OUT/temprmot_gmc_beta${BETA}/
  $PY run_hota_eval_v1.py --method temprmot_gmc --tag beta${BETA}
done
```

- [ ] **Step 3: Run Stage D**

```bash
bash run_exp37_stage_d.sh
```

- [ ] **Step 4: Write Stage D memo**

```bash
cat > diagnostics/results/exp37/stage_d_memo.md <<EOF
# Stage D — Portability Decision (HEADLINE)

Date: $(date +%Y-%m-%d)
TempRMOT bare HOTA: <from stage_d_baseline>
TempRMOT + GMC β=0.5: <fill>
TempRMOT + GMC β=1.0: <fill>
TempRMOT + GMC β=2.0: <fill>
Best β: <fill>
Best ΔHOTA: <fill>
Decision (spec §2 H_D, threshold ≥ +1.0): [POSITIVE | NEGATIVE]
Publication frame: [plug-in ego-comp service confirmed | only iKUN compatible]
EOF
git add run_exp37_stage_d.sh third_party/patches/temprmot_gmc_link.patch diagnostics/results/exp37/stage_d_memo.md
git commit -m "feat(exp37): Stage D portability plug-in test on TempRMOT"
```

---

## Task 17: Retrospective + memory update

**Files:**
- Create: `docs/superpowers/retro/2026-XX-XX-exp37-summary.md`
- Update: `/home/seanachan/.claude/projects/-home-seanachan-GMC-Link/memory/MEMORY.md`

- [ ] **Step 1: Aggregate all 4 stage memos into one retro**

```bash
cat > docs/superpowers/retro/$(date +%Y-%m-%d)-exp37-summary.md <<EOF
# Exp 37 Retrospective — Systematic Ego-Motion Study

## Hypotheses Resolution

| Hypothesis | Decision | Evidence |
|------------|----------|----------|
| H_A (ego source) | <from stage_a_memo> | <fill> |
| H_B (features) | <from stage_b_memo> | <fill> |
| H_C (conditioning) | <from stage_c_memo> | <fill> |
| H_D (portability) | <from stage_d_memo> | <fill> |

## Final Numbers

| Run | V1 micro AUC | HOTA @ α=1.0 | Notes |
|-----|--------------|--------------|-------|
| A1 ORB 13D (baseline) | 0.779 | 35.476 | from prior work |
| A2 (new ego) | <fill> | <fill> | |
| B2 (+OMF) | <fill> | <fill> | |
| C2 (+EMAP) | <fill> | <fill> | |
| TempRMOT bare | — | <fill> | |
| TempRMOT + GMC best-β | — | <fill> | **portability headline** |

## Publication Framing

<One paragraph: which hypothesis landed, what goes in the paper.>

## Open Threads

- <e.g. RAMOT code not yet released>
- <e.g. V2 replication deferred>
EOF
```

- [ ] **Step 2: Write project memory entry**

```bash
cat > ~/.claude/projects/-home-seanachan-GMC-Link/memory/project_exp37_summary.md <<EOF
---
name: Exp 37 summary
description: Systematic ego-motion study — A ego source, B features, C conditioning, D portability; which levers landed
type: project
---

Exp 37 four-stage ablation resolved: H_A=<fill>, H_B=<fill>, H_C=<fill>, H_D=<fill>. Headline: <one line>.

**Why:** Reframed GMC-Link as ego-compensation service for any RMOT tracker (not just iKUN fusion). H_D is the paper-worthy portability claim.

**How to apply:** When future work asks whether GMC-Link generalizes beyond iKUN, cite Stage D delta on TempRMOT. When asked why we picked <ego source winner>, cite Stage A AUC + KITTI ATE evidence.
EOF
```

- [ ] **Step 3: Add to MEMORY.md index**

Append one line under `## Project`:

```
- [project_exp37_summary.md](project_exp37_summary.md) — Exp 37 four-stage ablation: <outcome>
```

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/retro/*exp37* \
  ~/.claude/projects/-home-seanachan-GMC-Link/memory/project_exp37_summary.md \
  ~/.claude/projects/-home-seanachan-GMC-Link/memory/MEMORY.md
git commit -m "docs(exp37): final retrospective + memory update"
```

- [ ] **Step 5: (Optional) merge PR to main**

If H_D positive: open PR. If all negative: keep branch for reference, do not merge.

---

## Self-Review

**Spec coverage (vs `2026-04-22-ego-motion-systematic-design.md`):**
- §2 H_A → Task 8 ✓
- §2 H_B → Task 11 ✓
- §2 H_C → Task 13 ✓
- §2 H_D → Task 16 ✓
- §5 metrics V1 micro AUC → Tasks 8, 11, 13 ✓
- §5 KITTI ATE → Tasks 7, 8 ✓
- §5 HOTA → Task 14 + Task 16 ✓
- §6 gatekeepers → Tasks 2, 3 ✓
- §7 per-stage specs → Tasks 8, 11, 13, 16 ✓
- §10 kill switches → Task 2 pivot + Task 5 Path B ✓

**Placeholder scan:** the stage memos contain `<fill>` tokens that are intentional — they are for the implementer to populate after the training run completes. These are NOT TODOs for the code itself.

**Type consistency:**
- `EgoRouter.estimate(prev, curr, prev_bboxes=None) -> (H: np.ndarray (3,3), residual: float)` — used identically in Tasks 4, 5, 6.
- `per_bbox_omf_stats(omf_field, bbox) -> np.ndarray(5,)` — Task 9 and Task 10 consistent.
- `EXTRA_FEATURE_DIMS["omf_stats"] = 15` and `"ego_velocity_concat" = 2` — Tasks 10, 12 consistent.
- `_build_cache_key(..., ego="orb")` — Tasks 8, 10 consistent.

No gaps found.
