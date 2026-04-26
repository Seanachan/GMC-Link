# Exp 36A Rich Motion Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend motion representation from 13D to 25D (add multi-scale accel + sin/cos heading) via the existing `extra_features` mechanism; train + eval against stage1 baseline 0.779 micro-AUC.

**Architecture:** Reuse existing `EXTRA_FEATURE_DIMS` registry in `gmc_link/dataset.py` (training path) and mirror feature computation in `GMCLinkManager.compute_motion_vec` in `gmc_link/manager.py` (inference path). Aligner auto-scales `motion_dim` from `13 + compute_extra_dims(extra_features)`.

**Tech Stack:** PyTorch, NumPy, existing GMC-Link training harness. No new deps.

---

## File Structure

- `gmc_link/dataset.py` — add two feature names + compute functions (training-side)
- `gmc_link/manager.py` — mirror computation in inference-side motion-vector builder
- `run_exp36a.sh` — orchestrator: train + 3-seq eval + aggregate
- `docs/superpowers/specs/2026-04-20-exp36a-rich-motion-features-design.md` — reference spec (already committed)

No new files beyond the shell orchestrator. Single training config.

---

## Task 1: Register `accel_multiscale` and `heading_sincos` in dataset.py

**Files:**
- Modify: `gmc_link/dataset.py:115-124` (EXTRA_FEATURE_DIMS)
- Modify: `gmc_link/dataset.py:135-168` (compute_per_track_extras)

- [ ] **Step 1: Extend EXTRA_FEATURE_DIMS registry**

Find the dict at `gmc_link/dataset.py:115`:
```python
EXTRA_FEATURE_DIMS = {
    "speed_m": 1,       # F1: mid-scale speed magnitude
    "heading_m": 1,     # F2: mid-scale heading angle
    "accel": 2,         # F3: acceleration (long - short velocity)
    "ego_motion": 2,    # F4: ego/camera motion (dx, dy)
    "neighbor_mean_vel": 2,  # F5: mean velocity of other tracks
    "velocity_rank": 1,      # F6: speed rank among neighbors
    "heading_diff": 1,       # F7: heading vs mean flow
    "nn_dist": 1,            # F8: nearest neighbor distance
    "track_density": 1,      # F9: density of nearby tracks
}
```

Append two entries, so the final dict is:
```python
EXTRA_FEATURE_DIMS = {
    "speed_m": 1,
    "heading_m": 1,
    "accel": 2,
    "ego_motion": 2,
    "neighbor_mean_vel": 2,
    "velocity_rank": 1,
    "heading_diff": 1,
    "nn_dist": 1,
    "track_density": 1,
    "accel_multiscale": 6,   # exp36a: (dx,dy) × {2,5,10} second-order diff
    "heading_sincos": 6,     # exp36a: (sin,cos) × {2,5,10} heading
}
```

- [ ] **Step 2: Extend compute_per_track_extras()**

The function lives at `gmc_link/dataset.py:135`. It accepts `scale_velocities = [(dx_s, dy_s), (dx_m, dy_m), (dx_l, dy_l)]` and iterates feature names. Add two elif branches inside the `for feat in extra_features:` loop (before the `# F5-F9 are relational` comment):

```python
        elif feat == "accel_multiscale":
            # Second-order per-scale: use difference across the scale pair as
            # a proxy for d²x/dt² since we already have velocities at gaps
            # {2,5,10}. This is equivalent to (v_long - v_short) per axis,
            # normalized by scale. We emit all three axis-scale combos.
            # Short-mid, mid-long, short-long deltas capture different
            # temporal acceleration signatures.
            extras.extend([
                dx_m - dx_s,  # accel between short and mid
                dy_m - dy_s,
                dx_l - dx_m,  # accel between mid and long
                dy_l - dy_m,
                dx_l - dx_s,  # accel between short and long
                dy_l - dy_s,
            ])
        elif feat == "heading_sincos":
            # Smooth heading encoding per scale (avoids atan2 ±π discontinuity)
            # When |v|→0, atan2(0,0)=0 → sin=0, cos=1 (stationary encoded).
            for dx, dy in scale_velocities:
                theta = np.arctan2(dy, dx)
                extras.append(float(np.sin(theta)))
                extras.append(float(np.cos(theta)))
```

- [ ] **Step 3: Smoke test the registry additions**

Run:
```bash
~/miniconda/envs/RMOT/bin/python -c "
from gmc_link.dataset import EXTRA_FEATURE_DIMS, compute_extra_dims, compute_per_track_extras
assert EXTRA_FEATURE_DIMS['accel_multiscale'] == 6
assert EXTRA_FEATURE_DIMS['heading_sincos'] == 6
assert compute_extra_dims(['accel_multiscale', 'heading_sincos']) == 12
sv = [(1.0, 0.0), (2.0, 1.0), (3.0, 2.0)]
e = compute_per_track_extras(['accel_multiscale', 'heading_sincos'], sv)
print(f'len={len(e)} expected=12')
print(f'values={e}')
assert len(e) == 12
print('OK')
"
```
Expected: `len=12 expected=12 ... OK`

- [ ] **Step 4: Commit**

```bash
git add gmc_link/dataset.py
git commit -m "$(cat <<'EOF'
feat(exp36a): register accel_multiscale + heading_sincos features

Adds two new entries to EXTRA_FEATURE_DIMS and compute_per_track_extras
in dataset.py. accel_multiscale emits 6D (dx,dy × 3 scale-pair deltas);
heading_sincos emits 6D (sin,cos × 3 scales) for wraparound-free
heading encoding.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Mirror computation in GMCLinkManager (inference side)

**Files:**
- Modify: `gmc_link/manager.py:52` (aligner motion_dim default arg & load path)
- Modify: `gmc_link/manager.py:238-250` (motion-vector build block)

The manager currently hardcodes 13D. We need it to:
1. Read `extra_features` + `motion_dim` from checkpoint
2. Recompute features symmetrically to training side
3. Append to spatial_motion before aligner call

- [ ] **Step 1: Read extra_features from checkpoint**

Find the constructor block at `gmc_link/manager.py:38-62`. Locate:
```python
self.aligner = MotionLanguageAligner(
    motion_dim=13, lang_dim=lang_dim, embed_dim=256
).to(device)

self.temperature = 1.0  # default (no scaling)
if weights_path:
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        self.aligner.load_state_dict(checkpoint["model"])
        self.temperature = checkpoint.get("temperature", 1.0)
    else:
        self.aligner.load_state_dict(checkpoint)
self.aligner.eval()
```

Replace with a block that reads `motion_dim` + `extra_features` from the checkpoint and reinstantiates the aligner at the correct dim:
```python
self.extra_features: list[str] = []
motion_dim = 13
self.temperature = 1.0
checkpoint = None
if weights_path:
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        motion_dim = checkpoint.get("motion_dim", 13)
        self.extra_features = checkpoint.get("extra_features") or []
        self.temperature = checkpoint.get("temperature", 1.0)

self.aligner = MotionLanguageAligner(
    motion_dim=motion_dim, lang_dim=lang_dim, embed_dim=256
).to(device)
if checkpoint is not None:
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        self.aligner.load_state_dict(checkpoint["model"])
    else:
        self.aligner.load_state_dict(checkpoint)
self.aligner.eval()
```

- [ ] **Step 2: Append extras in motion-vector builder**

Find the block at `gmc_link/manager.py:238-250` (the 13D `spatial_motion` array construction). Add an import at top of file:
```python
from gmc_link.dataset import compute_per_track_extras
```

After the existing line:
```python
spatial_motion = np.array(
    [dx_s, dy_s, dx_m, dy_m, dx_l, dy_l,
     dw, dh, cx_n, cy_n, w_n, h_n, snr], dtype=np.float32
)
```

Insert:
```python
if self.extra_features:
    # Per-track (non-relational) extras only — manager has no neighbor context.
    per_track_names = [
        f for f in self.extra_features
        if f in {"speed_m", "heading_m", "accel", "ego_motion",
                 "accel_multiscale", "heading_sincos"}
    ]
    if per_track_names:
        scale_velocities = [(dx_s, dy_s), (dx_m, dy_m), (dx_l, dy_l)]
        extras = compute_per_track_extras(per_track_names, scale_velocities)
        spatial_motion = np.concatenate(
            [spatial_motion, np.array(extras, dtype=np.float32)]
        )
```

- [ ] **Step 3: Symmetry smoke test (manager ⇌ dataset)**

Run:
```bash
~/miniconda/envs/RMOT/bin/python -c "
import numpy as np
from gmc_link.dataset import compute_per_track_extras
sv = [(1.5, -0.5), (2.5, -1.0), (3.5, -1.5)]
names = ['accel_multiscale', 'heading_sincos']
a = np.array(compute_per_track_extras(names, sv), dtype=np.float32)
b = np.array(compute_per_track_extras(names, sv), dtype=np.float32)
assert a.shape == (12,), f'shape {a.shape}'
assert np.allclose(a, b)
# Manual accel check
assert np.isclose(a[0], 2.5 - 1.5)  # dx_m - dx_s
assert np.isclose(a[1], -1.0 - -0.5)  # dy_m - dy_s
# Manual heading check (mid scale)
theta_m = np.arctan2(-1.0, 2.5)
assert np.isclose(a[8], np.sin(theta_m))  # sin mid
assert np.isclose(a[9], np.cos(theta_m))  # cos mid
print('symmetry OK')
print(f'values: {a.tolist()}')
"
```
Expected: `symmetry OK` plus 12-element vector.

- [ ] **Step 4: Commit**

```bash
git add gmc_link/manager.py
git commit -m "$(cat <<'EOF'
feat(exp36a): mirror rich motion features in GMCLinkManager

Manager now reads extra_features + motion_dim from checkpoint metadata
and appends the same feature vector in compute_motion_vec() that
dataset.py emits during training. Only per-track (non-relational)
features are supported on the inference side — relational features
need neighbor context we don't build here.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Orchestrator script

**Files:**
- Create: `run_exp36a.sh`

- [ ] **Step 1: Write orchestrator**

```bash
#!/usr/bin/env bash
# Exp 36A: rich motion features (13D -> 25D) with MLP aligner.
# Trains stage1 with accel_multiscale + heading_sincos, evals on held-out
# seqs {0005, 0011, 0013}, aggregates against stage1 baseline.
#
# Usage: bash run_exp36a.sh > /tmp/exp36a.log 2>&1 &

set -euo pipefail

PY=~/miniconda/envs/RMOT/bin/python
DIAG=diagnostics/diag_gt_cosine_distributions.py
RESULTS_DIR=diagnostics/results
MULTISEQ_DIR="${RESULTS_DIR}/multiseq"
SEQS=(0005 0011 0013)
TAG="v1train_exp36a"
WEIGHTS="gmc_link_weights_${TAG}.pth"

echo "[$(date +%H:%M:%S)] Training ${WEIGHTS} (13D + accel_multiscale + heading_sincos = 25D)"
"${PY}" -m gmc_link.train \
    --split v1 \
    --extra-features accel_multiscale,heading_sincos \
    --epochs 5 --lr 3e-4 --batch-size 128 \
    --seed 42 \
    --save-path "${WEIGHTS}"

echo "[$(date +%H:%M:%S)] Evaluating on held-out seqs"
for seq in "${SEQS[@]}"; do
    "${PY}" "${DIAG}" --weights "${WEIGHTS}" --seq "${seq}" 2>&1 | tail -6
    mv "${RESULTS_DIR}/layer3_gt_cosine_${seq}.npz" \
       "${MULTISEQ_DIR}/layer3_${seq}_${TAG}.npz"
    if [[ -f "${RESULTS_DIR}/layer3_gt_cosine_${seq}.png" ]]; then
        mv "${RESULTS_DIR}/layer3_gt_cosine_${seq}.png" \
           "${MULTISEQ_DIR}/layer3_${seq}_${TAG}.png"
    fi
done

echo "[$(date +%H:%M:%S)] Aggregating against stage1 baseline"
ALL_WEIGHTS=(
    "v1train_stage1=gmc_link_weights_v1train_stage1.pth"
    "${TAG}=${WEIGHTS}"
)
"${PY}" diagnostics/aggregate_multiseq.py \
    --results-dir "${MULTISEQ_DIR}" \
    --output-dir "${MULTISEQ_DIR}" \
    --weights "${ALL_WEIGHTS[@]}" \
    --seqs "${SEQS[@]}"

echo "[$(date +%H:%M:%S)] Done. See ${MULTISEQ_DIR}/layer3_multiseq_comparison.md"
```

- [ ] **Step 2: Make executable + syntax check**

```bash
chmod +x run_exp36a.sh
bash -n run_exp36a.sh && echo "bash syntax OK"
```

- [ ] **Step 3: Commit**

```bash
git add run_exp36a.sh
git commit -m "$(cat <<'EOF'
chore(exp36a): orchestrator for train + 3-seq eval + aggregate

Trains v1train_exp36a with --extra-features accel_multiscale,heading_sincos
(25D motion), evaluates on held-out {0005, 0011, 0013}, aggregates against
stage1 baseline. Expected wall-clock ~35 min.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: End-to-end run + report

**Files:**
- Read: `diagnostics/results/multiseq/layer3_multiseq_comparison.md`

- [ ] **Step 1: Execute orchestrator**

```bash
bash run_exp36a.sh > /tmp/exp36a.log 2>&1 &
```

Tail log periodically, or wait for completion (~35 min). Check:
```bash
tail -30 /tmp/exp36a.log
```

- [ ] **Step 2: Inspect aggregated comparison**

```bash
cat diagnostics/results/multiseq/layer3_multiseq_comparison.md
```

Expected columns: `model_tag | mean_auc_micro | mean_auc_macro ± std | best_seq | worst_seq | max_gap`. Verify `v1train_exp36a` row present alongside `v1train_stage1` at 0.779.

- [ ] **Step 3: Apply decision rule (pre-registered in spec)**

Based on `mean_auc_micro` for `v1train_exp36a`:
- `≥ 0.80` → **win**. Tag commit, write retrospective memory, propose Exp 36C (features + transformer) as stretch.
- `0.79 ≤ x < 0.80` → **marginal**. Run seed sweep {123, 7}, report n=3 mean/std.
- `< 0.79` → **flat**. Write retrospective, proceed to Exp 36B (13D + transformer T=30).

Also inspect per-seq: even if mean flat, a ≥+0.03 lift on seq 0011 (worst baseline) is a targeted signal worth preserving.

- [ ] **Step 4: Retrospective memory**

Save to `~/.claude/projects/-home-seanachan-GMC-Link/memory/project_exp36a_<outcome>.md` where outcome ∈ {win, marginal, flat}. Template:

```markdown
---
name: Exp 36A rich motion features (<outcome>)
description: <one-line result summary>
type: project
---

<result table — micro/macro AUC per seq, delta vs stage1>

**Why:** <what the outcome tells us about the ceiling>

**How to apply:** <next experiment recommendation>
```

- [ ] **Step 5: Commit retrospective + weights**

Weights file (`.pth`) is gitignored. Commit only the diagnostic PNGs (if generated), npz is also gitignored. Commit the retrospective memory update via MEMORY.md pointer.

```bash
git add -p diagnostics/results/multiseq/  # diagnostic PNGs if any
# Update MEMORY.md pointer to new retrospective file (manual edit)
git commit -m "docs(exp36a): <outcome> retrospective + metrics"
```

---

## Self-Review

- [x] **Spec coverage:** Task 1 (register features) + Task 2 (manager mirror) + Task 3 (orchestrator) + Task 4 (run + decide) cover all spec requirements.
- [x] **Placeholder scan:** No TBD/TODO/vague sections. Every step has exact command or code.
- [x] **Type consistency:** `EXTRA_FEATURE_DIMS` dict key names (`accel_multiscale`, `heading_sincos`) are used identically in Task 1 Step 2 branches, Task 2 Step 2 whitelist, Task 3 Step 1 `--extra-features` flag, and Task 4 smoke test.
- [x] **Symmetry:** Task 2 Step 3 explicitly tests manager/dataset produce identical vectors for synthetic input; manager reuses `compute_per_track_extras` from dataset.py rather than reimplementing.
- [x] **Decision rule alignment:** Task 4 Step 3 matches the spec's pre-registered thresholds (0.80 / 0.79-0.80 / <0.79).
