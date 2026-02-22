# GMC-Link Training Research Notes

## Objective
Train a `MotionLanguageAligner` to match 2D velocity vectors with natural language motion descriptions on the Refer-KITTI dataset.

---

## Experiment Log

### Exp 1: Baseline CLIP-Style Contrastive Loss
- **Data:** Seq 0011 only, 40,898 samples from `labels_with_ids` (single-frame velocity)
- **Loss:** CLIP symmetric cross-entropy (N×N matrix, diagonal = ground truth)
- **Config:** batch=16, lr=1e-4, epochs=50
- **Result:** Loss stuck at ~6.88/6.93. Near random.
- **Issue:** Batch too small for contrastive learning, single-frame velocities too tiny (~0.001)

### Exp 2: Multi-Frame Velocity + Larger Batch
- **Data:** All 4 seqs, 15,542 samples, 5-frame gap velocities from KITTI tracking labels
- **Config:** batch=1024, lr=1e-4, epochs=100
- **Result:** Loss 6.88 → 6.44
- **Issue:** Many samples share same sentence → false negatives in contrastive batch

### Exp 3: Explicit Negative Sampling (BROKEN)
- **Change:** Added 1:1 negative pairs (same motion, wrong sentence) to dataset
- **Result:** Loss stuck at 6.80
- **Root Cause:** CLIP loss assumes diagonal = ground truth. Injecting wrong pairs into dataset and shuffling means ~50% of diagonal entries are intentionally wrong → contradictory supervision

### Exp 4: Positive-Only + Smaller Batch
- **Change:** Removed negative sampling, batch=128 (fewer sentence collisions), lr=1e-3, cosine LR scheduler
- **Result:** Loss 6.88 → 6.44 (with all seqs) → 4.39 (with motion-filtered expressions)
- **Issue:** Loss floor near `ln(128)=4.85`. With 40 unique sentences in batch of 128, many "false negatives" remain

### Exp 5: Motion Keyword Filtering
- **Change:** Only expressions with motion keywords (moving, parking, turning, counter/same direction, braking, slower)
- **Data:** 40 motion expressions, 3,038 samples → filtered down from 91 expressions
- **Result:** Loss 4.58 → 4.50 (plateaued)
- **Issue:** Still CLIP loss bottleneck + tiny velocity magnitudes

### Exp 6: Velocity Scaling (100x)
- **Change:** Added `VELOCITY_SCALE=100` in `utils.py`, applied in both training and inference
- **Velocity stats before scaling:** mean |dx|=0.016, mean |dy|=0.012
- **After scaling:** mean |dx|=1.6, mean |dy|=1.2 (much better for MLP)
- **Result:** Loss 4.54 → 4.50 (still plateaued due to CLIP loss limit)

### Exp 7: Switch to BCE Loss ✅
- **Change:** Complete loss function redesign:
  - `losses.py`: `BCEWithLogitsLoss` replacing CLIP cross-entropy
  - `alignment.py`: Added `score_pairs()` for per-pair cosine similarity
  - `dataset.py`: Each sample now has label (1.0=positive, 0.0=negative)
  - 3:1 negative ratio (same motion, random wrong sentence)
- **Data:** Train: seqs 15/16/18, Test: seq 11 (proper split)
- **Config:** batch=128, lr=1e-3, cosine scheduler, 200 epochs
- **Result:** Loss 0.29, Accuracy 82.2% ✅
- **E2E on seq 0011:** GT avg=0.52, non-GT avg=0.56 (separation: -0.04 ❌)
- **Issue:** Model doesn't generalize to held-out seq 0011. Possible causes:
  - Different velocity distributions between sequences
  - IoU matching only finds 19 GT matches out of ~660 expected
  - `forward()` (N×N matrix) vs `score_pairs()` (element-wise) may behave differently at inference

---

## Key Bugs Fixed Along the Way

| File | Bug | Fix |
|------|-----|-----|
| `core.py` | `len(cv2.DMatch)` crash in Lowe's ratio test | Check `len(match_pair)==2` first |
| `core.py` | Mask initialized to all zeros (no features detected) | Changed to `np.ones * 255` |
| `manager.py` | `prev_centroid=None` passes `hasattr` check | Added `or track.prev_centroid is None` |
| `alignment.py` | `vis_dim` parameter misleading | Renamed to `motion_dim` |
| `visualize.py` | `GMCLink` import mismatch | Changed to `GMCLinkManager` |
| `train.py` | Relative imports fail as script | Changed to absolute imports + `sys.path.insert` |
| `train.py` | `num_workers=4` deadlocks with HF tokenizers on MPS | Removed `num_workers` |

---

## Current Architecture

```
                    ┌─────────────┐
                    │  YOLO + BT  │ → Detect & track objects
                    └──────┬──────┘
                           │ bboxes + track IDs
                    ┌──────▼──────┐
                    │  GMC Engine │ → ORB features + RANSAC homography
                    │  (core.py)  │   to estimate camera motion
                    └──────┬──────┘
                           │ homography H
                    ┌──────▼──────┐
                    │   Manager   │ → Warp prev positions by H
                    │ (manager.py)│   Compute world velocity
                    └──────┬──────┘
                           │ compensated velocity (dx, dy) × 100
                    ┌──────▼──────┐     ┌──────────────┐
                    │   Aligner   │ ◄── │ TextEncoder   │
                    │(alignment.py)│     │ (MiniLM-L6)  │
                    └──────┬──────┘     └──────────────┘
                           │ similarity score → sigmoid → [0, 1]
                    ┌──────▼──────┐
                    │ Visualization│
                    └─────────────┘
```

## Open Questions
1. Why does IoU matching only find 19 GT matches? (YOLO boxes vs GT boxes misalignment?)
2. Why doesn't the 82% training accuracy generalize to seq 0011?
3. Is the velocity distribution fundamentally different between train/test sequences?
