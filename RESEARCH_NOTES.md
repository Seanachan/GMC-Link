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
- **Issue:** Model doesn't generalize to held-out seq 0011

### Exp 8: GMC Object Masking Fix ✅
- **Change:** Passed YOLO bounding boxes to `gmc_engine.estimate()` in `manager.py`. ORB features on tracked objects were contaminating the background homography.
- **Result (with old 82% model):**
  - FP reduced by 50% (1176 → 586)
  - Precision increased 2.2x (0.84% → 1.86%)
  - GT scores slightly improved, non-GT also rose — retraining needed

### Exp 9: Deeper MLP + Hard Negatives + Object Masking ✅
- **Change:**
  - Motion projector: 2→64→256 → 2→64→128→256 with dropout(0.1)
  - Hard negatives: zero-velocity + correct sentence, inverted velocity + correct sentence
  - 4 negatives per positive (up from 3): 2 hard + 2 random
- **Config:** batch=128, lr=1e-3, cosine scheduler, 300 epochs
- **Training:** Loss 0.2039, Accuracy **89.27%** (up from 82.5%)
- **E2E on seq 0011 (ORB+Homography):**
  - GT avg score: **0.7344** | Non-GT avg: **0.2115** | Separation: **+0.5229** ✅
  - FP: 352 | TP: 15
- **Best result so far** — ORB masking + deeper MLP + hard negatives

### Exp 10: Dense Optical Flow (Farneback) — Full Pipeline
- **Change:** Replaced ORB+homography GMC with `cv2.calcOpticalFlowFarneback`. Per-pixel flow, per-object bbox averaging, background median subtraction for camera compensation. Training also uses flow-derived velocities.
- **Training:** Loss 0.2108, Accuracy 88.93%
- **E2E on seq 0011:**
  - GT avg score: 0.6088 | Non-GT avg: 0.3338 | Separation: **+0.2750**
  - FP: 637 | TP: 17
- **Analysis:** Worse than ORB. Farneback flow is noisy; ORB+RANSAC rejects outliers. KITTI's planar road scenes suit the homography assumption well.

### Exp 11: RAFT Optical Flow (Learned, GPU-Accelerated)
- **Change:** Replaced Farneback with `torchvision.models.optical_flow.raft_small` (pretrained). Runs on MPS/CUDA for GPU acceleration. Auto-pads frames to multiple of 8.
- **Training:** Loss **0.1943**, Accuracy **89.91%** (best training accuracy)
- **E2E on seq 0011:**
  - GT avg score: 0.6000 | Non-GT avg: 0.4944 | Separation: **+0.1056**
  - FP: 1301 | TP: 17
- **Analysis:** Despite best training metrics, worst test-time separation. The RAFT flow field may contain too much fine-grained detail (textures, edges) that pollutes the per-bbox velocity average. The ORB+RANSAC approach, being sparser, actually produces more stable and discriminative velocity signals for this task.

### Exp 12: Resolving Training-Inference Domain Gap & Velocity Scaling
- **Change:** Unified pipeline to use centroid-difference velocities with `frame_gap=1` (replaced RAFT with centroid diffs during inference to match training). Increased `VELOCITY_SCALE` from `100` to `500` to properly amplify the tiny 1-frame pixel distances. Lowered E2E threshold to `0.4`.
- **Training:** Loss 0.3405, Accuracy 84.36% (50 epochs, all 4 sequences)
- **E2E on seq 0011 (Centroid-diff `frame_gap=1`):**
  - GT avg score: 0.4803 | Non-GT avg: 0.3336 | Separation: **+0.1466** ✅
  - FP: 723 | TP: 18 (with threshold 0.4)
- **Analysis:** Domain gap Fixed. Dataset confirmed to be actual GT tracking boxes, not predictions. While signal is much better, pure 1-frame centroid differences are heavily distorted by YOLO bounding-box jitter, leading to high FP. Next step is multi-frame velocity windowing (`frame_gap=5`).

---

## Experiment Comparison Table

| Exp | Method | Train Loss | Train Acc | GT Score | Non-GT Score | Separation | FP |
|-----|--------|-----------|-----------|----------|-------------|-----------|-----|
| 9 | **ORB + Homography** | 0.2039 | 89.27% | **0.7344** | **0.2115** | **+0.5229** | **352** |
| 10 | Farneback Dense Flow | 0.2108 | 88.93% | 0.6088 | 0.3338 | +0.2750 | 637 |
| 11 | RAFT Dense Flow | **0.1943** | **89.91%** | 0.6000 | 0.4944 | +0.1056 | 1301 |
| 12 | Centroid-diff (`gap=1`) | 0.3405 | 84.36% | 0.4803 | 0.3336 | +0.1466 | 723 |

**Conclusion:** ORB + Homography with object masking remains the best approach for KITTI driving scenes. Dense flow methods introduce noise that sparse keypoint matching avoids. Centroid differences are fast and stable but highly susceptible to YOLO detection jitter when `frame_gap=1`.

---

## Key Bugs Fixed Along the Way

| File | Bug | Fix |
|------|-----|-----|
| `core.py` | `len(cv2.DMatch)` crash in Lowe's ratio test | Check `len(match_pair)==2` first |
| `core.py` | Mask initialized to all zeros (no features detected) | Changed to `np.ones * 255` |
| `manager.py` | Object bboxes not passed to GMC engine | Added `detections` parameter to `process_frame` |
| `manager.py` | `prev_centroid=None` passes `hasattr` check | Added `or track.prev_centroid is None` |
| `alignment.py` | `vis_dim` parameter misleading | Renamed to `motion_dim` |
| `visualize.py` | `GMCLink` import mismatch | Changed to `GMCLinkManager` |
| `train.py` | Relative imports fail as script | Changed to absolute imports + `sys.path.insert` |
| `train.py` | `num_workers=4` deadlocks with HF tokenizers on MPS | Removed `num_workers` |
| `demo_inference.py` | `draw_frame_visualization` truncated mid-function | Fixed missing else/label/HUD overlay |

---

## Current Architecture

```
                    ┌─────────────┐
                    │  YOLO + BT  │ → Detect & track objects (ByteTrack)
                    └──────┬──────┘
                           │ bboxes + track IDs
                    ┌──────▼──────┐
                    │  Flow Engine│ → RAFT / ORB+Homography / Farneback
                    │  (core.py)  │   for ego-motion compensation
                    └──────┬──────┘
                           │ per-pixel flow / homography
                    ┌──────▼──────┐
                    │   Manager   │ → Extract per-object velocity
                    │ (manager.py)│   Subtract background flow
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
1. Why does IoU matching only find ~20 GT matches? (YOLO boxes vs GT boxes misalignment?)
2. Can ORB results be further improved with better hyperparameters or ensemble methods?
3. Would fine-tuning RAFT on KITTI-specific flow help, or is the issue fundamental to dense flow averaging?

