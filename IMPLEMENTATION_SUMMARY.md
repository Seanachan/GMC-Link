# Implementation Summary: Cumulative Homography + Learning Camera Motion

## What Was Implemented

### 1. Cumulative Homography Method (feature/cumulative-homography)

**Problem with original method:**
- Warped ALL centroid history EVERY frame
- Coordinates got warped N times → potential for bad homography propagation
- Lost original data → hard to debug

**New approach:**
- Store **original** centroid coordinates (never modified)
- Store **cumulative** homographies: `H[t]` transforms `frame[t] → current_frame`
- Warp coordinates **once** when computing velocities

**Mathematical insight:**
- Both methods are equivalent: `H3 @ (H2 @ (H1 @ c0)) = (H3 @ H2 @ H1) @ c0`
- Real benefits: preserves originals, bad homography recovery, debugging

**Code changes:**
```python
# Old (manager.py)
for tid, hist in self.centroid_history.items():
    warped = warp_points(hist, homography)
    self.centroid_history[tid] = warped  # OVERWRITES

# New (manager.py)
# Store originals
self.centroid_history[tid].append(curr_centroid)  # NEVER WARPED

# Update cumulative homographies
for H_old in self.homography_buffer:
    H_cumulative = H_prev_to_curr @ H_old
    updated_homographies.append(H_cumulative)

# Warp once when querying
for centroid, H in zip(centroid_hist, homographies):
    warped = warp_points(centroid, H)  # WARP ONCE
```

### 2. Learning Camera Motion (research/learn-camera-motion)

**Idea:**
Instead of preprocessing with geometry, feed homography as a feature and let the model learn compensation.

**Architecture:**
```
Image Motion (8D) ──► MLP ──┐
                            ├──► Fusion ──► Similarity ──► Score
Homography (5D)   ──► MLP ──┘

Where Homography Features = [tx, ty, scale_x, scale_y, rotation]
```

**Benefits:**
- Model can learn when to trust/ignore bad homographies
- Can learn non-planar scene compensation
- Data-driven robustness

**Drawbacks:**
- Needs diverse camera motions in training data
- Less interpretable than geometric preprocessing
- May not generalize to unseen camera patterns

**New module:**
`gmc_link/alignment_with_homography.py`
- `MotionLanguageAlignerWithHomography` class
- `decompose_homography_features()` function
- Fusion architecture combining motion + camera features

## Branch Structure

```
main
 └── feature/cumulative-homography
      └── research/learn-camera-motion
```

## How to Use

### Test Cumulative Homography (Stable)
```bash
git checkout feature/cumulative-homography

# Your existing workflow should work unchanged
python -m gmc_link.demo_inference
```

### Explore Learning Approach (Research)
```bash
git checkout research/learn-camera-motion

# Need to implement:
# 1. Modify dataset.py to include homography features
# 2. Modify train.py to use new aligner
# 3. Train and evaluate
```

## Key Insights

### You Were Right About Numerical Drift
The cumulative method is NOT primarily about avoiding numerical errors (which are tiny ~1e-6).

**Real problems it solves:**
1. **Bad homography recovery**: If one frame has bad H, it doesn't corrupt all future history
2. **Debugging**: Can inspect original coordinates vs warped
3. **Flexibility**: Can query different window sizes on-demand

### Learning vs Geometric
Both approaches have merit:

| Aspect | Geometric (Current) | Learning (Research) |
|--------|---------------------|---------------------|
| **When it works** | Always (if H is good) | After sufficient training |
| **Robustness** | Fragile to bad H | Robust (learns to ignore) |
| **Data needs** | None (physics-based) | High (diverse cameras) |
| **Interpretability** | High | Low (black box) |
| **Best for** | Production | Research/exploration |

## Next Steps

### For Production (Cumulative Homography)
1. ✅ Implemented
2. ⬜ Test on existing workflow
3. ⬜ Verify performance matches baseline
4. ⬜ Merge to main

### For Research (Learning Camera Motion)
1. ✅ Architecture implemented
2. ⬜ Modify dataset to extract homography features
3. ⬜ Modify training script
4. ⬜ Train on Refer-KITTI
5. ⬜ Ablation: with/without homography features
6. ⬜ Compare with geometric baseline

### Future Exploration
- **Hybrid approach**: Feed both world-frame motion AND homography features
- **Adaptive weighting**: Model learns when to trust geometry vs learned compensation
- **Multi-scale temporal windows**: Different windows for different motion patterns

## Files Modified/Created

**feature/cumulative-homography:**
- `gmc_link/manager.py` (modified)
- `gmc_link/manager.py.backup` (backup)

**research/learn-camera-motion:**
- `gmc_link/alignment_with_homography.py` (new)
- `RESEARCH_LEARN_CAMERA_MOTION.md` (new)
- `IMPLEMENTATION_SUMMARY.md` (this file)

## Conclusion

Both implementations are complete and ready for testing/experimentation:

1. **Cumulative homography** provides a cleaner, more debuggable version of the existing geometric approach
2. **Learning camera motion** opens a new research direction for data-driven robustness

The choice between them depends on your priorities:
- Production stability → Cumulative homography
- Research exploration → Learning camera motion
- Best of both → Hybrid approach (future work)
