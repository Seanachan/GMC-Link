# Research Branch: Learning Camera Motion Compensation

This branch explores **learning camera ego-motion compensation** instead of using geometric preprocessing.

## Motivation

Instead of:
```python
# Geometric preprocessing
world_coords = warp(image_coords, homography)
score = MLP(world_motion_vector, language)
```

We try:
```python
# Learned compensation
homography_features = extract_features(homography)
score = MLP(image_motion_vector, homography_features, language)
```

The model learns how to compensate for camera motion from data.

## Approach

### Input Features

**Motion Vector (8D)** - Image-frame coordinates:
- `dx, dy` - Centroid velocity in image space
- `dw, dh` - Depth scaling velocity
- `cx, cy` - Normalized centroid position
- `w, h` - Normalized bbox size

**Homography Features (5D)** - Geometric camera motion:
- `tx, ty` - Translation (camera pan)
- `sx, sy` - Scale (zoom in/out)
- `theta` - Rotation angle

### Architecture

```
┌─────────────────────┐
│  Image Motion (8D)  │
│  [dx, dy, ...]      │
└──────────┬──────────┘
           │
        MLP (64→128→256)
           │
           ├──────────────┐
           │              │
┌──────────▼──────────┐   │
│ Homography Feat (5D)│   │
│ [tx, ty, sx, sy, θ] │   │
└──────────┬──────────┘   │
           │              │
        MLP (32→64)       │
           │              │
           └──────┬───────┘
                  │
           Fusion (320→256)
                  │
           ┌──────▼──────┐
           │   L2 Norm   │
           └──────┬──────┘
                  │
           Cosine Similarity
                  │
                  ▼
            Language Embed
```

### Training

Same as original GMC-Link:
- Dataset: Refer-KITTI motion expressions
- Loss: BCE on (motion, homography, language) triplets
- Negative sampling: Wrong sentences + hard negatives

### Expected Benefits

✅ **Robustness**: Model learns when to trust/ignore noisy homographies  
✅ **Flexibility**: Can learn non-planar scene compensation  
✅ **Data-driven**: Discovers optimal compensation from data  

### Potential Drawbacks

❌ **Data requirements**: Needs diverse camera motions in training  
❌ **Interpretability**: Black-box compensation vs geometric  
❌ **Generalization**: May not work on unseen camera patterns  

## Files

- `gmc_link/alignment_with_homography.py` - New aligner module
- `gmc_link/manager_with_homography.py` - Manager using learned compensation
- `gmc_link/train_with_homography.py` - Training script (TODO)
- `gmc_link/dataset_with_homography.py` - Dataset with homography features (TODO)

## Usage

```python
from gmc_link.alignment_with_homography import (
    MotionLanguageAlignerWithHomography,
    decompose_homography_features
)

# Initialize model
aligner = MotionLanguageAlignerWithHomography(
    motion_dim=8,
    homography_dim=5,
    lang_dim=384,
    embed_dim=256
)

# Prepare features
motion_vector = np.array([dx, dy, dw, dh, cx, cy, w, h])  # Image frame
homography_features = decompose_homography_features(H_matrix)

# Inference
scores = aligner(
    torch.tensor(motion_vector),
    torch.tensor(homography_features),
    language_embedding
)
```

## Next Steps

1. ✅ Implement `MotionLanguageAlignerWithHomography`
2. ⬜ Modify dataset to include homography features
3. ⬜ Modify training script to use new aligner
4. ⬜ Train and evaluate on Refer-KITTI
5. ⬜ Compare with geometric preprocessing baseline
6. ⬜ Analyze learned camera compensation behavior

## Comparison with Baseline

| Method | Preprocessing | Learning | Interpretability | Robustness |
|--------|--------------|----------|------------------|------------|
| **Baseline (Cumulative Homography)** | Geometric warp | Motion semantics only | ✅ High | ⚠️ Fragile to bad H |
| **This Branch (Learned)** | None | Motion + camera jointly | ❌ Low | ✅ Data-driven robust |

## Experiment Ideas

1. **Ablation**: Train with/without homography features
2. **Visualization**: t-SNE of learned camera representations
3. **Stress test**: Inject bad homographies, measure degradation
4. **Transfer**: Train on KITTI, test on MOT17 (different camera)

## Branch Structure

```
main
 └── feature/cumulative-homography (merged)
      └── research/learn-camera-motion (this branch)
```

To switch back to baseline:
```bash
git checkout feature/cumulative-homography
```
