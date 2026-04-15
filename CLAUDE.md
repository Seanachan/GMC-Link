# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GMC-Link is a plug-and-play module for **Referring Multi-Object Tracking (RMOT)** that bridges object motion (geometry) with natural language (semantics). Given a video and a description like "moving cars", it scores which tracked objects match that description by reasoning about physical motion, not just visual appearance.

**Key Result**: +8.4% F1 improvement (0.5730→0.6569) when fused with iKUN using InfoNCE+FNM loss and a learned fusion head.

## Common Commands

### Training

```bash
# Train the Motion-Language Aligner (main model)
python -m gmc_link.train

# Train the Fusion Head (3-stage pipeline)
python gmc_link/fusion_head.py --collect  # Step 1: collect iKUN logits + GMC scores
python gmc_link/fusion_head.py --train    # Step 2: train MLP
python gmc_link/fusion_head.py --eval     # Step 3: evaluate on validation split
```

### Inference & Evaluation

```bash
# Generate iKUN baseline predictions (vision-only, no GMC)
python run_ikun_baseline_video.py

# Generate fusion predictions (iKUN + GMC-Link)
python run_fusion_video.py --expr moving-cars

# Evaluate with HOTA metrics
python run_hota_eval.py                  # both methods
python run_hota_eval.py --method baseline
python run_hota_eval.py --method fusion

# Multi-expression demo inference
python gmc_link/demo_inference.py --multi
```

### Ablation Studies

```bash
# Run structured ablation (multi-config evaluation)
python run_ablation_study.py

# Shell wrapper for batch ablation runs
bash run_ablation_proper.sh
```

### Package Installation

```bash
pip install -e .
# Dependencies: torch, torchvision, numpy, opencv-python, sentence-transformers, tqdm, scipy
```

## Architecture

### Pipeline Stages

**Stage 1 — Ego-Motion Compensation** (`gmc_link/core.py`):
- `ORBHomographyEngine` extracts ORB features, matches with BFMatcher (Hamming, Lowe's ratio=0.7), estimates RANSAC homography
- Foreground masking prevents tracking object features instead of static background
- Output: 3×3 homography matrix mapping previous frame → current frame

**Stage 2 — Cumulative Homography & Velocity** (`gmc_link/manager.py`):
- `GMCLinkManager` stores *original* (never-warped) centroid coordinates in history deques
- Maintains cumulative composed homographies: H[t-k→t] = H[t-1→t] @ ... @ H[t-k→t-k+1]
- Computes **multi-scale residual velocity** at three temporal gaps (2, 5, 10 frames) to capture different motion patterns
- Residual velocity = raw velocity − ego velocity, isolating true object movement
- EMA smoothing: `MotionBuffer` (α=0.3) and `ScoreBuffer` (α=0.4) in `utils.py`
- Outputs a **13D motion vector**: `[res_dx_s, res_dy_s, res_dx_m, res_dy_m, res_dx_l, res_dy_l, dw, dh, cx, cy, w, h, snr]`

**Stage 3 — Motion-Language Alignment** (`gmc_link/alignment.py`):
- `MotionLanguageAligner`: dual MLP projecting 13D motion → 256D and 384D language → 256D into shared embedding space
- Cosine similarity remapped to [0, 1] as alignment score
- Trained with symmetric InfoNCE loss + False-Negative Masking (`gmc_link/losses.py`)
- Language embeddings: SentenceTransformer (all-MiniLM-L6-v2, 384D) via `gmc_link/text_utils.py`

**Stage 4 — Fusion Head** (`gmc_link/fusion_head.py`):
- Tiny MLP: `[ikun_logit, gmc_score, is_motion_flag]` → 3→32→16→1 (sigmoid)
- `is_motion_flag`: 1.0 for motion expressions, 0.5 for stationary, 0.0 for appearance-only
- Replaces hand-tuned `min(vision, kinematic)` heuristic with learned combination

### Data Flow

```
Video Frames
    ↓
ORBHomographyEngine → frame-to-frame H matrices
    ↓
GMCLinkManager → compose cumulative H, warp original coords, compute multi-scale residual velocity
    ↓
13D motion vector [res_dx×3scales, res_dy×3scales, dw, dh, cx, cy, w, h, snr]
    ↓
MotionLanguageAligner ←── TextEncoder("moving cars") → 384D embedding
    ↓
Cosine similarity score ∈ [0, 1]
    ↓
FusionHead([ikun_logit, gmc_score, is_motion_flag]) → P(match)
```

### Training Data Pipeline (`gmc_link/dataset.py`)

- Loads Refer-KITTI V2 expressions and ground-truth centroid tracks
- Multi-scale frame gaps `[2, 5, 10]` matching `GMCLinkManager.FRAME_GAPS`
- Applies synthetic positional jitter (±2px) for robustness
- Normalizes velocity: `v_norm = (v_pixel / img_dims) × 100` (resolution-invariant)
- Generates positive (motion_vector, language_embedding) pairs for InfoNCE training

### Key Constants

- `VELOCITY_SCALE = 100` (`utils.py`) — multiplier for normalized velocities so MLP inputs are ~1.0 magnitude
- `FRAME_GAPS = [2, 5, 10]` (`manager.py`) — must match between `GMCLinkManager` and `dataset.py`
- InfoNCE temperature: `0.07` (`losses.py`)
- EMA alphas: `MotionBuffer(α=0.3)`, `ScoreBuffer(α=0.4)`
- Embedding dimensions: motion 13D → 256D, language 384D → 256D (shared space)
- Fusion Head architecture: 3→32→16→1 with sigmoid output

### Project Layout Notes

- `gmc_link/` — the installable package (core library)
- `run_*.py` — top-level experiment/evaluation scripts (not part of the package)
- `build/` — stale `setuptools` build artifacts; do not edit files here
- Weight files (`*.pth`) and data files (`*.npz`) are gitignored

## Data Paths

- Refer-KITTI dataset: `/home/seanachan/data/Dataset/refer-kitti-v2` (also symlinked as `refer-kitti/` and `Refer-KITTI/`)
- Full annotation JSON: `Refer-KITTI_labels.json`
- iKUN precomputed scores: `iKUN/`
- NeuralSORT track detections: `NeuralSORT/`

## Important Design Decisions

- **ORB over optical flow**: ORB+Homography outperforms Farneback and RAFT on KITTI planar scenes; better outlier rejection via RANSAC
- **Decision-level fusion only**: Feature-level injection (injecting motion into CLIP) caused catastrophic regression (−21.7% F1); always fuse at decision level
- **False-Negative Masking**: Multiple training samples can share the same expression; FNM prevents same-sentence pairs from being penalized as negatives
- **Cumulative homography**: Store original coordinates, warp once with composed H — more numerically stable than warping iteratively each frame
- **Multi-scale temporal velocity**: Three frame gaps (2, 5, 10) capture short/mid/long motion patterns; dominant improvement in ablations (+0.047 separation)
- **SNR feature**: Signal-to-noise ratio doesn't improve mean separation but reduces variance (±0.010 → ±0.007), stabilizing predictions
- **Motion keyword detection**: ~38 motion-related keywords (moving, turning, parking, etc.) determine `is_motion_flag` in the fusion head
- **Not for temporal trackers**: GMC-Link is designed for spatially-ignorant vision-language frameworks (e.g., TransRMOT, iKUN). Cascading onto trackers with native temporal memory (e.g., TempRMOT) causes structural regression due to redundant temporal constraints

## Experiment Log

Detailed experiment history (Exp 1–24+) is in `RESEARCH_NOTES.md`, including ablations, loss comparisons, and architectural decisions with exact metric values.
