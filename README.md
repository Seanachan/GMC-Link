# GMC-Link: Global Motion Compensation for Referring Multi-Object Tracking
A plug-and-play module that helps modern RMOT models to compensate camera's ego-motion when referring motion-related expressions.

## What It Does

GMC-Link answers the question: **"Given a video and a sentence like _'moving cars'_, which tracked objects match that description?"**

It bridges the gap between **object motion** (geometry) and **language** (semantics) by:

1. **Compensating for camera motion** so that only true object movement remains.
2. **Encoding that motion** into an 8D geometric spatio-temporal vector (`[dx, dy, dw, dh, cx, cy, w, h]`).
3. **Aligning motion with language** using a learned neural network to produce a match score.

---

## Architecture & Pipeline

```
Video Frame ──► GMC (Homography) ──► Compensated Velocity ──► MLP Aligner ──► Score [0, 1]
                                                                   ▲
Natural Language Prompt ──► SentenceTransformer Embedding ─────────┘
```

### Key Components

| Module | File | Role |
|---|---|---|
| **GlobalMotion** | `core.py` | Detects camera movement via ORB/SIFT feature matching and RANSAC homography estimation. Masks out tracked objects so only background features contribute. |
| **Utilities** | `utils.py` | `warp_points()` transforms previous positions into the current frame's coordinate system. `normalize_velocity()` makes velocities scale-invariant. `MotionBuffer` applies EMA smoothing to reduce jitter. |
| **MotionLanguageAligner** | `alignment.py` | A small MLP that projects an 8D spatio-temporal vector and a 384-dim language embedding into a shared space, then computes a similarity score via dot product. |
| **TextEncoder** | `text_utils.py` | Wraps `all-MiniLM-L6-v2` (SentenceTransformers) to encode natural language prompts into 384-dim embeddings. |
| **GMCLinkManager** | `manager.py` | The orchestrator. For each frame: runs GMC, computes compensated velocities for all tracks, and queries the aligner for alignment scores. |
| **Dataset & Training** | `dataset.py`, `train.py`, `losses.py` | Builds (motion, language) training pairs from the [Refer-KITTI](https://github.com/wudongming97/RMOT) dataset using BCE loss. Trains on sequences 0015/0016/0018, tests on 0011. |
| **Demo Inference** | `demo_inference.py` | End-to-end evaluation using YOLOv8 + ByteTrack for real detections, then GMC-Link for motion-language alignment scoring. |
| **Visualization** | `visualize.py` | Renders annotated frames with bounding boxes, velocity arrows, and alignment scores. |

---

## How It Works (Step by Step)

1. **Feature-based GMC**: Between consecutive frames, ORB/SIFT keypoints are matched on the *background* (tracked objects are masked out). A homography matrix is estimated via RANSAC to represent pure camera motion.

2. **Motion Compensation**: Each track's previous centroid is warped through the homography into the current frame's coordinate system. The difference `current_pos - warped_prev_pos` yields **world velocity** — the object's true motion with camera movement canceled out.

3. **Normalization & Smoothing**: Velocities are normalized by frame dimensions (scale-invariance) and smoothed with an exponential moving average (temporal stability).

4. **Language Encoding**: The user's text prompt (e.g., *"moving cars"*) is encoded once into a 384-dim vector using a SentenceTransformer.

5. **Alignment Scoring**: The MLP aligner projects the 8D motion/geometry vector and the 384-dim language vector into a shared 256-dim embedding space. A dot product + sigmoid produces a score in `[0, 1]` indicating how well the object's kinematics matches the description.

---

## Training

- **Dataset**: [Refer-KITTI](https://github.com/wudongming97/RMOT) — KITTI tracking sequences annotated with natural language expressions describing object motion.
- **Supervision**: BCE loss on (motion, language) pairs. Positive pairs come from ground-truth matches; negative pairs use mismatched sentences.
- **Motion keywords filtered**: Only expressions involving motion concepts (`moving`, `turning`, `parking`, `approaching`, etc.) are used — since the model only sees velocity vectors, not appearance.

---

## Usage

### Inference (Notebook)

```python
encoder = TextEncoder(device="cuda")
linker = GMCLinkManager(device="cuda", lang_dim=384)

language_embedding = encoder.encode("moving cars")
scores, velocities = linker.process_frame(frame, active_tracks, language_embedding)
# scores = {track_id: 0.87, ...}
```

### Training

```bash
python -m gmc_link.train
```

---

## Key Design Decisions

- **Geometry over appearance**: GMC-Link reasons purely about *motion*, making it complementary to vision-language models that reason about *appearance*.
- **Plug-and-play**: Works with any tracker (ByteTrack, BoT-SORT, etc.) — just provide track centroids.
- **Lightweight**: The aligner MLP is tiny (~few hundred KB), adding negligible overhead to an existing tracking pipeline.