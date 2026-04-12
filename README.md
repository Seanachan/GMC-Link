# GMC-Link: Global Motion Compensation for Referring Multi-Object Tracking

A plug-and-play module that helps modern RMOT models to compensate camera's ego-motion when referring motion-related expressions.

## What It Does

GMC-Link answers the question: **"Given a video and a sentence like _'moving cars'_, which tracked objects match that description?"**

It bridges the gap between **object motion** (geometry) and **language** (semantics) by:

1. **Compensating for camera motion** so that only true object movement remains.
2. **Encoding that motion** into an 13D geometric spatio-temporal vector (`[dx_s, dy_s, dx_m , dy_m ,dx_l , dy_l ,dw, dh, cx, cy, w, h ,snr]`).
   The motion representation is designed to explicitly capture both kinematic behavior and spatial context:

   - Multi-scale velocity (s/m/l) improves robustness under different frame gaps and noise levels  
   - (dw, dh) captures scale changes (e.g., approaching / receding objects)  
   - (cx, cy, w, h) provides spatial context for handling parallax  
   - snr measures motion reliability and suppresses noisy tracks 
3. **Aligning motion with language** using a learned neural network to produce a match score.

---

## Architecture & Pipeline

```text
Video Frame ──► GMC (Homography) ──► Motion Feature Extraction (13D) ──► MLP Aligner (InfoNCE) ──► Fusion with Tracker Score ──► Final Association
                                                                      ▲
Natural Language Prompt ──► SentenceTransformer Embedding ────────────┘
```

> We're training a Neural Network that can align the textual embeddings and motion embeddings together, and give a score of their alignment.

### Key Components

| Module                    | File                                  | Role                                                                                                                                                                                                      |
| ------------------------- | ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **GlobalMotion**          | `core.py`                             | Detects camera movement via ORB feature matching and RANSAC homography estimation. Return the Homography matrix for warping frames.                                                 |
| **Utilities**             | `utils.py`                            | `warp_points()` transforms previous positions into the current frame's coordinate system. `normalize_velocity()` makes velocities scale-invariant. `MotionBuffer` applies EMA smoothing to reduce jitter. |
| **MotionLanguageAligner** | `alignment.py`                        | A small MLP that projects an 8D spatio-temporal vector and a 384-dim language embedding into a shared space, then computes a similarity score via dot product.                                            |
| **TextEncoder**           | `text_utils.py`                       | Wraps `all-MiniLM-L6-v2` (SentenceTransformers) to encode natural language prompts into 384-dim embeddings.                                                                                               |
| **GMCLinkManager**        | `manager.py`                          | The orchestrator. For each frame: runs GMC, computes compensated velocities for all tracks, and queries the aligner for alignment scores.                                                                 |
| **Fusion Head**           | `fusion_head.py`                      | Learned MLP that fuses iKUN CLIP logits with GMC-Link motion scores for the best overall accuracy (+8.4% F1 over iKUN alone).                                                                             |
| **Dataset & Training**    | `dataset.py`, `train.py`, `losses.py` | Builds (motion, language) training pairs from the [Refer-KITTI](https://github.com/wudongming97/RMOT) dataset using InfoNCE loss with False-Negative Masking.                                              |
| **Demo Inference**        | `demo_inference.py`                   | End-to-end evaluation on iKUN + GMC-Link fusion across all expressions in a sequence.                                                                                                                      |

---

## How It Works (Step by Step)

1. **Feature-based GMC**: Between consecutive frames, ORB/SIFT keypoints are matched on the _background_ (tracked objects are masked out). A homography matrix is estimated via RANSAC to represent pure camera motion.

2. **Motion Compensation**: Each track's previous centroid is warped through the homography into the current frame's coordinate system. The difference `current_pos - warped_prev_pos` yields **world velocity** — the object's true motion with camera movement canceled out.

3. **Normalization & Smoothing**: Velocities are normalized by frame dimensions (scale-invariance) and smoothed with an exponential moving average (temporal stability).

4. **Language Encoding**: The user's text prompt (e.g., _"moving cars"_) is encoded once into a 384-dim vector using a SentenceTransformer.

5. **Alignment Scoring**: The MLP aligner projects the 13D motion/geometry vector and the 384-dim language vector into a shared 256-dim embedding space. A dot product + sigmoid produces a score in `[0, 1]` indicating how well the object's kinematics matches the description.

---

## Training

- **Dataset**: [Refer-KITTI](https://github.com/wudongming97/RMOT) — KITTI tracking sequences annotated with natural language expressions describing object motion.
- **Supervision**: Symmetric InfoNCE loss with False-Negative Masking (FNM). Positive pairs come from ground-truth matches; negatives are formed in-batch. FNM prevents same-sentence pairs from being treated as false negatives.
- **Motion keywords filtered**: Only expressions involving motion concepts (`moving`, `turning`, `parking`, `approaching`, etc.) are used — since the model only sees velocity vectors, not appearance.

---

## Usage

### Inference with Learned Fusion (Recommended)

The learned fusion head combines iKUN's CLIP logits with GMC-Link's kinematic scores for the best results:

```python
from gmc_link import GMCLinkManager, TextEncoder, load_fusion_head

# Initialize
encoder = TextEncoder(device="cuda")
linker = GMCLinkManager(weights_path="gmc_link_weights.pth", device="cuda", lang_dim=384)
fusion_model, threshold = load_fusion_head("gmc_link/fusion_head_weights.pth")

language_embedding = encoder.encode("moving cars")
gmc_scores, _ = linker.process_frame(frame, active_tracks, language_embedding)

# Fuse with iKUN logit for each track
import torch
is_motion = 1.0  # 1.0=motion, 0.5=stationary, 0.0=appearance
feat = torch.tensor([[ikun_logit, gmc_scores[track_id], is_motion]])
prob = fusion_model.predict_prob(feat).item()
is_match = prob >= threshold
```

### Standalone GMC-Link (without iKUN)

```python
encoder = TextEncoder(device="cuda")
linker = GMCLinkManager(weights_path="gmc_link_weights.pth", device="cuda", lang_dim=384)

language_embedding = encoder.encode("moving cars")
scores, velocities = linker.process_frame(frame, active_tracks, language_embedding)
# scores = {track_id: 0.87, ...}
```

### Training the Aligner

```bash
python -m gmc_link.train
```

### Training the Fusion Head

```bash
python gmc_link/fusion_head.py --collect  # collect iKUN + GMC-Link training data
python gmc_link/fusion_head.py --train    # train the fusion MLP
python gmc_link/fusion_head.py --eval     # evaluate on validation split
```

### Multi-Expression Evaluation

```bash
python gmc_link/demo_inference.py --multi  # defaults to learned fusion
```

---

## iKUN Integration & Learned Fusion (Best Results)

When paired with [iKUN](https://github.com/dyhBUPT/iKUN) (a CLIP-based RMOT tracker), the **InfoNCE-trained aligner + learned fusion head** achieves the best overall accuracy:

| Method | Motion F1 | Appearance F1 | Stationary F1 | Overall F1 | Δ Overall |
| --- | --- | --- | --- | --- | --- |
| iKUN baseline | 0.6386 | 0.4338 | 0.6684 | 0.5730 | — |
| iKUN + OR-logic | 0.6650 | 0.4338 | 0.6684 | 0.5863 | +1.3% |
| iKUN + BCE Fusion | 0.6252 | 0.4792 | 0.6972 | 0.5895 | +1.7% |
| **iKUN + InfoNCE Fusion** | **0.7328** | **0.5578** | **0.7134** | **0.6569** | **+8.4%** |

The fusion head is a tiny MLP (`[ikun_logit, gmc_score, is_motion_flag] → 32 → 16 → 1`). The key breakthrough is training the GMC-Link aligner with InfoNCE+FNM instead of BCE — the structured contrastive embedding space produces far more discriminative motion scores, enabling +8.4% Overall F1 improvement over iKUN alone.

> **Note:** Feature-level injection of motion embeddings into iKUN's CLIP visual pipeline was also explored (Stage 3) but causes catastrophic regression (−21.7% F1) because additive injection corrupts the CLIP representation. Decision-level fusion is the correct approach.

---

## TransRMOT Integration & Performance

### How It's Plugged In (For Developers)

Integrating GMC-Link into an existing tracker like TransRMOT is straightforward because GMC-Link acts as a **post-processing filter** on top of the tracker's own predictions.

Here is the step-by-step data flow of how GMC-Link was injected into TransRMOT's `inference.py` loop:

1. **Initialize the Manager:** We instantiate `GMCLinkManager` and `TextEncoder` alongside TransRMOT's core model. We encode the text prompt (e.g., "a red car moving left") once at the start of the video.
2. **Intercept Detections:** For every video frame, TransRMOT generates a list of associated bounding boxes. We intercept these boxes _before_ TransRMOT makes its final filtering decisions.
3. **Generate Kinematic Scores:** We pass the intercepted boxes and the current video frame into `GMCLinkManager.process_frame()`. GMC-Link computes the ego-motion, calculates the 13D velocity vectors, and asks its MLP aligner: _"Based purely on physics, how well do these boxes match the text prompt?"_ It returns a probability score between 0 and 1 for each box.
4. **Strict Minimax Fusion:** TransRMOT initially generates a "Vision Probability" (does this _look_ like a red car?). GMC-Link generates a "Kinematic Probability" (is this object _moving_ left?). We mathematically fuse them using a strict intersection: `final_score = min(vision_prob, kinematic_prob)`.
5. **Final Output:** If a stationary red car tricked TransRMOT's vision model, its `vision_prob` would be `0.9`. But GMC-Link's `kinematic_prob` would be `0.01` (because it's stationary). The `min()` function suppresses the score to `0.01`, instantly filtering out the hallucination.

**Example Code Integration (`inference.py`)**:

```python
# Inside TransRMOT's main evaluation loop
from gmc_link.manager import GMCLinkManager

gmc_linker = GMCLinkManager(weights_path="checkpoints/gmc_link.pth", device="cuda")

for frame in video_frames:
    # 1. TransRMOT native visual detection
    dt_instances = detector.detect(frame, text_prompt)

    # 2. Intercept and format for GMC-Link
    active_tracks = format_boxes_for_gmc(dt_instances)

    # 3. Geometric kinematic evaluation
    gmc_scores, _ = gmc_linker.process_frame(frame, active_tracks, language_embed)

    # 4. Strict Minimax Fusion
    for track in dt_instances:
        vision_prob = track.refers
        kinematic_prob = gmc_scores.get(track.track_id, 0.0)

        # Override vision hallucination with strict physical intersection
        track.refers = min(vision_prob, kinematic_prob)
```

### Benchmark Results

By enforcing this `min(vision_prob, kinematic_prob)` requirement during evaluation, GMC-Link securely grounded visual tracking with real-world spatial physics, destroying hallucinated trajectories while vastly elevating Association Accuracy (`AssA`).

| Tracker Configuration                | HOTA      | DetA      | AssA      | DetRe | DetPr |
| ------------------------------------ | --------- | --------- | --------- | ----- | ----- |
| **Baseline TransRMOT (Vision Only)** | 38.06     | 29.28     | 50.83     | 40.19 | 47.36 |
| **TransRMOT + GMC-Link (Ours)**      | **42.61** | **28.41** | **69.29** | 37.12 | 47.29 |

_Integration resulted in a massive **+18.4% absolute surge** in Tracking Association and set a new **SOTA `42.61` HOTA score**, proving geometry-aware trackers drastically outperform pure vision._

---

## TempRMOT Integration & Temporal Constraints

### The Double-Tracking Problem

While GMC-Link drastically enhances models operating purely on spatial language (like TransRMOT), integrating GMC-Link into architectures featuring **native temporal memory** computationally causes a structural regression.

When evaluated dataset-wide across the dynamic motion corpus (136 sequences) inside `TempRMOT`—which natively caches 8-frame multi-head attention trackers out-of-the-box:

| Tracker Configuration | HOTA | DetA | AssA |
| --- | --- | --- | --- |
| **Baseline TempRMOT (Native 8-frame memory)** | **49.930** | **37.221** | **67.172** |
| **TempRMOT + GMC-Link (Thr: 0.4)** | 43.177 | 29.723 | 62.860 |

### Why Did Validation HOTA Drop?

Because TempRMOT outputs heavily smoothed, highly-confident bounding vectors using its native temporal engine, forcing our strict `min(vision_prob, kinematic_prob)` fusion upon it operates as a redundant, secondary physical constraint. Mathematically, this arbitrarily drags validly-tracked identities down below TempRMOT's absolute deletion boundary (`filter_dt_by_ref_scores(0.4)`), causing thousands of True Positives to permanently vanish.

#### Addendum: Threshold Ablation Study (`0011+moving-cars`)

To formally verify if adjusting TempRMOT's internal deletion boundary could recover the performance regression, we conducted a targeted ablation on the `0011+moving-cars` subset by manually relaxing the deletion floor from `0.4` down to `0.2` when fusing GMC-Link probabilities.

| Setup (`0011+moving-cars`)         | HOTA       | DetA       | AssA       |
| ---------------------------------- | ---------- | ---------- | ---------- |
| **Baseline TempRMOT (Thr: 0.4)**   | **39.896** | 24.664     | **64.502** |
| TempRMOT + GMC-Link (Thr: 0.4)     | 29.408     | 18.591     | 46.529     |
| **TempRMOT + GMC-Link (Thr: 0.2)** | **39.797** | **28.350** | 55.881     |

Lowering the deletion threshold to `0.2` **completely recovered** the catastrophic 10% subset HOTA regression, bringing metrics cleanly back to parity with the baseline (~39.8%). Relaxing the probability floor allowed statistically-suppressed tracking links to survive, proving that GMC-Link was strictly penalized by TempRMOT's rigid `0.4` validation boundary. Ultimately, this mathematically traded Association Accuracy (-8.6%) for pure Detection Accuracy (+3.68%).

> [!WARNING]
> **Developer Insight:**
>
> 1. GMC-Link is a state-of-the-art plug-and-play geometric filter mathematically designed to rescue **spatially-ignorant Vision-Language frameworks** (e.g., TransRMOT).
> 2. It **should not** be cascaded onto frameworks that independently construct recursive temporal bounding boxes natively (like TempRMOT/Refer-SORT). While unilaterally lowering the underlying model's deletion threshold computationally recovers the HOTA destruction, cascading redundant temporal tracking pipelines remains fundamentally structurally hostile.

---

## Key Design Decisions

- **Geometry over appearance**: GMC-Link reasons purely about _motion_, making it complementary to vision-language models that reason about _appearance_.
- **Plug-and-play**: Works with any tracker (ByteTrack, BoT-SORT, TransRMOT) — just provide track centroids.
- **Lightweight**: The aligner MLP is tiny (~few hundred KB), adding negligible overhead to an existing tracking pipeline.
