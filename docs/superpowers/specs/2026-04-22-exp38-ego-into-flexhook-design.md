# Exp 38 — Ego-Motion-Aware FlexHook (Beat SOTA Attempt)

**Date:** 2026-04-22
**Author:** Seanachan (with Claude Opus 4.7)
**Supersedes:** Exp 37 final retrospective next-experiment hooks
**Branch target:** `exp/flexhook-ego` (off `main`)
**Related experiments:** Exp 35 (FlexHook-adjacent aligner, NEG), Exp 37 (ego-motion systematic, all NEG)
**External refs:**
- FlexHook (arXiv:2503.07516, CVPR'26, SOTA on Refer-KITTI V1/V2/LaMOT)
- Dynamic Registration (arXiv:2204.12769, 2022) — iterative joint ego + moving object
- MotionNet (arXiv:2003.06754, CVPR'20) — BEV ego-compensated motion, "compensation is critical"

---

## 0. One-Sentence Pitch

Fork FlexHook, replace its raw-displacement `speed` tensor at `mymodel.py:288` with our multi-scale ego-compensated residual velocity from `GMCLinkManager`, finetune from `refer-kitti-best.pth`, and measure HOTA delta on Refer-KITTI V1 + V2. Target: beat FlexHook SOTA (V1 53.82 / V2 42.53) by ≥ +1.0 HOTA.

---

## 1. Hypotheses & Falsification

### H₁ — Primary (Feature-level ego replacement)

**H₀:** Replacing FlexHook's raw per-frame bbox displacement with our 13D ego-compensated multi-scale residual velocity (gaps {2, 5, 10}) does not move HOTA on Refer-KITTI V1 / V2 beyond the reproduction noise band (±0.3 HOTA) of the published SOTA.

**H₁:** Ego-compensation inside FlexHook's full architecture (PCD + ROPE-Swin-T + RoBERTa) lifts HOTA by ≥ +1.0 on V1 and ≥ +0.5 on V2. Key premise: the PCD decoder is large enough to exploit richer motion (which Exp 35's MLP proxy could not), and FlexHook's absent ego compensation is a genuine structural gap.

**Pre-registered decision rule on V1 HOTA** (V2 confirmed secondary):

| Band | Δ HOTA vs 53.82 baseline | Outcome |
|---|---|---|
| Strong positive | ≥ +2.0 | SOTA attack; push V2 + write paper; add Hooks B/C for ablation depth |
| Positive | +1.0 to +2.0 | Publishable — run ablations (38-B, 38-C) to decompose contribution |
| Inconclusive | [−0.3, +1.0] | Repro noise range; per-seq and per-expression breakdown to decide |
| Negative | ≤ −0.3 | Ego compensation still doesn't help even with rich decoder. Falsifies "moat" hypothesis. Pivot to appearance fusion (prior next-step option). |

### H₂ — Iterative ego refinement (Hook A, from Dynamic Registration)

**H₀:** Iterative ego-motion refinement (1-shot vs 2-iter vs 3-iter ORB+homography, masking already-detected track bboxes from the prior frame before each subsequent estimation) provides no incremental gain over 1-shot on top of H₁'s winner config.

**H₁:** Iterative refinement adds ≥ +0.3 HOTA on top of H₁'s winner, measured by 38-B vs 38-A headline comparison.

**Decision:** Only run 38-B if 38-A is positive or inconclusive. If 38-A is strong positive, 38-B adds ablation depth. If negative, 38-B is suspended.

### H₃ — Dense motion field (Hook B, from MotionNet)

**H₀:** Replacing per-bbox 13D with dense per-pixel ego-compensated residual velocity map (pooled per-bbox) does not exceed 38-A's headline.

**H₁:** Dense field grants FlexHook's PCD decoder richer spatial conditioning, lifting HOTA by ≥ +0.5 over 38-A.

**Decision:** Conditional on 38-A positive. If 38-A negative, 38-C is suspended.

---

## 2. Scope: What Changes, What Doesn't

### Does NOT change

- `gmc_link/core.py` `ORBHomographyEngine` — reuse as-is for ego estimation.
- `gmc_link/manager.py` `GMCLinkManager` multi-scale residual velocity — reuse as-is.
- `gmc_link/utils.py` `MotionBuffer`, `ScoreBuffer` — not used in FlexHook context (no EMA over referring scores; FlexHook PCD handles temporal via self-attention).
- `gmc_link/alignment.py`, `gmc_link/losses.py`, `gmc_link/fusion_head.py` — irrelevant for Exp 38 (we're not using our aligner; we're injecting ego velocity directly into FlexHook).
- FlexHook's tracker outputs (`Temp-NeuralSORT-kitti{1,2}`) — reused as-is.
- FlexHook's pretrained weights (ROPE-Swin-T, RoBERTa, CLIP variants) — reused as-is.

### Does change (FlexHook fork)

- **Fork location:** `~/FlexHook-ego/` (sibling to `~/FlexHook/`; clean copy of commit-current state).
- **`models/mymodel.py`** — modify C-Hook `speed` computation:
  - Env-var gated `FLEXHOOK_EGO={off,on_13d,on_iter,on_dense}`.
  - `off` → original `speed = cur_pos_raw[:,1:] - cur_pos_raw[:,:-1]` (bare FlexHook).
  - `on_13d` → load precomputed ego-compensated residual velocity from cache, map to FlexHook's `cur_pos_raw` coordinate grid (3 scale-indexed pos_avg_pool levels).
  - `on_iter` → same as `on_13d` but cache built with iterative ORB refinement.
  - `on_dense` → load dense per-pixel velocity map, grid_sample under bbox grid.
- **New cache pipeline:** `tools/build_ego_cache.py` — precompute per-frame ego-compensated velocity tensors aligned to FlexHook's bbox/grid format. Cache location: `~/FlexHook-ego/cache/ego_speed/{seq}/{frame_id}.npz`.
- **Training script:** `finetune_ego.sh` — finetune from `refer-kitti-best.pth` with `FLEXHOOK_EGO=on_13d` for N epochs.
- **Inference script:** `infer_ego.sh` — single-GPU (3060 Ti constraint) eval variant.

### Does change (GMC-Link side)

- **New:** `tools/flexhook_ego_extractor.py` — standalone script that runs `GMCLinkManager` over a seq and writes ego-velocity tensors in FlexHook's cache format.
- **Branch:** `exp/flexhook-ego` off `main`, merged onto `exp/hn-mining` post-success.

### Explicit non-goals for Exp 38

- **NOT** re-training FlexHook from scratch (cost prohibitive on 3060 Ti; rely on finetune).
- **NOT** modifying FlexHook's PCD decoder architecture (only the `speed` input changes).
- **NOT** touching FlexHook's tracker upstream (Temp-NeuralSORT outputs are inputs).
- **NOT** fusing our aligner cosine at decision level (Stage D showed temporal-aware consumers regress; 38 is feature-level).
- **NOT** evaluating on Refer-Dance or LaMOT (KITTI V1 + V2 only — our tooling + ORB cache scoped to KITTI).

---

## 3. Architecture

### 3.1 Baseline FlexHook motion signal (bare, `FLEXHOOK_EGO=off`)

Pre-existing at `mymodel.py:283-297`:

```python
for i, output in enumerate(outputs):
    cur_pos_raw = self.pos_avg_pool[i](pos_raw)  # (B, T, 2, H_i, W_i)
    b, t, _, qh, qw = cur_pos_raw.shape

    speed = cur_pos_raw[:, 1:] - cur_pos_raw[:, :-1]  # raw displacement
    speed = torch.cat([speed, cur_pos_raw[:, -1:]], dim=1)  # replicate last
    cur_pos_raw = cur_pos_raw.flatten(0, 1)
    speed = speed.flatten(0, 1)

    obj_f = F.grid_sample(output, cur_pos_raw.permute(0, 2, 3, 1),
                          padding_mode='zeros', align_corners=False)
    obj_f = torch.cat([obj_f, speed], dim=1)  # concat speed as channels
```

- `pos_raw`: normalized bbox grid coordinates `∈ [-1, 1]`.
- `pos_avg_pool[i]` downsamples to scales `H_i × W_i ∈ {(28,84), (14,42), (7,21)}` (three PCD scales).
- `speed`: per-frame delta in normalized coordinates — **no ego compensation**.

### 3.2 Ego-compensated variant (`FLEXHOOK_EGO=on_13d`)

Pipeline:

```
Raw video frames
    ↓
ORBHomographyEngine (ego-motion H_{t-1 → t}, foreground-masked)
    ↓
GMCLinkManager.step(bbox_xyxy)  per-track, per-frame
    ↓
13D motion vector cached per (seq, frame, track_id):
  [res_dx_s, res_dy_s, res_dx_m, res_dy_m, res_dx_l, res_dy_l,
   dw, dh, cx, cy, w, h, snr]
    ↓
For each FlexHook sample (B, T=4 frames, track tokens):
  Look up cached 13D per (seq, frame, track_id)
  Project residual velocities [res_dx_{s,m,l}, res_dy_{s,m,l}] to normalized-grid delta units
  Broadcast to FlexHook's (H_i × W_i) grid per scale i ∈ {0,1,2}
    ↓
Replace `speed` in mymodel.py:288 with ego-compensated projection
```

**Coordinate alignment** (critical):
- FlexHook's `speed` lives in normalized grid coords `[-1, 1]` per frame.
- Our residual velocity lives in `(v_pixel / img_dims) × VELOCITY_SCALE` where `VELOCITY_SCALE=100`.
- Conversion: `speed_normgrid[i] = residual_velocity[i] * (img_dim / 100) * (2 / img_dim) = residual_velocity[i] * 0.02`.
- Scale mapping: FlexHook's PCD pos_avg_pool levels ↔ our FRAME_GAPS ∈ {2, 5, 10}. Use short gap (2) for PCD scale 0, mid (5) for scale 1, long (10) for scale 2.

### 3.3 Iterative ego (`FLEXHOOK_EGO=on_iter`)

Per frame pair `(t-1, t)`:

```python
H = orb_engine.estimate(frame_t_1, frame_t, mask=None)
for iter in range(max_iter):
    # Reproject all known track bboxes at t to t-1 using H
    bbox_masks = project_bboxes_to_prev_frame(tracks_t, H)
    # Re-estimate H masking out dynamic regions
    H = orb_engine.estimate(frame_t_1, frame_t, mask=bbox_masks)
```

- `max_iter ∈ {1, 2, 3}` sweep in 38-B.
- `tracks_t` = Temp-NeuralSORT-kitti1 outputs for frame t.
- Converges when `||H_new - H_old||_F < 1e-4` or max_iter hit.

### 3.4 Dense motion field (`FLEXHOOK_EGO=on_dense`)

Per frame:

```python
# Dense warped flow
prev_warped = cv2.warpPerspective(frame_t_1, H, frame_t.shape[:2])
raw_flow = farneback(prev_warped, frame_t)  # OR raft model
# Residual = raw flow
# Cache as (H, W, 2) tensor per (seq, frame)
```

At FlexHook inference time:
- Grid-sample dense field under `cur_pos_raw` scaled grid.
- Replace `speed` with the grid-sampled dense value.

Risk: dense field cache ~100GB for KITTI (compute on 3060 Ti feasible ~10 min/seq).

---

## 4. Data Flow

### Training (finetune) sample

```
1. Dataset: FlexHook's existing (B=80, T=4 frames, N=8 expressions) samples.
2. Per-(seq, frame, track_id), load cached ego-compensated 13D → (B, T, 13).
3. Project to 3 PCD scale grids: (B, T, 2, H_i, W_i) for i ∈ {0,1,2}.
4. Forward through FlexHook C-Hook with `obj_f = cat([vit_feat, ego_speed], dim=1)`.
5. Forward through PCD decoder as usual.
6. Standard FlexHook loss (InfoNCE + dispersion + task-head losses).
```

### Inference

```
1. For each seq ∈ {0005, 0011, 0013} (V1 held-out):
   a. Run ORB on frames → per-frame H_t.
   b. Apply GMCLinkManager with Temp-NeuralSORT-kitti1 tracks → 13D per (frame, track).
   c. Cache to npz.
2. Run FlexHook infer_ego.sh with FLEXHOOK_EGO=on_13d → eval results.
3. HOTA via TrackEval with FlexHook's seqmap.
```

### Cache specs

- Path: `~/FlexHook-ego/cache/ego_speed/{seq}/ego_13d.npz`
- Key schema: `{frame_id}_{track_id}` → np.float16 array shape (13,)
- Size estimate: V1 has 3 held-out seqs × ~500 frames × ~30 tracks × 13 × 2 bytes = ~1.2 MB. Trivial.
- V2 full val: ~50 seqs × 500 × 30 × 13 × 2 = ~20 MB. Still trivial.

---

## 5. Training

### Finetune regime

| Knob | Value | Notes |
|---|---|---|
| Init | `SOTA_ckpts/refer-kitti-best.pth` | ROPE-Swin-T backbone |
| Optimizer | AdamW | Match FlexHook default (`config.TRAIN.OPTIMIZER`) |
| LR | 3e-5 decoder / 0.0 backbone | Match FlexHook kitti1.yaml BASE_LR |
| LR schedule | Multistep @ {15, 30} | Match kitti1.yaml |
| Epochs | 10 | Finetune short; full FlexHook is 40 epochs, we do 1/4 |
| Batch size | 40 per GPU (single GPU 3060 Ti) | Half of FlexHook default (80) due to memory |
| Precision | fp16 autocast | Needed on 3060 Ti 6GB |
| Cache regeneration | Once per ego mode | |
| Seed | 0 | Match FlexHook SEED in yaml |

### Single-GPU constraint

FlexHook's `infer.sh` uses `--nproc_per_node=2` (distributed). On our 3060 Ti we run single-GPU:
- Replace `torch.distributed.launch --nproc_per_node=2` with single-process python.
- Halve batch size 80 → 40.
- Gradient accumulation 2× to preserve effective batch.

### Data

- Refer-KITTI V1 train split (FlexHook's `datasets/refer-kitti` layout).
- Validation: V1 held-out seqs 0005, 0011, 0013 (our same held-out).
- Expressions: FlexHook's existing expression pipeline.

### Training time estimate

Full repro finetune: ~10 epochs × 30 min/epoch (single GPU, batch 40) ≈ 5 hrs per run.
Total for 38-A + 38-B (3 iter variants) + 38-C ≈ 25 hrs of GPU compute.

---

## 6. Evaluation

### Primary gate

HOTA on Refer-KITTI V1 via FlexHook's `eval.sh` pipeline (TrackEval + their seqmap).

Report:
- Headline HOTA (primary gate; apply §1 decision rule).
- DetA, AssA, DetRe, DetPr, AssRe, AssPr (full HOTA decomposition).
- Per-seq HOTA on 0005, 0011, 0013 — the 0011 "hard" seq per prior memory is the critical test.

### Secondary

- V2 HOTA confirmation (if V1 positive).
- Per-expression breakdown — which expression types benefit most from ego compensation (expected: motion verbs like "turning", "parking", "approaching").
- Attention-weight diagnostic inside PCD: does the decoder attend more to our ego `speed` channels than it did to raw `speed` channels?

### Artifacts

- `~/FlexHook-ego/retest-kitti-1/exp38_{a,b_iter1,b_iter2,b_iter3,c_dense}/` — inference outputs per arm.
- `diagnostics/results/exp38/{stage_a_memo,stage_b_memo,stage_c_memo,retrospective}.md` — written to GMC-Link repo.
- `run_exp38_stage_{a,b,c}.sh` — drivers in GMC-Link root.

---

## 7. Ablations (Required for 38-A Positive Path)

### A — 38-A headline (no ablation, main run)

FlexHook + `FLEXHOOK_EGO=on_13d` finetune. Reported against §1 decision rule.

### A1 — Zero-velocity control

`FLEXHOOK_EGO=zero` (new flag): set `speed = torch.zeros_like(original_speed)`. Purpose: lower bound — how much of FlexHook's HOTA comes from motion at all? If A1 ≈ bare, motion signal contributes ~0 and Exp 38 is moot.

### A2 — Scrambled-velocity control

`FLEXHOOK_EGO=scramble`: shuffle the time axis of ego-compensated speed across batch. Purpose: verify signal structure matters (not just magnitude).

### A3 — Raw-displacement head-to-head (inside same finetune regime)

Rerun bare FlexHook with our finetune regime (LR=3e-5, 10 epochs) and `FLEXHOOK_EGO=off` to control for finetune effects. This gives **bare_finetuned** baseline, isolated from "checkpoint init" confound.

Report table row: `bare_finetuned`, `a_ego_13d`, `b_iter2`, `b_iter3`, `c_dense`, `a1_zero`, `a2_scramble`, `a3_bare_finetuned`.

---

## 8. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| FlexHook bare repro fails on single GPU 3060 Ti (OOM, arch mismatch) | **Medium-high** | Stage 0 gate: verify bare repro produces published HOTA within ±0.5 before touching ego integration. If fails, reduce batch to 20 + grad accum 4. |
| Coord alignment bug: normalized grid ↔ residual velocity projection wrong | Medium | Unit test `tools/flexhook_ego_extractor.py` with synthetic known-H case. Assert grid-sampled dummy flow at bbox center = bbox velocity. |
| Ego cache drift: Temp-NeuralSORT tracks don't match FlexHook's internal (seq, frame) indexing | Medium | Dump FlexHook's internal (seq, frame, track_id) iterator keys; build cache keyed identically. |
| Finetune from `refer-kitti-best.pth` destabilizes with new `speed` distribution | Medium | Warmup: first 2 epochs at LR=3e-6 with `speed` set to zero; then ramp up to full ego. |
| CLAUDE.md tracker-class dichotomy applies (FlexHook PCD temporal attention = temporal-aware consumer) | Low-medium | Feature-level replace ≠ decision-level stack. PCD still gets only one motion signal (ours), not two (ours + its own). Stage D's pattern ("two temporal smoothers over-regularize") does not apply here. |
| GPU memory: ROPE-Swin-T + RoBERTa + B=40 + T=4 + ego cache exceeds 6GB free | Medium | fp16 autocast + gradient checkpointing on backbone; if still OOM, drop T to 3 frames. |
| Ego signal structurally incompatible with FlexHook's PCD (speed was raw-grid, ego is in different dim) | Low | §3.2 conversion accounts for this; sanity-check grid histograms at train start. |
| Exp 35's result ("raw-disp beats ego under FlexHook-adjacent arch") generalizes to full FlexHook PCD | **Medium** | This is the key risk — the whole Exp 38 bets against Exp 35's finding. Mitigation: run 38-A headline ASAP; if NEG, kill Exp 38 fast and pivot to appearance fusion. |
| V2 dataset path broken (our cache infra only tested on V1) | Low | V2 is confirmation-only; defer V2 until V1 positive. |

---

## 9. Open Questions

1. **V1-first vs V2-first:** V1 has our prior baselines (all 12-lever NEG); V2 is FlexHook's headline benchmark. → **V1 first** (infra reuse). V2 follows after V1 positive.
2. **`pos_avg_pool` scale ↔ FRAME_GAPS mapping:** three scales `{(28,84), (14,42), (7,21)}` vs three gaps `{2, 5, 10}`. Map fine→short, coarse→long. → Confirmed in §3.2.
3. **Finetune epochs:** 10 vs 20 vs 40 (full). → **10 as default**; if 38-A inconclusive, try 20. 40 epochs only if single run produces clear positive.
4. **Do we unfreeze anything?** → **No**; only the C-Hook projection layers that consume `speed` (trivially small, ~K params) are trainable. Keeps finetune cheap + reproducible.
5. **Dense field backend (38-C):** Farneback vs RAFT vs PWC-Net. → **RAFT** (SOTA, manageable inference cost on 3060 Ti via small variant).

---

## 10. Kill-switches / Go-NoGo Gates

| Gate | Condition | Action |
|---|---|---|
| G0 — bare repro | FlexHook bare HOTA on V1 within ±0.5 of 53.82 | If fail → diagnose env/config before Stage 38-A |
| G1 — cache sanity | Grid histogram of ego-projected `speed` overlaps raw `speed` within ±3σ per channel | If fail → coord conversion bug, fix before training |
| G2 — 38-A headline | Δ HOTA per §1 decision rule | If NEG → skip 38-B, 38-C; pivot to appearance fusion |
| G3 — V1 vs V2 consistency | If V1 pos and V2 flat/neg, claim scope-limited to V1 | If V2 also pos → full SOTA claim |
| G4 — Exp 35 contradiction | If 38-A pos strongly, re-examine Exp 35 to explain architecture gap | Write note for paper's related work |

---

## 11. Paper Narrative

**Working title:** *"Ego-Motion is Not Optional: Camera-Compensated Motion for Referring Multi-Object Tracking"*

**Contributions (provisional, bound by experimental outcome):**

1. **Identify** ego-motion omission in RMOT SOTA (FlexHook, TempRMOT, iKUN baselines).
2. **Bring** LiDAR-community ego-compensation practice (MotionNet CVPR'20, Dynamic Registration 2022) to 2D camera RMOT.
3. **Show** feature-level injection of ego-compensated residual velocity into FlexHook's PCD lifts HOTA by +X on Refer-KITTI V1 / V2.
4. **Ablate** iterative ego refinement (Hook A) and dense motion field (Hook B) to decompose contribution.
5. **Report** tracker-class dichotomy (Exp 37 Stage D): ego-fusion helps spatially-ignorant consumers (iKUN), hurts temporally-aware ones (TempRMOT decision-level). Feature-level replacement (Exp 38) avoids this regression by not stacking temporal smoothers.

**Target venues:** CVPR'27 (May 2026 deadline), ICCV'26 (March deadline — tight), ECCV'26 (Nov deadline — comfortable), AAAI'27.

**Negative-outcome fallback:** If 38-A NEG, downgrade to design-guideline workshop paper built on Exp 35 + Exp 37 negative evidence + tracker-class dichotomy.

---

## 12. Exit Criteria

Exp 38 is complete when **all** of the following are true:

- [ ] G0 bare repro passed (baseline HOTA reproduced within ±0.5 on V1).
- [ ] 38-A headline finetune + eval done on V1.
- [ ] Decision rule applied (§1).
- [ ] If positive: 38-B iterative (at least iter=2) + 38-C dense + all A1/A2/A3 controls.
- [ ] V2 confirmation run (only if V1 positive).
- [ ] Retrospective written to `diagnostics/results/exp38/retrospective.md`.
- [ ] Memory files updated: new `project_exp38_*.md` + `MEMORY.md` pointer.
- [ ] Git commit on `exp/flexhook-ego` branch with all artifacts.

---

## 13. Timeline (realistic, 3060 Ti solo)

**Note — G0 pre-cleared:** FlexHook bare V1 repro at `53.824` HOTA already exists
at `/home/seanachan/FlexHook/retest-kitti-1/refer-kitti-best/results/pedestrian_summary.txt`
(paper 53.82, match within 0.004). Stage 0 dropped from schedule.

### Risk-weighted milestones

| Milestone | Day | What's done |
|---|---|---|
| M1 | T+5 | Ego extractor + ORB cache built for V1 seqs 0005, 0011, 0013 |
| M2 | T+8 | `mymodel.py` mod + coord-alignment unit tests pass |
| M3 | T+12 | 38-A headline finetune + HOTA eval done — decision gate applied (§1) |
| M4 | T+14 | A3 `bare_finetuned` control + 38-A memo written |
| M5 (cond.) | T+21 | 38-B iterative ego sweep (iter ∈ {1,2,3}) if 38-A positive |
| M6 (cond.) | T+28 | 38-C dense motion field + V2 confirmation run |
| M7 | T+30 | Retrospective + paper outline + memory update |

**Headline-only path (M1–M4): ~14 days realistic.**
**Full ablation path (M1–M7): ~30 days.**

### Risk-weighted outcome distribution (self-estimate)

| Scenario | Prob | Headline-only days |
|---|---|---|
| Smooth — no scale mismatch, coord OK, finetune converges first try | 15% | 7 |
| Normal — one risk triggers (coord bug, finetune tuning, or GPU OOM fallback) | 50% | 12–14 |
| Scale mismatch — FlexHook `pos_avg_pool` spatial scales ≠ our temporal gaps, needs redesign of injection point | 25% | 18–21 |
| Blow up — 38-A NEG, Exp 38 terminates early, pivot to appearance fusion | 10% | 5 + pivot |

**Expected value ≈ 13 days headline-only.** Full ablation adds ~2 weeks on top.

### Dominant schedule risks (ranked by impact)

1. **FlexHook scale semantics.** `pos_avg_pool[i]` is **spatial** (28×84 / 14×42 / 7×21 grid), our `FRAME_GAPS={2,5,10}` are **temporal**. §3.2's "fine↔short, coarse↔long" mapping is an assumption without theoretical grounding. If mismatch hurts results, injection point needs redesign (possibly at `conditional_f` or `obj_f` cat layer, not at `speed`). **Cost: +3–5 days.**
2. **FlexHook internal iterator keys.** Track ID and frame-index alignment between Temp-NeuralSORT outputs and FlexHook's dataloader batching needs reverse-engineering. **Cost: +1–2 days.**
3. **Resolution mismatch.** FlexHook operates on 224×672 resize; our ORB runs on KITTI native 1242×375. Conversion `residual_velocity * 0.02` assumes unit img_dim; needs ratio-aware in extractor. **Cost: +1 day.**
4. **6GB GPU OOM.** Free 6.2 GB current; batch 40 may OOM under fp16. Fallback batch 20 + grad accum doubles training time. **Cost: +1 day compute.**
5. **Finetune instability.** `speed` distribution shift from raw → ego may destabilize from-checkpoint init. §8 warmup (2 ep zero-speed) mitigates but needs testing. **Cost: +1–2 days hyperparam sweep.**

These risks are not serial — multiple can trigger simultaneously. Timeline estimate above assumes at most one triggers in the "normal" case.

---

## 14. Non-goals (explicit)

- Not improving FlexHook tracker upstream (Temp-NeuralSORT).
- Not adding CLIP visual features (that's the alternative next-step; Exp 38 is orthogonal).
- Not publishing without at least 38-A headline + A3 bare_finetuned control.
- Not extending to Refer-Dance or LaMOT.
