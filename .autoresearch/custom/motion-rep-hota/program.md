# autoresearch — motion-rep-hota

## Goal
Maximize `pooled_hota` on `gmc_link/alignment.py`. Higher is better.

## What the Agent Can Change
- **PHASE 2 (2026-06-13, scope-widened):** motion-feature engineering — the 13D motion vector.
  Editable: `gmc_link/manager.py` (inference/cache build, vector at ~L395-414) AND
  `gmc_link/dataset.py` (train build, `compute_motion_vector` ~L1088 + helpers). `gmc_link/alignment.py`
  still editable (e.g. to thread a new `motion_dim`), but its architecture is EXHAUSTED (phase 1, 18 runs).
  ALSO editable: `gmc_link/train.py` — holds the base-13 constant + `motion_dim = 13 + compute_extra_dims(...)`
  (L480) + the `--extra-features` argparse default (L589). The eval runs train with NO feature flag, so a
  feature is enabled by changing that default. Checkpoint carries `extra_features` + `motion_dim`
  (train.py:535-536); manager.py:78 + the cache builder read them back → enabling at train auto-syncs
  inference. Existing per-track extras share `compute_per_track_extras` (dataset.py:305) — math identical
  in train + inference by construction. NEVER touch run_*.py (eval/cache scripts) or the fusion recipe.
- **PHASE 1 (converged, do not re-probe):** `gmc_link/alignment.py` aligner architecture — fully mapped,
  ship `9c000d6` (Dropout 0.05 + trunk 768) = 44.739. All width/depth/dropout/activation/norm/residual/
  adapter/gating axes bracketed or dead. Re-probing = ~0 expected lift.

## ⚠ TRAIN/INFERENCE SYNC HAZARD (phase 2 — read every iteration)
The 13D vector is computed in TWO places that MUST stay byte-identical:
  1. `gmc_link/dataset.py` — TRAIN features (what the aligner learns on)
  2. `gmc_link/manager.py` — INFERENCE features (what the GMC cache is built from)
A feature change in one but not the other = silent garbage (aligner trains on X, scored on Y → HOTA collapse).
RULES:
- Any feature edit changes BOTH files together, with matching math + matching slot order.
- `FRAME_GAPS = [2,5,10]` must match between the two (dataset.py:71, manager.py:38).
- ⚠ CONFOUND (found run 18a3bf5): changing the LONG gap ALSO requires bumping `frame_gap` default
  (manager.py:50, currently 10) to ≥ max(FRAME_GAPS), because it sizes `homography_buffer` +
  `centroid_history` (maxlen=frame_gap+1) at inference. The cache builder constructs the manager with
  the DEFAULT frame_gap (doesn't pass it), so the default must cover the longest gap. If not bumped,
  inference can't look back far enough → long-scale velocity zero-fills → train/infer desync → MOVING
  craters (18a3bf5: [2,5,15] w/ frame_gap=10 gave MOVING 28.2, −2.7; result INVALID not a real NEG).
  Clean gap re-test = change FRAME_GAPS (both files) + frame_gap default (manager.py:50) together.
- Dim change (13→N): set `motion_dim` consistently; manager reads `checkpoint["motion_dim"]` (manager.py:77),
  aligner `motion_dim` param flows from there. Verify the aligner's `motion_adapter` in-dim matches.
- Stage ALL changed files into ONE git commit (eval reverts via `reset --hard HEAD~1` — atomic only if one commit).
- Before committing, sanity-check the two builders produce the same vector for a shared input if feasible.

## What the Agent Cannot Change
- The evaluation script (`evaluate.py` or the eval command). It is read-only.
- Dependencies — do not add new packages or imports that aren't already available.
- Any other files in the project unless explicitly noted here.
- Additional constraints: Edit ONLY gmc_link/alignment.py (shared_weight aligner). ONE change per iteration. GOAL: raise iKUN MOVING DetRe+DetPr (oracle headroom: ship MOVING 20.5/26.1 -> ceiling 69.2/80.3, +6.13 pooled reachable WITHOUT new tracker) while holding STATIC. Metric pooled_hota already bakes the STATIC>=43.2 guardrail (penalized -100 on miss). Single-seed = diagnostic screen ONLY; n=3 STATIC>=43.2 is the real promotion gate, run manually before any SOTA claim. DO NOT touch the fusion recipe (exhausted: 24 NEG levers) or eval scripts. Aligner levers: trunk width/depth, norm (LN/BN), dropout, activation, residual, projection dim, output L2-norm, InfoNCE temperature. Prior arch (Exp36 transformer / MLP-25D) NEG at AUC 0.779 -- HOTA-direct keep/discard is the NEW variable, re-screen them. Iteration ~22min (train 2m + 3 GMC caches 18m + fuse 1m). Baseline: seed0 pooled 44.561 / STATIC 43.240 / MOVING 28.885.

## Strategy
1. First run: establish baseline. Do not change anything.
2. Profile/analyze the current state — understand why the metric is what it is.
3. Try the most obvious improvement first (low-hanging fruit).
4. If that works, push further in the same direction.
5. If stuck, try something orthogonal or radical.
6. Read the git log of previous experiments. Don't repeat failed approaches.

## Simplicity Rule
A small improvement that adds ugly complexity is NOT worth it.
Equal performance with simpler code IS worth it.
Removing code that gets same results is the best outcome.

## Stop When
You don't stop. The human will interrupt you when they're satisfied.
If no improvement in 20+ consecutive runs, change strategy drastically.

## Weak-point ledger (run 10, 2026-06-13)

### Ship now
`9c000d6` = Dropout 0.05 + trunk 768 → pooled **44.739** (+0.178 vs baseline). Single-seed; n=3 STATIC≥43.2 gate not yet run.

### Probed axes (DEAD = won't revisit)
- **Width**: 512→768 POS, 768→1024 NEG. **Sweet at 768.** DEAD beyond 768 at current reg.
- **Dropout dose at 512 trunk**: 0→0.1 (component+ pooled tied) → 0.05 (pooled +0.135). **Sweet 0.05.**
- **Activation swap**: ReLU→GELU NEG (MOVING −0.778). DEAD smooth-activation axis.
- **Adapter capacity**: deepen 13→128→256 MLP NEG (STATIC crash). DEAD.
- **Per-modality LN before trunk**: NEG (MOVING −0.763 / over-normalized motion variance). DEAD.
- **Residual skip in trunk (post-LN repositioned)**: catastrophic MOVING −1.973 (raw adapter dominates summed signal). DEAD residual-skip without scale matching.

### Pattern
- Capacity adds without reg → STATIC crash (overfit).
- Smooth activation (GELU) → MOVING drop (loses sparse residual signal).
- Asymmetric per-modality changes (LN, deeper adapter) → MOVING drop (over-normalizes or scale-shifts motion-distinctive features).
- Sweet spot is **narrow** around the symmetric trunk with mild reg.

### Bottleneck hypothesis
MOVING DetRe (oracle 20.5→69.2) is not reachable via aligner-only architectural tweaks at this representational ceiling. Current 30.959 vs oracle 69.2 = 38pt gap; aligner perturbations only swing ±2pt. Real lever may be **input representation** (motion features themselves), but locked to alignment.py.

### Next-3 candidate directions
1. **Dropout 0.04 at trunk 768** — finer dose-probe between known POS 0.05 and zero. May find sub-0.05 sweet spot for current width.
2. **LeakyReLU(0.01) in trunk** — preserve signed residual-velocity activations without GELU's smoothness penalty.
3. **Trunk depth +1 layer** (768×3 hidden) — capacity in depth instead of width. Tests whether 768 saturation is width-specific or general.

## Weak-point ledger (run 17, 2026-06-13) — ALIGNER SURFACE FULLY MAPPED

### Ship unchanged
`9c000d6` = Dropout 0.05 + trunk 768 → pooled **44.739** (+0.178 vs baseline). 11 consecutive discards since. Single-seed; **n=3 STATIC≥43.2 gate NOT yet run** — required before any claim.

### All axes now bracketed/DEAD (single-seed screen)
| axis | result | peak |
|------|--------|------|
| trunk width | 512 (44.696) < **768 (44.739)** > 1024 (44.530 NEG) | 768 |
| inter-layer dropout | 0.04 (44.615) < **0.05 (44.739)** > 0.07 (44.549) | 0.05 |
| input dropout (pre-trunk) | 44.649 NEG | — |
| activation | **ReLU** > LeakyReLU (44.636) > GELU (44.344); all smooth NEG MOVING | ReLU |
| depth | **2 hidden** > 3 hidden (STATIC-trip NEG) | 2 |
| norm | **output-LN** ; internal-LN 44.638 NEG / per-mod-LN 44.693 NEG / BN 43.646 NEG (train/eval shift) | output-LN only |
| residual skip | 44.428 NEG (MOVING −1.97) | — |
| adapter capacity | deepen 13→128→256 STATIC-trip NEG | bare Linear |

### Conclusion (the answer to "what is the structure lacking")
MOVING never broke **31.238** (7c550fe) across 11 architectural variants; oracle ceiling is 69.2. The aligner architecture sits at a tight local optimum — **±2pt swing only**. The deficit is REPRESENTATION-BOUND, not architecture-bound: it lives in the 13D motion features upstream (manager.py/dataset.py), OUTSIDE the locked alignment.py scope. No pointwise-activation / capacity / norm / reg tweak reaches it.

### Next-3 (genuinely-distinct mechanisms only — near-peak dose re-probes have ~0 expected lift)
1. **Gated FFN / GLU variant** (ReGLU: a·relu(b)) — multiplicative input-dependent gating, structurally distinct from pointwise activation. Last in-scope architectural family. ← THIS RUN
2. **Learnable logit-scale before L2-norm** — risk: breaks fusion-recipe cos calibration; likely NEG.
3. **SCOPE-WIDEN (out of current charter)**: motion-feature engineering in manager.py / dataset.py to break the representation ceiling. The only path with real MOVING headroom. Requires user sign-off to unlock scope.

## Weak-point ledger (run 19, 2026-06-13) — ALIGNER SPACE CLOSED, LOOP CONVERGED

### Gating family closed (the last untried mechanism)
- 2-block ReGLU (5d863ae): NaN divergence → −100 (stability artifact, not a verdict).
- single-block ReGLU, bounded input (8b7f806): **44.299 NEG** (STATIC 43.230 / MOVING 29.700). Clean divergence-controlled test → gating does NOT help.

### Status: 13 non-improvements since ship `9c000d6` (44.739). Every aligner architectural family now bracketed or dead:
width · depth · inter-layer dropout · input dropout · activation (ReLU/GELU/LeakyReLU) · norm (LN-out/LN-in/per-mod-LN/BN) · residual · adapter capacity · gated FFN (ReGLU).

### Decision (surfaced to user)
Aligner-only search is EXHAUSTED within the locked charter. No remaining in-scope lever has non-trivial expected lift. MOVING is representation-bound (peak 31.2 vs oracle 69.2). Two forward paths, both needing user input:
1. **Promote**: pause loop, run manual **n=3 STATIC≥43.2 gate** on ship `9c000d6`; if it holds, it's a real +0.178 single-arch ship (Dropout 0.05 + trunk 768).
2. **Scope-widen**: unlock motion-feature engineering (manager.py / dataset.py 13D vector) — the only direction with real MOVING headroom. Requires charter change.

Continuing the loop unchanged = re-probing bracketed axes (dropout 0.06, width 640, SiLU) at ~0 expected lift. Not recommended.

## Weak-point ledger (phase-2 CONVERGED, 2026-06-14)

### Phase-2 motion-feature engineering: NEG across all sync-safe families
Best unchanged: ship `9c000d6` (Dropout 0.05 + trunk 768, 13D [2,5,10]) = 44.739. n=3 gate still owed.
| feature | +dims | pooled | MOVING | verdict |
|---------|-------|--------|--------|---------|
| speed_m | 1 | 44.713 | 30.696 | DISCARD (−0.026, closest) |
| accel | 2 | 44.685 | 30.201 | DISCARD |
| depth (17D) | 4 | 44.675 | 30.209 | DISCARD |
| FRAME_GAPS[2,5,15] clean | 0 | 44.597 | 29.433 | DISCARD |
| heading_sincos | 6 | 44.523 | 29.414 | DISCARD |

Plus 1 crash (expr_class.py infra, fixed) + 1 invalid (FRAME_GAPS buffer-desync confound).

### Conclusion
Monotonic dilution: pooled drop ∝ added dims; every kinematic/depth/directional/horizon signal is at-best neutral (speed_m −0.026), never positive. The 13D[2,5,10] vector is INFORMATION-SATURATED for what the aligner+fusion can exploit. neighbor-relational is the only untested family but is BLOCKED (manager has no inference-side neighbor context; adding it = major infra + desync risk).

BOTH phases exhausted: aligner architecture (phase 1, 18 runs) + motion-feature representation (phase 2, 5 valid runs). The MOVING oracle gap (31→69) is classification-bound, not extractable from this 2D-trajectory+depth pipeline — matches seq-encoder + signal-decomp memos. Real headroom needs a different signal source (appearance/LVLM — both explored/NEG/blocked per memory).

### Decision surfaced to user (2026-06-14): promote ship via n=3 gate / try non-sync-safe lever / stop loop.

## Weak-point ledger (2026-06-21) — NEIGHBOR-RELATIONAL TESTED, NEG. LOOP FULLY EXHAUSTED.

### velocity_rank (the last untested family) = NEG
Wired `velocity_rank` (1-dim: fraction of same-frame neighbors slower than me; moving-among-parked
ranks high, parked-among-traffic low) into BOTH builders. Correction to the 2026-06-14 note: manager
DOES have neighbor context — `active_tracks` passed to `process_frame` IS the full frame (cache builder
feeds every detection). Train infra (`compute_relational_extras`, `_extract_all_track_centroids`) already
existed; manager wiring was ~15 lines (post-loop concat mirroring the depth path), NOT "major infra".
- **Result: pooled 44.440 = −0.121 vs unmod baseline 44.561, −0.299 vs ship 9c000d6 44.739. DISCARD.**
- Harness emitted a FALSE "KEEP" — results.tsv history was absent on this branch → Best=None → any valid
  number kept. Corrected: reverted commit 0f606ae (`reset --hard`), results.tsv row fixed to discard.
- Soft-desync (expected, contributes to the drop): train neighbors = GT all-tracks vs inference
  neighbors = NeuralSORT detections → velocity_rank distribution differs train↔infer.

### Conclusion: ALL motion-feature families now closed.
Magnitude/kinematic/depth/directional/horizon (2026-06-14) + neighbor-relational (today), every one ≤0,
monotonic dilution holds. The 13D[2,5,10] vector is information-saturated; the MOVING oracle gap is
classification-bound, not extractable from this 2D-trajectory pipeline. Aligner arch (phase 1) +
motion features (phase 2) BOTH exhausted. No in-charter lever remains. Loop stopped 2026-06-21.
Owed before any ship claim: n=3 STATIC≥43.2 gate on 9c000d6 (44.739 single-seed, +0.178).
