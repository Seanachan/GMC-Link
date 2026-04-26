# Hard-Negative Mining Finetune on Stage 1 — Design Spec

**Date:** 2026-04-18
**Author:** Seanachan (with Claude Opus 4.7)
**Related:** Exp 33 multi-seq retrospective (commit `ce8a07d`), `RESEARCH_NOTES.md` §Exp 33
**Experiment ID:** Exp 34 (tentative)

---

## 1. Hypothesis & Falsification Criterion

**Null hypothesis (H₀):** The 0.779 seq-0011 micro AUC ceiling observed across all 11 V1 weights in Exp 33 is caused by the motion representation (13D multi-scale residual velocity vector being insufficiently expressive), not by the contrastive loss.

**Alternative (H₁):** The ceiling is caused by InfoNCE treating easy and hard negatives equally, allowing the aligner to ignore fine-grained discrimination between e.g. "moving cars" and "moving trucks" or "moving" vs "parking" semantics.

**Test:** Finetune the existing `gmc_link_weights_v1train_stage1.pth` with an HN-InfoNCE loss at three β values. Evaluate on the canonical V1 held-out seqs (0005, 0011, 0013) using the existing multi-seq pipeline added in commit `fd628fb`.

**Decision rule** (read against best-β micro AUC):
- **≥ 0.795** → reject H₀; ceiling is loss-bound. Justifies a full-budget (γ) warmup-then-switch run and more ambitious loss exploration (ArcFace, supervised contrastive).
- **0.779 ± 0.015** → inconclusive. Ceiling is probably representation-bound, but a larger-batch or full-budget run might change this. Do not pursue further hard-negative work on the 13D representation.
- **< 0.764** → H₀ strongly supported; hard-negative mining actively harms. Pivot to representation-side experiments (richer motion features, temporal context, cross-seq augmentation).

The ±0.015 band is the Exp 33 ±1σ representativeness threshold and matches the observed weight-to-weight spread on micro AUC.

## 2. Scope: What Changes, What Doesn't

**Does NOT change:**
- Motion encoder architecture (MLP or temporal transformer)
- Language encoder (SentenceTransformer all-MiniLM-L6-v2, 384D)
- Embedding head (256D shared space)
- 13D motion input schema, `FRAME_GAPS = [2, 5, 10]`, `VELOCITY_SCALE = 100`
- `train.py` control flow — same `_run_single_stage` entry point
- Dataset generation, cache layer, collate helpers
- `run_multiseq_eval.sh` driver logic
- `diagnostics/aggregate_multiseq.py` aggregator logic

**Does change (additive, no breaking changes to existing paths):**
- `gmc_link/losses.py` — add `HardNegativeInfoNCE` class alongside existing `AlignmentLoss`
- `gmc_link/train.py` — add `--loss {infonce,hninfo}` and `--beta FLOAT` args; route loss construction based on `--loss`
- New driver `run_hn_finetune.sh` — loops β values, then runs multi-seq eval with new weights added

Existing `--loss infonce` (default) path is bit-for-bit identical to current behavior.

## 3. Loss Formulation

For each anchor `i` in a batch of size B, with similarity matrix `S[i,j] = cos(motion_i, lang_j)`:

```
positive_mask[i,j] = (sentence_id[i] == sentence_id[j])        # False-Negative Masking
negative_mask[i,j] = ~positive_mask[i,j] & (i != j)

# Hardness weights over negatives only
w_raw[i,j]  = exp(β * S[i,j]) * negative_mask[i,j]
w_norm[i,j] = w_raw[i,j] / w_raw[i,:].sum()                    # per-anchor normalization
w[i,j]      = w_norm[i,j] * negative_mask[i,:].sum()           # rescale so Σⱼ w[i,j] = N_neg

# Weighted denominator
logits      = S / τ
den[i]      = exp(logits[i,i]) + Σⱼ w[i,j] * exp(logits[i,j])
L_m2l[i]    = -log( exp(logits[i,i]) / den[i] )

# Symmetric
L = (L_m2l.mean() + L_l2m.mean()) / 2
```

**Key properties:**
- β = 0 recovers standard InfoNCE (uniform weights = 1 after rescaling).
- Higher β concentrates loss on hardest negatives. β → ∞ approaches triplet behavior.
- Gradient flows through `w` because β scales logits that remain tensors (no `.detach()`).
- `positive_mask` at the sentence level handles both the diagonal and off-diagonal same-sentence duplicates. Same-motion-group pairs (e.g. "moving cars" vs "moving trucks") are NOT masked — those are the hard negatives we want.

**Temperature:** fixed at `τ = 0.07`, matching stage1. Not learnable in this experiment (one variable at a time).

## 4. Training & Evaluation Flow

```
[1] Verify gmc_link_weights_v1train_stage1.pth exists (pre-flight)

[2] For β in {0.5, 1.0, 2.0}:
    [2a] Build V1 dataset with use_group_labels=False (sentence-level IDs)
    [2b] Construct HardNegativeInfoNCE(temperature=0.07, beta=β, fnm=True)
    [2c] Load stage1 weights into MotionLanguageAligner
    [2d] Finetune 30 epochs, LR=1e-4, batch=256, cosine decay, AdamW, no warmup
    [2e] Save gmc_link_weights_v1train_hninfo_beta{β}.pth
    [2f] Save training curves PNG

[3] Run run_multiseq_eval.sh with the 3 new weight tags appended to the default list

[4] aggregate_multiseq.py regenerates diagnostics/results/multiseq/layer3_multiseq_comparison.md
    with 3 new rows (one per β)

[5] Interpret:
    - Pick best β by micro AUC
    - Check all 3 per-seq AUCs (0005 / 0011 / 0013) and macro ± std
    - Apply §1 decision rule against the canonical ±0.015 band
    - Record findings in RESEARCH_NOTES.md as "Exp 34: Hard-Negative Mining Finetune"
```

**Expected wall clock:** ~3 × 15 min training + ~5 min eval ≈ 50 min on the V1 training machine.

## 5. File Structure

**Created:**
- `tests/test_hninfo_loss.py` — unit tests for the new loss (see §7)
- `run_hn_finetune.sh` — shell driver for the 3-β sweep + eval
- `gmc_link_weights_v1train_hninfo_beta0.5.pth` (generated, gitignored)
- `gmc_link_weights_v1train_hninfo_beta1.0.pth` (generated, gitignored)
- `gmc_link_weights_v1train_hninfo_beta2.0.pth` (generated, gitignored)
- `gmc_link_weights_v1train_hninfo_beta{β}_curves.png` (generated, gitignored)
- `diagnostics/results/multiseq/layer3_multiseq_v1train_hninfo_beta{β}.{json,md}` (generated)

**Modified:**
- `gmc_link/losses.py` — new class `HardNegativeInfoNCE`, existing `AlignmentLoss` untouched
- `gmc_link/train.py` — `--loss` and `--beta` flags, construction routing, nothing else
- `run_multiseq_eval.sh` — add three new weight tags to the default weight list (non-breaking: existing tags still run)
- `RESEARCH_NOTES.md` — post-experiment, add "Exp 34" section with results and interpretation

## 6. Dataset & Labeling Path

The existing `build_training_data` already supports `use_group_labels: bool`. When false (default), each unique sentence gets a stable integer ID via `sentence_to_id = {s: i for i, s in enumerate(all_sentences)}`. This is exactly the sentence-level ID we need for FNM — same-sentence-different-track samples share an ID.

**No new dataset code.** The loss receives `expr_ids` (renamed conceptually to `sentence_ids` inside the new loss for clarity) and builds the FNM mask from equality: `positive_mask = expr_ids[:, None] == expr_ids[None, :]`.

For curriculum/group-level training (existing feature), `use_group_labels=True` is incompatible with HN-InfoNCE because group IDs conflate different sentences. The `--loss hninfo` path will enforce `use_group_labels=False` and error on the combination.

## 7. Testing Strategy

Follow project TDD pattern. Tests live in `tests/test_hninfo_loss.py`.

**Unit tests:**

1. **β=0 reduces to standard InfoNCE.** Construct a batch of random motion/language features and distinct sentence_ids. Run `HardNegativeInfoNCE(beta=0, fnm=False)` and `AlignmentLoss(temperature=0.07)` on the same similarity matrix. Assert losses match within `1e-5`.

2. **FNM masks same-sentence pairs.** Construct a batch where samples 0 and 3 share sentence_id=7 and samples 1, 2 have distinct IDs. Forward with `beta=1.0, fnm=True`. Assert the FNM mask zeros entries `[0,3]` and `[3,0]` in the weights tensor (verify by reading the loss's internal mask — expose it for test purposes or verify behaviorally that a large similarity at those positions doesn't affect the loss).

3. **Higher β amplifies gradients on the hardest negative.** Construct a batch with one anchor and 4 candidates: 1 positive (cos=0.9), 1 easy negative (cos=-0.5), 2 hard negatives (cos=0.7, 0.8). Run forward+backward at β=0.5 and β=2.0. Assert the ratio of `|grad|` at the hardest-negative embedding (cos=0.8) to the easy-negative embedding (cos=-0.5) is strictly larger under β=2.0.

4. **Per-anchor weight normalization sums correctly.** With a batch of 8, after masking, assert `w.sum(dim=1)` equals `negative_mask.sum(dim=1)` (i.e., N_neg per anchor) up to float tolerance, for any β > 0.

**Integration test:**

5. **End-to-end finetune smoke test.** Use the existing synthetic dataset pattern from `tests/` to run 1 epoch of `python -m gmc_link.train --loss hninfo --beta 1.0 --epochs 1 --resume <tiny_weights.pth>`. Assert the output `.pth` exists and contains `model` + `temperature` keys. Assert no NaN in final loss.

**No test required for multi-seq eval pipeline** — that's already covered by existing `tests/test_aggregate_multiseq.py` (13 tests, all passing). The new weights feed into the existing pipeline unchanged.

## 8. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Finetuning collapses (loss → 0, AUC drops) | Curves PNG reveals collapse; if β=2.0 collapses, report it as a finding (not a failure) and focus analysis on β=0.5 and β=1.0 |
| Sentence-ID plumbing breaks existing v2 training or curriculum paths | New `--loss hninfo` path is gated; existing `--loss infonce` (default) is untouched. Verified by unit test #1 equivalence |
| "No change in AUC" misinterpreted | §1 decision rule explicitly defines the ±0.015 inconclusive band; interpretation uses multi-seq micro, never seq 0011 alone |
| Hard-negative training oscillates early | LR=1e-4 (10× below stage1); cosine decay; finetune inherits stage1's converged geometry, sidestepping cold-start instability |
| FNM mask shape mismatch when batch has no duplicate sentences | Guard: if `negative_mask[i,:].sum() == 0` for any row (shouldn't happen at B=256 but possible at tiny-batch smoke test), skip that anchor's gradient contribution and warn |
| Micro AUC improves but macro worsens | Report both; favor consistency across seqs (macro) as secondary criterion. Exp 33 established that micro is the load-bearing metric, but a micro-only improvement with macro regression is a pyrrhic win worth calling out |

## 9. What This Experiment Does NOT Answer

Explicit out-of-scope — do not let these creep in:

- Whether a larger-batch regime (batch=512 or 1024 for more mining candidates) would help. Separate experiment.
- Whether a different motion representation (F1–F9 features, temporal context, cross-object relational features) breaks the ceiling. Exp 33 already mostly falsified the F1–F9 direction; temporal remains open.
- Whether the ceiling survives V2 training (16 seqs vs V1's 15). V1-only experiment, consistent with Exp 30–33.
- Whether the result transfers to fusion HOTA (Stage 4 fusion head metrics). This is a Stage 1 alignment-quality experiment only.
- Whether other loss families (triplet, SupCon, ArcFace) outperform HN-InfoNCE. Only pursued if H₁ is confirmed.

## 10. Success Criteria (restated for clarity)

**Primary:** best-β micro AUC on (0005, 0011, 0013) ≥ 0.795 → H₁ confirmed.
**Secondary:** macro AUC ± std; per-seq AUC on 0011 specifically (the dominant seq).
**Artifact success:** `layer3_multiseq_comparison.md` regenerated cleanly with 3 new rows; `RESEARCH_NOTES.md` has an "Exp 34" entry with the verdict.
