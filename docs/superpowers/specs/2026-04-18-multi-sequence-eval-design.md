# Multi-Sequence Evaluation for GT Cosine Diagnostic

**Date:** 2026-04-18
**Branch:** `exp/multi-seq-eval`, branched from `main`. This is evaluation-only tooling (no model or training changes), so it is logically independent from the motion-encoder experiment branches and deserves its own branch for clean merge semantics.
**Motivation:** Exp 30 (MLP baseline) reported AUC 0.779 on seq 0011. Exp 31 tested 9 feature enrichments, none crossed 0.800 on seq 0011. Exp 32 (temporal transformer) reported AUC 0.770 on seq 0011. All three experiments concluded from a single sequence's worth of evidence, which may be hiding real effects behind seq-0011-specific variance.

Before committing to a bigger architectural bet (CLIP text encoder swap, cross-attention fusion, loss function changes), we need to know whether the ~0.79 "ceiling" is a real property of the model or a property of seq 0011. This spec defines a multi-sequence evaluation protocol that re-evaluates all 11 prior weight files across the 3 V1 held-out sequences, separating signal from seq-specific noise.

## Goals

1. Aggregate GT-vs-non-GT cosine AUC across the 3 V1 held-out sequences (`0005`, `0011`, `0013`) to produce a lower-variance headline metric.
2. Retrospectively re-evaluate all 11 existing V1 weight files (Exp 30 Stage 1 + F1-F9 ablations + Exp 32 temporal transformer) under the new metric.
3. Produce a single Markdown comparison report that lets a reader answer:
   - Is Exp 30's baseline AUC 0.779 on seq 0011 representative or an outlier?
   - Did any Exp 31 feature cross 0.800 on some seq even though it didn't on 0011?
   - Is Exp 32's regression consistent across seqs or seq-0011-specific?
   - What are the per-expression AUCs when pooled across the 3 held-out seqs?
4. Establish multi-sequence eval as the primary reporting metric for the next ~3 experiments (alongside legacy seq-0011 reporting for continuity). After that transition window, drop single-seq reporting.

## Non-Goals

- Changing the training protocol, the training data, or any weights.
- Changing the language encoder or motion encoder.
- Changing the per-sequence diagnostic math. `diag_gt_cosine_distributions.py` receives one additive change: the raw `gt_cosines` / `nongt_cosines` arrays (already computed inside the loop) are now persisted in the output `.npz` alongside the existing `results` summary. All computation, logging, and plots stay identical.
- Evaluating on train sequences. Held-out only.
- Computing bootstrap confidence intervals or other second-order statistics. Out of scope for this pass; the spread across 3 seqs is the uncertainty signal.

## What AUC Means Here

The existing diagnostic reports a **ranking AUC**, not a classification AUC. Specifically, for a given expression:

- Every track visible in an active frame gets a cosine similarity score against the expression's language embedding.
- Tracks labeled as matching the expression in ground truth go into the **GT pool**.
- Tracks visible in the same frames but *not* matching go into the **non-GT pool**.
- AUC = probability that a randomly chosen GT-pool score exceeds a randomly chosen non-GT-pool score, computed from the Mann-Whitney U statistic.

Interpretation:
- **0.50** = random chance (no discrimination)
- **0.80** = the model ranks a GT track above a non-GT track 80% of the time
- **1.00** = perfect separation
- **< 0.50** = model is inverted (worse than chance)

This interpretation is baked into the aggregate Markdown report header so future readers don't have to re-derive it.

## Evaluation Set

The 3 V1 held-out sequences: **`0005`, `0011`, `0013`**.

- V1 training uses 15 seqs: `0001-0004, 0006-0010, 0012, 0014-0016, 0018, 0020`.
- Sequences `0017` and `0019` have labels but no V1 expression files and are excluded.
- The held-out set is small (n=3). We will *not* over-interpret tight numerical differences; the point of multi-seq eval is to detect *directional* discrepancies and variance, not to produce a p-value.

## Weights to Evaluate (Initial Retrospective Run)

11 weight files from prior V1 experiments:

| Exp | Tag | Weight file |
|---|---|---|
| 30 | `v1train_stage1` | `gmc_link_weights_v1train_stage1.pth` |
| 31 F1 | `v1train_F1_speed` | `gmc_link_weights_v1train_F1_speed.pth` |
| 31 F2 | `v1train_F2_heading` | `gmc_link_weights_v1train_F2_heading.pth` |
| 31 F3 | `v1train_F3_accel` | `gmc_link_weights_v1train_F3_accel.pth` |
| 31 F4 | `v1train_F4_ego` | `gmc_link_weights_v1train_F4_ego.pth` |
| 31 F5 | `v1train_F5_nbrmean` | `gmc_link_weights_v1train_F5_nbrmean.pth` |
| 31 F6 | `v1train_F6_velrank` | `gmc_link_weights_v1train_F6_velrank.pth` |
| 31 F7 | `v1train_F7_headdiff` | `gmc_link_weights_v1train_F7_headdiff.pth` |
| 31 F8 | `v1train_F8_nndist` | `gmc_link_weights_v1train_F8_nndist.pth` |
| 31 F9 | `v1train_F9_density` | `gmc_link_weights_v1train_F9_density.pth` |
| 32 | `v1train_temporal` | `gmc_link_weights_v1train_temporal.pth` |

`model_tag` is the weight filename stem (without `gmc_link_weights_` prefix and `.pth` suffix).

## Architecture

Three-component pipeline. Data flows only through the file system — no Python imports between components.

```
┌─────────────────────────────────────────────────────────┐
│ run_multiseq_eval.sh (new, ~40 LoC)                     │
│   for seq in 0005 0011 0013:                            │
│     for weight in <11 weight files>:                    │
│       1. python diagnostics/diag_gt_cosine_             │
│          distributions.py --weights <w> --seq <s>       │
│       2. mv diagnostics/results/layer3_gt_cosine_       │
│          {seq}.npz →                                    │
│          diagnostics/results/multiseq/                  │
│          layer3_{seq}_{model_tag}.npz                   │
│   python diagnostics/aggregate_multiseq.py \            │
│     --results-dir diagnostics/results/multiseq/ \       │
│     --weights <all 11> --seqs 0005 0011 0013 \          │
│     --output-dir diagnostics/results/multiseq/          │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ diag_gt_cosine_distributions.py (EXISTING, UNCHANGED)   │
│   — computes per-expression GT/non-GT cosine dists      │
│   — saves one .npz per invocation                       │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ aggregate_multiseq.py (new, ~250 LoC)                   │
│   — reads per-seq .npz files                            │
│   — computes macro + micro aggregate AUC                │
│   — writes per-weight .json + .md + .png                │
│   — writes 1 comparison .md across all weights          │
└─────────────────────────────────────────────────────────┘
```

### Loop order rationale

Outer loop is `seq`, inner loop is `weight`. This preserves OS page-cache warmth on the sequence's frames and homography data across the 11 diagnostic invocations — important because each invocation re-runs ORB homography estimation from scratch. A future optimization could add a disk cache for homographies keyed by `(seq, frame_pair)`; out of scope for this spec.

### Why the diagnostic stays unchanged

The existing `diag_gt_cosine_distributions.py` is used directly by RESEARCH_NOTES entries for Exp 30, 31, 32. Modifying it risks breaking artifact compatibility and single-seq debugging workflows. The shell wrapper invokes it as a black-box subprocess and relocates its outputs — a clean separation.

### Why the aggregator is separate

The aggregator does pure post-processing. It can be re-run any time on existing `.npz` files without re-running inference:
- When a new experiment is added, we aggregate existing + new results in one call.
- When the report format changes, we regenerate all reports without ~15 min of diagnostic runs.

## Output Artifacts

All artifacts live under `diagnostics/results/multiseq/`.

### Per-(seq, weight) `.npz` files

`layer3_{seq}_{model_tag}.npz` — 33 files for the initial 11 × 3 run. Contents identical to the existing single-seq `.npz` schema. Namespaced filenames prevent clobbering.

### Per-weight JSON

`layer3_multiseq_{model_tag}.json` — structured data, one per weight. Schema:

```json
{
  "model_tag": "v1train_temporal",
  "weights_path": "gmc_link_weights_v1train_temporal.pth",
  "sequences": ["0005", "0011", "0013"],
  "per_expression": {
    "moving cars": {
      "auc_per_seq": {"0005": 0.81, "0011": 0.83, "0013": 0.79},
      "auc_macro_mean": 0.810,
      "auc_macro_std": 0.020,
      "auc_micro": 0.812,
      "gt_count_per_seq": {"0005": 22, "0011": 15, "0013": 31},
      "nongt_count_per_seq": {"0005": 84, "0011": 112, "0013": 95}
    }
  },
  "headline": {
    "mean_auc_micro": 0.783,
    "mean_auc_macro": 0.781,
    "std_across_seqs": 0.012,
    "n_expressions": 23
  }
}
```

`gt_count_per_seq` is load-bearing for interpretation: a 0.95 AUC with 2 GT samples is noise; the same number with 40 is real signal.

### Per-weight Markdown report

`layer3_multiseq_{model_tag}.md` — the human-readable artifact. Template:

```markdown
# Multi-Sequence Eval: {model_tag}

## What this measures
AUC = probability that a randomly chosen GT-matching track gets a
higher cosine score than a randomly chosen non-matching track,
for a given expression. 0.50 = chance, 1.00 = perfect.

## Headline
- Mean AUC (micro, pooled across 3 seqs): **0.783**
- Mean AUC (macro, per-seq averaged):     **0.781** ± 0.012
- Seq-0011 only (legacy, for continuity): **0.779**

## Per-expression breakdown
| Expression | 0005 | 0011 | 0013 | macro μ ± σ | micro | GT counts |
|---|---|---|---|---|---|---|
| moving cars | 0.81 | 0.83 | 0.79 | 0.81±0.02 | 0.81 | 22/15/31 |
| parking cars | 0.67 | 0.52 | 0.71 | 0.63±0.10 | 0.64 | 18/9/14 |
```

### Per-weight box-plot PNG

`layer3_multiseq_{model_tag}.png` — one box per expression on the y-axis, showing the 3 per-seq AUC values. Tall boxes = seq-dependent expressions; short boxes = stable findings. Makes seq-0011 outliers visible at a glance.

### Comparison report across all weights

`layer3_multiseq_comparison.md` — one file for the 11-weight retrospective run. Columns:
- `model_tag`
- `mean_auc_micro` (the headline number)
- `mean_auc_macro ± std` (cross-seq consistency)
- `best_seq` (formatted as `"{seq_id}: {auc:.3f}"`, e.g. `"0013: 0.812"`) — the sequence where this model's mean AUC across expressions was highest
- `worst_seq` (same format) — the sequence where the mean AUC was lowest
- `max_gap` = `best_seq_auc − worst_seq_auc`, flagging models whose performance is seq-dependent

Rows: one per `model_tag`, ordered by `mean_auc_micro` descending. This is the single artifact that answers the primary questions from the Goals section.

## Aggregation Math

### Raw inputs per (seq, weight, expression)

From the existing diagnostic's `.npz`:
- `gt_cosines_{s,w,e}`: array of cosine scores for GT-matching tracks
- `nongt_cosines_{s,w,e}`: array of cosine scores for non-matching tracks

Per (seq, weight, expression):
```
auc[s,w,e] = MannWhitneyU(gt_cosines_{s,w,e}, nongt_cosines_{s,w,e})
             / (|gt| × |nongt|)
```

### Macro aggregation (per-seq-averaged)

For each (weight, expression):
```
auc_macro_mean[w,e] = mean_s( auc[s,w,e] )  over seqs with defined AUC
auc_macro_std[w,e]  = std_s( auc[s,w,e] )
```

For the headline:
```
mean_auc_macro[w] = mean_e( auc_macro_mean[w,e] )
                    over expressions present in ≥2 seqs after filtering
```

**Interpretation:** "On average across sequences, how well does this model rank GT vs non-GT for this expression?" Each seq weighted equally. Noisy on low-support expressions.

### Micro aggregation (pooled across seqs)

For each (weight, expression), pool scores across seqs before computing AUC:
```
gt_pooled_{w,e}    = concat_s( gt_cosines_{s,w,e} )
nongt_pooled_{w,e} = concat_s( nongt_cosines_{s,w,e} )
auc_micro[w,e]     = MannWhitneyU(gt_pooled, nongt_pooled)
                     / (|gt_pooled| × |nongt_pooled|)
```

For the headline:
```
mean_auc_micro[w] = mean_e( auc_micro[w,e] )
```

**Interpretation:** "If we pick one random GT and one random non-GT from the entire held-out pool, how often is the ranking correct?" Larger-support seqs dominate.

### Which number is headline

**Micro** is the headline one-number metric. It has lower variance than macro and is more stable for cross-experiment comparison. Macro (with std) is reported alongside to surface cross-seq inconsistency.

### Edge cases

| Condition | Macro behavior | Micro behavior |
|---|---|---|
| Expression not present in a seq | Skip that (seq, expression) | Empty concat — seq contributes no samples |
| 0 GT or 0 non-GT in a seq | AUC undefined; skip seq for macro mean | Only skip if pooled totals are 0 |
| Expression in <2 seqs after filtering | Exclude from macro headline (no meaningful std) | Include if ≥1 seq has data |

## Reporting Protocol Going Forward

For the next ~3 experiments after this spec lands (the "transition window"):
- Experiments report **both** legacy seq-0011 AUC (for continuity with Exp 30-32) and multi-seq micro AUC.
- RESEARCH_NOTES entries include a "multi-seq results" table alongside the existing single-seq table.
- After 3 experiments under both metrics, the transition window ends and we drop legacy seq-0011 reporting. Multi-seq micro becomes the canonical metric.

## Work Items

| Order | File | Purpose | LoC |
|---|---|---|---|
| 1 | `diagnostics/diag_gt_cosine_distributions.py` | Additive: persist raw `gt_cosines` / `nongt_cosines` arrays in the output `.npz`. No change to math, logging, or plots. | ~10 |
| 2 | `diagnostics/aggregate_multiseq.py` (new) | Read .npz files, compute aggregates, write JSON/MD/PNG/comparison | ~250 |
| 3 | `run_multiseq_eval.sh` (new) | Shell driver: 33 invocations + rename + call aggregator | ~40 |
| 4 | `diagnostics/results/multiseq/` (new dir) | Output location | — |
| 5 | `.gitignore` | Ignore `multiseq/*.npz` and `multiseq/*.png`; keep `*.md` tracked | 2 lines |

### Order of work

1. Write `aggregate_multiseq.py` with a synthetic `.npz` fixture for unit testing — can be validated without waiting for real diagnostic runs.
2. Write `run_multiseq_eval.sh`.
3. Smoke test: 1 weight × 1 seq end-to-end, verify artifacts land in the right place.
4. Full retrospective: 11 weights × 3 seqs (~15-45 min compute).
5. Commit the 11 per-weight `.md` reports and the single `layer3_multiseq_comparison.md` to git.
6. Add a short retrospective section to `RESEARCH_NOTES.md` interpreting the new numbers against Exp 30-32 conclusions.

### Success criteria for the tooling

- `bash run_multiseq_eval.sh` with the 11 weights produces 11 JSONs, 11 MDs, 11 PNGs, and 1 comparison MD without manual intervention.
- The aggregator is deterministic: same `.npz` inputs → bitwise-identical JSON outputs.
- A reader can answer all four primary questions (see Goals section) by reading only `layer3_multiseq_comparison.md`.

### Interpretive thresholds for the analysis

These are not pass/fail criteria for the tooling — they're the reading thresholds we'll apply when interpreting results:

- **"Seq 0011 is representative"** = for ≥70% of expressions, seq-0011 AUC is within ±1 std of the macro mean. <70% means seq 0011 is an outlier and prior conclusions need revisiting.
- **"Feature enrichment is dead"** (confirming Exp 31) = neither the overall micro AUC nor any single-seq mean AUC (i.e., `mean_e(auc[s,w,e])` for any fixed `s`) of any F1-F9 weight exceeds 0.800. If any single-seq mean AUC crosses 0.800 even while the micro is below, that feature is reopened for investigation under the new metric.
- **"Temporal transformer is a genuine regression"** = Exp 32 micro AUC is <(Exp 30 micro AUC) by ≥0.010 across the 3-seq pooled data. Within ±0.010 means the regression is noise-level; the transformer result becomes inconclusive rather than negative.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Re-running diagnostic 33 times re-does ORB homography work for each (weight, seq) pair | Accepted — seq-outer loop order preserves OS page-cache warmth. A homography disk cache is out of scope for this spec but noted as a future optimization. |
| Existing `layer3_gt_cosine_{seq}.npz` files (from Exp 30-32) could be overwritten by the shell wrapper | Wrapper renames to `multiseq/layer3_{seq}_{model_tag}.npz` *immediately* after each invocation, before the next run overwrites the source file. Single-seq legacy files are untouched because they live outside the `multiseq/` subdir. |
| Tiny GT sample counts make per-seq AUC numerically unstable | `gt_count_per_seq` is included in every JSON and MD so readers can weigh small-N AUCs appropriately. The interpretive thresholds in the success criteria acknowledge this. |
| Aggregate metrics could conflict with legacy seq-0011 metrics during the transition window | Transition protocol explicitly reports both for the next 3 experiments, letting us build a translation table before dropping legacy reporting. |

## Future Scaling (if successful)

If multi-seq reporting surfaces signals that seq-0011 alone hid:
- Extend to V2 held-out seqs for a larger evaluation pool.
- Add bootstrap confidence intervals per expression (requires `scipy.stats.bootstrap` or equivalent).
- Add a homography disk cache to make re-runs near-instant.
- Build a historical dashboard that shows AUC trends across experiments over time.

If multi-seq confirms seq-0011 was representative all along:
- Lock in seq-0011-equivalent multi-seq baseline numbers for future comparisons.
- Move directly to the next experimental direction (A1 CLIP text encoder swap, B1 hard negative mining, C2 cross-attention — see the prior options tree).
