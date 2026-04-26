# Label Noise Audit via GT-Rendered Video — Design

**Date:** 2026-04-20
**Branch:** `exp/hn-mining`
**Author:** Seanachan (human) + Claude (agent)
**Status:** Spec, pending user review

## Goal

Determine whether the Refer-KITTI V1 held-out ceiling at micro-AUC 0.779 is
**data-bound** (label noise) or **model-bound**. Four model-side attacks
(Exp 31 feature ablation, Exp 32 temporal transformer, Exp 34 HN-InfoNCE,
Exp 35 FlexHook-adjacent) have all plateaued at or below 0.779. Before
committing another ~2-hour training sweep (Exp 36 rich-feature transformer),
verify the GT labels themselves are trustworthy for motion-verb expressions.

## Why Now

- Exp 34 pre-registered decision rule returned representation-bound
- Memory: `project_exp34_representation_bound.md` — "positive-EV = richer
  motion features, stronger language encoder, OR wider/deeper motion head"
- None of those positive-EV directions was data-audit; an audit is orthogonal
  and cheap (~30 min render + ~30 min user viewing)
- If labels noisy, all four prior negatives are confounded — the bound may
  be lower than 0.779 would suggest

## Scope

### Sequences

- **0005** (baseline-best held-out, `mean_auc_micro = 0.821` on v1train_stage1)
- **0011** (worst, `mean_auc_micro = 0.779`, systematically pessimistic per
  memory)
- **0013 skipped** — only n=2 expressions, audit is statistically useless

### Expressions

Filter to motion-verb only via `MOTION_KEYWORDS` in
`gmc_link/dataset.py:357`. Non-motion (appearance-only, e.g. "red cars")
is out of GMC-Link's remit and not what we need to audit.

Estimated volume per seq: 15–30 motion expressions after filter.
Total clips: 30–60.

## Deliverables

### Per-expression MP4

- **Path:** `diagnostics/results/label_audit/<seq>_<expr_slug>.mp4`
- **FPS:** 5 (half KITTI capture rate for readability)
- **Frame range:** first→last frame in GT label dict (tight window)

### Overlay

- **Green bbox + track ID:** tracks in this expression's GT `label`
  field for the current frame
- **Light gray bbox:** all NeuralSORT predicted tracks at this frame
  (context — "what else is in the scene?")
- **Top-left text:** expression sentence
- **Top-right text:** `frame N / total`
- **Bottom-left text:** `seq <S>   verb family: <turning|moving|parking|...>`

### Review tally (user fills)

- **Path:** `diagnostics/results/label_audit/review.csv`
- **Columns:** `seq,expr_slug,sentence,verdict,note` where
  verdict ∈ {`accept`, `flag-wrong`, `ambiguous`}

## Data sources

| Resource | Path |
|---|---|
| Frames | `/home/seanachan/data/Dataset/refer-kitti/KITTI/training/image_02/<seq>/` |
| GT labels | `/home/seanachan/data/Dataset/refer-kitti/expression/<seq>/<expr>.json`  — `{label: {frame_id: [track_ids]}, sentence}` |
| Context tracks | `NeuralSORT/<seq>/car/predict.txt`, `NeuralSORT/<seq>/pedestrian/predict.txt` |
| GT track bboxes | `Refer-KITTI/gt_template/<seq>/<expr>/gt.txt` (MOT format: `frame,id,x,y,w,h,...`) |

## Architecture (single script)

**File:** `run_gt_audit_video.py` (~100 lines)

Model off `run_ikun_baseline_video.py` rendering loop:

```
for seq in [0005, 0011]:
    for expr_json in glob(expression/<seq>/*.json):
        sentence = expr_json['sentence']
        if not is_motion_expression(sentence):
            continue
        gt_by_frame = expr_json['label']          # {fid: [tids]}
        gt_boxes = load_mot('gt_template/<seq>/<expr>/gt.txt')
        context_tracks = load_neuralsort('<seq>')
        frames = sorted(gt_by_frame.keys(), as int)
        render_video(frames, gt_boxes, context_tracks, sentence, out_mp4)
```

## Review protocol

1. User opens each mp4 sequentially
2. For each: judge "does this track exhibit the described motion?"
3. Mark `accept` / `flag-wrong` / `ambiguous` in review.csv
4. Add free-form note for ambiguous cases

## Decision rule

Let `noise_rate = (flag-wrong) / total_clips`.

- **< 5%** → labels clean; ceiling is genuinely model-bound. Proceed with
  Exp 36 rich-feature transformer.
- **5–20%** → targeted cleanup: remove flagged expressions from train +
  eval sets, retrain stage1 on cleaned data, check if ceiling moves.
  If it does, audit is the finding.
- **> 20%** → major label quality problem. Priority shifts from modelling
  to dataset hygiene: relabel motion-verb subset (or pivot to V2 if V2
  is cleaner), then restart.

`ambiguous` cases count as 0.5 toward noise_rate (soft noise).

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Motion verbs are genuinely ambiguous (e.g., "moving slowly" vs "stopped at light") | `ambiguous` verdict is a first-class outcome, not a forced-choice |
| User fatigue at 60 clips | Allow spread across sessions; keep review.csv checkpointable |
| Confirmation bias (user flags to justify prior belief) | Render shuffled order; shuffle-key saved so audit is reproducible |
| Context tracks (NeuralSORT) missing IDs seen in GT | GT bbox (green) always drawn directly from gt_template, independent of NeuralSORT |

## Non-goals

- No automatic physics-consistency check this round (user preferred visual)
- No V2 audit (V1 is the ceiling we're auditing; V2 is next project)
- No numeric IoU comparisons — this is a semantic label audit, not
  tracking-quality audit

## Self-review (inline)

- [x] Placeholder scan: no TBD / TODO / vague sections
- [x] Internal consistency: decision rule matches goal; data sources
      cited in architecture match data-source table
- [x] Scope check: single script, one user activity — focused enough for
      one plan
- [x] Ambiguity check: `flag-wrong` vs `ambiguous` explicitly defined;
      noise_rate formula explicit about soft-counting ambiguous
