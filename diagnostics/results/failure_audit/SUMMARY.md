# Failure-Mode Audit — Pre-T8 Reconnaissance Verdict (CORRECTED 2026-05-14)

**Date:** 2026-05-14 (corrected same day)
**Branch:** `exp/ego-motion-systematic`
**Scope:** 3 cells flagged "unrecoverable" by `project_phase5b_per_expr_recovery_rate`

## Correction notice

The first version of this document (committed earlier today) concluded
`FN_ikun_coverage` dominated all three cells. That conclusion was the
product of a **schema-misread bug** in
`diagnostics/failure_audit/loaders.py::load_ikun_logits`.

The iKUN cascade cache layout — confirmed against
`iKUN/test.py::test_tracking` — is

```
{seq: {obj_id: {frame: {expr: [logit]}}}}
```

The loader assumed `{seq: {frame: {track_id: {expr: [logit]}}}}`, which
collapsed obj_id 1-114 into "frames 1-114" and produced the spurious
"iKUN sub-samples 0011 to frames 1-114" finding. After the fix, iKUN
covers all 371 frames of seq 0011 for every test expression including
`turning-cars`. Coverage is **100 %** for all three cells.

## TL;DR (corrected)

The three "unrecoverable on cascade iKUN" cells are NOT
`FN_ikun_coverage`-bound. Per-row attribution after the loader fix:

| cell | dominant FN class | n | pct | next lever |
|---|---|---:|---:|---|
| `turning-cars` × 0011               | **FN_fusion** | 18 | 60.0 % | fusion-threshold |
| `turning-vehicles` × 0011           | **FN_fusion** | 18 | 60.0 % | fusion-threshold |
| `pedestrian-who-are-walking` × 0011 | **FN_tracker** | 40 | 44.4 % | IoU bridge / tracker |

Already-admitted (TP) rows per cell:
- turning-cars: 10/30 (33 %)
- turning-vehicles: 10/30 (33 %)
- pedestrian-who-are-walking: 42/90 (47 %)

## Coverage table (corrected)

| expr | seq | n_gt_rows | GT frame range | iKUN frame range | overlap | coverage |
|---|---|---:|---|---|---:|---:|
| `turning-cars`               | 0011 |   30 | 309-338 | 1-371 | 30 | 100.0 % |
| `turning-vehicles`           | 0011 |   30 | 309-338 | 1-371 | 30 | 100.0 % |
| `pedestrian-who-are-walking` | 0011 |   90 |   1-56  | 1-371 | 56 | 100.0 % |
| `moving-cars` (control)      | 0011 |  650 |   1-371 | 1-371 | 316 | 100.0 % |
| `parking-vehicles` (control) | 0011 | 2598 |   1-355 | 1-371 | 354 | 100.0 % |

## Turning-cars / turning-vehicles — fusion-threshold story

Both cells track the same GT trajectory (gt_track_id 58) which the IoU
bridge maps to NeuralSORT track 106 across all 30 frames. Cache + tracker
behave identically on this trajectory:

- iKUN cascade logit range: `[-0.883, -0.838]` — uniformly very negative
- GMC depth-aug score range: `[0.310, 0.724]` — mid-range with motion-tail
- Ship recipe fused-gate (`logit + 0.9·gmc + 0.17`) range across the 30 GT rows:
  - 10 rows ≥ 0 → admitted (TP)
  - 18 rows in `[-0.418, -0.017]` → FN_fusion (just below 0)
  -  2 rows have no IoU-bridged tracker box → FN_tracker

The phase5b memory's claim that "cascade logits all below -0.5 for turning
exprs → no admit at any threshold" stands on the logits, but the practical
consequence is different than that memory inferred: with the ship recipe's
α·sc·gmc additive term, 10 rows already clear the gate, and a +0.5 bias on
`thr` would clear all 18 remaining FN_fusion rows. Threshold-lever
sensitivity is concentrated, not absent.

Counter-factual table (FN_fusion rows that would convert to admitted at a
larger ship `thr`):

| thr increment | turning-cars admits | turning-vehicles admits |
|---:|---:|---:|
| +0.5 | 18 | 18 |
| +0.3 |  9 |  9 |
| +0.2 |  5 |  5 |
| +0.1 |  2 |  2 |

Whether this translates to HOTA gain depends on FP regression elsewhere
in V1 at the same threshold — not measured here.

## Pedestrian-who-are-walking — tracker-bound

- 42 TP / 40 FN_tracker / 6 FN_detector / 2 FN_aligner
- 2 GT trajectories (53, 117) both bridge to NS track 2 on TP frames
- 40 FN_tracker rows: GT bbox at the frame, no NS box with IoU ≥ 0.5
- Lever: IoU threshold relaxation OR upstream tracker recall on
  pedestrian class

## What the new audit DOES NOT show

- Phase5b memory states "cascade iKUN stack does not recover these cells
  in HOTA." That is a HOTA-level observation about the ship-recipe stack.
  This per-row audit does NOT re-evaluate HOTA — it shows per-row
  admittance under the same recipe and asks where the lever sensitivity
  is. The 10 TPs per turning-verb cell are per-row predictions; HOTA
  could still be zero if the 10 admits scatter across IDs or the GT
  trajectory's tracker-side ID-switching dominates IDF1.
- The TP rate of 33-47 % means the ship recipe is already partially
  admitting these cells at per-row level. Phase5b's "no recovery" claim
  must therefore be about HOTA aggregation, not per-row admittance.

## Decision (corrected)

The original "door closes" decision was based on a non-existent
upstream-coverage limitation. With the corrected verdict:

1. **turning-cars / turning-vehicles × 0011** — fusion-threshold-sensitive,
   not driver-bound. The phase5b "unrecoverable" framing is too strong;
   what's true is that the ship `thr=+0.17` stops short of admitting 18
   of 30 GT rows whose fused gate sits in `[-0.418, -0.017]`. A
   per-class threshold relaxation OR a per-expr threshold-search would
   be a normal recipe-side lever, not an iKUN-driver change.
2. **pedestrian-who-are-walking × 0011** — tracker-bound, dominated by
   IoU-bridge failure on pedestrian boxes. Recipe lever: IoU threshold
   relaxation in the audit code itself, OR upstream tracker (NS) recall
   improvement on the pedestrian class.

The earlier "no recipe lever can address these cells" conclusion is
withdrawn. The earlier "iKUN eval-driver change required" recommendation
is withdrawn. The pooled-HOTA ceiling (44.564) memory is unaffected — it
was established by independent multi-seed HOTA work, not by this audit.

## Artifacts

- `diagnostics/failure_audit/coverage_recon.py` — recon script
- `diagnostics/failure_audit/build_table.py` — per-row joiner
- `diagnostics/failure_audit/attribute.py`   — 7-class decision tree
- `diagnostics/failure_audit/loaders.py`     — **corrected 2026-05-14**
- `diagnostics/results/failure_audit/coverage_recon.md` — coverage table (corrected)
- `diagnostics/results/failure_audit/attribution.md`    — per-row counts (corrected)
- `diagnostics/results/failure_audit/audit_turning-cars_0011.csv` (new)
- `diagnostics/results/failure_audit/audit_turning-vehicles_0011.csv` (new)
- `diagnostics/results/failure_audit/audit_pedestrian-who-are-walking_0011.csv` (re-run)
