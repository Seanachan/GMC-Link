# Failure-Mode Audit — Coverage Reconnaissance

Frame-level coverage check **before** the full per-row build_table.
Tests whether each cell's GT-active frames fall inside the iKUN
cascade cache window. Cells with `coverage_pct=0%` mean the
cascade dump never predicts on the frames where the verb fires —
failure is upstream of detector/tracker/aligner/fusion levers.

| expr | seq | cls | n_gt_rows | gt_frames | gt_range | ikun_frames | ikun_range | overlap | coverage_pct |
|---|---|---|---|---|---|---|---|---|---|
| turning-cars | 0011 | car | 30 | 30 | 309-338 | 371 | 1-371 | 30 | 100.0% |
| turning-vehicles | 0011 | car | 30 | 30 | 309-338 | 371 | 1-371 | 30 | 100.0% |
| pedestrian-who-are-walking | 0011 | pedestrian | 90 | 56 | 1-56 | 371 | 1-371 | 56 | 100.0% |
| moving-cars | 0011 | car | 650 | 316 | 1-371 | 371 | 1-371 | 316 | 100.0% |
| parking-vehicles | 0011 | car | 2598 | 354 | 1-355 | 371 | 1-371 | 354 | 100.0% |

## Interpretation

- `coverage_pct = 0%` → 100% of FN attributable to **FN_ikun_coverage**;
  no recipe lever (det/track/aligner/fusion) can fire on these frames.
- `coverage_pct > 0%` → proceed to per-row build_table for residual cells.
