# Spatial-gated rerank (cx side) — the position wall was the residual error

16/18 color exprs are position-qualified (left/right). crop-CLIP is position-blind → after
appearance rerank, residual FPs were wrong-side right-color cars. Added per-box cx gate
(RERANK_SPATIAL=1): "left" keep cx<0.5, "right" keep cx>0.5 (IMG_W=1242). cx = tracker box,
legit at inference. GT-validated: "left" exprs GT cx<0.5 frac=1.00, "right" frac=0.00.

## n=3 (tau=0.205), ship n=3 mean 44.580
| seed | ship | spatial rerank | stack (Path B + spatial) |
|------|--------|--------|--------|
| 0 | 44.561 | 45.251 | 45.716 |
| 1 | 44.513 | 45.201 | 45.636 |
| 2 | 44.667 | 45.359 | 45.483 |
| mean | 44.580 | 45.270 (+0.690) | **45.612 (+1.032)** |

vs appearance-only rerank +0.263 → spatial 2.6×. DetPr 28→46.
Additive: Path B +0.339 ⊕ spatial rerank +0.690 = +1.029 ≈ stack +1.032.
+1.048 vs paper 44.564. First iKUN pooled >45.5.

## Held-out tau (2-fold CV, color A/B, spatial)
A-opt tau 0.205, B-opt tau 0.205 (identical → robust). Held-out full-color gain +0.697
(val A@0.205 +0.397, val B@0.205 +0.300) ≈ in-sample +0.690 → ZERO overfit.
Both folds contribute (deterministic gate generalizes; appearance-only had fold B ~0).

LESSON: bottleneck = unused signal mis-routed (cx → damped GMC / position-blind crop-CLIP),
NOT encoder capacity. Spatial gate (one inequality, no model) beat 5× bigger CLIP. Cheapest
lever of campaign. iKUN-only (deficit-conditional; FH V2 color subset already 51).
