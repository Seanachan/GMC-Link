# Information-Cap Ladder Diagnostic (paper appendix)

_V1 test ['0005', '0011', '0013'], gap=5 frames, n=2807 (frame,track) instances, expression-semantic MOVING/STATIC label._

## Methods & caveats

- **Levels**: L1 raw pixel `||v_pix||` (px/frame) · L2 `||v_pix·Z/f||` no ego (m/frame) · L3 ego-comp pixel `||v_pix−ego||` (px/frame) · L4 full metric residual `||(vres·Z/f, dZ−ego_dZ)||` (m/frame).

- **Ego (no oxts)**: component-wise median over the slower-half cohort of in-frame tracks (no-homography stand-in for the ship's post-homography stationary cohort). Same estimator at L3 & L4.

- **Depth**: Depth-Anything-V2 metric (GT-track cache `z_track_gt`). No LiDAR/oxts available.

- **AUC label**: expression-semantic MOVING vs STATIC (velocity-asserting vs rest-asserting referring expressions). Independent of L1–L4, ego-uncontaminated.

- **CAVEAT 1 — scale-confound**: L2/L4 magnitudes are inflated by the `×Z/f` rescale, which world-XY proved is a *model no-op*. Magnitudes are descriptive only; **AUC (rank-based) is the inferential metric**.

- **CAVEAT 2 — SDF circularity**: SDF is defined via L4-derived kinematics. (a) L4-AUC on SDF has a built-in floor (dZ_res near τ_z excluded); (b) L1/L3-AUC on SDF is also non-randomly conditioned by the dZ_res filter. Least-circular evidence = **compare L3→L4 AUC delta in SDF vs NOT-SDF** (relative comparison cancels much of the bias).

- **CAVEAT 3**: ego = stationary-cohort proxy (not ORB ship feature, not oxts); canonical 2011_09_26 intrinsics. 228 frames skipped (cohort < 3 tracks). No GT class field exists → SDF has no class filter (pedestrian leak possible).

- **CAVEAT 4**: constant frame-gap dt (no oxts timestamps); cancels in AUC & ratios.

- **CAVEAT 5 — AUC is NOT fully scale-invariant across levels**: AUC is invariant to rescaling a *single* feature, but L2/L4 multiply velocity by depth Z — injecting Z as an extra (confounding) variable. So an L2/L4 AUC lift over L1/L3 may be a depth/scene-layout confound, NOT recovered motion. The §1b control-feature AUCs quantify this. The genuinely scale-honest motion contrast is **L1 vs L3** (both pure pixel, same units).

- **CAVEAT 6 — cohort-median ego**: when many tracks move together (following traffic), the slower-half median absorbs that common motion as 'ego', suppressing real object motion in L3. So a near-chance L3 is partly an estimator limitation, not proof ego-comp is worthless.


## 1. Full motion subset

### Magnitude ladder (descriptive — scale-confounded)

| Level | median | p25 | p75 |
|---|---|---|---|
| L1 | 4.4552 | 1.4142 | 11.1341 |
| L2 | 0.1747 | 0.0523 | 0.3239 |
| L3 | 3.6031 | 1.3998 | 9.6531 |
| L4 | 0.4798 | 0.2269 | 0.9281 |

### AUC ladder (inferential — MOVING vs STATIC, n=2807: 1758 MOV / 1049 STA)

| L1 | L2 | L3 | L4 |
|---|---|---|---|
| 0.5121 | 0.6489 | 0.5147 | 0.7218 |

_Scale-honest motion contrast (same units): **L1→L3 = 0.5121→0.5147**. L2/L4 lift is depth-weighted — see §1b._

### 1b. Control: static (non-velocity) feature AUC

_If bare depth/size/position separate MOVING from STATIC, the L2/L4 lift is a scene-layout confound, not motion._

| depth Z_t | bbox_area | lateral |cx−CX| |
|---|---|---|
| 0.6622 | 0.4005 | 0.3970 |

## 2. By keyword subtype — AUC vs shared STATIC pool

_pos = subtype instances (all MOVING); neg = global parked/stopped pool._

| subtype | n_pos | n_neg | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|---|
| turning | 26 | 1049 | 0.8399 | 0.9980 | 0.8488 | 0.9583 |
| braking | 75 | 1049 | 0.0754 | 0.0690 | 0.4775 | 0.6254 |
| walking | 441 | 1049 | 0.6431 | 0.7413 | 0.5297 | 0.5758 |
| moving | 1216 | 1049 | 0.4846 | 0.6437 | 0.5044 | 0.7757 |

## 3. dZ_res distribution (τ_z sanity-check)

τ_z = 0.5 × median|dZ_res| = **0.1901** m/frame (median|dZ_res| = 0.3801).

| p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|
| -1.5644 | -0.5045 | -0.0601 | 0.2505 | 1.3284 |

## 4. SDF (Same-Direction-Faster) analysis

**OK (>=50).**  n_sdf=517 (MOV 404 / STA 113), n_not_sdf=2290.

**Headline (Fix 1 — L3→L4 AUC delta, SDF vs NOT-SDF):**

```
{
  "delta_L3_to_L4_sdf": -0.09478226583720306,
  "delta_L3_to_L4_not_sdf": 0.19396391192919993,
  "hypothesis_supported": false
}
```

**τ_z sensitivity sweep:**

| τ_z (×median) | τ_z value | n_sdf | MOV | STA |
|---|---|---|---|---|
| 0.25x_median | 0.0950 | 557 | 427 | 130 |
| 0.5x_median | 0.1901 | 517 | 404 | 113 |
| 1.0x_median | 0.3801 | 427 | 335 | 92 |
| 1.5x_median | 0.5702 | 347 | 280 | 67 |

## 5. Cross-tab: keyword subtype × SDF fraction

| subtype | n | n_sdf | frac_sdf |
|---|---|---|---|
| turning | 26 | 1 | 0.0385 |
| braking | 75 | 38 | 0.5067 |
| walking | 441 | 49 | 0.1111 |
| moving | 1216 | 316 | 0.2599 |
| other-motion | 1049 | 113 | 0.1077 |

## 6. Bbox-size strata (size proxy, NOT class)


**full** (small = bbox_h < 60.0px):

| stratum | n | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| small_pedlike | 1818 | 0.6405 | 0.7056 | 0.6870 | 0.7566 |
| large_carlike | 989 | 0.3908 | 0.5852 | 0.3720 | 0.6320 |

**sdf** (small = bbox_h < 60.0px):

| stratum | n | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| small_pedlike | 409 | 0.4240 | 0.4126 | 0.7127 | 0.6155 |
| large_carlike | 108 | 0.1417 | 0.2078 | 0.1184 | 0.5553 |

## 7. Temporal-gap sensitivity (full subset AUC)

| gap | n | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| 5 (primary) | 2807 | 0.5121 | 0.6489 | 0.5147 | 0.7218 |
| 2 | 3095 | 0.5059 | 0.6416 | 0.4897 | 0.6774 |
| 10 | 2343 | 0.5121 | 0.6507 | 0.5389 | 0.7777 |
