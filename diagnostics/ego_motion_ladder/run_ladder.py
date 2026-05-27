"""Information-cap ladder diagnostic (paper appendix).

Computes a 4-level velocity ladder (L1 raw pixel -> L4 full metric residual) on
the V1-test motion-expression subset, with a scale-invariant ROC-AUC per level
(label = expression-semantic MOVING vs STATIC) as the inferential metric, plus a
kinematically-derived Same-Direction-Faster (SDF) sub-analysis.

Read-only: GT bboxes (labels_with_ids), GT depth cache (DAv2, z_track_gt), V1
expressions. No training, no model, no GPU.

Usage:  python diagnostics/ego_motion_ladder/run_ladder.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import ladder_lib as L

SEQS = ["0005", "0011", "0013"]          # V1 test split
PRIMARY_GAP = 5
SENS_GAPS = [2, 10]                       # sensitivity footnote
SMALL_BBOX_H = 60.0                        # px; height < => pedestrian-like (size proxy, NOT class)
MIN_CLASS_FOR_AUC = 5                      # min instances per class to compute AUC
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "ego_motion_ladder")

FX, FY, CX, CY = L.INTRINSICS


# ── instance construction ─────────────────────────────────────────────────

def build_instances(seq: str, gap: int) -> List[dict]:
    """One record per (frame, track) labelled MOVING/STATIC by expression semantics."""
    geom = L.load_geometry(seq)
    depth = L.load_depth(seq)
    referents = L.load_motion_referents(seq)

    # Aggregate motion state + subtype per (frame, track) across all exprs; drop conflicts.
    states: Dict[tuple, set] = defaultdict(set)
    subtypes: Dict[tuple, set] = defaultdict(set)
    for _sent, state, subtype, refs in referents:
        for fid, tids in refs.items():
            for tid in tids:
                states[(fid, tid)].add(state)
                subtypes[(fid, tid)].add(subtype)

    # Precompute per-frame ego (over ALL tracks present at both t and t-gap with depth).
    ego_by_frame: Dict[int, tuple] = {}
    n_frames_unreliable = 0
    for fid, frame in geom.items():
        prev = geom.get(fid - gap)
        if prev is None:
            continue
        vels, dzs = {}, {}
        for tid, (cx, cy, _w, _h) in frame.items():
            if tid not in prev:
                continue
            if tid not in depth or fid not in depth[tid] or (fid - gap) not in depth[tid]:
                continue
            pcx, pcy, _pw, _ph = prev[tid]
            vels[tid] = ((cx - pcx) / gap, (cy - pcy) / gap)
            dzs[tid] = (depth[tid][fid] - depth[tid][fid - gap]) / gap
        if len(vels) >= 3:
            ego_by_frame[fid] = L.estimate_ego(vels, dzs)
        else:
            n_frames_unreliable += 1

    PRIORITY = ["turning", "braking", "walking", "moving", "other-motion"]
    out = []
    for (fid, tid), st in states.items():
        if len(st) != 1:                       # MOVING+STATIC conflict -> exclude
            continue
        label = next(iter(st))
        if fid not in ego_by_frame:
            continue
        prev = geom.get(fid - gap)
        if prev is None or tid not in geom.get(fid, {}) or tid not in prev:
            continue
        if tid not in depth or fid not in depth[tid] or (fid - gap) not in depth[tid]:
            continue
        cx, cy, w, h = geom[fid][tid]
        pcx, pcy, _pw, _ph = prev[tid]
        v_pix = ((cx - pcx) / gap, (cy - pcy) / gap)
        dz = (depth[tid][fid] - depth[tid][fid - gap]) / gap
        lv = L.compute_levels(v_pix, depth[tid][fid], dz, ego_by_frame[fid], FX, FY)
        subtype = next((p for p in PRIORITY if p in subtypes[(fid, tid)]), "other-motion")
        out.append({
            "seq": seq, "frame": fid, "track": tid, "label": label,
            "subtype": subtype, "bbox_h": h,
            "L1": lv["L1"], "L2": lv["L2"], "L3": lv["L3"], "L4": lv["L4"],
            "v_metric_x": lv["v_metric_x"], "v_metric_z": lv["v_metric_z"],
            "dz_res": lv["dz_res"],
            # static (non-velocity) control features — to expose depth/layout confound
            "z_t": depth[tid][fid], "cx_abs": abs(cx - CX), "bbox_area": w * h,
        })
    return out, n_frames_unreliable


# ── aggregation helpers ────────────────────────────────────────────────────

def mag_stats(rows: List[dict]) -> dict:
    out = {}
    for lvl in ("L1", "L2", "L3", "L4"):
        v = np.array([r[lvl] for r in rows], dtype=float)
        out[lvl] = {
            "median": float(np.median(v)) if len(v) else None,
            "p25": float(np.percentile(v, 25)) if len(v) else None,
            "p75": float(np.percentile(v, 75)) if len(v) else None,
        }
    return out


def auc_ladder(rows: List[dict]) -> dict:
    labels = [1 if r["label"] == "MOVING" else 0 for r in rows]
    out = {"n": len(rows), "n_moving": sum(labels), "n_static": len(labels) - sum(labels)}
    for lvl in ("L1", "L2", "L3", "L4"):
        out[lvl] = L.safe_auc(labels, [r[lvl] for r in rows])
    return out


def control_auc(rows: List[dict]) -> dict:
    """AUC of STATIC (non-velocity) features for MOVING vs STATIC.

    If bare depth Z / bbox size / lateral position separate the classes on their
    own, then the L2/L4 (depth-scaled) AUC lift is a scene-layout confound, not
    recovered motion.
    """
    labels = [1 if r["label"] == "MOVING" else 0 for r in rows]
    return {
        "z_t": L.safe_auc(labels, [r["z_t"] for r in rows]),
        "bbox_area": L.safe_auc(labels, [r["bbox_area"] for r in rows]),
        "cx_abs": L.safe_auc(labels, [r["cx_abs"] for r in rows]),
    }


def auc_subtype_vs_static(sub_rows: List[dict], static_pool: List[dict]) -> dict:
    """Per-level AUC: pos = subtype instances (MOVING), neg = global STATIC pool.

    Subtypes are class-pure by construction, so within-subtype AUC is undefined;
    contrasting against the shared STATIC pool makes 'can level L tell <subtype>
    from parked?' computable.
    """
    rows = sub_rows + static_pool
    labels = [1] * len(sub_rows) + [0] * len(static_pool)
    out = {"n_pos": len(sub_rows), "n_neg": len(static_pool)}
    for lvl in ("L1", "L2", "L3", "L4"):
        out[lvl] = L.safe_auc(labels, [r[lvl] for r in rows])
    return out


def fmt(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


# ── main ───────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Primary-gap instances across seqs.
    rows: List[dict] = []
    unreliable = 0
    for seq in SEQS:
        r, nu = build_instances(seq, PRIMARY_GAP)
        rows.extend(r)
        unreliable += nu
    print(f"[ladder] gap={PRIMARY_GAP}: {len(rows)} instances "
          f"({sum(1 for r in rows if r['label']=='MOVING')} MOVING / "
          f"{sum(1 for r in rows if r['label']=='STATIC')} STATIC); "
          f"{unreliable} frames skipped (ego cohort < 3 tracks)")

    # tau_z = 0.5 * median(|dz_res|) over full subset.
    abs_dzres = np.array([abs(r["dz_res"]) for r in rows], dtype=float)
    med_abs_dzres = float(np.median(abs_dzres))
    tau_z = 0.5 * med_abs_dzres
    for r in rows:
        r["sdf"] = L.is_sdf(r["dz_res"], r["v_metric_x"], r["v_metric_z"], tau_z)

    sdf = [r for r in rows if r["sdf"]]
    notsdf = [r for r in rows if not r["sdf"]]

    report = {
        "config": {
            "seqs": SEQS, "primary_gap": PRIMARY_GAP, "sens_gaps": SENS_GAPS,
            "intrinsics_fx_fy": [FX, FY], "tau_z": tau_z,
            "median_abs_dz_res": med_abs_dzres, "small_bbox_h_px": SMALL_BBOX_H,
            "n_instances": len(rows), "n_frames_skipped_ego": unreliable,
        },
        "full": {"mag": mag_stats(rows), "auc": auc_ladder(rows),
                 "control_auc": control_auc(rows)},
        "by_subtype": {},
        "dz_res_distribution": {
            p: float(np.percentile([r["dz_res"] for r in rows], p))
            for p in (5, 25, 50, 75, 95)
        },
        "sdf": {},
        "bbox_strata": {},
        "sens_gaps": {},
        "crosstab_subtype_sdf": {},
    }

    # by subtype: AUC vs the shared STATIC pool (subtypes are class-pure)
    static_pool = [r for r in rows if r["label"] == "STATIC"]
    for sub in ("turning", "braking", "walking", "moving"):
        sr = [r for r in rows if r["subtype"] == sub]   # all MOVING by construction
        if sr:
            report["by_subtype"][sub] = {
                "mag": mag_stats(sr),
                "auc_vs_static": auc_subtype_vs_static(sr, static_pool),
            }

    # SDF analysis with gates
    n_sdf = len(sdf)
    sdf_block = {
        "tau_z": tau_z, "n_sdf": n_sdf, "n_not_sdf": len(notsdf),
        "n_sdf_moving": sum(1 for r in sdf if r["label"] == "MOVING"),
        "n_sdf_static": sum(1 for r in sdf if r["label"] == "STATIC"),
        "median_features_sdf": mag_stats(sdf) if sdf else None,
    }
    if n_sdf < 20:
        sdf_block["verdict"] = "POWER-FAIL (<20): counts + median features only, no AUC."
    else:
        if n_sdf < 50:
            sdf_block["verdict"] = "LOW-POWER (<50): AUC reported but treat as case study; see tau_z sweep."
        else:
            sdf_block["verdict"] = "OK (>=50)."
        sdf_auc = auc_ladder(sdf)
        notsdf_auc = auc_ladder(notsdf)
        sdf_block["auc_sdf"] = sdf_auc
        sdf_block["auc_not_sdf"] = notsdf_auc
        # Fix-1 headline: L3->L4 AUC delta, SDF vs NOT-SDF (cancels conditioning bias).
        deg = (sdf_auc["n_moving"] < MIN_CLASS_FOR_AUC or sdf_auc["n_static"] < MIN_CLASS_FOR_AUC)
        if deg or sdf_auc["L3"] is None or sdf_auc["L4"] is None:
            sdf_block["headline"] = (
                "DEGENERATE: SDF subset is near single-class "
                f"(MOVING={sdf_auc['n_moving']}, STATIC={sdf_auc['n_static']}) -> "
                "within-SDF MOVING/STATIC AUC not computable. This is the predicted "
                "limitation: SDF is defined by dZ_res>tau_z, which selects movers; "
                "monocular depth + linguistic labels cannot validate the SDF claim "
                "(only LiDAR/oxts metric-motion GT could).")
        else:
            d_sdf = sdf_auc["L4"] - sdf_auc["L3"]
            d_not = (notsdf_auc["L4"] - notsdf_auc["L3"]
                     if notsdf_auc["L3"] is not None and notsdf_auc["L4"] is not None else None)
            sdf_block["headline"] = {
                "delta_L3_to_L4_sdf": d_sdf,
                "delta_L3_to_L4_not_sdf": d_not,
                "hypothesis_supported": (d_not is not None and d_sdf > d_not),
            }
    # tau_z sensitivity sweep (always, cheap; primary when SDF small)
    sweep = {}
    for k in (0.25, 0.5, 1.0, 1.5):
        t = k * med_abs_dzres
        s = [r for r in rows if L.is_sdf(r["dz_res"], r["v_metric_x"], r["v_metric_z"], t)]
        sweep[f"{k}x_median"] = {
            "tau_z": t, "n_sdf": len(s),
            "n_moving": sum(1 for r in s if r["label"] == "MOVING"),
            "n_static": sum(1 for r in s if r["label"] == "STATIC"),
        }
    sdf_block["tau_z_sensitivity"] = sweep
    report["sdf"] = sdf_block

    # bbox-size strata (full + SDF only)
    for name, subset in (("full", rows), ("sdf", sdf)):
        small = [r for r in subset if r["bbox_h"] < SMALL_BBOX_H]
        large = [r for r in subset if r["bbox_h"] >= SMALL_BBOX_H]
        report["bbox_strata"][name] = {
            "small_pedlike": {"n": len(small), "auc": auc_ladder(small) if len(small) >= 10 else None},
            "large_carlike": {"n": len(large), "auc": auc_ladder(large) if len(large) >= 10 else None},
        }

    # crosstab subtype x SDF
    for sub in ("turning", "braking", "walking", "moving", "other-motion"):
        sr = [r for r in rows if r["subtype"] == sub]
        if sr:
            report["crosstab_subtype_sdf"][sub] = {
                "n": len(sr), "n_sdf": sum(1 for r in sr if r["sdf"]),
                "frac_sdf": sum(1 for r in sr if r["sdf"]) / len(sr),
            }

    # sensitivity gaps (AUC ladder, full subset only)
    for g in SENS_GAPS:
        gr = []
        for seq in SEQS:
            r, _ = build_instances(seq, g)
            gr.extend(r)
        report["sens_gaps"][f"gap_{g}"] = auc_ladder(gr)

    # ── write artifacts ────────────────────────────────────────────────────
    with open(os.path.join(OUT_DIR, "ladder_per_instance.json"), "w") as f:
        json.dump(rows, f)
    with open(os.path.join(OUT_DIR, "ladder_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    write_markdown(report, rows, os.path.join(OUT_DIR, "ladder_report.md"))
    make_plot(rows, report, os.path.join(OUT_DIR, "ladder_boxplot.png"))
    print(f"[ladder] wrote artifacts to {OUT_DIR}")
    return report


def write_markdown(report, rows, path):
    c = report["config"]
    f = report["full"]
    lines = []
    lines.append("# Information-Cap Ladder Diagnostic (paper appendix)\n")
    lines.append(f"_V1 test {c['seqs']}, gap={c['primary_gap']} frames, "
                 f"n={c['n_instances']} (frame,track) instances, "
                 f"expression-semantic MOVING/STATIC label._\n")
    lines.append("## Methods & caveats\n")
    lines.append("- **Levels**: L1 raw pixel `||v_pix||` (px/frame) · "
                 "L2 `||v_pix·Z/f||` no ego (m/frame) · "
                 "L3 ego-comp pixel `||v_pix−ego||` (px/frame) · "
                 "L4 full metric residual `||(vres·Z/f, dZ−ego_dZ)||` (m/frame).\n")
    lines.append("- **Ego (no oxts)**: component-wise median over the slower-half "
                 "cohort of in-frame tracks (no-homography stand-in for the ship's "
                 "post-homography stationary cohort). Same estimator at L3 & L4.\n")
    lines.append("- **Depth**: Depth-Anything-V2 metric (GT-track cache `z_track_gt`). "
                 "No LiDAR/oxts available.\n")
    lines.append("- **AUC label**: expression-semantic MOVING vs STATIC "
                 "(velocity-asserting vs rest-asserting referring expressions). "
                 "Independent of L1–L4, ego-uncontaminated.\n")
    lines.append("- **CAVEAT 1 — scale-confound**: L2/L4 magnitudes are inflated by "
                 "the `×Z/f` rescale, which world-XY proved is a *model no-op*. "
                 "Magnitudes are descriptive only; **AUC (rank-based) is the inferential metric**.\n")
    lines.append("- **CAVEAT 2 — SDF circularity**: SDF is defined via L4-derived "
                 "kinematics. (a) L4-AUC on SDF has a built-in floor (dZ_res near τ_z "
                 "excluded); (b) L1/L3-AUC on SDF is also non-randomly conditioned by "
                 "the dZ_res filter. Least-circular evidence = **compare L3→L4 AUC delta "
                 "in SDF vs NOT-SDF** (relative comparison cancels much of the bias).\n")
    lines.append("- **CAVEAT 3**: ego = stationary-cohort proxy (not ORB ship feature, "
                 "not oxts); canonical 2011_09_26 intrinsics. "
                 f"{c['n_frames_skipped_ego']} frames skipped (cohort < 3 tracks). "
                 "No GT class field exists → SDF has no class filter (pedestrian leak possible).\n")
    lines.append("- **CAVEAT 4**: constant frame-gap dt (no oxts timestamps); cancels in AUC & ratios.\n")
    lines.append("- **CAVEAT 5 — AUC is NOT fully scale-invariant across levels**: AUC is "
                 "invariant to rescaling a *single* feature, but L2/L4 multiply velocity by "
                 "depth Z — injecting Z as an extra (confounding) variable. So an L2/L4 AUC "
                 "lift over L1/L3 may be a depth/scene-layout confound, NOT recovered motion. "
                 "The §1b control-feature AUCs quantify this. The genuinely scale-honest motion "
                 "contrast is **L1 vs L3** (both pure pixel, same units).\n")
    lines.append("- **CAVEAT 6 — cohort-median ego**: when many tracks move together (following "
                 "traffic), the slower-half median absorbs that common motion as 'ego', "
                 "suppressing real object motion in L3. So a near-chance L3 is partly an "
                 "estimator limitation, not proof ego-comp is worthless.\n")

    lines.append("\n## 1. Full motion subset\n")
    lines.append("### Magnitude ladder (descriptive — scale-confounded)\n")
    lines.append("| Level | median | p25 | p75 |\n|---|---|---|---|")
    for lvl in ("L1", "L2", "L3", "L4"):
        m = f["mag"][lvl]
        lines.append(f"| {lvl} | {fmt(m['median'])} | {fmt(m['p25'])} | {fmt(m['p75'])} |")
    a = f["auc"]
    lines.append(f"\n### AUC ladder (inferential — MOVING vs STATIC, "
                 f"n={a['n']}: {a['n_moving']} MOV / {a['n_static']} STA)\n")
    lines.append("| L1 | L2 | L3 | L4 |\n|---|---|---|---|")
    lines.append(f"| {fmt(a['L1'])} | {fmt(a['L2'])} | {fmt(a['L3'])} | {fmt(a['L4'])} |")
    lines.append(f"\n_Scale-honest motion contrast (same units): **L1→L3 = "
                 f"{fmt(a['L1'])}→{fmt(a['L3'])}**. L2/L4 lift is depth-weighted — see §1b._\n")

    lines.append("### 1b. Control: static (non-velocity) feature AUC\n")
    lines.append("_If bare depth/size/position separate MOVING from STATIC, the L2/L4 lift is "
                 "a scene-layout confound, not motion._\n")
    ca = report["full"]["control_auc"]
    lines.append("| depth Z_t | bbox_area | lateral |cx−CX| |\n|---|---|---|")
    lines.append(f"| {fmt(ca['z_t'])} | {fmt(ca['bbox_area'])} | {fmt(ca['cx_abs'])} |")

    lines.append("\n## 2. By keyword subtype — AUC vs shared STATIC pool\n")
    lines.append("_pos = subtype instances (all MOVING); neg = global parked/stopped pool._\n")
    lines.append("| subtype | n_pos | n_neg | L1 | L2 | L3 | L4 |\n|---|---|---|---|---|---|---|")
    for sub, blk in report["by_subtype"].items():
        a = blk["auc_vs_static"]
        lines.append(f"| {sub} | {a['n_pos']} | {a['n_neg']} | "
                     f"{fmt(a['L1'])} | {fmt(a['L2'])} | {fmt(a['L3'])} | {fmt(a['L4'])} |")

    lines.append("\n## 3. dZ_res distribution (τ_z sanity-check)\n")
    lines.append(f"τ_z = 0.5 × median|dZ_res| = **{fmt(c['tau_z'])}** m/frame "
                 f"(median|dZ_res| = {fmt(c['median_abs_dz_res'])}).\n")
    lines.append("| p5 | p25 | p50 | p75 | p95 |\n|---|---|---|---|---|")
    d = report["dz_res_distribution"]
    lines.append("| " + " | ".join(fmt(d[p]) for p in (5, 25, 50, 75, 95)) + " |")

    lines.append("\n## 4. SDF (Same-Direction-Faster) analysis\n")
    s = report["sdf"]
    lines.append(f"**{s['verdict']}**  n_sdf={s['n_sdf']} "
                 f"(MOV {s['n_sdf_moving']} / STA {s['n_sdf_static']}), n_not_sdf={s['n_not_sdf']}.\n")
    if "headline" in s:
        lines.append(f"**Headline (Fix 1 — L3→L4 AUC delta, SDF vs NOT-SDF):**\n")
        lines.append(f"```\n{json.dumps(s['headline'], indent=2)}\n```\n")
    lines.append("**τ_z sensitivity sweep:**\n")
    lines.append("| τ_z (×median) | τ_z value | n_sdf | MOV | STA |\n|---|---|---|---|---|")
    for k, v in s["tau_z_sensitivity"].items():
        lines.append(f"| {k} | {fmt(v['tau_z'])} | {v['n_sdf']} | {v['n_moving']} | {v['n_static']} |")

    lines.append("\n## 5. Cross-tab: keyword subtype × SDF fraction\n")
    lines.append("| subtype | n | n_sdf | frac_sdf |\n|---|---|---|---|")
    for sub, v in report["crosstab_subtype_sdf"].items():
        lines.append(f"| {sub} | {v['n']} | {v['n_sdf']} | {fmt(v['frac_sdf'])} |")

    lines.append("\n## 6. Bbox-size strata (size proxy, NOT class)\n")
    for name in ("full", "sdf"):
        lines.append(f"\n**{name}** (small = bbox_h < {SMALL_BBOX_H}px):\n")
        lines.append("| stratum | n | L1 | L2 | L3 | L4 |\n|---|---|---|---|---|---|")
        for strat in ("small_pedlike", "large_carlike"):
            b = report["bbox_strata"][name][strat]
            if b["auc"]:
                a = b["auc"]
                lines.append(f"| {strat} | {b['n']} | {fmt(a['L1'])} | {fmt(a['L2'])} | "
                             f"{fmt(a['L3'])} | {fmt(a['L4'])} |")
            else:
                lines.append(f"| {strat} | {b['n']} | (n<10, no AUC) | | | |")

    lines.append("\n## 7. Temporal-gap sensitivity (full subset AUC)\n")
    lines.append("| gap | n | L1 | L2 | L3 | L4 |\n|---|---|---|---|---|---|")
    a = report["full"]["auc"]
    lines.append(f"| {c['primary_gap']} (primary) | {a['n']} | {fmt(a['L1'])} | "
                 f"{fmt(a['L2'])} | {fmt(a['L3'])} | {fmt(a['L4'])} |")
    for g, a in report["sens_gaps"].items():
        lines.append(f"| {g.split('_')[1]} | {a['n']} | {fmt(a['L1'])} | "
                     f"{fmt(a['L2'])} | {fmt(a['L3'])} | {fmt(a['L4'])} |")

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def make_plot(rows, report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: magnitude box (log y; mixed units -> annotate confound)
    data = [[r[lvl] for r in rows] for lvl in ("L1", "L2", "L3", "L4")]
    ax1.boxplot(data, labels=["L1\npx/frame", "L2\nm/frame", "L3\npx/frame", "L4\nm/frame"],
                showfliers=False)
    ax1.set_yscale("symlog")
    ax1.set_title("Magnitude ladder (DESCRIPTIVE — scale-confounded)\nL2/L4 ×Z/f rescale is a model no-op")
    ax1.set_ylabel("magnitude (mixed units, symlog)")

    # Panel 2: AUC bars (inferential)
    labels_auc = []
    full = [report["full"]["auc"][l] or 0.5 for l in ("L1", "L2", "L3", "L4")]
    x = np.arange(4)
    width = 0.35
    ax2.bar(x - width / 2, full, width, label="full subset")
    s = report["sdf"]
    if "auc_sdf" in s and s["auc_sdf"]["L1"] is not None:
        sdf_v = [s["auc_sdf"][l] or 0.5 for l in ("L1", "L2", "L3", "L4")]
        ax2.bar(x + width / 2, sdf_v, width, label="SDF subset")
    ax2.axhline(0.5, color="gray", ls="--", lw=1, label="chance")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["L1", "L2", "L3", "L4"])
    ax2.set_ylim(0.3, 1.0)
    ax2.set_title("AUC ladder (INFERENTIAL — MOVING vs STATIC)\nrank-based, scale-invariant")
    ax2.set_ylabel("ROC-AUC")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
