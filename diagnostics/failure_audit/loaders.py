"""Source loaders for the failure-mode audit.

Each loader returns a tidy pandas DataFrame keyed on a subset of
(seq, frame, track_id, expr). Joins happen in build_table.py.
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np


def load_gt(repo_root: Path, seq: str, expr: str) -> pd.DataFrame:
    """Load GT track + bbox for (seq, expr) from refer-kitti/gt_template_old/.

    KITTI-MOT-style: frame, track_id, x, y, w, h, vis, ?, ?
    Returns columns: frame, track_id, x, y, w, h, gt_match=1.
    """
    gt_path = repo_root / "refer-kitti" / "gt_template_old" / seq / expr / "gt.txt"
    if not gt_path.exists():
        return pd.DataFrame(columns=["frame","track_id","x","y","w","h","gt_match"])
    rows = []
    for line in gt_path.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        rows.append((int(parts[0]), int(parts[1]),
                     float(parts[2]), float(parts[3]),
                     float(parts[4]), float(parts[5])))
    df = pd.DataFrame(rows, columns=["frame","track_id","x","y","w","h"])
    df["gt_match"] = 1
    return df


def load_ikun_logits(repo_root: Path, seq: str, expr: str) -> pd.DataFrame:
    """Flatten the nested iKUN cascade logit cache to (frame, track_id, ikun_logit).

    JSON layout: {seq: {obj_id: {frame: {expr: [logit]}}}} — confirmed by
    iKUN/test.py write path `OUTPUTS[video][obj_id][frame_id][expression]`.
    For pedestrian-walking-* expression family, returns rows where the expr
    name *startswith* the given family token (e.g. pedestrian-walking-women,
    pedestrian-walking-men); ikun_logit is the mean across matched exprs.
    """
    cache_path = repo_root / "iKUN" / "ikun_results_v1_cascade_full.json"
    cache = json.loads(cache_path.read_text())
    seq_data = cache.get(seq, {})
    rows = []
    for obj_str, frame_dict in seq_data.items():
        track_id = int(obj_str)
        for frame_str, expr_dict in frame_dict.items():
            frame = int(frame_str)
            matched = [v[0] for k, v in expr_dict.items() if _expr_match(k, expr)]
            if matched:
                rows.append((frame, track_id, float(np.mean(matched))))
    return pd.DataFrame(rows, columns=["frame", "track_id", "ikun_logit"])


def load_detector_hits(repo_root: Path, seq: str, expr: str) -> pd.DataFrame:
    """Per (frame, track_id) detector-hit indicator.

    Reads det_cache/DDETR-kitti/<seq>/<class>/dets.json. Class is inferred from
    expression family ('car' for *-cars / *-vehicles, 'pedestrian' for
    pedestrian-*).

    Real cache schema wraps detections under a 'frames' key:
      {"seq": ..., "frames": {frame_str: [[x1,y1,x2,y2,score], ...]}}
    Flat dict schema (test fixture + legacy): {frame_str: [[x1,y1,x2,y2,score,track_id?], ...]}
    When track_id is absent (5-element det), track_id is set to -1.
    """
    cls = _expr_class(expr)
    det_path = repo_root / "det_cache" / "DDETR-kitti" / seq / cls / "dets.json"
    if not det_path.exists():
        return pd.DataFrame(columns=["frame", "track_id", "detector_hit"])
    raw = json.loads(det_path.read_text())
    # Handle real cache format where frame data lives under 'frames' key
    if "frames" in raw and isinstance(raw["frames"], dict):
        frame_dict = raw["frames"]
    else:
        frame_dict = {k: v for k, v in raw.items()
                      if k not in {"seq", "class", "img_h", "img_w", "score_thr", "schema"}}
    rows = []
    for frame_str, dets in frame_dict.items():
        frame = int(frame_str)
        for det in dets:
            # det = [x1,y1,x2,y2,score,track_id] OR [x1,y1,x2,y2,score]
            track_id = int(det[5]) if len(det) >= 6 else -1
            rows.append((frame, track_id, 1))
    return pd.DataFrame(rows, columns=["frame", "track_id", "detector_hit"])


def _expr_class(expr: str) -> str:
    if expr.startswith("pedestrian"):
        return "pedestrian"
    return "car"


def load_tracker_assoc(repo_root: Path, seq: str, expr: str) -> pd.DataFrame:
    """Per (frame, track_id) assoc state in {stable, switched, lost}.

    Rule:
      stable  — track existed in frame-1 (gap == 1) or has no prior history (fresh)
      lost    — track existed previously but had a gap > 1 frame
      switched— track existed in frame-1 and bbox center jumped > 2x diag
                (impossible with single predict.txt without ID-tracking detector
                output; reduced to lost-vs-stable for the v1 audit)

    NeuralSORT layout: NeuralSORT/<seq>/<class>/predict.txt where class is
    inferred from expression family (car for *-cars/*-vehicles, pedestrian for
    pedestrian-*).
    """
    cls = _expr_class(expr)
    pred_path = repo_root / "NeuralSORT" / seq / cls / "predict.txt"
    if not pred_path.exists():
        return pd.DataFrame(columns=["frame","track_id","x","y","w","h","tracker_assoc"])
    raw = []
    for line in pred_path.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        raw.append({
            "frame":    int(parts[0]),
            "track_id": int(parts[1]),
            "x":        float(parts[2]),
            "y":        float(parts[3]),
            "w":        float(parts[4]),
            "h":        float(parts[5]),
        })
    df = pd.DataFrame(raw)
    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)
    df["prev_frame"] = df.groupby("track_id")["frame"].shift(1)
    df["tracker_assoc"] = np.where(
        df["prev_frame"].isna(),                        "stable",
        np.where(df["frame"] - df["prev_frame"] == 1,  "stable",
                                                        "lost")
    )
    return df[["frame","track_id","x","y","w","h","tracker_assoc"]]


def load_gmc_scores(repo_root: Path, seq: str, expr: str,
                    cache_tpl: str = "gmc_scores_v1_{seq}_depth_seed1_cache.json"
                    ) -> pd.DataFrame:
    """Per (frame, track_id) GMC aligner cosine from a JSON cache.

    Cache schema: {expr: {frame_str: {track_id_str: score_float}}}.
    For family targets (e.g. pedestrian-walking), score is the mean across
    matched exprs for that (frame, track_id).

    cache_tpl selects the aligner cache; default is the depth-aug seed-1 cache
    used by the legacy 3-cell audit. Pass the ship template, e.g.
    "gmc_scores_v1_{seq}_sharedweight_seed0_rawcos_cache.json", to decompose the
    canonical ship signal.
    """
    cache_path = repo_root / "gmc_link" / cache_tpl.format(seq=seq)
    if not cache_path.exists():
        return pd.DataFrame(columns=["frame", "track_id", "aligner_gmc_score"])
    cache = json.loads(cache_path.read_text())
    matched_exprs = [k for k in cache.keys() if _expr_match(k, expr)]
    if not matched_exprs:
        return pd.DataFrame(columns=["frame", "track_id", "aligner_gmc_score"])
    rows: dict[tuple[int, int], list[float]] = {}
    for me in matched_exprs:
        for frame_str, track_dict in cache[me].items():
            frame = int(frame_str)
            for track_str, score in track_dict.items():
                key = (frame, int(track_str))
                rows.setdefault(key, []).append(float(score))
    return pd.DataFrame([
        {"frame": k[0], "track_id": k[1], "aligner_gmc_score": float(np.mean(v))}
        for k, v in rows.items()
    ])


def _expr_match(cache_expr: str, target: str) -> bool:
    """True if cache_expr matches the target family.

    Exact for non-family targets, prefix-match for pedestrian-walking-*.
    """
    if target == "pedestrian-walking":
        return cache_expr.startswith("pedestrian-walking") or \
               cache_expr in {"persons-who-are-walking", "people-who-are-walking",
                              "walking-pedestrian"}
    return cache_expr == target


SHIP_ALPHA_MOTION  = 1.0
SHIP_SCALE_MOTION  = 0.9
SHIP_THR_MOTION    = 0.17


def compute_fusion_gate(ikun_logit: float, gmc_score: float, expr: str) -> float:
    """Apply ship-recipe motion-axis fusion. All 3 target families are MOVING.

    Recipe (memory: project_ikun_scalematched_positive):
        fused = ikun_logit + alpha * scale * gmc_score + thr
    """
    return (ikun_logit
            + SHIP_ALPHA_MOTION * SHIP_SCALE_MOTION * gmc_score
            + SHIP_THR_MOTION)
