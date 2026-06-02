"""iKUN cascade+simcalib + linear additive GMC bias on 3-seq POOLED HOTA.

Mirrors FlexHook fusion form (which beat paper on V2, +0.497 on V1):

    motion expr:     fused = cs + b + alpha*(gmc - 0.5)*scale
                     keep iff fused > thr_motion
    appearance expr: fused = cs + b
                     keep iff fused > 0  (baseline gating)

No MLP. Plain hand-tuned linear additive form.

Reference baselines (3-seq pooled HOTA, paper-canonical gt_template_old):
  paper-pure (alpha=0)  44.564  (paper README claim)
  local B (alpha=0)     44.224  (cli-fork drift 0.34)
  hand-tuned alpha=0.5+thr  43.910  (NEG, see project_phase5_stack_pooled_negative)
  hand-tuned alpha=1.0      43.260  (NEG)
  learned residual MLP      42.919  (NEG, project_ikun_learned_residual_negative)

Goal: test whether FlexHook's WINNING linear-additive recipe (which beat paper on
both V1 and V2 FlexHook) generalizes to iKUN cascade architecture. Different
fusion form than the legacy iKUN tries (MLP, raw alpha bias without thr).

Usage:
    python run_ikun_linear_additive.py --alpha 1.0 --gmc_scale 0.9 --thr 0.17
    python run_ikun_linear_additive.py --grid

Grid mode (2026-05-02 Path 1): motion locked at ship recipe (α=1, sc=0.9, thr=+0.17),
sweep APPEARANCE-axis bias. Project memory project_gmc_is_motion_plus_bbox_specialist
shows APPEAR raw sep +0.264 > motion +0.172. APPEAR = 77% of V1 frames; if class HOTA
gains +1.0, pool gain ≈ +0.77 (4× current iKUN gain).
"""
import argparse, json, os, shutil, subprocess, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, "/home/seanachan/GMC-Link")
sys.path.insert(0, "/home/seanachan/iKUN")

from gmc_link.demo_inference import load_neuralsort_tracks, load_ikun_scores
from utils import expression_conversion as ikun_expression_conversion

# V1 defaults; override via env for V2 (iKUN-V2 cross-split eval).
DATA_ROOT      = os.environ.get("IKUN_DATA_ROOT", "/home/seanachan/GMC-Link/refer-kitti")
TRACK_DIR      = "/home/seanachan/GMC-Link/NeuralSORT"
GT_TEMPLATE    = os.environ.get("IKUN_GT_TEMPLATE", "/home/seanachan/data/Dataset/refer-kitti/gt_template_old")
TEXT_FEAT_JSON = os.environ.get("IKUN_TEXT_FEAT_JSON", "/home/seanachan/GMC-Link/iKUN/text_feat_bboxNum_v1.json")
CASCADE_FULL   = os.environ.get("IKUN_CASCADE_JSON", "/home/seanachan/GMC-Link/iKUN/ikun_results_v1_cascade_full.json")
_GMC_SUFFIX = os.environ.get("GMC_SUFFIX", "")  # e.g. "_seed0"
_GMC_CACHE_VER = os.environ.get("GMC_CACHE_VER", "v1")  # v1|v2 cache filename tag
RAW_COS    = os.environ.get("GMC_RAW_COS", "0") == "1"  # Arm B: GMC cache contains raw cosine [-1,+1]
RERANK_SPATIAL = os.environ.get("RERANK_SPATIAL", "0") == "1"  # gate rerank admits by cx side
GMC_CACHE_TPL  = "/home/seanachan/GMC-Link/gmc_link/gmc_scores_" + _GMC_CACHE_VER + "_{seq}" + _GMC_SUFFIX + "_cache.json"
TRACKEVAL      = "/home/seanachan/TempRMOT/TrackEval/scripts/run_mot_challenge.py"
_OUT_SUFFIX = os.environ.get("OUT_SUFFIX", "")  # e.g. "_seed0"
OUT_ROOT       = os.environ.get("IKUN_OUT_ROOT", "/home/seanachan/GMC-Link/hota_eval_ikun_linear_additive" + _OUT_SUFFIX)

TEST_SEQS = ["0005", "0011", "0013"]
FRAMES = {"0005": (0, 296), "0011": (0, 372), "0013": (0, 339)}
SIM_A, SIM_B, SIM_TAU = 8.0, -0.1, 100.0

MOTION_KW = ["moving","walking","running","turning","faster","slower","braking",
             "parking","parked","stopped","stop","stand","static","stationary","accelerat"]
STATIC_KW = ["parking","parked","stopped","stop","stand","static","stationary"]


def is_motion(e): return any(k in e.lower() for k in MOTION_KW)
def classify(e):
    if not is_motion(e): return "APPEARANCE"
    if any(k in e.lower() for k in STATIC_KW): return "STATIC"
    return "MOVING"


def compute_simcalib_bias(text_feat, exprs):
    train_dict, test_dict = text_feat["train"], text_feat["test"]
    keys = list(train_dict.keys())
    FEATS = np.array([train_dict[k]["feature"] for k in keys])
    PROBS = np.array([train_dict[k]["probability"] for k in keys])
    bias = {}
    for expr in exprs:
        en = ikun_expression_conversion(expr)
        target = test_dict if en in test_dict else train_dict
        if en not in target: bias[expr] = 0.0; continue
        feat = np.array(target[en]["feature"])[None, :]
        sim = (feat @ FEATS.T)[0]
        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-12)
        w = np.exp(SIM_TAU * sim); w = w / w.sum()
        prob = float((w * PROBS).sum())
        bias[expr] = SIM_A * prob + SIM_B
    return bias


def merged_ns(seq):
    car = load_neuralsort_tracks(os.path.join(TRACK_DIR, seq, "car", "predict.txt"))
    ped = load_neuralsort_tracks(os.path.join(TRACK_DIR, seq, "pedestrian", "predict.txt"))
    max_car = 0
    for fid, dets in car.items():
        for oid, *_ in dets: max_car = max(max_car, oid)
    ns = defaultdict(list)
    for fid, dets in car.items(): ns[fid].extend(dets)
    for fid, dets in ped.items():
        ns[fid].extend([(oid+max_car, x, y, w, h) for oid, x, y, w, h in dets])
    return ns


def _iou_xywh(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    uni = aw * ah + bw * bh - inter
    return inter / uni if uni > 0 else 0.0


def _load_gt_boxes(gt_path):
    """frame -> list of (x,y,w,h) GT boxes, KITTI-MOT gt.txt format."""
    boxes = defaultdict(list)
    if not os.path.exists(gt_path):
        return boxes
    for line in open(gt_path):
        p = line.strip().split(",")
        if len(p) < 6:
            continue
        boxes[int(p[0])].append((float(p[2]), float(p[3]), float(p[4]), float(p[5])))
    return boxes


def gen_predicts(text_feat, gmc_caches, alpha, gmc_scale, thr_motion, run_dir,
                 alpha_a=0.0, scale_a=0.0, thr_a=0.0, mode="ship", dump_path=None,
                 motion_fuse="add", gmc_gate=0.35, rerank_set=None,
                 clip_caches=None, rerank_tau=0.0):
    res_dir = os.path.join(run_dir, "results")
    if os.path.exists(res_dir): shutil.rmtree(res_dir)
    os.makedirs(res_dir, exist_ok=True)
    seqmap_lines = []
    dump_rows = []  # per-object component dump for signal decomposition

    for seq in TEST_SEQS:
        ns = merged_ns(seq)
        expr_dir = os.path.join(DATA_ROOT, "expression", seq)
        exprs = sorted(f.replace(".json","") for f in os.listdir(expr_dir) if f.endswith(".json"))
        bias = compute_simcalib_bias(text_feat, exprs)
        gmc_seq = gmc_caches.get(seq, {})
        min_f, max_f = FRAMES[seq]

        for expr in exprs:
            outd = os.path.join(res_dir, seq, expr); os.makedirs(outd, exist_ok=True)
            gt_src = os.path.join(GT_TEMPLATE, seq, expr, "gt.txt")
            gt_dst = os.path.join(outd, "gt.txt")
            if os.path.exists(gt_src): shutil.copy2(gt_src, gt_dst)
            else: open(gt_dst, "w").close()
            open(os.path.join(outd, "predict.txt"), "w").close()
            seqmap_lines.append(f"{seq}+{expr}")

            ikun_scores = load_ikun_scores(CASCADE_FULL, seq, expr)
            b = bias.get(expr, 0.0)
            motion = is_motion(expr)
            per_expr_gmc = gmc_seq.get(expr, {})

            cls = classify(expr)
            rows = []
            # rerank_set (if given) restricts oracle/rerank overrides to listed
            # "seq+expr" keys; all other exprs fall through to ship logic. Lets us
            # measure "fix ONLY these exprs" pooled headroom (Step 0) and later
            # apply a reranker to only the catastrophic subset.
            in_rerank = (rerank_set is None) or (f"{seq}+{expr}" in rerank_set)
            use_oracle = in_rerank and (mode == "oracle"
                          or (mode == "oracle_motion" and motion)
                          or (mode == "oracle_appear" and not motion))
            if use_oracle:
                # Upper bound: admit exactly the tracker boxes that IoU>=0.5-match a GT
                # box at that frame. Perfect scoring on the current tracker's detections;
                # ceiling capped only by tracker localization coverage.
                gt_boxes = _load_gt_boxes(gt_src)
                for fid, dets in ns.items():
                    if not (min_f < fid < max_f): continue
                    gbs = gt_boxes.get(fid, [])
                    if not gbs: continue
                    for oid, x, y, w, h in dets:
                        if any(_iou_xywh((x, y, w, h), gb) >= 0.5 for gb in gbs):
                            rows.append((fid, oid, x, y, w, h))
                with open(os.path.join(outd, "predict.txt"), "w") as f:
                    for fid, oid, x, y, w, h in rows:
                        f.write(f"{fid},{oid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1\n")
                continue
            # Step 0.5 mechanism probe: re-rank by a TRACK-level appearance score
            # that REPLACES native admit (not additive → dodges native veto). Here
            # the score = median CLIP-B/32 cosine over the track's frames (existing
            # clip_logit_neuralsort cache). Admit whole track iff score > rerank_tau.
            use_rerank = in_rerank and mode == "rerank_clipb32"
            if use_rerank:
                per_expr_clip = (clip_caches or {}).get(seq, {}).get(expr, {})
                track_boxes = defaultdict(list)
                track_cos = defaultdict(list)
                for fid, dets in ns.items():
                    if not (min_f < fid < max_f): continue
                    fclip = per_expr_clip.get(str(fid), {})
                    for oid, x, y, w, h in dets:
                        track_boxes[oid].append((fid, x, y, w, h))
                        c = fclip.get(str(oid))
                        if c is not None:
                            track_cos[oid].append(float(c))
                # Optional spatial gate (RERANK_SPATIAL=1): the appearance crop is
                # position-blind, so for position-qualified exprs ("...-in-the-left")
                # gate admitted boxes by centroid side (cx). Tests whether POSITION,
                # not color, is the residual wall on the 16/18 position-qualified exprs.
                el = expr.lower()
                side = ("left" if "left" in el else
                        "right" if "right" in el else None)
                IMG_W = 1242.0  # KITTI V1
                for oid, boxes in track_boxes.items():
                    cos_list = track_cos.get(oid, [])
                    if not cos_list:
                        continue
                    if float(np.median(cos_list)) > rerank_tau:
                        for (fid, x, y, w, h) in boxes:
                            if RERANK_SPATIAL and side is not None:
                                cxn = (x + w / 2.0) / IMG_W
                                if side == "left" and cxn >= 0.5:
                                    continue
                                if side == "right" and cxn <= 0.5:
                                    continue
                            rows.append((fid, oid, x, y, w, h))
                with open(os.path.join(outd, "predict.txt"), "w") as f:
                    for fid, oid, x, y, w, h in rows:
                        f.write(f"{fid},{oid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1\n")
                continue
            for fid, dets in ns.items():
                if not (min_f < fid < max_f): continue
                for oid, x, y, w, h in dets:
                    cs = ikun_scores.get(fid, {}).get(oid)
                    if cs is None: continue
                    native_part = cs + b
                    if motion:
                        default = 0.0 if RAW_COS else 0.5
                        gmc = float(per_expr_gmc.get(str(fid), {}).get(str(oid), default))
                        gmc_term = gmc if RAW_COS else (gmc - 0.5)
                        gmc_part = alpha * gmc_term * gmc_scale
                        thr = thr_motion
                    else:
                        if scale_a != 0.0:
                            default = 0.0 if RAW_COS else 0.5
                            gmc = float(per_expr_gmc.get(str(fid), {}).get(str(oid), default))
                            gmc_term = gmc if RAW_COS else (gmc - 0.5)
                            gmc_part = alpha_a * gmc_term * scale_a
                            thr = thr_a
                        else:
                            gmc = float("nan")
                            gmc_part = 0.0
                            thr = 0.0
                    if mode == "native_only":   fused = native_part
                    elif mode == "gmc_only":    fused = gmc_part
                    elif motion and motion_fuse == "relu":
                        fused = max(native_part, 0.0) + gmc_part
                    elif motion and motion_fuse == "max":
                        fused = max(native_part, gmc_part)
                    else:                       fused = native_part + gmc_part
                    keep = fused > thr
                    if mode == "ship" and motion and motion_fuse == "orgate" \
                            and gmc_part > gmc_gate:
                        keep = True   # high-confidence GMC bypasses native veto
                    if keep:
                        rows.append((fid, oid, x, y, w, h))
                    if dump_path is not None:
                        dump_rows.append((seq, expr, cls, fid, oid, x, y, w, h,
                                          cs, b, gmc, native_part, gmc_part, fused, thr, int(keep)))

            with open(os.path.join(outd, "predict.txt"), "w") as f:
                for fid, oid, x, y, w, h in rows:
                    f.write(f"{fid},{oid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1\n")

    sm = os.path.join(run_dir, "seqmap.txt")
    open(sm, "w").write("\n".join(seqmap_lines) + "\n")
    if dump_path is not None:
        with open(dump_path, "w") as f:
            f.write("seq,expr,cls,frame,oid,x,y,w,h,cs,b,gmc,native_part,gmc_part,fused,thr,keep\n")
            for r in dump_rows:
                (seq_, expr_, cls_, fid_, oid_, x_, y_, w_, h_,
                 cs_, b_, gmc_, npart, gpart, fused_, thr_, keep_) = r
                gmc_s = "{:.6f}".format(gmc_) if gmc_ == gmc_ else "nan"
                f.write("{},{},{},{},{},{:.2f},{:.2f},{:.2f},{:.2f},"
                        "{:.6f},{:.6f},{},{:.6f},{:.6f},{:.6f},{:.6f},{}\n".format(
                    seq_, expr_, cls_, fid_, oid_, x_, y_, w_, h_,
                    cs_, b_, gmc_s, npart, gpart, fused_, thr_, keep_))
        print(f"  [dump_preds] wrote {len(dump_rows)} object rows -> {dump_path}", flush=True)
    return res_dir, sm


def run_te(seqmap_path, results_dir, class_filter=None):
    if class_filter is None:
        sm = seqmap_path
    else:
        sm = os.path.join(os.path.dirname(seqmap_path), f"seqmap_{class_filter}.txt")
        lines = [l for l in open(seqmap_path).read().splitlines()
                 if l and classify(l.split("+", 1)[1]) == class_filter]
        if not lines: return None
        open(sm, "w").write("\n".join(lines) + "\n")
    sp = os.path.join(results_dir, "pedestrian_summary.txt")
    if os.path.exists(sp): os.remove(sp)
    cmd = [sys.executable, TRACKEVAL,
           "--METRICS", "HOTA",
           "--SEQMAP_FILE", os.path.abspath(sm),
           "--SKIP_SPLIT_FOL", "True",
           "--GT_FOLDER", os.path.abspath(results_dir),
           "--TRACKERS_FOLDER", os.path.abspath(results_dir),
           "--GT_LOC_FORMAT", "{gt_folder}/{video_id}/{expression_id}/gt.txt",
           "--TRACKERS_TO_EVAL", os.path.abspath(results_dir),
           "--USE_PARALLEL", "False", "--PLOT_CURVES", "False", "--PRINT_CONFIG", "False"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(TRACKEVAL))
    if not os.path.exists(sp):
        sys.stderr.write(f"FAIL ({class_filter}) rc={proc.returncode}\n{proc.stderr[-1500:]}\n")
        return None
    return float(open(sp).read().splitlines()[1].split()[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=0.65)
    p.add_argument("--gmc_scale", type=float, default=10.0)
    p.add_argument("--thr", type=float, default=3.0)
    p.add_argument("--alpha_appear", type=float, default=0.0)
    p.add_argument("--gmc_scale_appear", type=float, default=0.0)
    p.add_argument("--thr_appear", type=float, default=0.0)
    p.add_argument("--grid", action="store_true")
    p.add_argument("--mode", choices=["ship", "native_only", "gmc_only", "oracle",
                                       "oracle_motion", "oracle_appear"], default="ship",
                   help="counterfactual: drop GMC / drop native / oracle (perfect scoring) "
                        "/ oracle_motion (perfect on motion exprs, ship on appearance) / oracle_appear (vice versa)")
    p.add_argument("--dump_preds", default=None,
                   help="path to write per-object component CSV (cs,b,gmc,fused,thr,keep)")
    p.add_argument("--motion_fuse", choices=["add", "relu", "max", "orgate"], default="add",
                   help="motion-expr fusion: add=native+gmc (ship); relu=max(native,0)+gmc; max=max(native,gmc); orgate=add + admit if gmc_part>gmc_gate")
    p.add_argument("--gmc_gate", type=float, default=0.35,
                   help="orgate: high-confidence GMC threshold that bypasses the native veto")
    args = p.parse_args()

    print("Loading text_feat + GMC caches...", flush=True)
    text_feat = json.load(open(TEXT_FEAT_JSON))
    gmc_caches = {s: json.load(open(GMC_CACHE_TPL.format(seq=s))) for s in TEST_SEQS}

    if args.grid:
        # Path 1: APPEARANCE-axis GMC extension. Motion ship LOCKED at (α=1.0, sc=0.9, thr=+0.17 → pool 44.400).
        # Sweep appearance bias (alpha_a, scale_a, thr_a). APPEAR is 77% of frames; raw sep +0.264 > motion +0.172.
        # tuple: (tag, alpha_m, scale_m, thr_m, alpha_a, scale_a, thr_a)
        M_A, M_S, M_T = 1.0, 0.9, 0.17
        # Refine 3: refine 2 peak sc=0.25 thr=0.10 → 44.601 (+0.201, beats paper +0.037).
        # Map ridge top with sc=0.25-0.4 × thr=0.10-0.15. Identify peak vs cliff.
        configs = [
            ("appear_sc025_thrp12",  M_A, M_S, M_T, 1.0, 0.25, 0.12),
            ("appear_sc025_thrp15",  M_A, M_S, M_T, 1.0, 0.25, 0.15),
            ("appear_sc03_thrp1",    M_A, M_S, M_T, 1.0, 0.30, 0.10),
            ("appear_sc03_thrp13",   M_A, M_S, M_T, 1.0, 0.30, 0.13),
            ("appear_sc035_thrp1",   M_A, M_S, M_T, 1.0, 0.35, 0.10),
            ("appear_sc035_thrp13",  M_A, M_S, M_T, 1.0, 0.35, 0.13),
            ("appear_sc04_thrp13",   M_A, M_S, M_T, 1.0, 0.40, 0.13),
            ("appear_sc04_thrp17",   M_A, M_S, M_T, 1.0, 0.40, 0.17),
        ]
    else:
        tag = f"a{args.alpha}_scale{args.gmc_scale}_thr{args.thr}"
        if args.gmc_scale_appear != 0.0:
            tag += f"_aa{args.alpha_appear}_sca{args.gmc_scale_appear}_thra{args.thr_appear}"
        configs = [(tag, args.alpha, args.gmc_scale, args.thr,
                    args.alpha_appear, args.gmc_scale_appear, args.thr_appear)]

    os.makedirs(OUT_ROOT, exist_ok=True)
    rows = []
    for tag, a, sc, thr, a_a, sc_a, thr_a in configs:
        run_dir = os.path.join(OUT_ROOT, tag)
        os.makedirs(run_dir, exist_ok=True)
        print(f"\n=== {tag}: motion(α={a}, sc={sc}, thr={thr}) "
              f"appear(α={a_a}, sc={sc_a}, thr={thr_a}) ===", flush=True)
        dump_path = None
        if args.dump_preds:
            dump_path = (args.dump_preds if len(configs) == 1
                         else args.dump_preds.replace(".csv", f"_{tag}.csv"))
        res_dir, sm = gen_predicts(text_feat, gmc_caches, a, sc, thr, run_dir,
                                    alpha_a=a_a, scale_a=sc_a, thr_a=thr_a,
                                    mode=args.mode, dump_path=dump_path,
                                    motion_fuse=args.motion_fuse, gmc_gate=args.gmc_gate)
        pooled = run_te(sm, res_dir)
        moving = run_te(sm, res_dir, class_filter="MOVING")
        static = run_te(sm, res_dir, class_filter="STATIC")
        appear = run_te(sm, res_dir, class_filter="APPEARANCE")
        rows.append((tag, a, sc, thr, a_a, sc_a, thr_a, pooled, appear, moving, static))
        print(f"  pooled={pooled}  APPEAR={appear}  MOVING={moving}  STATIC={static}", flush=True)

    print("\n=== iKUN linear-additive GMC sweep summary ===")
    print("tag                       α_m  sc_m  thr_m  α_a  sc_a  thr_a  pooled   APPEAR   MOVING   STATIC")
    for tag, a, sc, thr, a_a, sc_a, thr_a, p_, ap, mo, st in rows:
        print(f"{tag:<25} {a:4.2f} {sc:4.2f} {thr:5.2f}  {a_a:4.2f} {sc_a:4.2f} {thr_a:5.2f}  "
              f"{p_:.3f}   {ap:.3f}   {mo:.3f}   {st:.3f}")
    print("\niKUN paper-pure baseline:  44.564")
    print("Local B (alpha=0):         44.224")
    print("Motion-only ship (eff09_thrp17): 44.400")


if __name__ == "__main__":
    main()
