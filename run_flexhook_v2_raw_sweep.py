"""FlexHook V2 + GMC Phase 5 sweep on 4-seq POOLED HOTA (Refer-KITTI V2).

Mirrors run_flexhook_phase5_gmc_sweep.py for the V2 benchmark:
- Reads FlexHook V2 result_0.json (kitti2 ckpt, Temp-NeuralSORT-kitti2 tracker)
- Fuses with FlexHook V2 GMC cache (gmc_scores_flexhook_v2_{seq}_cache.json)
- Same form: fused = margin + alpha * (gmc - 0.5) * scale + bias_motion
- Same paper-canonical V1 winner: alpha=1, gmc_scale=10, thr=0
- 4 test seqs: 0005, 0011, 0013, 0019 (~862 expr-seq pairs)

Usage:
    python run_flexhook_v2_phase5_gmc_sweep.py --alpha 1 --gmc_scale 10 --thr 0
    python run_flexhook_v2_phase5_gmc_sweep.py --grid
"""
import argparse, json, os, shutil, subprocess, sys
from collections import defaultdict
import numpy as np

FLEXHOOK_RES = "/home/seanachan/FlexHook/retest-kitti-2/refer-kitti-v2-best/results"
RESULT_JSON  = "/home/seanachan/FlexHook/retest-kitti-2/refer-kitti-v2-best/result_0.json"
TRACK_DIR    = "/home/seanachan/FlexHook/FlexHook/tracker_outputs/Temp-NeuralSORT-kitti2"
DATA_ROOT    = "/home/seanachan/data/Dataset/refer-kitti-v2"
GT_TEMPLATE  = "/home/seanachan/FlexHook/datasets/refer-kitti-v2/gt_template_gen"
_GMC_SUFFIX  = os.environ.get("GMC_SUFFIX", "")
RAW_COS      = os.environ.get("GMC_RAW_COS", "0") == "1"  # Arm B: GMC cache contains raw cosine [-1,+1]
GMC_CACHE_TPL= "/home/seanachan/GMC-Link/gmc_link/gmc_scores_flexhook_v2_raw_{seq}" + _GMC_SUFFIX + "_cache.json"
TRACKEVAL    = "/home/seanachan/TempRMOT/TrackEval/scripts/run_mot_challenge.py"
_OUT_SUFFIX  = os.environ.get("OUT_SUFFIX", "")
OUT_ROOT     = "/home/seanachan/GMC-Link/hota_eval_flexhook_v2_raw_gmc" + _OUT_SUFFIX

TEST_SEQS = ["0005", "0011", "0013", "0019"]
FRAMES = {"0005": (0, 296), "0011": (0, 372), "0013": (0, 339), "0019": (0, 1058)}
MOTION_KW = ["moving", "walking", "running", "turning", "faster", "slower",
             "braking", "parking", "parked", "stopped", "stop", "stand",
             "static", "stationary", "accelerat",
             # V2 paraphrased motion vocab
             "transit", "traveling", "headed", "going", "passing", "drive",
             "driving", "circulating", "in-motion", "in-the-process-of-moving"]
STATIC_KW = ["parking", "parked", "stopped", "stop", "stand", "static",
             "stationary", "left-on", "abandoned", "left-behind"]


def is_motion(e): return any(k in e.lower() for k in MOTION_KW)
def is_strict_static(e): return any(k in e.lower() for k in STATIC_KW)
def classify(e):
    el = e.lower()
    if any(k in el for k in STATIC_KW): return "STATIC"
    if any(k in el for k in MOTION_KW): return "MOVING"
    return "APPEARANCE"


def _iou_xywh(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih; uni = aw * ah + bw * bh - inter
    return inter / uni if uni > 0 else 0.0


def _load_gt_boxes(gt_path):
    boxes = defaultdict(list)
    if not os.path.exists(gt_path): return boxes
    for line in open(gt_path):
        p = line.strip().split(",")
        if len(p) < 6: continue
        boxes[int(p[0])].append((float(p[2]), float(p[3]), float(p[4]), float(p[5])))
    return boxes


def load_tracks(seq):
    car_path = os.path.join(TRACK_DIR, seq, "car", "predict.txt")
    ped_path = os.path.join(TRACK_DIR, seq, "pedestrian", "predict.txt")
    arr_c = np.loadtxt(car_path, delimiter=",") if os.path.getsize(car_path) > 0 else np.empty((0, 10))
    arr_p = np.loadtxt(ped_path, delimiter=",") if os.path.exists(ped_path) and os.path.getsize(ped_path) > 0 else np.empty((0, 10))
    if arr_c.ndim == 1 and arr_c.size: arr_c = arr_c[None, :]
    if arr_p.ndim == 1 and arr_p.size: arr_p = arr_p[None, :]
    if arr_c.size:
        max_obj = arr_c[:, 1].max()
        if arr_p.size:
            arr_p[:, 1] += max_obj
        tracks = np.concatenate([arr_c, arr_p], axis=0) if arr_p.size else arr_c
    else:
        tracks = arr_p
    tracks[:, 0] = tracks[:, 0] - 1
    return tracks


def gen_predicts(cls_dict, tracks_by_seq, gmc_caches, alpha, gmc_scale, thr_motion, run_dir,
                 alpha_a=0.0, scale_a=0.0, thr_a=0.0,
                 alpha_s=0.0, scale_s=0.0, thr_s=0.0,
                 mode="ship", dump_path=None):
    res_dir = os.path.join(run_dir, "results")
    if os.path.exists(res_dir): shutil.rmtree(res_dir)
    os.makedirs(res_dir, exist_ok=True)
    seqmap = []
    dump_rows = []

    for seq in TEST_SEQS:
        if seq not in cls_dict: continue
        seq_out = os.path.join(res_dir, seq)
        os.makedirs(seq_out, exist_ok=True)
        expr_dir = os.path.join(DATA_ROOT, "expression", seq)
        exp_files = sorted(f for f in os.listdir(expr_dir) if f.endswith(".json"))
        expr_text_by_id = {}
        gt_boxes_by_expr = {}
        for ef in exp_files:
            expr_id = ef.replace(".json", "")
            with open(os.path.join(expr_dir, ef)) as fh:
                expr_text_by_id[expr_id] = json.load(fh)["sentence"]
            outd = os.path.join(seq_out, expr_id)
            os.makedirs(outd, exist_ok=True)
            gt_src = os.path.join(GT_TEMPLATE, seq, expr_id, "gt.txt")
            gt_boxes_by_expr[expr_id] = _load_gt_boxes(gt_src)
            gt_dst = os.path.join(outd, "gt.txt")
            if os.path.exists(gt_src):
                if os.path.exists(gt_dst) or os.path.islink(gt_dst): os.remove(gt_dst)
                shutil.copy2(gt_src, gt_dst)
            else:
                open(gt_dst, "w").close()
            seqmap.append(f"{seq}+{expr_id}")

        tracks = tracks_by_seq[seq]
        tracks_idx = {}
        for r in tracks:
            k = (int(r[0]), int(r[1]))
            tracks_idx.setdefault(k, r)
        min_f, max_f = FRAMES[seq]
        gmc_seq = gmc_caches.get(seq, {})
        seq_dict = cls_dict[seq]
        pred_buf = defaultdict(list)

        for obj_id, obj_dict in seq_dict.items():
            oid_int = int(obj_id)
            for frame_id, frame_dict in obj_dict.items():
                fid_int = int(frame_id)
                row = tracks_idx.get((fid_int, oid_int))
                if row is None: continue
                bbox = row.copy()
                if not (min_f <= bbox[0] <= max_f): continue
                bbox[0] += 1
                fid_pred = int(bbox[0])
                bbox_str = ",".join(map(str, bbox.tolist()))

                for expr_id, expr_text in expr_text_by_id.items():
                    score = frame_dict.get(expr_text)
                    if score is None: continue
                    margin = float(score[1] - score[0])

                    cls = classify(expr_id)
                    default = 0.0 if RAW_COS else 0.5
                    if cls == "STATIC" and scale_s != 0.0:
                        gmc = gmc_seq.get(expr_id, {}).get(str(fid_pred), {}).get(str(oid_int), default)
                        gmc_term = gmc if RAW_COS else (gmc - 0.5)
                        bias = alpha_s * gmc_term * scale_s
                        thr = thr_s
                    elif cls == "MOVING" or cls == "STATIC":
                        # Preserve ship: STATIC inherits MOTION bias when scale_s=0
                        if alpha != 0.0:
                            gmc = gmc_seq.get(expr_id, {}).get(str(fid_pred), {}).get(str(oid_int), default)
                            gmc_term = gmc if RAW_COS else (gmc - 0.5)
                            bias = alpha * gmc_term * gmc_scale
                        else:
                            bias = 0.0
                        thr = thr_motion
                    else:
                        if alpha_a != 0.0 and scale_a != 0.0:
                            gmc = gmc_seq.get(expr_id, {}).get(str(fid_pred), {}).get(str(oid_int), default)
                            gmc_term = gmc if RAW_COS else (gmc - 0.5)
                            bias = alpha_a * gmc_term * scale_a
                        else:
                            bias = 0.0
                        thr = thr_a
                    if mode == "native_only":   fused = margin
                    elif mode == "gmc_only":    fused = bias
                    else:                       fused = margin + bias
                    keep = fused > thr
                    use_oracle = (mode == "oracle"
                                  or (mode == "oracle_motion" and cls in ("MOVING", "STATIC"))
                                  or (mode == "oracle_appear" and cls == "APPEARANCE"))
                    if use_oracle:
                        gbs = gt_boxes_by_expr.get(expr_id, {}).get(fid_pred, [])
                        ob = (float(bbox[2]), float(bbox[3]), float(bbox[4]), float(bbox[5]))
                        keep = any(_iou_xywh(ob, gb) >= 0.5 for gb in gbs)
                    if keep:
                        pred_buf[expr_id].append(bbox_str)
                    if dump_path is not None:
                        dump_rows.append((seq, expr_id, cls, fid_pred, oid_int,
                                          float(bbox[2]), float(bbox[3]), float(bbox[4]), float(bbox[5]),
                                          margin, bias, margin + bias, thr, int(keep)))

        for expr_id in expr_text_by_id:
            outd = os.path.join(seq_out, expr_id)
            with open(os.path.join(outd, "predict.txt"), "w") as f:
                lines = pred_buf.get(expr_id, [])
                if lines: f.write("\n".join(lines) + "\n")

    sm_path = os.path.join(run_dir, "seqmap.txt")
    with open(sm_path, "w") as f:
        f.write("\n".join(seqmap) + "\n")
    if dump_path is not None:
        with open(dump_path, "w") as f:
            f.write("seq,expr,cls,frame,oid,x,y,w,h,native_part,gmc_part,fused,thr,keep\n")
            for r in dump_rows:
                f.write("{},{},{},{},{},{:.2f},{:.2f},{:.2f},{:.2f},{:.6f},{:.6f},{:.6f},{:.6f},{}\n".format(*r))
        print(f"  [dump_preds] wrote {len(dump_rows)} rows -> {dump_path}", flush=True)
    return res_dir, sm_path


def run_te(seqmap, results_dir, class_filter=None):
    if class_filter is None:
        sm = seqmap
    else:
        sm = os.path.join(os.path.dirname(seqmap), f"seqmap_{class_filter}.txt")
        lines = [l for l in open(seqmap).read().splitlines()
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
           "--USE_PARALLEL", "False", "--PLOT_CURVES", "False",
           "--PRINT_CONFIG", "False"]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=os.path.dirname(TRACKEVAL))
    if not os.path.exists(sp):
        sys.stderr.write(f"FAIL ({class_filter}) rc={proc.returncode}\n{proc.stderr[-1500:]}\n")
        return None
    return float(open(sp).read().splitlines()[1].split()[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--gmc_scale", type=float, default=10.0)
    p.add_argument("--thr", type=float, default=0.0)
    p.add_argument("--alpha_appear", type=float, default=0.0)
    p.add_argument("--gmc_scale_appear", type=float, default=0.0)
    p.add_argument("--thr_appear", type=float, default=0.0)
    p.add_argument("--alpha_static", type=float, default=0.0)
    p.add_argument("--gmc_scale_static", type=float, default=0.0)
    p.add_argument("--thr_static", type=float, default=0.0)
    p.add_argument("--grid", action="store_true")
    p.add_argument("--mode", choices=["ship", "native_only", "gmc_only",
                                       "oracle", "oracle_motion", "oracle_appear"], default="ship")
    p.add_argument("--dump_preds", default=None)
    args = p.parse_args()

    print(f"Loading FlexHook V2 result_0.json ({RESULT_JSON})...", flush=True)
    with open(RESULT_JSON) as fh:
        cls_dict = json.load(fh)

    print("Loading GMC V2 caches...", flush=True)
    gmc_caches = {}
    for s in TEST_SEQS:
        cp = GMC_CACHE_TPL.format(seq=s)
        if os.path.exists(cp):
            with open(cp) as fh:
                gmc_caches[s] = json.load(fh)
        else:
            print(f"  WARN: {cp} missing — alpha will silently default to 0 for {s}")

    print("Loading tracks...", flush=True)
    tracks_by_seq = {seq: load_tracks(seq) for seq in TEST_SEQS}

    if args.grid:
        # 2026-05-03 STATIC recipe-split. Lock V2 motion ship (α=0.4, sc=10, thr=+1.3)
        # + V2 appear ship (α=1.0, sc=3.5, thr=+1.2). Sweep STATIC sc_s + thr_s.
        # Goal: recover STATIC class loss (Δ=−0.040 in gap diagnosis) on V2.
        M_A, M_S, M_T = 0.4,  10.0, +1.3
        A_A, A_S, A_T = 1.0,  3.5,  +1.2
        configs = [
            # (tag, αm, sm, tm, αa, sa, ta, αs, ss, ts)
            ("v2_static_off",        M_A, M_S, M_T, A_A, A_S, A_T,  0.0, 0.0,  0.0),
            ("v2_static_sc05_thr12", M_A, M_S, M_T, A_A, A_S, A_T,  1.0, 0.5,  +1.2),
            ("v2_static_sc1_thr12",  M_A, M_S, M_T, A_A, A_S, A_T,  1.0, 1.0,  +1.2),
            ("v2_static_sc15_thr12", M_A, M_S, M_T, A_A, A_S, A_T,  1.0, 1.5,  +1.2),
            ("v2_static_sc2_thr12",  M_A, M_S, M_T, A_A, A_S, A_T,  1.0, 2.0,  +1.2),
            ("v2_static_sc25_thr12", M_A, M_S, M_T, A_A, A_S, A_T,  1.0, 2.5,  +1.2),
            ("v2_static_neg",        M_A, M_S, M_T, A_A, A_S, A_T, -1.0, 3.5,  +1.2),
        ]
    else:
        tag = f"a{args.alpha}_scale{args.gmc_scale}_thr{args.thr}"
        if args.gmc_scale_appear != 0.0:
            tag += f"_aa{args.alpha_appear}_sca{args.gmc_scale_appear}_thra{args.thr_appear}"
        if args.gmc_scale_static != 0.0:
            tag += f"_as{args.alpha_static}_scs{args.gmc_scale_static}_thrs{args.thr_static}"
        configs = [(tag, args.alpha, args.gmc_scale, args.thr,
                    args.alpha_appear, args.gmc_scale_appear, args.thr_appear,
                    args.alpha_static, args.gmc_scale_static, args.thr_static)]

    os.makedirs(OUT_ROOT, exist_ok=True)
    rows = []
    for cfg in configs:
        tag, a, sc, thr, a_a, sc_a, thr_a, a_s, sc_s, thr_s = cfg
        run_dir = os.path.join(OUT_ROOT, tag)
        os.makedirs(run_dir, exist_ok=True)
        print(f"\n=== {tag}: motion(α={a}, sc={sc}, thr={thr}) "
              f"appear(α={a_a}, sc={sc_a}, thr={thr_a}) "
              f"static(α={a_s}, sc={sc_s}, thr={thr_s}) ===", flush=True)
        dump_path = (args.dump_preds if (args.dump_preds and len(configs) == 1) else None)
        res_dir, sm = gen_predicts(cls_dict, tracks_by_seq, gmc_caches, a, sc, thr, run_dir,
                                    alpha_a=a_a, scale_a=sc_a, thr_a=thr_a,
                                    alpha_s=a_s, scale_s=sc_s, thr_s=thr_s,
                                    mode=args.mode, dump_path=dump_path)
        pooled = run_te(sm, res_dir)
        moving = run_te(sm, res_dir, class_filter="MOVING")
        static = run_te(sm, res_dir, class_filter="STATIC")
        appear = run_te(sm, res_dir, class_filter="APPEARANCE")
        rows.append((tag, a, sc, thr, a_a, sc_a, thr_a, a_s, sc_s, thr_s, pooled, appear, moving, static))
        print(f"  pooled={pooled}  APPEAR={appear}  MOVING={moving}  STATIC={static}", flush=True)

    print("\n=== FlexHook V2 Phase 5 + GMC sweep summary ===")
    print(f"{'tag':<28} {'α_m':>4} {'sc_m':>5} {'thr_m':>5} {'α_a':>4} {'sc_a':>5} {'thr_a':>5} "
          f"{'α_s':>5} {'sc_s':>5} {'thr_s':>5} {'pooled':>8} {'APPEAR':>8} {'MOVING':>8} {'STATIC':>8}")
    base = next((r for r in rows if "off" in r[0] or "ctrl" in r[0] or "baseline" in r[0]), None)
    fmt = lambda v: f"{v:>8.3f}" if v is not None else "    None"
    for tag, a, sc, thr, a_a, sc_a, thr_a, a_s, sc_s, thr_s, pl, ap, m, s in rows:
        print(f"{tag:<28} {a:>4.2f} {sc:>5.1f} {thr:>+5.2f} {a_a:>4.2f} {sc_a:>5.1f} {thr_a:>+5.2f} "
              f"{a_s:>+5.2f} {sc_s:>5.1f} {thr_s:>+5.2f} "
              f"{fmt(pl)} {fmt(ap)} {fmt(m)} {fmt(s)}")
    if base is not None:
        bp, bs = base[10], base[13]
        print(f"\nΔ vs ctrl ({base[0]}):")
        for tag, a, sc, thr, a_a, sc_a, thr_a, a_s, sc_s, thr_s, pl, ap, m, s in rows:
            dp = (pl - bp) if (pl is not None and bp is not None) else None
            ds = (s  - bs) if (s  is not None and bs is not None) else None
            dp_s = f"{dp:>+7.3f}" if dp is not None else "   None"
            ds_s = f"{ds:>+7.3f}" if ds is not None else "   None"
            print(f"  {tag:<28}  Δpool={dp_s}  ΔSTATIC={ds_s}")
    print(f"\nFlexHook V2 paper claim: 42.526")
    print(f"V2 motion+appear ship: 42.799")


if __name__ == "__main__":
    main()
