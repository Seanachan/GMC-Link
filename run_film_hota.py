"""HOTA macro 3-seq + per-class eval for FiLM-trained iKUN cascade.

Pipeline:
  1. Load FiLM-trained ckpt (motion_film_head trained, rest frozen)
  2. Cascade KUM inference on YOLOv8-NS tracks {0005,0011,0013}
  3. Apply sim_calib bias (paper-strength baseline B recipe)
  4. Generate predict.txt per (seq, expr) where fused = cascade_logit + sim_calib_bias > 0
  5. Run TrackEval HOTA per seq, aggregate macro + per-class

Usage:
    conda activate RMOT
    python run_film_hota.py --ckpt /home/seanachan/GMC-Link/save/film_v1/epoch19.pth
"""
import argparse, os, sys, json, shutil, subprocess, statistics, csv
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, "/home/seanachan/GMC-Link")
sys.path.insert(0, "/home/seanachan/iKUN")

from gmc_link.demo_inference import load_neuralsort_tracks, load_ikun_scores
from utils import expression_conversion as ikun_expression_conversion

DATA_ROOT = "/home/seanachan/GMC-Link/refer-kitti"
TRACK_DIR = "/home/seanachan/GMC-Link/NeuralSORT"
GT_TEMPLATE = "/home/seanachan/data/Dataset/refer-kitti/gt_template_old"
TEXT_FEAT_JSON = "/home/seanachan/GMC-Link/iKUN/text_feat_bboxNum_v1.json"
MOTION_13D_DIR = "/home/seanachan/GMC-Link/iKUN/motion_13d_cache_v1"
OUTPUT_ROOT = "/home/seanachan/GMC-Link/hota_eval_film_v1"
TRACKEVAL = "/home/seanachan/TempRMOT/TrackEval/scripts/run_mot_challenge.py"
SIM_A, SIM_B, SIM_TAU = 8.0, -0.1, 100.0

STATIC_KW = ["parking", "parked", "stopped", "stop", "stand", "static", "stationary"]
MOVING_NARROW = ["moving", "walking", "running", "turning", "faster", "slower", "braking", "accelerat"]


def classify(e):
    el = e.lower()
    if any(k in el for k in STATIC_KW): return "STATIC"
    if any(k in el for k in MOVING_NARROW): return "MOVING"
    return "OTHER"


def compute_simcalib_bias(text_feat, exprs):
    train_dict, test_dict = text_feat["train"], text_feat["test"]
    keys = list(train_dict.keys())
    FEATS = np.array([train_dict[k]["feature"] for k in keys])
    PROBS = np.array([train_dict[k]["probability"] for k in keys])
    bias = {}
    for expr in exprs:
        en = ikun_expression_conversion(expr)
        target = test_dict if en in test_dict else train_dict
        if en not in target:
            bias[expr] = 0.0; continue
        feat = np.array(target[en]["feature"])[None, :]
        sim = (feat @ FEATS.T)[0]
        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-12)
        w = np.exp(SIM_TAU * sim); w = w / w.sum()
        prob = float((w * PROBS).sum())
        bias[expr] = SIM_A * prob + SIM_B
    return bias


def run_cascade_inference_with_film(ckpt_path, target_seqs, motion_13d_dir=MOTION_13D_DIR,
                                    kum_mode="cascade attention"):
    """Run KUM inference with motion_13d FiLM injection (FiLM = no-op if ckpt has no motion_film_head)."""
    sys.argv = [sys.argv[0]]
    from opts import opts as OptsClass
    opt = OptsClass().parse()
    opt.save_root = "/home/seanachan/GMC-Link"
    opt.data_root = os.path.abspath(DATA_ROOT)
    opt.track_root = os.path.abspath(TRACK_DIR)
    opt.kum_mode = kum_mode
    opt.test_ckpt = ckpt_path
    opt.motion_13d_dir = motion_13d_dir

    import utils as ikun_utils
    _orig = ikun_utils.VIDEOS.copy()
    ikun_utils.VIDEOS["test"] = target_seqs

    from model import get_model
    from utils import load_from_ckpt, tokenize
    from dataloader import get_dataloader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(opt, "Model")
    model, _ = load_from_ckpt(model, ckpt_path)
    model = model.to(device).eval()

    dl = get_dataloader("test", opt, "Track_Dataset")
    OUT = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    try:
        with torch.no_grad():
            for batch in tqdm(dl, desc="cascade-FiLM inference"):
                inputs = dict(
                    local_img=batch["cropped_images"].to(device),
                    global_img=batch["global_images"].to(device),
                    exp=tokenize(batch["expression_new"]).to(device),
                    motion_13d=batch["motion_13d"].to(device),
                )
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(inputs)["logits"]
                logits = logits.cpu()
                for idx in range(len(batch["video"])):
                    for fid in range(int(batch["start_frame"][idx]),
                                     int(batch["stop_frame"][idx]) + 1):
                        OUT[batch["video"][idx]][int(batch["obj_id"][idx])][fid][
                            batch["expression_raw"][idx]
                        ].append(logits[idx].item())
    finally:
        ikun_utils.VIDEOS.clear()
        ikun_utils.VIDEOS.update(_orig)
    return OUT


def merged_ns(seq):
    car = load_neuralsort_tracks(os.path.join(TRACK_DIR, seq, "car", "predict.txt"))
    ped = load_neuralsort_tracks(os.path.join(TRACK_DIR, seq, "pedestrian", "predict.txt"))
    max_car = 0
    for fid, dets in car.items():
        for oid, *_ in dets: max_car = max(max_car, oid)
    ns = defaultdict(list)
    for fid, dets in car.items():
        ns[fid].extend(dets)
    for fid, dets in ped.items():
        ns[fid].extend([(oid + max_car, x, y, w, h) for oid, x, y, w, h in dets])
    return ns


def gen_predict(seq, simcalib_bias, ns, exprs, run_dir, raw_logits_per_seq):
    res_dir = os.path.join(run_dir, "results_film")
    seq_out = os.path.join(res_dir, seq)
    os.makedirs(seq_out, exist_ok=True)
    seqmap = []
    for expr in exprs:
        outd = os.path.join(seq_out, expr); os.makedirs(outd, exist_ok=True)
        gt_src = os.path.join(GT_TEMPLATE, seq, expr, "gt.txt")
        gt_dst = os.path.join(outd, "gt.txt")
        if os.path.exists(gt_src): shutil.copy2(gt_src, gt_dst)
        else: open(gt_dst, "w").close()
        bias = simcalib_bias.get(expr, 0.0)
        lines = []
        per_seq = raw_logits_per_seq.get(seq, {})
        for fid, dets in ns.items():
            for oid, x, y, w, h in dets:
                logits = per_seq.get(oid, {}).get(fid, {}).get(expr, [])
                if not logits: continue
                cs = float(np.mean(logits))
                fused = cs + bias
                if fused > 0.0:
                    lines.append(f"{fid},{oid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1")
        with open(os.path.join(outd, "predict.txt"), "w") as f:
            f.write("\n".join(lines))
            if lines: f.write("\n")
        seqmap.append(f"{seq}+{expr}")
    sm_path = os.path.join(run_dir, "seqmap.txt")
    with open(sm_path, "w") as f:
        f.write("\n".join(seqmap) + "\n")
    return res_dir, sm_path


def run_te(seqmap, results_dir):
    cmd = [sys.executable, TRACKEVAL,
           "--METRICS", "HOTA", "CLEAR", "Identity",
           "--SEQMAP_FILE", os.path.abspath(seqmap),
           "--SKIP_SPLIT_FOL", "True",
           "--GT_FOLDER", os.path.abspath(results_dir),
           "--TRACKERS_FOLDER", os.path.abspath(results_dir),
           "--GT_LOC_FORMAT", "{gt_folder}/{video_id}/{expression_id}/gt.txt",
           "--TRACKERS_TO_EVAL", os.path.abspath(results_dir),
           "--USE_PARALLEL", "False", "--PLOT_CURVES", "False", "--PRINT_CONFIG", "False"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(TRACKEVAL))
    sp = os.path.join(results_dir, "pedestrian_summary.txt")
    if not os.path.exists(sp):
        sys.stderr.write(f"[run_te] TrackEval rc={proc.returncode}, no summary at {sp}\n")
        if proc.stderr:
            sys.stderr.write(proc.stderr[-2000:] + "\n")
        return None, None
    with open(sp) as f:
        pooled = float(f.read().splitlines()[1].split()[0])
    csv_p = os.path.join(results_dir, "pedestrian_detailed.csv")
    g = {"STATIC": [], "MOVING": [], "OTHER": []}
    cols = [f"HOTA___{a}" for a in [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85]]
    with open(csv_p) as f:
        for r in csv.DictReader(f):
            sx = r.get("seq", "")
            if "+" not in sx: continue
            e = sx.split("+", 1)[1]
            vs = [float(r[c]) for c in cols if r.get(c) not in (None, "")]
            h = (statistics.mean(vs) * 100) if vs else 0.0
            g[classify(e)].append(h)
    per_class = {k: statistics.mean(v) if v else 0.0 for k, v in g.items()}
    return pooled, per_class


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="FiLM-trained ckpt path")
    p.add_argument("--seqs", nargs="+", default=["0005","0011","0013"])
    p.add_argument("--motion_13d_dir", default=MOTION_13D_DIR)
    p.add_argument("--kum_mode", default="cascade attention",
                   choices=["cascade attention", "cross correlation", "text-first modulation"])
    p.add_argument("--out_root", default=OUTPUT_ROOT)
    args = p.parse_args()

    out_root = args.out_root
    with open(TEXT_FEAT_JSON) as f:
        text_feat = json.load(f)
    os.makedirs(out_root, exist_ok=True)

    raw_logits = run_cascade_inference_with_film(args.ckpt, args.seqs, args.motion_13d_dir, args.kum_mode)

    results = {}
    for seq in args.seqs:
        print(f"\n=== Seq {seq} ===")
        expr_dir = os.path.join(DATA_ROOT, "expression", seq)
        exprs = sorted(f.replace(".json", "") for f in os.listdir(expr_dir) if f.endswith(".json"))
        ns = merged_ns(seq)
        bias = compute_simcalib_bias(text_feat, exprs)
        rd = os.path.join(out_root, seq, "B")
        if os.path.exists(rd): shutil.rmtree(rd)
        results_dir, sm = gen_predict(seq, bias, ns, exprs, rd, raw_logits)
        pooled, pc = run_te(sm, results_dir)
        results[seq] = (pooled, pc)
        if pooled is None: print(f"  FAIL"); continue
        print(f"  pooled={pooled:>7.3f}  STATIC={pc['STATIC']:>6.2f}  "
              f"MOVING={pc['MOVING']:>6.2f}  OTHER={pc['OTHER']:>6.2f}")

    print("\n=== Macro 3-seq ===")
    pools = [r[0] for r in results.values() if r[0] is not None]
    if pools:
        print(f"  macro pooled = {statistics.mean(pools):.3f}")
    print(f"\nReference baselines (cascade B, no FiLM):")
    print(f"  YOLOv8-NS macro 39.414, 0011 47.085 (Phase 5C/5E B)")


if __name__ == "__main__":
    main()
