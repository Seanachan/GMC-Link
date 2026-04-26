"""
V1 Fair-Split HOTA Evaluation
==============================
Train split: seq 0011 (GMC-Link aligner + fusion head both trained here)
Test  split: seq 0005, 0013 (never seen during training)

Steps:
  1. Run iKUN inference on 0005, 0011, 0013 (val split) → save ikun_results_v1.json
  2. For test seqs (0005, 0013): generate predict.txt using NeuralSORT tracks
     - baseline:  iKUN logit > 0
     - fusion:    learned fusion head (ikun_logit, gmc_score, is_motion_flag)
                  using gmc_link_weights_v1train.pth
  3. Run TrackEval HOTA

Usage:
    python run_hota_eval_v1.py                 # both methods
    python run_hota_eval_v1.py --method baseline
    python run_hota_eval_v1.py --method fusion
    python run_hota_eval_v1.py --skip-ikun     # skip iKUN inference if already done
"""

import os
import sys
import json
import math
import shutil
import argparse
import subprocess
from collections import defaultdict

import cv2
import numpy as np
import torch

# ── Paths ───────────────────────────────────────────────────────────────
IKUN_DIR       = "/home/seanachan/iKUN"
DATA_ROOT      = "refer-kitti"           # symlink to Refer-KITTI V1
TRACK_DIR      = "NeuralSORT"
WEIGHTS_PATH   = "gmc_link_weights_v1train.pth"
FUSION_WEIGHTS = "gmc_link/fusion_head_weights_v1.pth"
GT_TEMPLATE    = os.path.join("Refer-KITTI", "gt_template")
IKUN_WEIGHTS   = "iKUN.pth"

IKUN_RESULTS_PATH = "iKUN/ikun_results_v1.json"

TRACKEVAL_SCRIPT = "/home/seanachan/TempRMOT/TrackEval/scripts/run_mot_challenge.py"
OUTPUT_ROOT      = "hota_eval_v1_clean"
MOTION_ONLY      = False  # set via --motion-only flag

# ── Fusion mode config ──────────────────────────────────────────────────
FUSION_MODE = "additive"   # "additive" (2-param) or "mlp" (learned head)
ALPHA       = 0.07         # weight for GMC logit in additive mode

# V1 split
TRAIN_SEQS = ["0011"]
TEST_SEQS  = ["0005", "0011", "0013"]  # held-out set
ALL_SEQS   = ["0005", "0011", "0013"]

FRAMES = {
    "0005": (0, 296),
    "0011": (0, 372),
    "0013": (0, 339),
}

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


# ── iKUN inference ───────────────────────────────────────────────────────

def run_ikun_inference():
    """Run iKUN on all val sequences (0005, 0011, 0013) and save results."""
    if os.path.exists(IKUN_RESULTS_PATH):
        print(f"iKUN results already exist: {IKUN_RESULTS_PATH}")
        return

    print("Running iKUN inference on val sequences (0005, 0011, 0013)...")

    _saved_argv = sys.argv
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, IKUN_DIR)
    try:
        from opts import opts as OptsClass
        opt = OptsClass().parse()
    finally:
        sys.argv = _saved_argv

    opt.save_root  = os.path.abspath(".")
    opt.data_root  = os.path.abspath(DATA_ROOT)
    opt.track_root = os.path.abspath(TRACK_DIR)

    # Patch VIDEOS to run on all val sequences
    import utils as ikun_utils
    _orig_videos = ikun_utils.VIDEOS.copy()
    ikun_utils.VIDEOS["test"] = ALL_SEQS

    from model import get_model
    from utils import load_from_ckpt, tokenize, EXPRESSIONS
    from dataloader import get_dataloader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(opt, "Model")
    try:
        model, _ = load_from_ckpt(model, os.path.abspath(IKUN_WEIGHTS))
        print("iKUN weights loaded.")
    except Exception as e:
        print(f"Warning: Could not load iKUN weights: {e}")
    model.eval()

    dataloader = get_dataloader("test", opt, "Track_Dataset")
    print(f"Track_Dataset: {len(dataloader.dataset)} samples")

    from tqdm import tqdm
    OUTPUTS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="iKUN inference"):
            inputs = dict(
                local_img=batch["cropped_images"].to(device),
                global_img=batch["global_images"].to(device),
                exp=tokenize(batch["expression_new"]).to(device),
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                similarity = model(inputs)["logits"].cpu()
            for idx in range(len(batch["video"])):
                for frame_id in range(int(batch["start_frame"][idx]),
                                      int(batch["stop_frame"][idx]) + 1):
                    OUTPUTS[batch["video"][idx]][int(batch["obj_id"][idx])][frame_id][
                        batch["expression_raw"][idx]
                    ].append(similarity[idx].item())

    # Restore VIDEOS
    ikun_utils.VIDEOS.update(_orig_videos)

    os.makedirs(os.path.dirname(IKUN_RESULTS_PATH), exist_ok=True)
    with open(IKUN_RESULTS_PATH, "w") as f:
        json.dump(OUTPUTS, f)
    print(f"iKUN results saved → {IKUN_RESULTS_PATH}")


# ── Helpers ──────────────────────────────────────────────────────────────

def load_neuralsort_tracks(path):
    """Load NeuralSORT predict.txt → {frame_id: [(obj_id, x, y, w, h), ...]}"""
    tracks = defaultdict(list)
    if not os.path.exists(path):
        return tracks
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 0 or data.size == 0:
        return tracks
    if data.ndim == 1:
        data = data[np.newaxis]
    for row in data:
        fid, oid, x, y, w, h = int(row[0]), int(row[1]), row[2], row[3], row[4], row[5]
        tracks[fid].append((oid, x, y, w, h))
    return tracks


def load_ikun_scores(results_path, sequence, expression):
    """Returns {frame_id: {obj_id: logit}}"""
    with open(results_path) as f:
        all_data = json.load(f)
    out = defaultdict(dict)
    seq_data = all_data.get(sequence, {})
    for obj_id_str, obj_dict in seq_data.items():
        obj_id = int(obj_id_str)
        for frame_id_str, frame_dict in obj_dict.items():
            frame_id = int(frame_id_str)
            if expression in frame_dict:
                vals = frame_dict[expression]
                out[frame_id][obj_id] = float(np.mean(vals))
    return out


def load_tracks_for_seq(sequence):
    """Merge car + pedestrian NeuralSORT tracks for a sequence."""
    car_path = os.path.join(TRACK_DIR, sequence, "car", "predict.txt")
    ped_path = os.path.join(TRACK_DIR, sequence, "pedestrian", "predict.txt")
    car_tracks = load_neuralsort_tracks(car_path)
    ped_tracks = load_neuralsort_tracks(ped_path)

    # Offset pedestrian IDs to avoid collision
    max_car_id = 0
    for dets in car_tracks.values():
        for oid, *_ in dets:
            max_car_id = max(max_car_id, oid)

    merged = defaultdict(list)
    for fid, dets in car_tracks.items():
        merged[fid].extend(dets)
    for fid, dets in ped_tracks.items():
        merged[fid].extend([(oid + max_car_id, x, y, w, h) for oid, x, y, w, h in dets])
    return merged


def classify_expression(sentence):
    MOTION_KW = ["moving", "driving", "parking", "parked", "turning", "braking",
                 "following", "approaching", "slower", "faster", "walking",
                 "heading", "traveling", "travelling", "same direction",
                 "counter direction", "opposite direction"]
    STATIC_KW = ["stationary", "stopped", "standing", "at rest", "still"]
    s = sentence.lower()
    if any(k in s for k in STATIC_KW):
        return "stationary"
    if any(k in s for k in MOTION_KW):
        return "motion"
    return "appearance"


# ── Prediction generation ─────────────────────────────────────────────────

def generate_predictions(method: str):
    from gmc_link.manager import GMCLinkManager
    from gmc_link.text_utils import TextEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Read encoder name + lang_dim from checkpoint (fallback to MiniLM/384 for legacy weights)
    ckpt_enc_name, ckpt_lang_dim = "all-MiniLM-L6-v2", 384
    if method == "fusion":
        try:
            _ck = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
            if isinstance(_ck, dict):
                ckpt_enc_name = _ck.get("text_encoder", ckpt_enc_name)
                ckpt_lang_dim = int(_ck.get("lang_dim", ckpt_lang_dim))
        except Exception as e:
            print(f"Warning: could not inspect {WEIGHTS_PATH} metadata ({e}); using defaults")

    encoder, fusion_model, fusion_thr = None, None, 0.5
    if method == "fusion":
        encoder = TextEncoder(model_name=ckpt_enc_name, device=device)
        if FUSION_MODE == "mlp":
            from gmc_link.fusion_head import load_fusion_head
            fusion_model, fusion_thr = load_fusion_head(FUSION_WEIGHTS)
            print(f"Fusion head loaded (thr={fusion_thr:.2f}) | GMC weights: {WEIGHTS_PATH} | enc: {ckpt_enc_name} ({ckpt_lang_dim}D)")
        else:
            print(f"Additive logit fusion (α={ALPHA:.3f}) | GMC weights: {WEIGHTS_PATH} | enc: {ckpt_enc_name} ({ckpt_lang_dim}D)")

    for sequence in TEST_SEQS:
        print(f"\n── Sequence {sequence} [{method}] ──")
        seq_out = os.path.join(OUTPUT_ROOT, f"results_{method}", sequence)
        os.makedirs(seq_out, exist_ok=True)

        ns_tracks = load_tracks_for_seq(sequence)
        frame_dir = os.path.join(DATA_ROOT, "KITTI", "training", "image_02", sequence)
        frame_files = sorted(f for f in os.listdir(frame_dir) if f.endswith(".png"))

        expr_dir_v1 = os.path.join(DATA_ROOT, "expression", sequence)
        all_expressions = sorted(f.replace(".json", "") for f in os.listdir(expr_dir_v1)
                                 if f.endswith(".json"))

        # Optionally filter to motion-only expressions
        if MOTION_ONLY:
            expressions = []
            for expr in all_expressions:
                sent = expr.replace("-", " ")
                ejson = os.path.join(expr_dir_v1, f"{expr}.json")
                if os.path.exists(ejson):
                    with open(ejson) as _f:
                        sent = json.load(_f)["sentence"]
                if classify_expression(sent) in ("motion", "stationary"):
                    expressions.append(expr)
            print(f"  Motion-only filter: {len(expressions)}/{len(all_expressions)} expressions")
        else:
            expressions = all_expressions
        print(f"  {len(expressions)} expressions, {len(frame_files)} frames")

        gt_template_seq = os.path.join(GT_TEMPLATE, sequence)

        for expr_idx, expression in enumerate(expressions):
            expr_out = os.path.join(seq_out, expression)
            os.makedirs(expr_out, exist_ok=True)

            # Copy GT
            gt_src = os.path.join(gt_template_seq, expression, "gt.txt")
            gt_dst = os.path.join(expr_out, "gt.txt")
            if os.path.exists(gt_src):
                shutil.copy2(gt_src, gt_dst)
            else:
                open(gt_dst, "w").close()

            ikun_scores = load_ikun_scores(IKUN_RESULTS_PATH, sequence, expression)

            sentence = expression.replace("-", " ")
            expr_json = os.path.join(DATA_ROOT, "expression", sequence, f"{expression}.json")
            if os.path.exists(expr_json):
                with open(expr_json) as f:
                    sentence = json.load(f)["sentence"]
            expr_type = classify_expression(sentence)

            # Fresh GMC manager per expression (stateful)
            linker = None
            lang_emb = None
            if method == "fusion":
                linker = GMCLinkManager(weights_path=WEIGHTS_PATH, device=device, lang_dim=ckpt_lang_dim)
                lang_emb = encoder.encode(sentence)

            predict_lines = []
            for frame_0idx, fname in enumerate(frame_files):
                frame_1idx = frame_0idx + 1
                if frame_1idx not in ns_tracks:
                    continue
                detections = ns_tracks[frame_1idx]

                gmc_scores = {}
                gmc_cosines = {}
                if method == "fusion" and linker is not None:
                    frame_img = cv2.imread(os.path.join(frame_dir, fname))
                    if frame_img is not None:
                        class _T:
                            def __init__(self, oid, x, y, w, h):
                                self.id = oid
                                self.bbox = [x, y, x+w, y+h]
                                self.centroid = (x+w/2, y+h/2)
                        active = [_T(o, x, y, w, h) for o, x, y, w, h in detections]
                        det_arr = np.array([[x,y,x+w,y+h] for _,x,y,w,h in detections]) if detections else None
                        try:
                            gmc_scores, _, gmc_cosines = linker.process_frame(
                                frame_img, active, lang_emb, detections=det_arr)
                        except Exception:
                            pass

                for obj_id, x, y, w, h in detections:
                    logit = ikun_scores.get(frame_1idx, {}).get(obj_id)
                    if logit is None:
                        continue
                    if method == "baseline":
                        is_pos = logit > 0.0
                    elif FUSION_MODE == "additive":
                        gmc_cos = gmc_cosines.get(obj_id, 0.0)
                        if expr_type in ("motion", "stationary"):
                            final_score = logit + ALPHA * gmc_cos
                        else:
                            final_score = logit  # appearance-only: ignore GMC
                        is_pos = final_score > 0.0
                    else:  # mlp mode
                        gmc_prob = gmc_scores.get(obj_id, 0.0)
                        is_motion = 1.0 if expr_type == "motion" else (0.5 if expr_type == "stationary" else 0.0)
                        feat = torch.tensor([[logit, gmc_prob, is_motion]], dtype=torch.float32)
                        with torch.no_grad():
                            prob = fusion_model.predict_prob(feat).item()
                        is_pos = prob >= fusion_thr
                    if is_pos:
                        predict_lines.append(
                            f"{frame_1idx},{obj_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1"
                        )

            with open(os.path.join(expr_out, "predict.txt"), "w") as f:
                f.write("\n".join(predict_lines))
                if predict_lines:
                    f.write("\n")

            if (expr_idx + 1) % 20 == 0:
                print(f"  {expr_idx+1}/{len(expressions)} expressions done")

        print(f"  → {seq_out}")

    return os.path.join(OUTPUT_ROOT, f"results_{method}")


# ── HOTA evaluation ───────────────────────────────────────────────────────

def create_seqmap(method: str) -> str:
    seqmap_path = os.path.join(OUTPUT_ROOT, f"seqmap_{method}.txt")
    lines = []
    for seq in TEST_SEQS:
        seq_dir = os.path.join(OUTPUT_ROOT, f"results_{method}", seq)
        if not os.path.isdir(seq_dir):
            continue
        for expr in sorted(os.listdir(seq_dir)):
            if os.path.isdir(os.path.join(seq_dir, expr)):
                lines.append(f"{seq}+{expr}")
    with open(seqmap_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Seqmap: {seqmap_path} ({len(lines)} entries)")
    return seqmap_path


def run_trackeval(method: str, seqmap_path: str, results_dir: str):
    cmd = [
        sys.executable, TRACKEVAL_SCRIPT,
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--SEQMAP_FILE", os.path.abspath(seqmap_path),
        "--SKIP_SPLIT_FOL", "True",
        "--GT_FOLDER", os.path.abspath(results_dir),
        "--TRACKERS_FOLDER", os.path.abspath(results_dir),
        "--GT_LOC_FORMAT", "{gt_folder}/{video_id}/{expression_id}/gt.txt",
        "--TRACKERS_TO_EVAL", os.path.abspath(results_dir),
        "--USE_PARALLEL", "False",
        "--PLOT_CURVES", "False",
        "--PRINT_CONFIG", "False",
    ]
    print(f"\nRunning TrackEval [{method}]...")
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=os.path.dirname(TRACKEVAL_SCRIPT))
    print("\n" + "=" * 70)
    print(f"Results [{method.upper()}]")
    print("=" * 70)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-2000:] if result.stderr else "")
    return result.returncode


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    global OUTPUT_ROOT, MOTION_ONLY, FUSION_MODE, ALPHA, WEIGHTS_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["baseline", "fusion", "both"], default="both")
    parser.add_argument("--skip-ikun", action="store_true",
                        help="Skip iKUN inference (use existing ikun_results_v1.json)")
    parser.add_argument("--motion-only", action="store_true",
                        help="Evaluate only motion/stationary expressions (filter appearance-only)")
    parser.add_argument("--fusion-mode", choices=["additive", "mlp"], default="additive",
                        help="Fusion strategy: additive logit (default) or learned MLP")
    parser.add_argument("--alpha", type=float, default=None,
                        help="GMC logit weight for additive fusion (overrides ALPHA)")
    parser.add_argument("--weights", type=str, default=None,
                        help="GMC-Link aligner weights path (overrides WEIGHTS_PATH)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Suffix for OUTPUT_ROOT (e.g. 'stage1' → hota_eval_v1_stage1)")
    args = parser.parse_args()

    FUSION_MODE = args.fusion_mode
    if args.alpha is not None:
        ALPHA = args.alpha
    if args.weights is not None:
        WEIGHTS_PATH = args.weights

    if args.motion_only:
        MOTION_ONLY = True
        OUTPUT_ROOT = "hota_eval_v1_motion"
        print("Motion-only mode: filtering appearance-only expressions")

    if args.tag:
        OUTPUT_ROOT = f"hota_eval_v1_{args.tag}"
        print(f"Output root: {OUTPUT_ROOT}")

    if not args.skip_ikun:
        run_ikun_inference()
    else:
        print(f"Skipping iKUN inference, using {IKUN_RESULTS_PATH}")

    methods = ["baseline", "fusion"] if args.method == "both" else [args.method]
    for method in methods:
        print(f"\n{'='*70}\nGenerating predictions: {method}\n{'='*70}")
        results_dir = generate_predictions(method)
        seqmap_path = create_seqmap(method)
        run_trackeval(method, seqmap_path, results_dir)


if __name__ == "__main__":
    main()
