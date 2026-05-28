"""Build GMC cache for arbitrary V1 seq.

Adapted from run_cascade_alpha_sweep.py:precompute_gmc_scores() but
parameterized by sequence id. Used to extend Phase 4 stack sweep to
0005 + 0013 (cache only existed for 0011).

Usage:
    conda activate RMOT
    python run_build_gmc_cache.py 0005
    python run_build_gmc_cache.py 0013
"""
import os, sys, json
from collections import defaultdict
import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from gmc_link.manager import GMCLinkManager
from gmc_link.text_utils import TextEncoder
from gmc_link.demo_inference import load_neuralsort_tracks, DummyTrack
from gmc_link.depth_cache import DepthCache

DATA_ROOT   = os.environ.get("GMC_DATA_ROOT", "refer-kitti")
TRACK_DIR   = "NeuralSORT"
FRAME_DIR   = os.environ.get("GMC_FRAME_DIR", "/home/seanachan/data/Dataset/refer-kitti/KITTI/training/image_02")
CACHE_VER   = os.environ.get("GMC_CACHE_VER", "v1")  # cache filename version tag (v1|v2)
GMC_WEIGHTS = os.environ.get("GMC_WEIGHTS", "gmc_link_weights_v1train.pth")
GMC_SUFFIX  = os.environ.get("GMC_SUFFIX", "")  # appended before _cache.json: e.g. "_seed0"
GMC_RAW_COS = os.environ.get("GMC_RAW_COS", "0") == "1"  # Arm B: dump raw cosine (skip sigmoid+EMA)
GMC_DEPTH_ARCH = os.environ.get("GMC_DEPTH_ARCH", "ikun")  # depth cache arch tag
GMC_DEPTH_DIR  = os.environ.get("GMC_DEPTH_DIR",  "gmc_link/depth_cache")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


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


def build(seq):
    cache_path = f"gmc_link/gmc_scores_{CACHE_VER}_{seq}{GMC_SUFFIX}_cache.json"
    if os.path.exists(cache_path):
        print(f"[gmc] cache exists → {cache_path}, skip")
        return
    print(f"[gmc] building cache on {seq} → {cache_path}")

    text_encoder_name = "all-MiniLM-L6-v2"
    lang_dim = 384
    use_depth = False
    world_xy = False
    depth_source = "monocular"
    if os.path.exists(GMC_WEIGHTS):
        ckpt_meta = torch.load(GMC_WEIGHTS, map_location="cpu")
        if isinstance(ckpt_meta, dict) and "model" in ckpt_meta:
            text_encoder_name = ckpt_meta.get("text_encoder") or text_encoder_name
            lang_dim = ckpt_meta.get("lang_dim") or lang_dim
            use_depth = bool(ckpt_meta.get("use_depth", False))
            world_xy = bool(ckpt_meta.get("world_xy", False))
            depth_source = ckpt_meta.get("depth_source") or "monocular"
        del ckpt_meta
    print(f"  [gmc] text_encoder={text_encoder_name} lang_dim={lang_dim} use_depth={use_depth} world_xy={world_xy} depth_source={depth_source}")
    encoder = TextEncoder(model_name=text_encoder_name, device=DEVICE)

    depth_cache = None
    if use_depth:
        if depth_source in ("lidar_oxts", "lidar_cohort"):
            # Path B: LiDAR twin caches (z_track_lidar_<arch>_<seq>). Manager auto-uses
            # oxts ego ONLY for lidar_oxts (from ckpt meta); lidar_cohort keeps cohort ego.
            ddir = os.environ.get("GMC_DEPTH_DIR", "gmc_link/depth_cache_lidar")
            depth_path = os.path.join(ddir, f"z_track_lidar_{GMC_DEPTH_ARCH}_{seq}.json")
        else:
            depth_path = os.path.join(GMC_DEPTH_DIR, f"z_track_{GMC_DEPTH_ARCH}_{seq}.json")
        depth_cache = DepthCache.load(depth_path)
        print(f"  [gmc] loaded depth cache {depth_path} tracks={len(depth_cache.table)}")

    ns_tracks = merged_ns(seq)
    expr_dir = os.path.join(DATA_ROOT, "expression", seq)
    expressions = sorted(f.replace(".json","") for f in os.listdir(expr_dir) if f.endswith(".json"))
    seq_frame_dir = os.path.join(FRAME_DIR, seq)
    frame_files = sorted(f for f in os.listdir(seq_frame_dir) if f.endswith((".png",".jpg")))
    total_frames = len(frame_files)
    print(f"  {len(expressions)} exprs, {total_frames} frames")

    cache = {}
    clip_cache = {}  # frame_id -> CLIP feats; shared across exprs (CLIP is expr-independent)
    for expression in tqdm(expressions, desc=f"gmc-expr-{seq}"):
        text_emb = encoder.encode(expression.replace("-", " ")).to(DEVICE)
        linker = GMCLinkManager(weights_path=GMC_WEIGHTS, device=DEVICE, lang_dim=lang_dim, world_xy=world_xy)
        per_expr = {}
        for f0 in range(total_frames):
            f1 = f0 + 1
            dets = ns_tracks.get(f1, [])
            if not dets: continue
            frame_img = cv2.imread(os.path.join(seq_frame_dir, frame_files[f0]))
            if frame_img is None: continue
            active = [DummyTrack(oid, x, y, w, h) for oid, x, y, w, h in dets]
            det_arr = np.array([[x, y, x+w, y+h] for _, x, y, w, h in dets])
            depth_z_lookup = None
            if depth_cache is not None:
                depth_z_lookup = {}
                for oid, *_ in dets:
                    z = depth_cache.lookup(oid, f1)
                    if z is not None:
                        depth_z_lookup[oid] = float(z)
            scores, _, _ = linker.process_frame(frame_img, active, text_emb, detections=det_arr, raw_cos=GMC_RAW_COS, depth_z_lookup=depth_z_lookup, seq=seq, frame_id=f1, clip_feat_cache=clip_cache)
            for oid, g in scores.items():
                per_expr.setdefault(str(f1), {})[str(oid)] = float(g)
        cache[expression] = per_expr

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    json.dump(cache, open(cache_path, "w"))
    print(f"[gmc] cached → {cache_path}")


if __name__ == "__main__":
    seqs = sys.argv[1:] if len(sys.argv) > 1 else ["0005", "0013"]
    for s in seqs:
        build(s)
