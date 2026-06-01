"""Step 1 — build appearance re-rank caches (CLIP-L/14 cosine + HSV-lightness).

Both emitted in the existing rerank-cache format (expr -> fid -> oid -> score) so
run_ikun_linear_additive.gen_predicts(mode="rerank_clipb32", clip_caches=...) consumes
them unchanged. One pass over tracker-box crops:
  - CLIP-L/14 (ViT-L-14 datacomp_xl, SAME training data as the cached B/32 → clean
    encoder-only upgrade) zero-shot cosine(crop, expr-text).
  - HSV "lightness" = mean over central crop of (V/255)*(1 - S/255); high for
    silver/white/light, low for saturated/dark. For the silver+light-color subset,
    higher = match (color-specific; Step 2 generalizes per color word).

Frame indexing copies run_build_clip_logit_cache.py exactly (f1 = f0+1, center crop)
so boxes align with the B/32 cache + the eval's merged_ns.

Run (conda RMOT, GPU): python -m diagnostics.appearance_rerank.build_rerank_caches --seqs 0005 0011 0013
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/home/seanachan/GMC-Link")
from gmc_link.demo_inference import load_neuralsort_tracks
from run_build_clip_logit_cache import merged_neuralsort, crop_bbox, load_expressions

V1_FRAME_DIR = "/home/seanachan/data/Dataset/refer-kitti/KITTI/training/image_02"
V1_DATA_ROOT = "/home/seanachan/GMC-Link/refer-kitti"
OUT_DIR = "/home/seanachan/GMC-Link/gmc_link"


def hsv_lightness(pil):
    """Mean (V/255)*(1-S/255) over central 60% of the crop. High = light/desaturated."""
    rgb = np.asarray(pil)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, W = hsv.shape[:2]
    y0, y1 = int(H * 0.2), int(H * 0.8)
    x0, x1 = int(W * 0.2), int(W * 0.8)
    c = hsv[max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)]
    if c.size == 0:
        c = hsv
    S = c[:, :, 1].astype(np.float32) / 255.0
    V = c[:, :, 2].astype(np.float32) / 255.0
    return float(np.mean(V * (1.0 - S)))


def build_seq(seq, model, tokenizer, preprocess, device, batch_size=128):
    tracks = merged_neuralsort(seq)
    expressions = load_expressions(V1_DATA_ROOT, seq, use_raw=False)
    seq_frame_dir = os.path.join(V1_FRAME_DIR, seq)
    frame_files = sorted(f for f in os.listdir(seq_frame_dir) if f.endswith((".png", ".jpg")))

    img_feats = {}   # (fid,oid) -> 768D fp32
    hsv_score = {}   # (fid,oid) -> float
    batch_imgs, batch_keys = [], []

    def flush():
        if not batch_imgs:
            return
        imgs = torch.stack(batch_imgs, 0).to(device)
        with torch.no_grad():
            f = model.encode_image(imgs)
            f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        for k, v in zip(batch_keys, f.cpu().numpy().astype(np.float32)):
            img_feats[k] = v
        batch_imgs.clear(); batch_keys.clear()

    for f0, fname in enumerate(tqdm(frame_files, desc=f"  encode-{seq}")):
        f1 = f0 + 1
        dets = tracks.get(f1, [])
        if not dets:
            continue
        frame = cv2.imread(os.path.join(seq_frame_dir, fname))
        if frame is None:
            continue
        for oid, x, y, w, h in dets:
            cx, cy = x + w / 2.0, y + h / 2.0
            pil = crop_bbox(frame, cx, cy, w, h)
            if pil is None:
                continue
            hsv_score[(f1, oid)] = hsv_lightness(pil)
            batch_imgs.append(preprocess(pil)); batch_keys.append((f1, oid))
            if len(batch_imgs) >= batch_size:
                flush()
    flush()

    text_feats = {}
    for name, text in expressions.items():
        tok = tokenizer([text]).to(device)
        with torch.no_grad():
            tf = model.encode_text(tok)
            tf = tf / tf.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        text_feats[name] = tf.squeeze(0).cpu().numpy().astype(np.float32)

    keys = list(img_feats.keys())
    mat = np.stack([img_feats[k] for k in keys]) if keys else np.zeros((0, 768), np.float32)

    clip_cache, hsv_cache = {}, {}
    for name, tf in text_feats.items():
        sims = mat @ tf if len(keys) else []
        pe_clip, pe_hsv = {}, {}
        for (fid, oid), s in zip(keys, sims):
            pe_clip.setdefault(str(fid), {})[str(oid)] = float(s)
        # HSV is expr-independent but stored per-expr for format compat
        for (fid, oid), s in hsv_score.items():
            pe_hsv.setdefault(str(fid), {})[str(oid)] = float(s)
        clip_cache[name] = pe_clip
        hsv_cache[name] = pe_hsv

    cp = os.path.join(OUT_DIR, f"rerank_clipL14_neuralsort_{seq}_cache.json")
    hp = os.path.join(OUT_DIR, f"rerank_hsv_neuralsort_{seq}_cache.json")
    json.dump(clip_cache, open(cp, "w"))
    json.dump(hsv_cache, open(hp, "w"))
    print(f"[rerank] {seq}: {len(keys)} crops -> {cp} + {hp}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", default=["0005", "0011", "0013"])
    ap.add_argument("--clip-arch", default="ViT-L-14")
    ap.add_argument("--clip-pretrained", default="datacomp_xl_s13b_b90k")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import open_clip
    print(f"Loading CLIP {args.clip_arch} {args.clip_pretrained}...", flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_arch, pretrained=args.clip_pretrained)
    model = model.to(args.device).eval()
    for q in model.parameters():
        q.requires_grad = False
    tokenizer = open_clip.get_tokenizer(args.clip_arch)
    for seq in args.seqs:
        build_seq(seq, model, tokenizer, preprocess, args.device, args.batch_size)


if __name__ == "__main__":
    main()
