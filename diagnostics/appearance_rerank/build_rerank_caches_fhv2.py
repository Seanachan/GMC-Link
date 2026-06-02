"""Step 2 cross-host — CLIP-L/14 rerank cache on the FlexHook-V2 tracker.

Mirrors run_build_clip_logit_cache.py's flexhook_v2 convention EXACTLY (merged_flexhook
keyed by raw predict.txt frame f1=f0+1, image f0=f1-1, V2 frames). FH V2 eval
(run_flexhook_v2_raw_sweep.load_tracks) does frame-1 then +1 → fid_pred = raw predict
frame = f1. So keying the cache by f1 aligns with the rerank lookup by fid_pred.

Text = the expr's V2 "sentence" (paraphrase), keyed by expr_id (filename) to match the
FH V2 eval. Output: rerank_clipL14_flexhook_v2_{seq}_cache.json (expr_id -> f1 -> oid -> cos).

Run (GPU): python -m diagnostics.appearance_rerank.build_rerank_caches_fhv2 --seqs 0005 0011 0013 0019
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, "/home/seanachan/GMC-Link")
from run_build_clip_logit_cache import merged_flexhook, crop_bbox

FLEXHOOK_DIR = "/home/seanachan/FlexHook/FlexHook/tracker_outputs/Temp-NeuralSORT-kitti2"
V2_FRAME_DIR = "/home/seanachan/data/Dataset/refer-kitti-v2/KITTI/training/image_02"
V2_EXPR_DIR = "/home/seanachan/data/Dataset/refer-kitti-v2/expression"
OUT_DIR = "/home/seanachan/GMC-Link/gmc_link"


def load_v2_sentences(seq):
    """expr_id (filename) -> sentence (paraphrase), matching the FH V2 eval."""
    d = os.path.join(V2_EXPR_DIR, seq)
    out = {}
    for ef in sorted(f for f in os.listdir(d) if f.endswith(".json")):
        out[ef[:-5]] = json.load(open(os.path.join(d, ef)))["sentence"]
    return out


def build_seq(seq, model, tokenizer, preprocess, device, batch_size=128):
    tracks = merged_flexhook(seq, FLEXHOOK_DIR)   # keyed by raw predict frame f1
    sentences = load_v2_sentences(seq)
    seq_frame_dir = os.path.join(V2_FRAME_DIR, seq)
    frame_files = sorted(f for f in os.listdir(seq_frame_dir) if f.endswith((".png", ".jpg")))

    img_feats = {}
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
            batch_imgs.append(preprocess(pil)); batch_keys.append((f1, oid))
            if len(batch_imgs) >= batch_size:
                flush()
    flush()

    keys = list(img_feats.keys())
    mat = np.stack([img_feats[k] for k in keys]) if keys else np.zeros((0, 768), np.float32)
    text_feats = {}
    for eid, text in sentences.items():
        tok = tokenizer([text]).to(device)
        with torch.no_grad():
            tf = model.encode_text(tok)
            tf = tf / tf.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        text_feats[eid] = tf.squeeze(0).cpu().numpy().astype(np.float32)

    cache = {}
    for eid, tf in text_feats.items():
        sims = mat @ tf if len(keys) else []
        pe = {}
        for (fid, oid), s in zip(keys, sims):
            pe.setdefault(str(fid), {})[str(oid)] = float(s)
        cache[eid] = pe
    cp = os.path.join(OUT_DIR, f"rerank_clipL14_flexhook_v2_{seq}_cache.json")
    json.dump(cache, open(cp, "w"))
    print(f"[rerank-fhv2] {seq}: {len(keys)} crops, {len(text_feats)} exprs -> {cp}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", default=["0005", "0011", "0013", "0019"])
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
