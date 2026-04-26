#!/usr/bin/env bash
# Exp 37 Stage B — Motion feature lever (13D vs 13D + OMF).
#
# B1 (13D): re-uses Stage A winner's weights — no new training.
# B2 (13D + OMF 28D): re-trains aligner with --extra-features omf_stats.
#
# Depends on:
#   diagnostics/results/exp37/stage_a_winner      (e.g., "orb" | "recoverpose")
#
# Dense-flow source: spec §7 line 171 prescribes RAFT when SparseMFE is
# unavailable (our case). This runner assumes a precomputed OMF cache at
# ``cache/omf/<ego>/<seq>/<frame>.npz``. If the cache is absent the aligner
# falls back to zero-filled OMF block (harmless for layout, but silently
# no-ops the ablation). Implementer must populate OMF cache before training
# for the arm to be meaningful — see precompute step below.
#
# Spec: §§2 H_B, 3, 7 (Stage B), 10
# Plan: Task 11

set -euo pipefail

PY=~/miniconda/envs/RMOT/bin/python
DIAG=diagnostics/diag_gt_cosine_distributions.py
RESULTS_DIR=diagnostics/results
OUT="${RESULTS_DIR}/exp37"

if [[ ! -f "${OUT}/stage_a_winner" ]]; then
  echo "ERROR: ${OUT}/stage_a_winner missing — finish Task 8 first." >&2
  exit 1
fi
EGO=$(cat "${OUT}/stage_a_winner" | tr -d '[:space:]')

# ─── Pre-compute OMF cache (Farneback fallback since RAFT needs extra dep) ───
# Pure cv2 implementation: cv2.calcOpticalFlowFarneback per (frame, frame_gap)
# pair for every V1 training seq × FRAME_GAPS. Caches to cache/omf/<EGO>/.
OMF_CACHE="cache/omf/${EGO}"
if [[ ! -d "${OMF_CACHE}" ]]; then
  echo "============================================================"
  echo "Pre-computing Farneback OMF cache → ${OMF_CACHE}"
  echo "============================================================"
  "${PY}" - <<'PYEOF'
import os, sys, cv2, numpy as np
from pathlib import Path

EGO = os.environ.get("EGO", "orb")
DATA_ROOT = "refer-kitti"
V1_SEQS = ["0001","0002","0003","0004","0006","0007","0008","0009",
           "0010","0012","0014","0015","0016","0018","0020"]
FRAME_GAPS = (2, 5, 10)
OUT_ROOT = Path(f"cache/omf/{EGO}")

for seq in V1_SEQS:
    img_dir = Path(DATA_ROOT) / "KITTI" / "training" / "image_02" / seq
    if not img_dir.exists():
        print(f"  skip {seq}: no frames")
        continue
    out_seq = OUT_ROOT / seq
    out_seq.mkdir(parents=True, exist_ok=True)
    frames = sorted(int(p.stem) for p in img_dir.glob("*.png"))
    print(f"  {seq}: {len(frames)} frames")
    for fid in frames:
        for gap in FRAME_GAPS:
            tgt = fid + gap
            if tgt not in frames:
                continue
            out_npz = out_seq / f"{fid:06d}_gap{gap}.npz"
            if out_npz.exists():
                continue
            prev = cv2.imread(str(img_dir / f"{fid:06d}.png"), cv2.IMREAD_GRAYSCALE)
            curr = cv2.imread(str(img_dir / f"{tgt:06d}.png"), cv2.IMREAD_GRAYSCALE)
            if prev is None or curr is None:
                continue
            flow = cv2.calcOpticalFlowFarneback(
                prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            np.savez_compressed(out_npz, flow=flow.astype(np.float32))
print("OMF cache complete.")
PYEOF
fi

# ─── Train B2 ──────────────────────────────────────────────────────
B2_TAG="exp37_stage_b2_${EGO}_omf28d"
B2_W="gmc_link_weights_${B2_TAG}.pth"

echo "============================================================"
echo "Training B2: 13D + omf_stats (28D) → ${B2_W}"
echo "============================================================"
"${PY}" -m gmc_link.train \
  --split v1 \
  --ego "${EGO}" \
  --extra-features omf_stats \
  --epochs 100 \
  --lr 1e-3 \
  --batch-size 128 \
  --save-path "${B2_W}"

# ─── Eval ─────────────────────────────────────────────────────────
SEQS=(0005 0011 0013)
for seq in "${SEQS[@]}"; do
  echo "--- ${seq} / ${B2_TAG} ---"
  "${PY}" "${DIAG}" --weights "${B2_W}" --seq "${seq}"
  src="${RESULTS_DIR}/layer3_gt_cosine_${seq}.npz"
  dst="${OUT}/layer3_${seq}_${B2_TAG}.npz"
  mv "${src}" "${dst}"
  src_png="${RESULTS_DIR}/layer3_gt_cosine_${seq}.png"
  [[ -f "${src_png}" ]] && mv "${src_png}" "${OUT}/layer3_${seq}_${B2_TAG}.png"
done

"${PY}" diagnostics/aggregate_multiseq.py \
  --results-dir "${OUT}" \
  --output-dir "${OUT}" \
  --weights "${B2_TAG}=${B2_W}" \
  --seqs "${SEQS[@]}"

echo "Done. See ${OUT}/layer3_multiseq_${B2_TAG}.md"
