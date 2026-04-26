# Label Audit GT Video Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render per-expression MP4 audit clips for V1 motion-verb expressions on seqs {0005, 0011}, plus a review.csv template, so the user can visually verify GT label correctness.

**Architecture:** Single Python script `run_gt_audit_video.py` at repo root (follows existing `run_*.py` convention, e.g. `run_ikun_baseline_video.py`). Loads per-expression GT from Refer-KITTI V1 expression JSONs, bboxes from `Refer-KITTI/gt_template` MOT files, NeuralSORT predicted tracks as context. Uses OpenCV for frame overlay + MP4 encoding.

**Tech Stack:** Python 3, OpenCV (`cv2`), NumPy, stdlib (`json`, `csv`, `pathlib`, `collections`). Reuses `gmc_link.dataset.is_motion_expression` for filter.

**Spec:** `docs/superpowers/specs/2026-04-20-label-audit-gt-video-design.md`

---

## Context for the implementer

- The repo has no `tests/` directory and no pytest config. Smoke tests are inline as `python -c "..."` commands invoking the module's functions against real data. No new test framework is introduced for a one-off audit utility.
- Refer-KITTI V1 data roots:
  - Frames: `/home/seanachan/data/Dataset/refer-kitti/KITTI/training/image_02/<seq>/<fid:010d>.png`
  - Expressions: `/home/seanachan/data/Dataset/refer-kitti/expression/<seq>/<slug>.json` with keys `{label: {fid: [tids]}, ignore, video_name, sentence}`
  - GT bboxes for each expression: `Refer-KITTI/gt_template/<seq>/<slug>/gt.txt` (MOT format: `frame,id,x,y,w,h,...`)
  - Context tracks: `NeuralSORT/<seq>/{car,pedestrian}/predict.txt`
- Known motion expression on seq 0005 used in smoke tests: pick any `*.json` whose sentence contains a word from `MOTION_KEYWORDS` (e.g. `moving`, `parked`, `turning`).
- Output goes to `diagnostics/results/label_audit/` (directory created on run).
- The `gt_template` directory was regenerated on 2026-04-16 per `CLAUDE.md`; bbox coords are correct pixel values in MOT format.

---

## File structure

Only one file is created. All logic lives in `run_gt_audit_video.py`:

| Function | Responsibility |
|---|---|
| `load_expression(path)` | Parse expression JSON → `{sentence, gt_by_frame}` |
| `load_gt_template_mot(seq, slug)` | Parse `gt_template/.../gt.txt` → `{fid: [(tid,x,y,w,h)]}` |
| `load_neuralsort_context(seq)` | Merge `car/pedestrian/predict.txt` → `{fid: [(tid,x,y,w,h)]}` |
| `draw_box(img, x, y, w, h, color, label, thickness)` | Render one bbox with optional label tag |
| `draw_text_banner(img, text, pos)` | Render text with dark background |
| `render_expression_video(seq, slug, expr_data)` | Compose per-expression MP4 |
| `write_review_csv(clips)` | Write audit review template |
| `main()` | Loop seqs × motion-expressions, invoke renderer |

---

## Task 1: Scaffold + data loaders

**Files:**
- Create: `run_gt_audit_video.py`

- [ ] **Step 1: Create scaffold with paths, constants, and loader stubs**

```python
#!/usr/bin/env python
"""Render GT-annotated audit videos for Refer-KITTI V1 motion expressions."""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gmc_link.dataset import is_motion_expression

BASE = Path("/home/seanachan/GMC-Link")
DATA_ROOT = Path("/home/seanachan/data/Dataset/refer-kitti")
IMG_ROOT = DATA_ROOT / "KITTI" / "training" / "image_02"
EXPR_ROOT = DATA_ROOT / "expression"
GT_TEMPLATE = BASE / "Refer-KITTI" / "gt_template"
NEURALSORT_ROOT = BASE / "NeuralSORT"

OUT_DIR = BASE / "diagnostics" / "results" / "label_audit"

SEQS = ["0005", "0011"]
FPS = 5
COLOR_GT = (0, 220, 0)
COLOR_CONTEXT = (160, 160, 160)
COLOR_TEXT = (255, 255, 255)
COLOR_TEXT_BG = (30, 30, 30)


def load_expression(path: Path) -> dict:
    """Parse expression JSON into sentence and per-frame GT track IDs."""
    with open(path) as f:
        d = json.load(f)
    gt_by_frame = {int(fid): [int(t) for t in tids] for fid, tids in d["label"].items()}
    return {"sentence": d["sentence"], "gt_by_frame": gt_by_frame}


def load_gt_template_mot(seq: str, slug: str) -> dict:
    path = GT_TEMPLATE / seq / slug / "gt.txt"
    out: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    if not path.is_file():
        return out
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            fid = int(float(parts[0]))
            tid = int(float(parts[1]))
            x, y, w, h = (float(p) for p in parts[2:6])
            out[fid].append((tid, x, y, w, h))
    return out


def load_neuralsort_context(seq: str) -> dict:
    out: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    id_offset = 0
    for sub in ("car", "pedestrian"):
        path = NEURALSORT_ROOT / seq / sub / "predict.txt"
        if not path.is_file():
            continue
        try:
            arr = np.loadtxt(path, delimiter=",")
        except Exception:
            continue
        if arr.ndim != 2 or len(arr) == 0:
            continue
        max_id_here = int(arr[:, 1].max())
        for row in arr:
            fid = int(row[0])
            tid = int(row[1]) + id_offset
            x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
            out[fid].append((tid, x, y, w, h))
        id_offset += max_id_here + 1
    return out


if __name__ == "__main__":
    print("Loaders only — run main() added in Task 4.")
```

- [ ] **Step 2: Smoke test loaders against real files**

Run:
```bash
python -c "
from run_gt_audit_video import load_expression, load_gt_template_mot, load_neuralsort_context, EXPR_ROOT
import os
seq = '0005'
slugs = sorted(os.listdir(EXPR_ROOT / seq))
slug = slugs[0].replace('.json','')
data = load_expression(EXPR_ROOT / seq / f'{slug}.json')
print('sentence:', data['sentence'])
print('n_frames_in_label:', len(data['gt_by_frame']))
gt = load_gt_template_mot(seq, slug)
print('gt_frames:', len(gt))
ctx = load_neuralsort_context(seq)
print('ctx_frames:', len(ctx))
assert len(data['gt_by_frame']) > 0, 'expression label empty'
assert len(ctx) > 0, 'neuralsort context empty'
print('PASS')
"
```
Expected: non-empty sentence, positive `n_frames_in_label`, positive `gt_frames` (gt_template may be 0 for some slugs if not regenerated — fallback handled in Task 3), positive `ctx_frames`, final `PASS`.

- [ ] **Step 3: Commit**

```bash
git add run_gt_audit_video.py
git commit -m "feat(audit): scaffold run_gt_audit_video with data loaders

Paths: refer-kitti V1 frames + expression JSONs, Refer-KITTI/gt_template
MOT files, NeuralSORT predicted tracks as context. Loaders for each.
Smoke-tested against seq 0005.
"
```

---

## Task 2: Bbox and text overlay helpers

**Files:**
- Modify: `run_gt_audit_video.py`

- [ ] **Step 1: Add `draw_box` and `draw_text_banner` above the `if __name__` block**

```python
def draw_box(img: np.ndarray, x: float, y: float, w: float, h: float,
             color: tuple, label: str | None = None, thickness: int = 2) -> None:
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def draw_text_banner(img: np.ndarray, text: str, pos: tuple,
                     font_scale: float = 0.6) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    x, y = pos
    cv2.rectangle(img, (x - 4, y - th - 6), (x + tw + 4, y + 4),
                  COLOR_TEXT_BG, -1)
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_TEXT, 1)
```

- [ ] **Step 2: Smoke test the overlays produce non-black pixels**

Run:
```bash
python -c "
import numpy as np
from run_gt_audit_video import draw_box, draw_text_banner, COLOR_GT
img = np.zeros((200, 400, 3), dtype=np.uint8)
draw_box(img, 50, 50, 100, 80, COLOR_GT, label='id 7', thickness=2)
draw_text_banner(img, 'hello', (10, 30))
assert img.sum() > 0, 'overlay drew nothing'
import cv2
cv2.imwrite('/tmp/audit_overlay_smoke.png', img)
print('PASS: overlay pixels written to /tmp/audit_overlay_smoke.png')
"
```
Expected: `PASS: ...` and `/tmp/audit_overlay_smoke.png` exists and is a visible image with a green box + white-on-dark text banner.

- [ ] **Step 3: Commit**

```bash
git add run_gt_audit_video.py
git commit -m "feat(audit): overlay helpers draw_box + draw_text_banner"
```

---

## Task 3: Per-expression video renderer

**Files:**
- Modify: `run_gt_audit_video.py`

- [ ] **Step 1: Add `render_expression_video` above the `if __name__` block**

```python
def render_expression_video(seq: str, slug: str, expr_data: dict) -> Path | None:
    sentence = expr_data["sentence"]
    gt_by_frame = expr_data["gt_by_frame"]
    gt_boxes = load_gt_template_mot(seq, slug)
    context = load_neuralsort_context(seq)

    frames = sorted(gt_by_frame.keys())
    if not frames:
        return None

    img_dir = IMG_ROOT / seq
    first_img = img_dir / f"{frames[0]:010d}.png"
    if not first_img.is_file():
        print(f"  SKIP {seq}/{slug}: missing {first_img}")
        return None
    first = cv2.imread(str(first_img))
    H, W = first.shape[:2]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{seq}_{slug}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (W, H))

    total = len(frames)
    frames_written = 0
    for i, fid in enumerate(frames):
        img_path = img_dir / f"{fid:010d}.png"
        if not img_path.is_file():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Context tracks (gray, thin)
        for _tid, x, y, w, h in context.get(fid, []):
            draw_box(img, x, y, w, h, COLOR_CONTEXT, thickness=1)

        # GT tracks for this expression (green, thick, id-labelled)
        for tid, x, y, w, h in gt_boxes.get(fid, []):
            draw_box(img, x, y, w, h, COLOR_GT, label=f"id {tid}", thickness=2)

        draw_text_banner(img, sentence[:120], (10, 30))
        draw_text_banner(img, f"frame {fid}  ({i+1}/{total})", (max(W - 260, 10), 30))
        draw_text_banner(img, f"seq {seq}", (10, H - 15))

        writer.write(img)
        frames_written += 1

    writer.release()
    if frames_written == 0:
        out_path.unlink(missing_ok=True)
        return None
    return out_path
```

- [ ] **Step 2: Smoke test by rendering one real motion expression from seq 0005**

Run:
```bash
python -c "
import os
from pathlib import Path
from run_gt_audit_video import (load_expression, render_expression_video,
                                 is_motion_expression, EXPR_ROOT)
seq = '0005'
chosen = None
for p in sorted((EXPR_ROOT / seq).glob('*.json')):
    data = load_expression(p)
    if is_motion_expression(data['sentence']) and data['gt_by_frame']:
        chosen = (p.stem, data)
        break
assert chosen is not None, 'no motion expression found on 0005'
slug, data = chosen
print(f'rendering {seq}/{slug}: {data[\"sentence\"]}')
out = render_expression_video(seq, slug, data)
assert out is not None and out.is_file(), f'no mp4 produced'
size = out.stat().st_size
print(f'PASS: {out} size={size}')
assert size > 10_000, f'suspiciously small mp4: {size} bytes'
"
```
Expected: prints the sentence being rendered, then `PASS: <path> size=<bytes>` with bytes in the hundreds of KB to low MB.

- [ ] **Step 3: Commit**

```bash
git add run_gt_audit_video.py
git commit -m "feat(audit): render_expression_video composes per-expr MP4

Green GT bboxes (from gt_template) over gray context (NeuralSORT),
with sentence + frame counter + seq overlays. 5 fps mp4v."
```

---

## Task 4: Main loop + review.csv + end-to-end run

**Files:**
- Modify: `run_gt_audit_video.py`

- [ ] **Step 1: Add `write_review_csv` and `main`, replace the stub `if __name__` block**

```python
def write_review_csv(clips: list[tuple[str, str, str]]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "review.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq", "expr_slug", "sentence", "verdict", "note"])
        for seq, slug, sentence in clips:
            w.writerow([seq, slug, sentence, "", ""])
    return csv_path


def main() -> None:
    clips: list[tuple[str, str, str]] = []
    for seq in SEQS:
        expr_dir = EXPR_ROOT / seq
        if not expr_dir.is_dir():
            print(f"SKIP seq {seq}: {expr_dir} missing")
            continue
        jsons = sorted(expr_dir.glob("*.json"))
        n_motion = 0
        for expr_json in jsons:
            data = load_expression(expr_json)
            if not is_motion_expression(data["sentence"]):
                continue
            n_motion += 1
            slug = expr_json.stem
            print(f"[{seq}] {slug}: {data['sentence']}")
            out = render_expression_video(seq, slug, data)
            if out is not None:
                clips.append((seq, slug, data["sentence"]))
        print(f"  seq {seq}: {n_motion} motion expressions, "
              f"{len([c for c in clips if c[0]==seq])} rendered")

    csv_path = write_review_csv(clips)
    print(f"\nDone. {len(clips)} clips in {OUT_DIR}")
    print(f"Review template: {csv_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run end-to-end on seqs 0005 + 0011**

Run:
```bash
python run_gt_audit_video.py
```
Expected: prints per-expression sentence + summary per seq, ends with `Done. N clips in ...` and a `review.csv` path. Should take 2–10 minutes depending on frame counts. `N` expected between 20 and 80.

- [ ] **Step 3: Verify artifacts**

Run:
```bash
python -c "
from pathlib import Path
import csv
out = Path('diagnostics/results/label_audit')
mp4s = sorted(out.glob('*.mp4'))
csv_path = out / 'review.csv'
print(f'mp4 count: {len(mp4s)}')
assert len(mp4s) > 0, 'no mp4 produced'
assert csv_path.is_file(), 'review.csv missing'
rows = list(csv.reader(open(csv_path)))
print(f'csv rows (incl header): {len(rows)}')
assert len(rows) == len(mp4s) + 1, f'csv rows {len(rows)-1} != mp4s {len(mp4s)}'
for p in mp4s[:3]:
    s = p.stat().st_size
    assert s > 10_000, f'{p} suspiciously small: {s} bytes'
print('PASS: artifact set consistent')
"
```
Expected: `mp4 count: N`, `csv rows (incl header): N+1`, `PASS: artifact set consistent`.

- [ ] **Step 4: Commit script + outputs**

```bash
git add run_gt_audit_video.py diagnostics/results/label_audit/review.csv
git commit -m "feat(audit): end-to-end GT audit video renderer

Iterates seqs {0005, 0011}, filters motion-verb expressions via
MOTION_KEYWORDS, renders per-expression MP4 into
diagnostics/results/label_audit/, writes review.csv template.

Run: python run_gt_audit_video.py
"
```

Do **not** commit `*.mp4` outputs — they are large and regenerable. Confirm `.gitignore` covers `diagnostics/results/label_audit/*.mp4` (the existing `diagnostics/results/` tree is already large-artifact territory; add a rule if not).

- [ ] **Step 5: Check .gitignore and add mp4 ignore if missing**

Run:
```bash
grep -q 'label_audit/.*mp4\|label_audit/\*\.mp4\|diagnostics/results/label_audit' .gitignore || {
    printf '\ndiagnostics/results/label_audit/*.mp4\n' >> .gitignore
    git add .gitignore
    git commit -m "chore(gitignore): ignore audit mp4 outputs"
}
```
Expected: either silent (already covered) or one additional commit ignoring mp4s.

---

## Self-review

**1. Spec coverage:**
- Seqs {0005, 0011}, motion-verb filter → Task 4 `SEQS` constant + `is_motion_expression` call ✓
- Skip 0013 (n=2 data-thin) → `SEQS` excludes 0013 ✓
- Output path `diagnostics/results/label_audit/<seq>_<expr_slug>.mp4` → Task 3 `out_path` ✓
- Green GT + gray context overlays → Task 3 `draw_box` calls with `COLOR_GT` / `COLOR_CONTEXT` ✓
- Sentence top-left, frame top-right, seq bottom-left → Task 3 three `draw_text_banner` calls ✓
- 5 fps playback → `FPS = 5` constant used in `cv2.VideoWriter` ✓
- review.csv with `seq,expr_slug,sentence,verdict,note` → Task 4 `write_review_csv` ✓
- Decision rule / noise_rate / ambiguous handling → out of plan scope (user activity, not code)
- Shuffled order for confirmation-bias mitigation → spec lists this under *risks* as a nice-to-have; plan currently sorts alphabetically to keep reruns deterministic. **Acceptable departure** — user can shuffle the review.csv at review time. Flag for user in handoff.

**2. Placeholder scan:** No TBD / TODO / "similar to" / vague error-handling phrasing. All code blocks present. Commands explicit.

**3. Type consistency:** `load_expression` returns `{sentence, gt_by_frame}` — used consistently by `render_expression_video`. Track tuple shape `(tid, x, y, w, h)` stable across `load_gt_template_mot`, `load_neuralsort_context`, and `draw_box` call sites. `Path` used consistently (not mixed with `str`). CSV column order fixed: `seq, expr_slug, sentence, verdict, note`.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-20-label-audit-gt-video.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, spec + quality review between tasks. Stricter but slower.
2. **Inline Execution** — executing-plans skill, batch with checkpoints in this session. Faster, given auto-mode and user's speed preference.

Default choice given user context: **Inline Execution**. Ask once before proceeding.
