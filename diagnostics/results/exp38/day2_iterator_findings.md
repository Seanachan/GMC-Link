# Exp 38 Day 2 — FlexHook Iterator Reverse-Engineering

Date: 2026-04-22
Scope: Locate the ego-speed injection point in FlexHook's data iterator + model.

## Files reviewed

- `/home/seanachan/FlexHook/data/build.py` (95 lines)
- `/home/seanachan/FlexHook/data/mydataloader.py` (634 lines)
- `/home/seanachan/FlexHook/models/mymodel.py` L270-330 (speed block)

## Data flow (test / eval mode)

```
Temp-NeuralSORT predict.txt  (MOT: frame, tid, x, y, w, h, conf, ...)
     ↓  data/mydataloader.py L239-281  _parse_data
     ↓  frame_id = int(frame[0]) - 1   (TempRMOT 1-frame shift fix)
     ↓  obj_ids_car first → obj_key = f'{video}_{int(obj)}'
     ↓  obj_ids_ped  → obj_key = f'{video}_{int(obj+max(obj_ids_car))}'  ← ID shift
data[obj_key]['bbox']      list of [frame_id, x1, y1, x2, y2]  (pixel coords)
data[obj_key]['bbox_norm'] list of [frame_id, x1/W, y1/H, x2/W, y2/H]
     ↓  __getitem__ L436-615
     ↓  images: T×3×H×W  (PIL → transform)
     ↓  global_pe: T×2×16×48  (bbox grid in normgrid coords `*2-1`, for grid_sample)
     ↓  bbox_gt: T×4  (cxcywh normalized)
     ↓  returned: images, global_pe, bbox_gt, sampled_target_exp, ids, mask,
     ↓             video, str(obj), str(frame_id), last
     ↓  frame_id = data['bbox'][sampled_indices[-1]][0]   (anchor frame)
```

## Where raw speed is computed — `mymodel.py:288`

```python
# cur_pos_raw: (B, T, 2, 16, 48) positions from grid_sample onto the feature map
speed = cur_pos_raw[:,1:] - cur_pos_raw[:,:-1]          # (B, T-1, 2, 16, 48)
speed = torch.cat([speed, cur_pos_raw[:,-1:]], dim=1)   # (B, T, 2, 16, 48)  (last-frame padding)
# ...
obj_f = torch.cat([obj_f, speed], dim=1)                # concat into object feature
```

This is **raw bbox-centre displacement in normgrid coords across sampled T frames** — no ego compensation, no multi-scale, no residual subtraction. Our attack vector.

## Injection strategies

| Strategy | Scope | Risk | Notes |
|---|---|---|---|
| **A. Dataloader preprocess** | Add ego_speed tensor to `__getitem__` return | ↑ files (2) | Cleanest: emit ego_speed per sampled frame alongside global_pe. Model reads replacement tensor instead of computing. |
| **B. Model-side cache lookup** | Swap line 288 with `cache.lookup(video, frame, tid, cls)` | 1 file | Needs a singleton cache object with video/frame/id → 13D lookup. Strongly coupled to `video`/`obj`/`sampled_indices` which must be forwarded to model. |

**Chosen: Strategy A.** Dataloader returns an extra tensor `ego_speed_normgrid: T×2`, piped through to `mymodel.py` where we replace the subtraction with a broadcast to the H×W grid.

## ID mapping (must-fix)

Our cache is keyed by `(seq, frame, original_track_id, class)`. FlexHook merges into shifted namespace:

```
car tids: [1,2,3,...,max_c]
ped tids: [1,2,3,...,max_p] → shifted by max_c → [max_c+1,...,max_c+max_p]
obj_key:  f'{video}_{int(obj)}'   # post-shift
```

Injection-time reverse map:
```python
if shifted_id <= max_car_id:
    class = 'car'; original_tid = shifted_id
else:
    class = 'pedestrian'; original_tid = shifted_id - max_car_id
```

`max_car_id` is available from the `obj_ids_car` list computed in `_parse_data`.

## Frame indexing reconciliation

- predict.txt frame: 1-indexed
- FlexHook internal `data['bbox'][idx][0]`: `predict_frame - 1` (0-indexed, offset for TempRMOT)
- Image file: `{data_frame:06d}.png` (0-indexed)
- Our cache `frames` field: 1-indexed (matches raw predict.txt)

→ Injection-time cache lookup must add 1 to FlexHook's internal frame_id, OR we rewrite the cache to 0-indexed. Pick latter: keep cache FlexHook-native by storing `frame_id - 1`, simpler at lookup time.

**Action item:** Add a `--fh_frame_offset` flag to extractor (default 0) and regenerate with `=-1` so cache is FlexHook-native 0-indexed. Or: handle the +1 at lookup time. Decision: lookup-time handling — keeps extractor output schema stable.

## Ego-speed shape at injection

- Our 13D cache's cols 0..5 = residual dx/dy at gaps {2, 5, 10} in VELOCITY_SCALE=100 units.
- Per spec §3.2: `speed_normgrid = residual * 0.02`.
- `mymodel.py` expects `speed: (B, T, 2, 16, 48)` — spatial broadcast of per-object scalar across the 16×48 grid.
- **Open question Q1 from spec:** which gap(s) to map to which `pos_avg_pool[i]` (i=0,1,2, spatial scales 28×84 / 14×42 / 7×21). Heuristic: fine→short, coarse→long. No theoretical grounding; revisit after A1 (zero ablation) results.

For 38-A (first iteration), simplest plan: inject **gap=5 mid-scale residual** (cols 2,3) as the replacement for all spatial scales. Keep the ablation grid for gap-to-scale mapping.

## Null-test plan (task #127)

1. Fork FlexHook to `~/FlexHook-ego/` via `cp -rL`.
2. Add env-gated branch at `mymodel.py:288`:
   ```python
   if os.environ.get("FLEXHOOK_EGO", "off") == "off":
       speed = cur_pos_raw[:,1:] - cur_pos_raw[:,:-1]  # original
   else:
       speed = load_ego_cache(video, frame_id, obj_id, ...)  # new
   ```
3. Run infer.sh on V1 with `FLEXHOOK_EGO=off` — must yield HOTA=53.824±0.01.

## Day 2 eve — 38-A launch (task #128)

- Init: `refer-kitti-best.pth` checkpoint.
- Epochs: 10 (per realistic timeline §spec).
- Config: `FLEXHOOK_EGO=on_13d`, gap=5 injection, lookup from
  `~/GMC-Link/diagnostics/results/exp38/cache/ego_speed/`.
- Eval: V1 held-out seqs {0005, 0011, 0013}.
- Decision: ≥+1.0 HOTA = POSITIVE → publish. ≤−0.3 = NEG → pivot.
