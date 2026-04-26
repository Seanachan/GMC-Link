"""Per-bbox Object-Motion-Field (OMF) pooling statistics.

Given a dense optical-flow-like field ``omf_field`` of shape (H, W, 2) and a
bbox (x, y, w, h) in pixel coordinates, return the 5-number pooled summary
``[mean_dx, mean_dy, std_dx, std_dy, max_mag]``. The caller concatenates these
across ``FRAME_GAPS`` (= (2, 5, 10)) to produce the 15-dim OMF feature block
used in Stage B of Exp 37 (see
``docs/superpowers/specs/2026-04-22-ego-motion-systematic-design.md`` §3).

Out-of-frame and degenerate bboxes return zeros — a conservative default that
keeps InfoNCE well-behaved when crops are unavailable (e.g., track fragments
near frame edges).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def per_bbox_omf_stats(omf_field: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    if omf_field.ndim != 3 or omf_field.shape[2] != 2:
        raise ValueError(
            f"omf_field must have shape (H, W, 2); got {omf_field.shape}"
        )
    H, W, _ = omf_field.shape
    x, y, w, h = bbox
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(W, int(x + w))
    y1 = min(H, int(y + h))
    if x1 <= x0 or y1 <= y0:
        return np.zeros(5, dtype=np.float32)

    patch = omf_field[y0:y1, x0:x1, :]
    dx = patch[..., 0]
    dy = patch[..., 1]
    mag = np.sqrt(dx * dx + dy * dy)
    return np.array(
        [
            float(dx.mean()),
            float(dy.mean()),
            float(dx.std()),
            float(dy.std()),
            float(mag.max()),
        ],
        dtype=np.float32,
    )
