# FlexHook-Adjacent Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a joint cross-attention decoder (Exp 35 / Approach A) that consumes CLIP text tokens, CLIP ViT patch tokens pooled under each track's bbox neighborhood, and 3 ego-compensated motion tokens from a tiny temporal transformer; train it with the existing InfoNCE+FNM signal on Refer-KITTI V1; evaluate via the existing multi-seq Layer-3 micro AUC pipeline.

**Architecture:** Frozen HuggingFace CLIP ViT-B/16 (both text and vision towers) + three new trainable modules: `TemporalMotionEncoder` (13D×T=10 window → 3 scale tokens), `ClipTextEncoder` wrapper (frozen CLIP + 512→256 projector), `ClipPatchExtractor` wrapper (frozen CLIP + bbox-pool + 768→256 projector). These feed a 2-layer cross-attention decoder (`FlexHookAdjacentAligner`) that outputs a single 256D motion-side embedding per (track, expression). The language side is mean-pooled projected CLIP text tokens. Cosine similarity → InfoNCE loss. Everything trained in one stage, seed 42.

**Tech Stack:** PyTorch 2.6, `transformers` 4.57 (CLIPModel `openai/clip-vit-base-patch16`), pytest 8.4. Existing GMC-Link code paths (core.py, manager.py, utils.py, losses.py, fusion_head.py) unchanged.

**Spec:** `docs/superpowers/specs/2026-04-19-flexhook-adjacent-alignment-design.md` (commit `f0425b9`).

---

## Global Conventions

- Python interpreter: `~/miniconda/envs/RMOT/bin/python`. Tests: `~/miniconda/envs/RMOT/bin/python -m pytest <path> -v`.
- All paths are relative to repo root `/home/seanachan/GMC-Link/` unless stated otherwise.
- Seed for the headline run: `42`. Every training invocation in this plan must set `--seed 42` unless explicitly overridden.
- Commit style: follow existing `feat(exp35): ...` / `test(exp35): ...` / `fix(exp35): ...` with Co-Authored-By trailer.
- Work on branch `exp/flexhook-adjacent`. If you are implementing via subagent-driven-development, a worktree is recommended (see superpowers:using-git-worktrees).

---

## Task 1: Branch setup, gitignore, and CLIP probe

**Files:**
- Modify: `.gitignore`
- Create: `tests/test_clip_env_smoke.py`

- [ ] **Step 1: Create branch off current HEAD**

```bash
cd /home/seanachan/GMC-Link
git checkout -b exp/flexhook-adjacent
git status
```

Expected: "On branch exp/flexhook-adjacent".

- [ ] **Step 2: Extend .gitignore for Exp 35 artifacts**

Append to `.gitignore`:

```
# Exp 35 / FlexHook-adjacent
gmc_link_weights_v1train_flexhook_adj*.pth
gmc_link_weights_v1train_flexhook_a[123]*.pth
diagnostics/cache/vit_patches_*.npz
diagnostics/results/multiseq/layer3_*_v1train_flexhook_adj.*
diagnostics/results/multiseq/layer3_*_v1train_a[123]_*.*
diagnostics/results/multiseq/layer3_multiseq_v1train_flexhook_adj.*
diagnostics/results/multiseq/layer3_multiseq_v1train_a[123]_*.*
```

- [ ] **Step 3: Write env smoke test**

Create `tests/test_clip_env_smoke.py`:

```python
"""Smoke test that the CLIP model and processor load and produce expected shapes.

Runs once at the start of Exp 35 implementation to confirm the environment
has everything the plan assumes. Not part of regular CI — gated on an env
flag to keep CI light.
"""
import os
import pytest
import torch


@pytest.mark.skipif(os.environ.get("RUN_CLIP_SMOKE") != "1",
                    reason="CLIP smoke test is opt-in via RUN_CLIP_SMOKE=1")
def test_clip_vit_b16_loads_with_expected_dims():
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    assert model.vision_model.config.hidden_size == 768
    assert model.text_model.config.hidden_size == 512
    assert model.vision_model.config.image_size == 224
    assert model.vision_model.config.patch_size == 16

    dummy = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        vout = model.vision_model(pixel_values=dummy).last_hidden_state
    # 1 CLS + 14*14 = 197 tokens, 768 hidden
    assert vout.shape == (1, 197, 768), f"got {vout.shape}"
```

- [ ] **Step 4: Run the smoke test**

```bash
cd /home/seanachan/GMC-Link
RUN_CLIP_SMOKE=1 ~/miniconda/envs/RMOT/bin/python -m pytest tests/test_clip_env_smoke.py -v
```

Expected: PASS. If the CLIP weights aren't cached locally, first run downloads them (~600 MB to `~/.cache/huggingface/hub/`); subsequent runs are offline.

- [ ] **Step 5: Commit**

```bash
git add .gitignore tests/test_clip_env_smoke.py
git commit -m "$(cat <<'EOF'
chore(exp35): branch setup, gitignore Exp 35 artifacts, CLIP env smoke

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `TemporalMotionEncoder` module

**Files:**
- Create: `gmc_link/motion_encoder.py`
- Create: `tests/test_motion_encoder.py`

- [ ] **Step 1: Write the failing shape + mask tests**

Create `tests/test_motion_encoder.py`:

```python
import pytest
import torch

from gmc_link.motion_encoder import TemporalMotionEncoder


def test_output_shape_default():
    enc = TemporalMotionEncoder()
    x = torch.randn(4, 10, 13)
    out = enc(x)
    assert out.shape == (4, 3, 256), f"expected (4,3,256), got {out.shape}"


def test_output_shape_variable_T():
    enc = TemporalMotionEncoder(T_max=10)
    x = torch.randn(2, 5, 13)  # shorter window
    out = enc(x)
    assert out.shape == (2, 3, 256)


def test_mask_changes_output():
    torch.manual_seed(0)
    enc = TemporalMotionEncoder().eval()
    x = torch.randn(1, 10, 13)
    mask_all = torch.ones(1, 10, dtype=torch.bool)
    mask_half = torch.cat([torch.ones(1, 5, dtype=torch.bool),
                           torch.zeros(1, 5, dtype=torch.bool)], dim=1)
    y_full = enc(x, mask=mask_all)
    y_half = enc(x, mask=mask_half)
    assert not torch.allclose(y_full, y_half), \
        "masking half the frames should change the pooled output"


def test_gradient_flows_to_input_proj():
    enc = TemporalMotionEncoder()
    x = torch.randn(2, 10, 13, requires_grad=False)
    out = enc(x)
    out.sum().backward()
    assert enc.input_proj.weight.grad is not None
    assert enc.input_proj.weight.grad.abs().sum() > 0


def test_param_count_under_500k():
    enc = TemporalMotionEncoder()
    n = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    assert n < 500_000, f"motion encoder too large: {n} params"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/seanachan/GMC-Link
~/miniconda/envs/RMOT/bin/python -m pytest tests/test_motion_encoder.py -v
```

Expected: ImportError / ModuleNotFoundError for `gmc_link.motion_encoder`.

- [ ] **Step 3: Implement `TemporalMotionEncoder`**

Create `gmc_link/motion_encoder.py`:

```python
"""Tiny temporal transformer that ingests a motion window and emits 3 scale-tokens.

Input: (B, T, 13) multi-scale residual velocity + bbox geometry + SNR, as produced
by GMCLinkManager and gmc_link.dataset for Exp 35.

Output: (B, 3, 256) — one token per temporal scale (short / mid / long, aligned to
FRAME_GAPS = [2, 5, 10]). Consumed by FlexHookAdjacentAligner.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalMotionEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 13,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        T_max: int = 10,
        n_scales: int = 3,
        d_out: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.T_max = T_max
        self.n_scales = n_scales

        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(T_max, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.scale_queries = nn.Parameter(torch.randn(n_scales, d_model) * 0.02)
        self.scale_pool = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout,
        )

        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_out),
            nn.LayerNorm(d_out),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x:    (B, T, 13)   motion window
        mask: (B, T)       True = valid frame, False = padded. None => all valid.
        Returns: (B, 3, 256)
        """
        B, T, _ = x.shape
        assert T <= self.T_max, f"T={T} exceeds T_max={self.T_max}"

        h = self.input_proj(x) + self.pos_embed[:T].unsqueeze(0)  # (B, T, d)
        key_padding_mask = None if mask is None else ~mask  # (B, T), True = ignore
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)  # (B, T, d)

        q = self.scale_queries.unsqueeze(0).expand(B, -1, -1)  # (B, 3, d)
        tokens, _ = self.scale_pool(
            q, h, h, key_padding_mask=key_padding_mask, need_weights=False,
        )  # (B, 3, d)
        return self.out_proj(tokens)  # (B, 3, d_out)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/seanachan/GMC-Link
~/miniconda/envs/RMOT/bin/python -m pytest tests/test_motion_encoder.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add gmc_link/motion_encoder.py tests/test_motion_encoder.py
git commit -m "$(cat <<'EOF'
feat(exp35): TemporalMotionEncoder — 13D window → 3 scale tokens

Tiny temporal transformer (2 layers, d=128, 4 heads) over a multi-scale
residual-velocity window with three scale-specific pool queries. Output
(B, 3, 256) feeds the FlexHook-adjacent joint decoder.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `ClipTextEncoder` wrapper

**Files:**
- Create: `gmc_link/text_encoder_clip.py`
- Create: `tests/test_clip_text_encoder.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_clip_text_encoder.py`:

```python
import os
import pytest
import torch


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_CLIP_SMOKE") != "1",
    reason="CLIP wrapper tests require loading the CLIP model (gated by RUN_CLIP_SMOKE=1)",
)


def test_output_shape_and_mask():
    from gmc_link.text_encoder_clip import ClipTextEncoder
    enc = ClipTextEncoder(d_out=256).eval()
    out = enc(["moving cars", "left vehicles which are parking"])
    assert "tokens" in out and "mask" in out
    assert out["tokens"].ndim == 3
    assert out["tokens"].shape[0] == 2
    assert out["tokens"].shape[2] == 256
    # Mask: True for real tokens, False for padding
    assert out["mask"].dtype == torch.bool
    assert out["mask"].shape == out["tokens"].shape[:2]
    # First prompt has fewer real tokens than the second (after tokenization)
    assert out["mask"][0].sum() < out["mask"][1].sum()


def test_clip_params_frozen():
    from gmc_link.text_encoder_clip import ClipTextEncoder
    enc = ClipTextEncoder(d_out=256)
    trainable = [n for n, p in enc.named_parameters() if p.requires_grad]
    # Only the projector should be trainable.
    for n in trainable:
        assert n.startswith("projector."), f"unexpected trainable param: {n}"
    # And there must be at least one trainable param.
    assert len(trainable) > 0


def test_projector_grad_flows():
    from gmc_link.text_encoder_clip import ClipTextEncoder
    enc = ClipTextEncoder(d_out=256)
    out = enc(["moving cars"])
    out["tokens"].sum().backward()
    found_grad = False
    for n, p in enc.named_parameters():
        if n.startswith("projector.") and p.grad is not None and p.grad.abs().sum() > 0:
            found_grad = True
    assert found_grad
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/seanachan/GMC-Link
RUN_CLIP_SMOKE=1 ~/miniconda/envs/RMOT/bin/python -m pytest tests/test_clip_text_encoder.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `ClipTextEncoder`**

Create `gmc_link/text_encoder_clip.py`:

```python
"""Frozen CLIP text tower + trainable Linear→LayerNorm projector.

Returns all token hidden states (not just [EOS]), with a key-padding mask so the
joint decoder can attend over variable-length text. Projector output dim matches
the decoder's d_model (default 256).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizerFast


CLIP_NAME = "openai/clip-vit-base-patch16"


class ClipTextEncoder(nn.Module):
    def __init__(
        self,
        d_out: int = 256,
        max_length: int = 32,
        clip_name: str = CLIP_NAME,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.tokenizer = CLIPTokenizerFast.from_pretrained(clip_name)
        self.clip_text = CLIPTextModel.from_pretrained(clip_name)
        for p in self.clip_text.parameters():
            p.requires_grad_(False)
        self.clip_text.eval()

        d_hidden = self.clip_text.config.hidden_size  # 512 for ViT-B/16
        self.projector = nn.Sequential(
            nn.Linear(d_hidden, d_out),
            nn.LayerNorm(d_out),
        )

    def train(self, mode: bool = True):  # keep CLIP frozen even in .train()
        super().train(mode)
        self.clip_text.eval()
        return self

    @torch.no_grad()
    def _encode_frozen(self, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.projector.parameters()).device
        enc = self.tokenizer(
            list(texts),
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)
        out = self.clip_text(**enc)  # last_hidden_state: (B, L, 512)
        return out.last_hidden_state, enc.attention_mask.bool()

    def forward(self, texts: Sequence[str]) -> dict:
        hidden, mask = self._encode_frozen(texts)
        tokens = self.projector(hidden)  # (B, L, d_out)
        return {"tokens": tokens, "mask": mask}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/seanachan/GMC-Link
RUN_CLIP_SMOKE=1 ~/miniconda/envs/RMOT/bin/python -m pytest tests/test_clip_text_encoder.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add gmc_link/text_encoder_clip.py tests/test_clip_text_encoder.py
git commit -m "$(cat <<'EOF'
feat(exp35): ClipTextEncoder — frozen CLIP text + trainable projector

Returns all token hidden states with a padding mask, projected to d_out=256
via Linear→LayerNorm. CLIP text params frozen, .eval() pinned.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `ClipPatchExtractor` wrapper (bbox-pooled frozen ViT)

**Files:**
- Create: `gmc_link/visual_encoder.py`
- Create: `tests/test_clip_visual_encoder.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_clip_visual_encoder.py`:

```python
import os
import pytest
import torch


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_CLIP_SMOKE") != "1",
    reason="CLIP wrapper tests require loading the CLIP model (gated by RUN_CLIP_SMOKE=1)",
)


def test_forward_patches_shape():
    from gmc_link.visual_encoder import ClipPatchExtractor
    enc = ClipPatchExtractor(d_out=256).eval()
    # Two dummy 224x224 RGB images (already normalized or not — encoder applies its own normalization)
    images = torch.rand(2, 3, 224, 224)
    patches = enc.forward_patches(images)  # (B, 196, 768)
    assert patches.shape == (2, 196, 768)


def test_pool_under_bbox_selects_expected_count():
    from gmc_link.visual_encoder import ClipPatchExtractor
    enc = ClipPatchExtractor(d_out=256).eval()
    patches = torch.randn(1, 196, 768)  # stand-in for CLIP output (pre-projection)
    # bbox covers the top-left 48x48 pixel region on a 224x224 image
    # → 3x3 = 9 patches before expansion; with 1.5x expansion → ~16 patches (clipped)
    bbox = torch.tensor([[0.0, 0.0, 48.0, 48.0]])  # (1, 4) xyxy
    image_hw = torch.tensor([[224, 224]])
    pooled, mask = enc.pool_under_bbox(patches, bbox, image_hw, expand=1.5)
    assert pooled.shape[0] == 1
    n_selected = int(mask.sum())
    assert 4 <= n_selected <= 25, f"expected 4..25 selected patches, got {n_selected}"


def test_bbox_off_frame_falls_back_to_center_9():
    from gmc_link.visual_encoder import ClipPatchExtractor
    enc = ClipPatchExtractor(d_out=256).eval()
    patches = torch.randn(1, 196, 768)
    bbox = torch.tensor([[-50.0, -50.0, -20.0, -20.0]])  # wholly off-frame
    image_hw = torch.tensor([[224, 224]])
    pooled, mask = enc.pool_under_bbox(patches, bbox, image_hw, expand=1.5)
    assert int(mask.sum()) == 9, "off-frame bbox must fall back to 9 center patches"


def test_forward_returns_projected_tokens_and_mask():
    from gmc_link.visual_encoder import ClipPatchExtractor
    enc = ClipPatchExtractor(d_out=256).eval()
    images = torch.rand(2, 3, 224, 224)
    bboxes = torch.tensor([[30.0, 30.0, 100.0, 100.0],
                           [0.0, 0.0, 224.0, 224.0]])
    image_hw = torch.tensor([[224, 224], [224, 224]])
    out = enc(images, bboxes, image_hw)
    assert out["tokens"].shape[0] == 2
    assert out["tokens"].shape[2] == 256
    assert out["mask"].dtype == torch.bool
    assert out["tokens"].shape[:2] == out["mask"].shape


def test_clip_params_frozen():
    from gmc_link.visual_encoder import ClipPatchExtractor
    enc = ClipPatchExtractor(d_out=256)
    trainable = [n for n, p in enc.named_parameters() if p.requires_grad]
    for n in trainable:
        assert n.startswith("projector."), f"unexpected trainable param: {n}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/seanachan/GMC-Link
RUN_CLIP_SMOKE=1 ~/miniconda/envs/RMOT/bin/python -m pytest tests/test_clip_visual_encoder.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `ClipPatchExtractor`**

Create `gmc_link/visual_encoder.py`:

```python
"""Frozen CLIP ViT-B/16 vision tower with bbox-pooled patch selection.

The ViT processes a 224x224 image into a (197, 768) token sequence: 1 CLS + 14x14
patches. We drop CLS and keep the 196 patch tokens. For each (image, bbox), we
expand the bbox by a factor (default 1.5x), clamp to image bounds, and select the
subset of patches whose 16x16 grid cell overlaps the expanded bbox. If the bbox
is entirely off-frame, we fall back to the 9 center patches.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import CLIPVisionModel


CLIP_NAME = "openai/clip-vit-base-patch16"
PATCH = 16
GRID = 14  # 224 / 16
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


class ClipPatchExtractor(nn.Module):
    def __init__(
        self,
        d_out: int = 256,
        clip_name: str = CLIP_NAME,
    ) -> None:
        super().__init__()
        self.clip_vision = CLIPVisionModel.from_pretrained(clip_name)
        for p in self.clip_vision.parameters():
            p.requires_grad_(False)
        self.clip_vision.eval()

        d_hidden = self.clip_vision.config.hidden_size  # 768
        self.projector = nn.Sequential(
            nn.Linear(d_hidden, d_out),
            nn.LayerNorm(d_out),
        )
        self.register_buffer("clip_mean", CLIP_MEAN, persistent=False)
        self.register_buffer("clip_std", CLIP_STD, persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip_vision.eval()
        return self

    @torch.no_grad()
    def forward_patches(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, 224, 224) in [0, 1]. Returns (B, 196, 768) patch tokens."""
        x = (images - self.clip_mean) / self.clip_std
        out = self.clip_vision(pixel_values=x).last_hidden_state  # (B, 197, 768)
        return out[:, 1:, :].contiguous()  # drop CLS → (B, 196, 768)

    @staticmethod
    def pool_under_bbox(
        patches: torch.Tensor,
        bbox_xyxy: torch.Tensor,
        image_hw: torch.Tensor,
        expand: float = 1.5,
        max_select: int = 40,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        patches:   (B, 196, D)
        bbox_xyxy: (B, 4) in pixel coords of the ORIGINAL image (pre-resize)
        image_hw:  (B, 2)  original image (H, W)
        Returns: (B, N_max, D) padded, (B, N_max) bool mask (True = valid).
        """
        B, _, D = patches.shape
        device = patches.device
        # Map bbox to the 224x224 resized coordinate system, then to patch grid.
        H = image_hw[:, 0].to(device).float()
        W = image_hw[:, 1].to(device).float()

        bx = bbox_xyxy.to(device).float()
        cx = (bx[:, 0] + bx[:, 2]) * 0.5
        cy = (bx[:, 1] + bx[:, 3]) * 0.5
        w = (bx[:, 2] - bx[:, 0]) * expand
        h = (bx[:, 3] - bx[:, 1]) * expand
        x0 = (cx - w * 0.5) / W * 224.0
        y0 = (cy - h * 0.5) / H * 224.0
        x1 = (cx + w * 0.5) / W * 224.0
        y1 = (cy + h * 0.5) / H * 224.0

        # Clamp to image bounds.
        x0c = x0.clamp(0, 224); x1c = x1.clamp(0, 224)
        y0c = y0.clamp(0, 224); y1c = y1.clamp(0, 224)

        # Select indices of patches that overlap the bbox.
        selected_rows, pooled = [], []
        max_n = 0
        for b in range(B):
            if x1c[b] <= x0c[b] or y1c[b] <= y0c[b]:
                # Off-frame fallback: center 3x3 = 9 patches
                center = GRID // 2
                idx = torch.tensor(
                    [r * GRID + c for r in range(center - 1, center + 2)
                                  for c in range(center - 1, center + 2)],
                    device=device, dtype=torch.long,
                )
            else:
                pc0 = int((x0c[b] / PATCH).floor().item())
                pr0 = int((y0c[b] / PATCH).floor().item())
                pc1 = int((x1c[b] / PATCH).ceil().item())
                pr1 = int((y1c[b] / PATCH).ceil().item())
                pc0 = max(0, pc0); pr0 = max(0, pr0)
                pc1 = min(GRID, pc1); pr1 = min(GRID, pr1)
                idx = torch.tensor(
                    [r * GRID + c for r in range(pr0, pr1)
                                  for c in range(pc0, pc1)],
                    device=device, dtype=torch.long,
                )
            if idx.numel() > max_select:
                idx = idx[:max_select]
            selected_rows.append(idx)
            pooled.append(patches[b, idx, :])  # (n_b, D)
            max_n = max(max_n, idx.numel())

        # Right-pad to (B, max_n, D).
        out = patches.new_zeros(B, max_n, D)
        mask = torch.zeros(B, max_n, dtype=torch.bool, device=device)
        for b, t in enumerate(pooled):
            n = t.shape[0]
            out[b, :n, :] = t
            mask[b, :n] = True
        return out, mask

    def forward(
        self,
        images: torch.Tensor,
        bboxes: torch.Tensor,
        image_hw: torch.Tensor,
        expand: float = 1.5,
    ) -> dict:
        patches = self.forward_patches(images)  # (B, 196, 768)
        pooled, mask = self.pool_under_bbox(patches, bboxes, image_hw, expand=expand)
        tokens = self.projector(pooled)  # (B, N, d_out)
        return {"tokens": tokens, "mask": mask}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/seanachan/GMC-Link
RUN_CLIP_SMOKE=1 ~/miniconda/envs/RMOT/bin/python -m pytest tests/test_clip_visual_encoder.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add gmc_link/visual_encoder.py tests/test_clip_visual_encoder.py
git commit -m "$(cat <<'EOF'
feat(exp35): ClipPatchExtractor — frozen ViT + bbox-pooled patch selection

Drops CLS, selects patches whose 16x16 grid cell overlaps a 1.5x-expanded bbox,
falls back to the 9 center patches when bbox is off-frame. Output projected to
d_out=256. CLIP vision params frozen, .eval() pinned.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `FlexHookAdjacentAligner` joint decoder

**Files:**
- Create: `gmc_link/aligner_v2.py`
- Create: `tests/test_aligner_v2.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_aligner_v2.py`:

```python
import os
import pytest
import torch


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_CLIP_SMOKE") != "1",
    reason="Aligner tests require loading CLIP (gated by RUN_CLIP_SMOKE=1)",
)


def _make_aligner():
    from gmc_link.aligner_v2 import FlexHookAdjacentAligner
    return FlexHookAdjacentAligner(d_model=256).eval()


def test_forward_returns_embeddings():
    aligner = _make_aligner()
    motion = torch.randn(2, 10, 13)
    images = torch.rand(2, 3, 224, 224)
    bboxes = torch.tensor([[10.0, 10.0, 80.0, 80.0],
                           [50.0, 50.0, 200.0, 200.0]])
    image_hw = torch.tensor([[224, 224], [224, 224]])
    texts = ["moving cars", "parking vehicles"]
    out = aligner(motion, images, bboxes, image_hw, texts)
    assert out["motion_emb"].shape == (2, 256)
    assert out["lang_emb"].shape == (2, 256)


def test_score_in_unit_interval():
    aligner = _make_aligner()
    motion = torch.randn(1, 10, 13)
    images = torch.rand(1, 3, 224, 224)
    bboxes = torch.tensor([[10.0, 10.0, 80.0, 80.0]])
    image_hw = torch.tensor([[224, 224]])
    texts = ["moving cars"]
    out = aligner(motion, images, bboxes, image_hw, texts)
    score = aligner.similarity(out["motion_emb"], out["lang_emb"])
    assert 0.0 <= float(score.min()) <= float(score.max()) <= 1.0


def test_integration_with_existing_alignment_loss():
    from gmc_link.losses import AlignmentLoss
    aligner = _make_aligner().train()
    motion = torch.randn(4, 10, 13)
    images = torch.rand(4, 3, 224, 224)
    bboxes = torch.tensor([[10, 10, 80, 80]] * 4, dtype=torch.float32)
    image_hw = torch.tensor([[224, 224]] * 4)
    texts = ["moving cars", "parking vehicles", "moving cars", "turning vehicles"]
    sentence_ids = torch.tensor([0, 1, 0, 2])

    out = aligner(motion, images, bboxes, image_hw, texts)
    loss_fn = AlignmentLoss()
    loss = loss_fn(out["motion_emb"], out["lang_emb"], sentence_ids=sentence_ids)
    assert torch.isfinite(loss)
    loss.backward()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/seanachan/GMC-Link
RUN_CLIP_SMOKE=1 ~/miniconda/envs/RMOT/bin/python -m pytest tests/test_aligner_v2.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `FlexHookAdjacentAligner`**

Create `gmc_link/aligner_v2.py`:

```python
"""Joint cross-attention decoder for Exp 35.

Architecture: one learned query attends to the concatenation of CLIP text tokens,
3 scale motion tokens, and bbox-pooled CLIP ViT patch tokens. Two decoder layers.
Output: a single 256D motion-side embedding per sample. The language-side embedding
is the mean-pooled projected CLIP text tokens (so cosine similarity stays comparable
to the existing Layer-3 protocol).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from gmc_link.motion_encoder import TemporalMotionEncoder
from gmc_link.text_encoder_clip import ClipTextEncoder
from gmc_link.visual_encoder import ClipPatchExtractor


class _DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float) -> None:
        super().__init__()
        self.ca = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, kv_mask: torch.Tensor | None) -> torch.Tensor:
        # kv_mask: (B, L_kv) True=valid → convert to PyTorch key_padding_mask (True=ignore)
        kpm = None if kv_mask is None else ~kv_mask
        a, _ = self.ca(self.n1(q), kv, kv, key_padding_mask=kpm, need_weights=False)
        q = q + a
        q = q + self.ffn(self.n2(q))
        return q


class FlexHookAdjacentAligner(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        dim_ff: int = 512,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_motion: bool = True,
        use_vision: bool = True,
        motion_input_dim: int = 13,
        motion_window: int = 10,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_motion = use_motion
        self.use_vision = use_vision

        self.text_enc = ClipTextEncoder(d_out=d_model)
        self.motion_enc = (
            TemporalMotionEncoder(in_dim=motion_input_dim, T_max=motion_window, d_out=d_model)
            if use_motion else None
        )
        self.visual_enc = ClipPatchExtractor(d_out=d_model) if use_vision else None

        # Learned query and type embeddings (text / motion / vision).
        self.learned_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.type_text = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.type_motion = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.type_vision = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.layers = nn.ModuleList([
            _DecoderLayer(d_model, n_heads, dim_ff, dropout) for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(d_model)

    # ----- building blocks -----
    def _encode_text(self, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.text_enc(texts)
        return out["tokens"], out["mask"]

    def _encode_motion(
        self,
        motion: torch.Tensor,
        motion_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mtokens = self.motion_enc(motion, mask=motion_mask)  # (B, 3, D)
        mmask = torch.ones(mtokens.shape[:2], dtype=torch.bool, device=mtokens.device)
        return mtokens, mmask

    def _encode_vision(
        self,
        images: torch.Tensor,
        bboxes: torch.Tensor,
        image_hw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.visual_enc(images, bboxes, image_hw)
        return out["tokens"], out["mask"]

    # ----- forward -----
    def forward(
        self,
        motion: torch.Tensor,
        images: torch.Tensor,
        bboxes: torch.Tensor,
        image_hw: torch.Tensor,
        texts: Sequence[str],
        motion_mask: torch.Tensor | None = None,
    ) -> dict:
        B = motion.shape[0]

        text_tokens, text_mask = self._encode_text(texts)
        text_tokens = text_tokens + self.type_text

        parts_tok = [text_tokens]
        parts_mask = [text_mask]

        if self.use_motion:
            mtok, mmask = self._encode_motion(motion, motion_mask)
            parts_tok.append(mtok + self.type_motion)
            parts_mask.append(mmask)

        if self.use_vision:
            vtok, vmask = self._encode_vision(images, bboxes, image_hw)
            parts_tok.append(vtok + self.type_vision)
            parts_mask.append(vmask)

        kv = torch.cat(parts_tok, dim=1)  # (B, L_total, D)
        kv_mask = torch.cat(parts_mask, dim=1)  # (B, L_total)

        q = self.learned_query.expand(B, -1, -1)  # (B, 1, D)
        for layer in self.layers:
            q = layer(q, kv, kv_mask)
        motion_emb = self.out_norm(q).squeeze(1)  # (B, D)

        # Language-side: mean-pool projected text tokens over real tokens only.
        tm = text_mask.unsqueeze(-1).float()
        lang_emb = (text_tokens * tm).sum(dim=1) / tm.sum(dim=1).clamp_min(1.0)

        return {"motion_emb": motion_emb, "lang_emb": lang_emb}

    @staticmethod
    def similarity(motion_emb: torch.Tensor, lang_emb: torch.Tensor) -> torch.Tensor:
        a = torch.nn.functional.normalize(motion_emb, dim=-1)
        b = torch.nn.functional.normalize(lang_emb, dim=-1)
        return 0.5 * ((a * b).sum(dim=-1) + 1.0)  # cosine remapped to [0, 1]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/seanachan/GMC-Link
RUN_CLIP_SMOKE=1 ~/miniconda/envs/RMOT/bin/python -m pytest tests/test_aligner_v2.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add gmc_link/aligner_v2.py tests/test_aligner_v2.py
git commit -m "$(cat <<'EOF'
feat(exp35): FlexHookAdjacentAligner joint cross-attention decoder

One learned query attends to concat(text, motion_scale_tokens, bbox_patches)
with type embeddings. 2 decoder layers, d=256, 4 heads, ffn=512. Exposes
use_motion / use_vision flags for ablations A2 and A3.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: ViT patch-token cache builder

**Files:**
- Create: `diagnostics/build_vit_patch_cache.py`
- Create: `tests/test_vit_patch_cache.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_vit_patch_cache.py`:

```python
import os
import pathlib
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_CLIP_SMOKE") != "1",
    reason="Patch cache test requires CLIP (gated by RUN_CLIP_SMOKE=1)",
)


def _make_fake_frames(dir_: pathlib.Path, n: int = 3) -> list[pathlib.Path]:
    paths = []
    for i in range(n):
        p = dir_ / f"frame_{i:06d}.png"
        arr = (np.random.rand(100, 200, 3) * 255).astype(np.uint8)
        Image.fromarray(arr).save(p)
        paths.append(p)
    return paths


def test_cache_roundtrip(tmp_path: pathlib.Path):
    from diagnostics.build_vit_patch_cache import build_cache, load_cache
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    paths = _make_fake_frames(frame_dir, n=3)

    cache_path = tmp_path / "patch_cache.npz"
    build_cache([str(p) for p in paths], str(cache_path), batch_size=2)
    assert cache_path.exists()

    cache = load_cache(str(cache_path))
    assert set(cache.keys()) == {"frame_ids", "patches", "image_hw"}
    assert cache["patches"].shape == (3, 196, 768)
    assert cache["patches"].dtype == np.float16
    assert cache["image_hw"].shape == (3, 2)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/seanachan/GMC-Link
RUN_CLIP_SMOKE=1 ~/miniconda/envs/RMOT/bin/python -m pytest tests/test_vit_patch_cache.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement the cache builder**

Create `diagnostics/build_vit_patch_cache.py`:

```python
"""Per-frame CLIP ViT patch-token cache.

Usage:
  python diagnostics/build_vit_patch_cache.py \
      --frame-list frames_v1_train.txt \
      --out diagnostics/cache/vit_patches_v1_train.npz

The cache is (N_frames, 196, 768) float16 patch tokens + frame_ids (str) + image_hw
(H, W in original image coords). Consumed by the dataset v2 path.
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Iterable

import numpy as np
import torch
from PIL import Image


IMG_SIZE = 224


def _load_and_resize(path: str) -> tuple[torch.Tensor, tuple[int, int]]:
    img = Image.open(path).convert("RGB")
    hw = (img.size[1], img.size[0])
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1), hw


def build_cache(
    frame_paths: Iterable[str],
    out_path: str,
    batch_size: int = 32,
    device: str | None = None,
) -> None:
    from gmc_link.visual_encoder import ClipPatchExtractor
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    enc = ClipPatchExtractor().to(dev).eval()

    frame_paths = list(frame_paths)
    patches = np.zeros((len(frame_paths), 196, 768), dtype=np.float16)
    hw = np.zeros((len(frame_paths), 2), dtype=np.int32)
    ids = np.array(frame_paths, dtype=object)

    buf_imgs: list[torch.Tensor] = []
    buf_idx: list[int] = []

    def _flush():
        if not buf_imgs:
            return
        x = torch.stack(buf_imgs, dim=0).to(dev)
        with torch.no_grad():
            p = enc.forward_patches(x)  # (B, 196, 768) fp32
        p = p.cpu().numpy().astype(np.float16)
        for j, idx in enumerate(buf_idx):
            patches[idx] = p[j]
        buf_imgs.clear()
        buf_idx.clear()

    for i, path in enumerate(frame_paths):
        img, origin_hw = _load_and_resize(path)
        buf_imgs.append(img)
        buf_idx.append(i)
        hw[i] = origin_hw
        if len(buf_imgs) == batch_size:
            _flush()
    _flush()

    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, frame_ids=ids, patches=patches, image_hw=hw)


def load_cache(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {"frame_ids": data["frame_ids"], "patches": data["patches"], "image_hw": data["image_hw"]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frame-list", required=True, help="text file, one frame path per line")
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    with open(args.frame_list) as f:
        frames = [ln.strip() for ln in f if ln.strip()]
    build_cache(frames, args.out, batch_size=args.batch_size)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/seanachan/GMC-Link
RUN_CLIP_SMOKE=1 ~/miniconda/envs/RMOT/bin/python -m pytest tests/test_vit_patch_cache.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add diagnostics/build_vit_patch_cache.py tests/test_vit_patch_cache.py
git commit -m "$(cat <<'EOF'
feat(exp35): ViT patch-token cache builder

Produces (N_frames, 196, 768) fp16 patch tokens + frame_ids + original image_hw,
consumable by dataset.py v2. One-off per split.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Dataset v2 path (motion windows + bbox + ref_frame)

**Files:**
- Modify: `gmc_link/dataset.py` (add a new builder alongside existing `_generate_positive_pairs`)
- Create: `tests/test_dataset_v2.py`

**Scene-setting for the implementer:** the existing `_generate_positive_pairs` in `gmc_link/dataset.py` emits one 13D vector per (track, valid frame). For Exp 35 we need a *window* of T=10 consecutive 13D vectors per sample, plus the track's bbox on the reference (last-in-window) frame, plus the reference frame path. We do **NOT** modify `_generate_positive_pairs`; we add a parallel `_generate_flexhook_pairs` and wire it through `build_dataset(...)` behind a new `mode="flexhook"` kwarg.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dataset_v2.py`:

```python
import os
import pathlib

import numpy as np
import pytest
import torch


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_V2_DATASET") != "1",
    reason="Gated: requires Refer-KITTI at the default path",
)


def test_v2_sample_schema_tiny_slice():
    """Smoke: build a tiny V1 train slice via mode='flexhook' and check schema."""
    from gmc_link.dataset import build_dataset

    ds = build_dataset(
        split="v1",
        mode="flexhook",
        T=10,
        max_samples=32,  # tiny for speed
        seed=42,
    )
    assert len(ds) > 0
    sample = ds[0]
    assert set(sample.keys()) >= {
        "motion_window", "motion_mask", "bbox_xyxy",
        "image_hw", "ref_frame_path", "text", "sentence_id",
    }
    assert sample["motion_window"].shape == (10, 13)
    assert sample["motion_mask"].dtype == torch.bool
    assert sample["bbox_xyxy"].shape == (4,)
    assert isinstance(sample["ref_frame_path"], str)
    assert isinstance(sample["text"], str)


def test_collate_flexhook_stacks_correctly():
    from gmc_link.dataset import build_dataset, collate_flexhook
    ds = build_dataset(split="v1", mode="flexhook", T=10, max_samples=8, seed=42)
    batch = collate_flexhook([ds[i] for i in range(4)])
    assert batch["motion_window"].shape == (4, 10, 13)
    assert batch["bbox_xyxy"].shape == (4, 4)
    assert batch["image_hw"].shape == (4, 2)
    assert len(batch["ref_frame_path"]) == 4
    assert len(batch["text"]) == 4
    assert batch["sentence_id"].shape == (4,)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/seanachan/GMC-Link
RUN_V2_DATASET=1 ~/miniconda/envs/RMOT/bin/python -m pytest tests/test_dataset_v2.py -v
```

Expected: fails because `build_dataset` does not yet accept `mode="flexhook"` and `collate_flexhook` does not exist.

- [ ] **Step 3: Add `FlexHookDataset`, `_generate_flexhook_pairs`, `collate_flexhook`, and route through `build_dataset`**

In `gmc_link/dataset.py`, **add** (do not replace any existing code):

```python
# ---- Exp 35 / FlexHook-adjacent additions -----------------------------------

class FlexHookDataset(Dataset):
    """Emits (motion_window, bbox, ref_frame_path, text) per sample."""

    def __init__(self, records: list[dict]) -> None:
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict:
        r = self._records[idx]
        return {
            "motion_window": torch.from_numpy(r["motion_window"]).float(),  # (T, 13)
            "motion_mask": torch.from_numpy(r["motion_mask"]).bool(),       # (T,)
            "bbox_xyxy": torch.from_numpy(r["bbox_xyxy"]).float(),          # (4,)
            "image_hw": torch.tensor(r["image_hw"], dtype=torch.int32),     # (2,)
            "ref_frame_path": r["ref_frame_path"],
            "text": r["text"],
            "sentence_id": int(r["sentence_id"]),
        }


def collate_flexhook(samples: list[dict]) -> dict:
    return {
        "motion_window": torch.stack([s["motion_window"] for s in samples], dim=0),
        "motion_mask": torch.stack([s["motion_mask"] for s in samples], dim=0),
        "bbox_xyxy": torch.stack([s["bbox_xyxy"] for s in samples], dim=0),
        "image_hw": torch.stack([s["image_hw"] for s in samples], dim=0),
        "ref_frame_path": [s["ref_frame_path"] for s in samples],
        "text": [s["text"] for s in samples],
        "sentence_id": torch.tensor([s["sentence_id"] for s in samples], dtype=torch.long),
    }


def _generate_flexhook_pairs(
    split: str,
    T: int,
    max_samples: int | None,
    seed: int,
    raw_displacement: bool = False,
) -> list[dict]:
    """Build per-(track, window) records.

    Reuses the same centroid/ego/multi-scale-velocity logic as the existing
    _generate_positive_pairs but emits a *window* of T consecutive 13D vectors
    plus the track's bbox on the reference (last-in-window) frame.

    If raw_displacement is True, replaces the 6 residual-velocity dims with
    single-frame raw bbox displacement, replicated across the 3 scale slots.
    This is the A1 ablation.
    """
    # Lift the geometry pipeline the existing generator uses. Keep the algorithmic
    # logic identical so A vs baselines remain comparable. The only change is that
    # for each valid anchor frame f, we assemble the last T 13D snapshots of the
    # same track ending at f, left-padding (with motion_mask=False) if fewer than
    # T are available. Shape details match _generate_positive_pairs' outputs.
    raise NotImplementedError(
        "Implementer: follow the exact structure of _generate_positive_pairs in "
        "this file (see line ~706 at time of writing). Keep FRAME_GAPS, "
        "VELOCITY_SCALE, and the 13D feature layout identical. Per valid sample: "
        "(1) walk the track to collect up to T prior 13D vectors ending at the "
        "anchor frame; (2) left-pad to exactly T with zeros + motion_mask=False; "
        "(3) record the anchor frame's bbox in original pixel coords; (4) record "
        "image_hw from the frame; (5) record the frame's filesystem path (the "
        "existing dataset loader already knows this); (6) reuse the same "
        "sentence_id assignment as _generate_positive_pairs. For raw_displacement "
        "mode, replace dims [0..5] with [raw_dx, raw_dy]*3 from the bbox "
        "displacement between the previous and current anchor frames."
    )


def build_dataset(
    *,
    split: str,
    mode: str = "mlp",
    T: int = 10,
    max_samples: int | None = None,
    seed: int = 42,
    raw_displacement: bool = False,
    **legacy_kwargs,
):
    """Unified dataset factory.

    mode='mlp'      → existing path (13D snapshots, identical to pre-Exp 35).
    mode='flexhook' → Exp 35 path (T-windows + bbox + ref_frame).
    """
    if mode == "flexhook":
        records = _generate_flexhook_pairs(
            split=split, T=T, max_samples=max_samples, seed=seed,
            raw_displacement=raw_displacement,
        )
        return FlexHookDataset(records)

    # Fallback to legacy path; import and call the existing function so the public
    # surface stays identical for `--arch mlp`.
    return _legacy_build_dataset(split=split, **legacy_kwargs)
```

> **Implementation note for the engineer:** `_legacy_build_dataset` is the name you should give to the current `build_dataset`-equivalent entry point (whatever it is called at HEAD) — rename/expose it so both paths coexist. Search for the existing training data factory (likely named `build_training_dataset` or used inline in `train.py`) and wire it through.
>
> The `_generate_flexhook_pairs` stub body intentionally raises — this is the task where the implementer reads the existing `_generate_positive_pairs` (line ~706 at time of writing) and replicates its logic with the window-plus-bbox-plus-frame-path extension. Keep `FRAME_GAPS`, `VELOCITY_SCALE`, sentence-ID assignment, and the 13D feature layout **identical** across both generators. No reshuffling of per-expression logic; only an additional window assembly step.

- [ ] **Step 4: Implement `_generate_flexhook_pairs` by mirroring `_generate_positive_pairs`**

The implementer must now:
1. Read `gmc_link/dataset.py` starting at the line where `_generate_positive_pairs` is defined.
2. Copy the track-walking structure into `_generate_flexhook_pairs`.
3. For each anchor frame, gather the last T 13D vectors (zero-left-pad + `motion_mask`).
4. Record `bbox_xyxy` (the anchor frame's bbox in original pixel coords, as already available on the GT centroid track), `image_hw`, and `ref_frame_path` (the existing loader knows the frame path pattern — look at how `diag_gt_cosine_distributions.py` locates frames).
5. Keep sentence-ID assignment identical to the existing generator.
6. If `raw_displacement=True`, replace dims [0..5] with `[raw_dx, raw_dy]` replicated 3 times, computed from the anchor frame's bbox center minus the previous frame's bbox center, in original pixel coords (no ego compensation, no multi-scale).

After implementation, remove the `raise NotImplementedError` and re-run the test from Step 2.

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /home/seanachan/GMC-Link
RUN_V2_DATASET=1 ~/miniconda/envs/RMOT/bin/python -m pytest tests/test_dataset_v2.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add gmc_link/dataset.py tests/test_dataset_v2.py
git commit -m "$(cat <<'EOF'
feat(exp35): dataset.py mode='flexhook' — window+bbox+frame_path samples

New FlexHookDataset and collate_flexhook emit T=10 windows of 13D vectors
plus the anchor frame's bbox and filesystem path. Reuses the exact track
walking / ego compensation / multi-scale velocity logic from
_generate_positive_pairs so A-vs-A1 and A-vs-MLP comparisons stay clean.
raw_displacement=True swaps dims[0..5] for raw bbox displacement (A1 ablation).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Training integration — `--arch flexhook`

**Files:**
- Modify: `gmc_link/train.py` (extend the existing `--architecture` choices)
- Create: `tests/test_train_flexhook_smoke.py`

- [ ] **Step 1: Write the failing 3-step smoke test**

Create `tests/test_train_flexhook_smoke.py`:

```python
import os
import pathlib
import subprocess
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_V2_DATASET") != "1" or os.environ.get("RUN_CLIP_SMOKE") != "1",
    reason="Smoke test needs CLIP + Refer-KITTI (gated by RUN_V2_DATASET=1, RUN_CLIP_SMOKE=1)",
)


def test_three_training_steps_do_not_crash(tmp_path: pathlib.Path):
    save = tmp_path / "smoke.pth"
    cmd = [
        os.path.expanduser("~/miniconda/envs/RMOT/bin/python"),
        "-m", "gmc_link.train",
        "--split", "v1",
        "--architecture", "flexhook",
        "--epochs", "1",
        "--max-steps", "3",
        "--batch-size", "4",
        "--lr", "3e-4",
        "--seed", "42",
        "--save-path", str(save),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert save.exists(), "checkpoint was not saved"
```

- [ ] **Step 2: Run the smoke test to verify it fails**

```bash
cd /home/seanachan/GMC-Link
RUN_CLIP_SMOKE=1 RUN_V2_DATASET=1 ~/miniconda/envs/RMOT/bin/python -m pytest tests/test_train_flexhook_smoke.py -v
```

Expected: fails because `--architecture flexhook` isn't a valid choice and/or `--max-steps` isn't recognized.

- [ ] **Step 3: Extend `train.py` argument parser and dispatcher**

In `gmc_link/train.py`:

1. Extend the existing `add_argument` for `--architecture`:

```python
parser.add_argument(
    "--architecture",
    default="mlp",
    choices=["mlp", "temporal_transformer", "flexhook"],
    help="mlp (default), temporal_transformer (Exp 27), or flexhook (Exp 35)",
)
```

2. Add a `--max-steps` arg (used only by smoke tests; default None → full epochs):

```python
parser.add_argument("--max-steps", type=int, default=None,
                    help="If set, stop training after this many optimizer steps (for smoke tests).")
parser.add_argument("--seed", type=int, default=42)
```

3. In `_run_single_stage` (or equivalent), add the `flexhook` branch that:
   - Uses `build_dataset(split=..., mode="flexhook", T=10, seed=args.seed, raw_displacement=args.raw_displacement)` and `collate_flexhook`.
   - Constructs `FlexHookAdjacentAligner(use_motion=args.use_motion, use_vision=args.use_vision)`.
   - Creates two AdamW param groups: `{LR=args.lr}` for trainable (decoder, projectors, motion encoder, queries/type embeds), `{LR=0.0}` for CLIP params. Use `filter(lambda p: p.requires_grad, ...)` to isolate trainable.
   - Loads images from `ref_frame_path` inside the training step: read with PIL, resize to 224×224, stack, to device.
   - Forward: `out = model(motion_window, images, bbox_xyxy, image_hw, text, motion_mask=...)`.
   - Loss: existing `AlignmentLoss()(out["motion_emb"], out["lang_emb"], sentence_ids=batch["sentence_id"])`.
   - Honor `--max-steps` to break out of the training loop early.

4. Add `--raw-displacement`, `--no-motion`, `--no-vision` flags (used by ablations):

```python
parser.add_argument("--raw-displacement", action="store_true",
                    help="A1 ablation: replace residual velocity with raw bbox displacement.")
parser.add_argument("--no-motion", action="store_true",
                    help="A2 ablation: drop motion tokens from the decoder.")
parser.add_argument("--no-vision", action="store_true",
                    help="A3 ablation: drop ViT patch tokens from the decoder.")
```

Pass `use_motion = not args.no_motion`, `use_vision = not args.no_vision` into the aligner constructor, and `raw_displacement=args.raw_displacement` into the dataset.

- [ ] **Step 4: Run smoke test to verify it passes**

```bash
cd /home/seanachan/GMC-Link
RUN_CLIP_SMOKE=1 RUN_V2_DATASET=1 ~/miniconda/envs/RMOT/bin/python -m pytest tests/test_train_flexhook_smoke.py -v
```

Expected: PASS. The 3-step smoke must complete in < 5 minutes (dominated by CLIP warm-up).

- [ ] **Step 5: Commit**

```bash
git add gmc_link/train.py tests/test_train_flexhook_smoke.py
git commit -m "$(cat <<'EOF'
feat(exp35): train.py --architecture flexhook + ablation flags

Adds --architecture flexhook path wiring FlexHookAdjacentAligner +
mode='flexhook' dataset + collate_flexhook + image loader. New flags
--raw-displacement, --no-motion, --no-vision drive ablations A1/A2/A3.
--max-steps allows smoke tests without running full epochs.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Diagnostic integration — `--arch flexhook` in `diag_gt_cosine_distributions.py`

**Files:**
- Modify: `diagnostics/diag_gt_cosine_distributions.py`

**Scene-setting:** the existing script loads an `MotionLanguageAligner` checkpoint and scores every (GT-track, expression) pair per frame, producing `layer3_gt_cosine_{seq}.{npz,png}`. Our Exp 35 checkpoint is a `FlexHookAdjacentAligner`. We add an `--arch {mlp,flexhook}` flag that dispatches the correct model constructor and scoring loop. The `.npz` schema emitted must be unchanged so `diagnostics/aggregate_multiseq.py` works without modification.

- [ ] **Step 1: Add `--arch` flag and dispatcher**

Modify `diagnostics/diag_gt_cosine_distributions.py`:

1. Add argument:

```python
p.add_argument("--arch", default="mlp", choices=["mlp", "flexhook"],
               help="mlp (default) or flexhook (Exp 35 checkpoint)")
```

2. At model load time:

```python
if args.arch == "flexhook":
    from gmc_link.aligner_v2 import FlexHookAdjacentAligner
    model = FlexHookAdjacentAligner().to(device).eval()
    sd = torch.load(args.weights, map_location=device)
    model.load_state_dict(sd, strict=True)
else:
    # existing MLP path
    ...
```

3. In the per-track scoring loop, branch: for `arch='flexhook'`, build the per-frame input just like training (motion window, bbox, image_hw, ref_frame path → loaded image) and call `model(...)`; take `similarity(motion_emb, lang_emb)` as the score. For `arch='mlp'`, keep the existing code path verbatim.

4. `.npz` output schema (`gt_match`, `cosine`, `expression_id`, `seq_id`, `track_id`, `frame_id`) must be identical to the current schema — `aggregate_multiseq.py` and `layer3_multiseq_*.md` rendering are unchanged.

- [ ] **Step 2: Run the diagnostic against the MLP path to prove no regression**

```bash
cd /home/seanachan/GMC-Link
~/miniconda/envs/RMOT/bin/python diagnostics/diag_gt_cosine_distributions.py \
    --weights gmc_link_weights_v1train_stage1.pth \
    --seq 0011 --arch mlp
```

Expected: same numbers as the pre-modification run. Diff `layer3_gt_cosine_0011.npz` cosines if unsure.

- [ ] **Step 3: Commit**

```bash
git add diagnostics/diag_gt_cosine_distributions.py
git commit -m "$(cat <<'EOF'
feat(exp35): diag_gt_cosine_distributions.py --arch flexhook dispatch

Adds --arch {mlp,flexhook} so Exp 35 checkpoints can reuse the existing
multi-seq pipeline with an identical .npz schema. MLP path unchanged.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Headline + ablation driver scripts

**Files:**
- Create: `run_flexhook_adjacent.sh`
- Create: `run_flexhook_ablations.sh`

- [ ] **Step 1: Write `run_flexhook_adjacent.sh`**

```bash
#!/usr/bin/env bash
# Exp 35 headline driver: train FlexHookAdjacentAligner (Approach A) on V1, then
# run the Layer 3 multi-seq diagnostic on {0005, 0011, 0013} and aggregate.
#
# Spec: docs/superpowers/specs/2026-04-19-flexhook-adjacent-alignment-design.md
# Plan: docs/superpowers/plans/2026-04-19-flexhook-adjacent-alignment.md
#
# Usage: bash run_flexhook_adjacent.sh

set -euo pipefail

PY=~/miniconda/envs/RMOT/bin/python
DIAG=diagnostics/diag_gt_cosine_distributions.py
RESULTS_DIR=diagnostics/results
MULTISEQ_DIR="${RESULTS_DIR}/multiseq"
SEQS=(0005 0011 0013)
TAG=v1train_flexhook_adj
WEIGHTS="gmc_link_weights_${TAG}.pth"

mkdir -p "${MULTISEQ_DIR}"

# ── Training ───────────────────────────────────────────────────────
echo "============================================================"
echo "Exp 35 headline: training ${WEIGHTS}"
echo "============================================================"
"${PY}" -m gmc_link.train \
    --split v1 \
    --architecture flexhook \
    --epochs 40 --lr 3e-4 --batch-size 128 \
    --seed 42 \
    --save-path "${WEIGHTS}"

# ── Evaluation loop ────────────────────────────────────────────────
for seq in "${SEQS[@]}"; do
    echo "--- ${seq} ---"
    "${PY}" "${DIAG}" --weights "${WEIGHTS}" --seq "${seq}" --arch flexhook
    src="${RESULTS_DIR}/layer3_gt_cosine_${seq}.npz"
    dst="${MULTISEQ_DIR}/layer3_${seq}_${TAG}.npz"
    mv "${src}" "${dst}"
    src_png="${RESULTS_DIR}/layer3_gt_cosine_${seq}.png"
    if [[ -f "${src_png}" ]]; then
        mv "${src_png}" "${MULTISEQ_DIR}/layer3_${seq}_${TAG}.png"
    fi
done

# ── Aggregation: add a 12th row next to Exp 33's 11 baselines ──────
ALL_WEIGHTS=(
    "v1train_stage1=gmc_link_weights_v1train_stage1.pth"
    "${TAG}=${WEIGHTS}"
)
LEGACY=("v1train_stage1=0.779")
echo "Aggregating headline row..."
"${PY}" diagnostics/aggregate_multiseq.py \
    --results-dir "${MULTISEQ_DIR}" \
    --output-dir "${MULTISEQ_DIR}" \
    --weights "${ALL_WEIGHTS[@]}" \
    --seqs "${SEQS[@]}" \
    --legacy-seq-0011 "${LEGACY[@]}"

echo "Done. See ${MULTISEQ_DIR}/layer3_multiseq_comparison.md"
```

Make executable:

```bash
chmod +x /home/seanachan/GMC-Link/run_flexhook_adjacent.sh
```

- [ ] **Step 2: Write `run_flexhook_ablations.sh`**

```bash
#!/usr/bin/env bash
# Exp 35 ablation driver: train A1 (raw displacement), A2 (no motion), A3 (no vision)
# with identical hyperparameters, then evaluate each on {0005, 0011, 0013}.
#
# Usage: bash run_flexhook_ablations.sh

set -euo pipefail

PY=~/miniconda/envs/RMOT/bin/python
DIAG=diagnostics/diag_gt_cosine_distributions.py
RESULTS_DIR=diagnostics/results
MULTISEQ_DIR="${RESULTS_DIR}/multiseq"
SEQS=(0005 0011 0013)

declare -A ABL_FLAGS=(
    ["a1_rawdisp"]="--raw-displacement"
    ["a2_nomotion"]="--no-motion"
    ["a3_novision"]="--no-vision"
)

for abl in "${!ABL_FLAGS[@]}"; do
    tag="v1train_${abl}"
    weights="gmc_link_weights_${tag}.pth"
    flags="${ABL_FLAGS[$abl]}"

    echo "============================================================"
    echo "Ablation ${abl}: training ${weights} [${flags}]"
    echo "============================================================"
    "${PY}" -m gmc_link.train \
        --split v1 \
        --architecture flexhook \
        ${flags} \
        --epochs 40 --lr 3e-4 --batch-size 128 \
        --seed 42 \
        --save-path "${weights}"

    for seq in "${SEQS[@]}"; do
        "${PY}" "${DIAG}" --weights "${weights}" --seq "${seq}" --arch flexhook
        mv "${RESULTS_DIR}/layer3_gt_cosine_${seq}.npz" \
           "${MULTISEQ_DIR}/layer3_${seq}_${tag}.npz"
        if [[ -f "${RESULTS_DIR}/layer3_gt_cosine_${seq}.png" ]]; then
            mv "${RESULTS_DIR}/layer3_gt_cosine_${seq}.png" \
               "${MULTISEQ_DIR}/layer3_${seq}_${tag}.png"
        fi
    done
done

# Aggregate all 5: stage1 baseline + headline + 3 ablations.
ALL_WEIGHTS=(
    "v1train_stage1=gmc_link_weights_v1train_stage1.pth"
    "v1train_flexhook_adj=gmc_link_weights_v1train_flexhook_adj.pth"
    "v1train_a1_rawdisp=gmc_link_weights_v1train_a1_rawdisp.pth"
    "v1train_a2_nomotion=gmc_link_weights_v1train_a2_nomotion.pth"
    "v1train_a3_novision=gmc_link_weights_v1train_a3_novision.pth"
)
LEGACY=("v1train_stage1=0.779")
"${PY}" diagnostics/aggregate_multiseq.py \
    --results-dir "${MULTISEQ_DIR}" \
    --output-dir "${MULTISEQ_DIR}" \
    --weights "${ALL_WEIGHTS[@]}" \
    --seqs "${SEQS[@]}" \
    --legacy-seq-0011 "${LEGACY[@]}"

echo "Done. See ${MULTISEQ_DIR}/layer3_multiseq_comparison.md"
```

Make executable:

```bash
chmod +x /home/seanachan/GMC-Link/run_flexhook_ablations.sh
```

- [ ] **Step 3: Commit**

```bash
git add run_flexhook_adjacent.sh run_flexhook_ablations.sh
git commit -m "$(cat <<'EOF'
feat(exp35): driver scripts for headline + A1/A2/A3 ablations

run_flexhook_adjacent.sh trains the headline Approach A checkpoint and
evaluates on {0005, 0011, 0013}. run_flexhook_ablations.sh trains A1/A2/A3
at identical hyperparams and aggregates a 5-row comparison against stage1.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Run headline training + multi-seq eval

- [ ] **Step 1: Build the patch-token cache once**

```bash
cd /home/seanachan/GMC-Link
# Produce a frame list covering V1 train split + eval seqs.
# Implementer: adapt this to the actual expression-annotation walker used by
# _generate_positive_pairs. The goal is one unique frame path per line.
~/miniconda/envs/RMOT/bin/python diagnostics/build_vit_patch_cache.py \
    --frame-list diagnostics/cache/frames_v1_all.txt \
    --out diagnostics/cache/vit_patches_v1_all.npz \
    --batch-size 64
```

Expected: `diagnostics/cache/vit_patches_v1_all.npz` exists, size ~3 GB.

> **Note:** if patch-token caching isn't wired into `train.py` yet at this point (i.e., the trainer is still doing per-step `CLIPVisionModel.forward`), the headline run will work but be ~10× slower. Integrating the cache into the training DataLoader is a "nice to have" that can be skipped for Exp 35 if wall-time remains acceptable (< 2 hours per training run). If it's too slow, add a cache-hit path in `collate_flexhook` or in the training step that keys into the `.npz` by `ref_frame_path`.

- [ ] **Step 2: Launch the headline run**

```bash
cd /home/seanachan/GMC-Link
bash run_flexhook_adjacent.sh 2>&1 | tee diagnostics/results/multiseq/_run_log_flexhook_adj.txt
```

Expected: ~20–90 min on RTX 3060 Ti depending on whether the patch cache is wired. Final micro AUC printed in the aggregated `layer3_multiseq_v1train_flexhook_adj.md`.

- [ ] **Step 3: Verify result against §1 decision rule**

```bash
grep -A2 "Headline" diagnostics/results/multiseq/layer3_multiseq_v1train_flexhook_adj.md
```

Read the micro AUC. Apply:
- ≥ 0.830 → Strong positive; proceed to Task 12 + 13 as planned.
- 0.795 – 0.829 → Positive; proceed but flag in retrospective.
- 0.764 – 0.794 → Inconclusive; proceed through ablations but expect the retrospective to recommend an architecture family change.
- < 0.764 → Strongly negative; pause and debug (check loss trajectory, attention weights) before burning compute on ablations.

- [ ] **Step 4: Commit headline artifacts**

```bash
git add diagnostics/results/multiseq/layer3_multiseq_v1train_flexhook_adj.md \
        diagnostics/results/multiseq/layer3_multiseq_comparison.md \
        diagnostics/results/multiseq/_run_log_flexhook_adj.txt
git commit -m "$(cat <<'EOF'
exp(exp35): headline FlexHookAdjacentAligner multi-seq result

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: A1 ablation — raw displacement (the ego-compensation proof)

- [ ] **Step 1: Train + evaluate A1 only (run the other two in Task 13)**

```bash
cd /home/seanachan/GMC-Link
~/miniconda/envs/RMOT/bin/python -m gmc_link.train \
    --split v1 --architecture flexhook \
    --raw-displacement \
    --epochs 40 --lr 3e-4 --batch-size 128 --seed 42 \
    --save-path gmc_link_weights_v1train_a1_rawdisp.pth

for seq in 0005 0011 0013; do
    ~/miniconda/envs/RMOT/bin/python diagnostics/diag_gt_cosine_distributions.py \
        --weights gmc_link_weights_v1train_a1_rawdisp.pth \
        --seq "${seq}" --arch flexhook
    mv diagnostics/results/layer3_gt_cosine_${seq}.npz \
       diagnostics/results/multiseq/layer3_${seq}_v1train_a1_rawdisp.npz
done
```

- [ ] **Step 2: Compute A vs A1 delta**

```bash
cd /home/seanachan/GMC-Link
~/miniconda/envs/RMOT/bin/python diagnostics/aggregate_multiseq.py \
    --results-dir diagnostics/results/multiseq \
    --output-dir diagnostics/results/multiseq \
    --weights \
        "v1train_flexhook_adj=gmc_link_weights_v1train_flexhook_adj.pth" \
        "v1train_a1_rawdisp=gmc_link_weights_v1train_a1_rawdisp.pth" \
    --seqs 0005 0011 0013 \
    --legacy-seq-0011 "v1train_stage1=0.779"
```

Read the two rows from `layer3_multiseq_comparison.md`. The **headline number for the paper** is `Δ = AUC(A) − AUC(A1)`. Record:
- If Δ > 0.01: positive ego-compensation evidence under the new architecture.
- If |Δ| ≤ 0.01: ego compensation is not load-bearing under this architecture — an honest negative result worth reporting.
- If Δ < −0.01: ego compensation is hurting under this architecture — the interesting case; investigate training dynamics before concluding.

- [ ] **Step 3: Commit A1 artifacts**

```bash
git add diagnostics/results/multiseq/layer3_0005_v1train_a1_rawdisp.npz \
        diagnostics/results/multiseq/layer3_0011_v1train_a1_rawdisp.npz \
        diagnostics/results/multiseq/layer3_0013_v1train_a1_rawdisp.npz \
        diagnostics/results/multiseq/layer3_multiseq_comparison.md
git commit -m "$(cat <<'EOF'
exp(exp35): A1 raw-displacement ablation — ego-compensation Δ

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: A2 and A3 ablations (no-motion, no-vision)

- [ ] **Step 1: Train A2 (no-motion) and evaluate**

```bash
cd /home/seanachan/GMC-Link
~/miniconda/envs/RMOT/bin/python -m gmc_link.train \
    --split v1 --architecture flexhook --no-motion \
    --epochs 40 --lr 3e-4 --batch-size 128 --seed 42 \
    --save-path gmc_link_weights_v1train_a2_nomotion.pth

for seq in 0005 0011 0013; do
    ~/miniconda/envs/RMOT/bin/python diagnostics/diag_gt_cosine_distributions.py \
        --weights gmc_link_weights_v1train_a2_nomotion.pth \
        --seq "${seq}" --arch flexhook
    mv diagnostics/results/layer3_gt_cosine_${seq}.npz \
       diagnostics/results/multiseq/layer3_${seq}_v1train_a2_nomotion.npz
done
```

- [ ] **Step 2: Train A3 (no-vision) and evaluate**

```bash
cd /home/seanachan/GMC-Link
~/miniconda/envs/RMOT/bin/python -m gmc_link.train \
    --split v1 --architecture flexhook --no-vision \
    --epochs 40 --lr 3e-4 --batch-size 128 --seed 42 \
    --save-path gmc_link_weights_v1train_a3_novision.pth

for seq in 0005 0011 0013; do
    ~/miniconda/envs/RMOT/bin/python diagnostics/diag_gt_cosine_distributions.py \
        --weights gmc_link_weights_v1train_a3_novision.pth \
        --seq "${seq}" --arch flexhook
    mv diagnostics/results/layer3_gt_cosine_${seq}.npz \
       diagnostics/results/multiseq/layer3_${seq}_v1train_a3_novision.npz
done
```

- [ ] **Step 3: Re-aggregate all rows including A2, A3**

```bash
cd /home/seanachan/GMC-Link
~/miniconda/envs/RMOT/bin/python diagnostics/aggregate_multiseq.py \
    --results-dir diagnostics/results/multiseq \
    --output-dir diagnostics/results/multiseq \
    --weights \
        "v1train_stage1=gmc_link_weights_v1train_stage1.pth" \
        "v1train_flexhook_adj=gmc_link_weights_v1train_flexhook_adj.pth" \
        "v1train_a1_rawdisp=gmc_link_weights_v1train_a1_rawdisp.pth" \
        "v1train_a2_nomotion=gmc_link_weights_v1train_a2_nomotion.pth" \
        "v1train_a3_novision=gmc_link_weights_v1train_a3_novision.pth" \
    --seqs 0005 0011 0013 \
    --legacy-seq-0011 "v1train_stage1=0.779"
```

- [ ] **Step 4: Commit A2/A3 artifacts**

```bash
git add diagnostics/results/multiseq/layer3_0005_v1train_a2_nomotion.npz \
        diagnostics/results/multiseq/layer3_0011_v1train_a2_nomotion.npz \
        diagnostics/results/multiseq/layer3_0013_v1train_a2_nomotion.npz \
        diagnostics/results/multiseq/layer3_0005_v1train_a3_novision.npz \
        diagnostics/results/multiseq/layer3_0011_v1train_a3_novision.npz \
        diagnostics/results/multiseq/layer3_0013_v1train_a3_novision.npz \
        diagnostics/results/multiseq/layer3_multiseq_comparison.md
git commit -m "$(cat <<'EOF'
exp(exp35): A2 no-motion + A3 no-vision ablations

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Retrospective — RESEARCH_NOTES.md §Exp 35

**Files:**
- Modify: `RESEARCH_NOTES.md` (append new section)

- [ ] **Step 1: Append Exp 35 retrospective**

Use this skeleton (fill values from the 5-row comparison table):

```markdown
## Exp 35 — FlexHook-adjacent aligner with ego-compensated motion tokens

**Date:** 2026-04-XX
**Branch:** `exp/flexhook-adjacent`
**Spec:** `docs/superpowers/specs/2026-04-19-flexhook-adjacent-alignment-design.md`
**Plan:** `docs/superpowers/plans/2026-04-19-flexhook-adjacent-alignment.md`

### Decision rule (pre-registered, from spec §1)
- ≥ 0.830 strong positive · 0.795–0.829 positive · 0.764–0.794 inconclusive · < 0.764 negative.

### Headline numbers (multi-seq micro AUC, V1 train → {0005, 0011, 0013})

| Variant | Micro AUC | Macro AUC | Δ vs stage1 (0.779) | Δ vs A | Notes |
|---|---|---|---|---|---|
| Stage 1 (Exp 33 baseline, 13D→MLP) | 0.779 | 0.838 | — | — | reference |
| A — FlexHook-adjacent (headline) | TBF | TBF | TBF | 0 | |
| A1 — raw displacement (no ego comp) | TBF | TBF | TBF | TBF | ego-comp proof |
| A2 — no motion tokens | TBF | TBF | TBF | TBF | motion necessity |
| A3 — no ViT patches | TBF | TBF | TBF | TBF | vision necessity |

### Interpretation
- Verdict against §1 decision rule: [STRONG POSITIVE | POSITIVE | INCONCLUSIVE | NEGATIVE].
- Ego-compensation contribution (Δ A − A1): [X].
- Motion contribution overall (Δ A − A2): [X].
- Vision contribution overall (Δ A − A3): [X].
- Attention-weight observations (if captured): …

### Next steps
- [If positive] Escalate to Exp 36: fine-tune CLIP top blocks, richer motion tokens (Approach B), reference-point conditioning (Approach C), or end-to-end F1 replacement (Framing Q).
- [If inconclusive/negative] Pivot to [specify direction based on ablation signals].
```

Replace `TBF` with actual values from `layer3_multiseq_comparison.md`. Fill in the interpretation section with real observations. Keep the retrospective honest — if the delta is negative, say so plainly.

- [ ] **Step 2: Update memory index**

If the headline and ablations produce a clear verdict, create a new memory file `/home/seanachan/.claude/projects/-home-seanachan-GMC-Link/memory/project_exp35_<verdict>.md` capturing the headline number and its implications, then add a one-line pointer to `MEMORY.md`. Follow the same structure as the existing `project_exp34_representation_bound.md`.

- [ ] **Step 3: Commit retrospective**

```bash
git add RESEARCH_NOTES.md
git commit -m "$(cat <<'EOF'
docs(exp35): retrospective — FlexHook-adjacent verdict + ablations

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Self-review checklist (for the plan author)

This section records the self-review I (plan author) performed against the spec. Issues found and fixed inline are listed here for traceability.

**Spec coverage:**
- Spec §3.1 TemporalMotionEncoder → Task 2 ✔
- Spec §3.2 CLIP text wrapper → Task 3 ✔
- Spec §3.3 CLIP vision wrapper with bbox pool → Task 4 ✔
- Spec §3.4 joint decoder → Task 5 ✔
- Spec §4 patch-token cache → Task 6 + Task 11 Step 1 ✔
- Spec §4 dataset v2 path → Task 7 ✔
- Spec §5 training regime (AdamW, LR=3e-4, 40 epochs, batch 128, seed 42) → Tasks 8 + 10 ✔
- Spec §6 primary + secondary evaluation → Tasks 9 + 11 ✔
- Spec §7 ablations A1/A2/A3 → Tasks 12 + 13 ✔
- Spec §10 exit criteria → covered by Tasks 11–14 ✔

**Placeholder scan:** No TBD / TODO / "implement later" in code or commands. One explicit TBF in Task 14 that is expected to be filled at that task's execution time (this is a runtime measurement, not a plan gap).

**Type consistency:**
- `TemporalMotionEncoder` output is `(B, 3, d_out=256)` in both Task 2 and Task 5's decoder concat. ✔
- `ClipTextEncoder.forward` returns `{"tokens", "mask"}` in Task 3 and is consumed as such in Task 5. ✔
- `ClipPatchExtractor.forward` returns `{"tokens", "mask"}` in Task 4 and is consumed as such in Task 5. ✔
- `FlexHookAdjacentAligner.forward` returns `{"motion_emb", "lang_emb"}` in Task 5 and is consumed by `AlignmentLoss` and `diag_gt_cosine_distributions.py` the same way in Tasks 8 + 9. ✔
- `collate_flexhook` and `build_dataset(mode="flexhook")` are named consistently across Tasks 7 + 8. ✔

**Known risks not resolved in the plan (inherited from spec §8):**
- ViT patch attention may drown motion attention — Task 11 Step 3 inspects the headline result first; mitigation strategies (aux loss, motion warm-start) are deferred to Exp 36 if triggered.
- Patch-token cache integration into the training DataLoader is flagged as optional in Task 11 Step 1. Acceptable wall-time is the gating condition.
