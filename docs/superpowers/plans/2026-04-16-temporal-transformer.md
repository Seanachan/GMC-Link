# Temporal Transformer Motion Encoder — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-frame MLP motion projector with a 1-layer transformer encoder over T=10 frame sequences to improve cosine separation (currently +0.195) and AUC (currently 0.779).

**Architecture:** A `TemporalMotionEncoder` module processes (batch, T, 13) sequences via a learnable input projection, [CLS] token, positional encoding, and one TransformerEncoderLayer (d_model=64, 4 heads). The [CLS] output is projected to 256D for cosine similarity with language embeddings. The training data pipeline produces sliding-window sequences from per-track per-frame 13D vectors.

**Tech Stack:** PyTorch (nn.TransformerEncoderLayer), existing GMC-Link dataset/training infrastructure, conda env `RMOT`

**Spec:** `docs/superpowers/specs/2026-04-16-temporal-transformer-design.md`

---

### Task 1: Create branch

**Files:** None (git only)

- [ ] **Step 1: Create and switch to new branch**

```bash
git checkout -b exp/temporal-transformer
```

- [ ] **Step 2: Verify branch**

```bash
git branch --show-current
```

Expected: `exp/temporal-transformer`

---

### Task 2: Implement TemporalMotionEncoder

**Files:**
- Modify: `gmc_link/alignment.py`

- [ ] **Step 1: Add TemporalMotionEncoder class**

Add this class **above** the existing `MotionLanguageAligner` class (before line 13):

```python
class TemporalMotionEncoder(nn.Module):
    """
    Transformer-based motion encoder that processes a sequence of T per-frame
    motion vectors and produces a single embedding via a [CLS] token.
    """

    def __init__(
        self,
        motion_dim: int = 13,
        d_model: int = 64,
        n_head: int = 4,
        dim_feedforward: int = 128,
        num_layers: int = 1,
        seq_len: int = 10,
        embed_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len

        # Per-frame projection: 13D → d_model
        self.input_proj = nn.Linear(motion_dim, d_model)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Learnable positional encoding for T+1 positions (CLS + T frames)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len + 1, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output projection: d_model → embed_dim
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, T, motion_dim) — sequence of per-frame motion vectors
            padding_mask: (batch, T+1) bool — True=padded/ignore, False=valid
                          Length T+1 because [CLS] is prepended. CLS position
                          should always be False (valid).
        Returns:
            (batch, embed_dim) — embedding from [CLS] token output
        """
        B, T, _ = x.shape

        # Project each frame to d_model
        x = self.input_proj(x)  # (B, T, d_model)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)  # (B, T+1, d_model)

        # Add positional encoding (handle variable T <= seq_len)
        x = x + self.pos_encoding[:, : T + 1, :]

        # Transformer encoder with optional padding mask
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Take [CLS] output (position 0)
        cls_out = x[:, 0, :]  # (B, d_model)

        # Project to shared embedding space
        return self.output_proj(cls_out)  # (B, embed_dim)
```

- [ ] **Step 2: Modify MotionLanguageAligner to support both architectures**

Replace the entire `MotionLanguageAligner` class with:

```python
class MotionLanguageAligner(nn.Module):
    """
    Reasoning Head of GMC-Link that aligns motion features with language features.

    Supports two architectures:
    - "mlp": single-frame MLP projector (original, default)
    - "temporal_transformer": sequence-based transformer encoder with [CLS] token

    Trained with Supervised InfoNCE. At inference, use encode() to get L2-normalized
    embeddings and compute cosine similarity directly.
    """

    def __init__(
        self,
        motion_dim: int = 13,
        lang_dim: int = 384,
        embed_dim: int = 256,
        architecture: str = "mlp",
        seq_len: int = 10,
    ) -> None:
        super().__init__()
        self.architecture = architecture

        if architecture == "temporal_transformer":
            self.motion_projector = TemporalMotionEncoder(
                motion_dim=motion_dim,
                seq_len=seq_len,
                embed_dim=embed_dim,
            )
        else:
            # Original MLP: 13D → 128 → 256 → 512 → embed_dim
            self.motion_projector = nn.Sequential(
                nn.Linear(motion_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, embed_dim),
                nn.LayerNorm(embed_dim),
            )

        # Language Projector: two-layer projection with bottleneck (unchanged)
        self.lang_projector = nn.Sequential(
            nn.Linear(lang_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def encode(
        self,
        motion_feats: torch.Tensor,
        lang_feats: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project motion and language inputs into the shared latent space.

        Args:
            motion_feats: (N, motion_dim) for MLP or (N, T, motion_dim) for transformer.
            lang_feats:   (N, lang_dim) or (M, lang_dim) language embeddings.
            padding_mask: (N, T+1) bool tensor for transformer. None for MLP.

        Returns:
            motion_emb:   (N, embed_dim) L2-normalized motion embeddings.
            lang_emb:     (N, embed_dim) or (M, embed_dim) L2-normalized language embeddings.
        """
        if self.architecture == "temporal_transformer":
            motion_emb = F.normalize(
                self.motion_projector(motion_feats, padding_mask), p=2, dim=-1
            )
        else:
            motion_emb = F.normalize(self.motion_projector(motion_feats), p=2, dim=-1)
        lang_emb = F.normalize(self.lang_projector(lang_feats), p=2, dim=-1)
        return motion_emb, lang_emb

    def forward(
        self,
        motion_feats: torch.Tensor,
        lang_feats: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute cosine similarity scores between motion and language embeddings.

        Args:
            motion_feats: (N, motion_dim) for MLP or (N, T, motion_dim) for transformer.
            lang_feats:   (M, L_dim) language embeddings.
            padding_mask: (N, T+1) bool tensor for transformer. None for MLP.

        Returns:
            scores: (N, M) cosine similarity in [-1, 1].
        """
        motion_emb, lang_emb = self.encode(motion_feats, lang_feats, padding_mask)
        return torch.matmul(motion_emb, lang_emb.t())
```

- [ ] **Step 3: Commit**

```bash
git add gmc_link/alignment.py
git commit -m "feat: add TemporalMotionEncoder with [CLS] token and transformer self-attention"
```

---

### Task 3: Smoke test the new model

**Files:** None (verification only)

- [ ] **Step 1: Test MLP architecture still works**

```bash
conda run -n RMOT python -c "
import torch
from gmc_link.alignment import MotionLanguageAligner

model = MotionLanguageAligner(motion_dim=13, architecture='mlp')
motion = torch.randn(4, 13)
lang = torch.randn(4, 384)
sim = model(motion, lang)
print(f'MLP: sim shape={sim.shape}, range=[{sim.min():.3f}, {sim.max():.3f}]')
assert sim.shape == (4, 4), 'MLP output shape wrong'
print('MLP backward compatibility OK')
"
```

Expected: `MLP: sim shape=torch.Size([4, 4])` and `MLP backward compatibility OK`

- [ ] **Step 2: Test transformer architecture**

```bash
conda run -n RMOT python -c "
import torch
from gmc_link.alignment import MotionLanguageAligner

model = MotionLanguageAligner(motion_dim=13, architecture='temporal_transformer', seq_len=10)
motion_seq = torch.randn(4, 10, 13)  # (batch=4, T=10, 13D)
lang = torch.randn(4, 384)

# No padding mask (full sequences)
sim = model(motion_seq, lang)
print(f'Transformer (no mask): sim shape={sim.shape}')
assert sim.shape == (4, 4)

# With padding mask (track with only 4 frames)
mask = torch.zeros(4, 11, dtype=torch.bool)  # T+1=11, all valid
mask[0, 1:7] = True  # first sample: positions 1-6 are padded (only 4 valid frames)
sim2 = model(motion_seq, lang, padding_mask=mask)
print(f'Transformer (with mask): sim shape={sim2.shape}')
assert sim2.shape == (4, 4)

# Check gradient flow
loss = sim2.sum()
loss.backward()
print(f'Gradient flow OK, CLS grad norm: {model.motion_projector.cls_token.grad.norm():.4f}')

params = sum(p.numel() for p in model.motion_projector.parameters())
print(f'TemporalMotionEncoder params: {params}')
print('All transformer tests passed')
"
```

Expected: All assertions pass, gradient flows through [CLS] token, ~20K params.

---

### Task 4: Add sequence generation to dataset.py

**Files:**
- Modify: `gmc_link/dataset.py`

- [ ] **Step 1: Modify `_generate_positive_pairs` to return track boundary info**

Add a new parameter `track_boundaries` (default `None`) to the function signature. When provided as an empty list, it gets populated during the iteration. Add this to the signature at line 582:

```python
def _generate_positive_pairs(
    track_centroids: Dict[int, Dict[int, Tuple[float, float, float, float]]],
    embedding: np.ndarray,
    expression_id: int,
    frame_gaps: List[int],
    frame_shape: Tuple[int, int],
    seq: str = None,
    frame_dir: str = None,
    orb_engine: Any = None,
    extra_features: List[str] = None,
    all_track_centroids_for_frame: Dict[int, Dict[int, Tuple]] = None,
    track_boundaries: List = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
```

Inside the function, at the start of the track loop (inside `for tid, centroids in track_centroids.items():`), add:

```python
        track_start_idx = len(motion_data)
        track_frame_ids = []
```

After each successful `motion_data.append(motion_vec)` (right after `labels.append(expression_id)`), add:

```python
            track_frame_ids.append(curr_fid)
```

At the end of the track loop (after the inner frame loop, before the next track iteration), add:

```python
        if track_boundaries is not None and track_frame_ids:
            track_boundaries.append({
                "tid": tid,
                "start": track_start_idx,
                "end": len(motion_data),
                "frame_ids": track_frame_ids,
            })
```

- [ ] **Step 2: Add `_vectors_to_sequences` function**

Add this after `_generate_positive_pairs` (before `build_training_data`):

```python
def _vectors_to_sequences(
    motion_data: List[np.ndarray],
    language_data: List[np.ndarray],
    labels: List[int],
    track_boundaries: List[dict],
    seq_len: int = 10,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Convert per-frame motion vectors into sliding-window sequences.

    Args:
        motion_data: flat list of (D,) vectors, ordered by track then frame
        language_data: parallel list of (384,) embeddings
        labels: parallel list of expression IDs
        track_boundaries: list of {tid, start, end, frame_ids} from _generate_positive_pairs
        seq_len: window length T

    Returns:
        seq_motion: list of (seq_len, D) arrays
        seq_masks: list of (seq_len+1,) bool arrays (True=padded, False=valid; +1 for CLS)
        seq_language: list of (384,) arrays
        seq_labels: list of int
    """
    dim = motion_data[0].shape[0] if motion_data else 13
    seq_motion = []
    seq_masks = []
    seq_language = []
    seq_labels = []

    for tb in track_boundaries:
        start, end = tb["start"], tb["end"]
        n_frames = end - start
        if n_frames == 0:
            continue

        vectors = motion_data[start:end]  # already in frame order
        embedding = language_data[start]   # same for all frames of this track
        label = labels[start]

        # Create a sliding window for each valid end position
        for win_end in range(1, n_frames + 1):
            win_start = max(0, win_end - seq_len)
            window = vectors[win_start:win_end]
            n_valid = len(window)

            # Left-pad with zeros if window < seq_len
            if n_valid < seq_len:
                pad = [np.zeros(dim, dtype=np.float32)] * (seq_len - n_valid)
                window = pad + list(window)

            seq = np.stack(window)  # (seq_len, D)

            # Mask: True=padded, False=valid. +1 for CLS (always valid = False)
            mask = np.zeros(seq_len + 1, dtype=bool)
            n_padded = seq_len - n_valid
            if n_padded > 0:
                mask[1 : n_padded + 1] = True  # positions 1..n_padded are padded

            seq_motion.append(seq)
            seq_masks.append(mask)
            seq_language.append(embedding.copy())
            seq_labels.append(label)

    return seq_motion, seq_masks, seq_language, seq_labels
```

- [ ] **Step 3: Add SequenceMotionLanguageDataset and collate function**

Add after `collate_fn`:

```python
class SequenceMotionLanguageDataset(Dataset):
    """
    PyTorch Dataset for temporal contrastive motion-language training.
    Each sample: (motion_sequence, padding_mask, language_embedding, expression_id)
    """

    def __init__(self, motion_seqs, masks, language_data, labels):
        assert len(motion_seqs) == len(masks) == len(language_data) == len(labels)
        self.motion_seqs = motion_seqs
        self.masks = masks
        self.language_data = language_data
        self.labels = labels

    def __len__(self):
        return len(self.motion_seqs)

    def __getitem__(self, idx):
        seq = torch.tensor(self.motion_seqs[idx], dtype=torch.float32)
        mask = torch.tensor(self.masks[idx], dtype=torch.bool)
        lang = torch.tensor(self.language_data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return seq, mask, lang, label


def sequence_collate_fn(batch):
    """Stack into (B, T, D) motion, (B, T+1) mask, (B, 384) language, (B,) labels."""
    seq_batch = torch.stack([item[0] for item in batch], dim=0)
    mask_batch = torch.stack([item[1] for item in batch], dim=0)
    lang_batch = torch.stack([item[2] for item in batch], dim=0)
    label_batch = torch.stack([item[3] for item in batch], dim=0)
    return seq_batch, mask_batch, lang_batch, label_batch
```

- [ ] **Step 4: Modify `build_training_data` to support sequence output**

Add `seq_len: int = 0` parameter to the signature. When `seq_len > 0`, collect track boundaries, convert to sequences, and return sequences instead of flat vectors. Change the return type annotation to handle both cases.

At the signature (around line 742), add the parameter:

```python
def build_training_data(
    data_root: str,
    sequences: List[str],
    text_encoder: Any,
    frame_gaps: List[int] = None,
    frame_shape: Tuple[int, int] = (375, 1242),
    use_group_labels: bool = False,
    extra_features: List[str] = None,
    seq_len: int = 0,
) -> Tuple:
```

Before the expression loop, add:

```python
    all_track_boundaries = [] if seq_len > 0 else None
```

In the `_generate_positive_pairs` call, pass the new parameter:

```python
        m_data, l_data, lbls = _generate_positive_pairs(
            track_centroids,
            embedding,
            expression_id,
            frame_gaps,
            actual_shape,
            seq=seq,
            frame_dir=frame_dir,
            orb_engine=orb_engine,
            extra_features=extra_features,
            all_track_centroids_for_frame=all_frame_track_data,
            track_boundaries=all_track_boundaries if seq_len > 0 else None,
        )
```

Wait — `all_track_boundaries` collects boundaries across all expressions, but the `start`/`end` indices need to be relative to the **global** `motion_data` list (after `.extend()`). Fix by offsetting. Right before the `motion_data.extend(m_data)` call, add:

```python
        # Offset track boundaries to global indices
        if seq_len > 0 and all_track_boundaries is not None:
            offset = len(motion_data)
            # The last len(m_data)-related boundaries were just added to all_track_boundaries
            # by the _generate_positive_pairs call. Offset them.
```

Actually, simpler approach: pass a fresh list to each `_generate_positive_pairs` call, then offset and accumulate:

Replace the `_generate_positive_pairs` call block with:

```python
        per_expr_boundaries = []
        m_data, l_data, lbls = _generate_positive_pairs(
            track_centroids,
            embedding,
            expression_id,
            frame_gaps,
            actual_shape,
            seq=seq,
            frame_dir=frame_dir,
            orb_engine=orb_engine,
            extra_features=extra_features,
            all_track_centroids_for_frame=all_frame_track_data,
            track_boundaries=per_expr_boundaries if seq_len > 0 else None,
        )

        # Offset boundaries to global indices before extending
        if seq_len > 0:
            offset = len(motion_data)
            for tb in per_expr_boundaries:
                tb["start"] += offset
                tb["end"] += offset
            all_track_boundaries.extend(per_expr_boundaries)
```

At the end of `build_training_data`, before the return, add:

```python
    # Convert to sequences if requested
    if seq_len > 0:
        print(f"  Converting to sequences (seq_len={seq_len})...")
        seq_motion, seq_masks, seq_language, seq_labels = _vectors_to_sequences(
            motion_data, language_data, labels, all_track_boundaries, seq_len=seq_len,
        )
        print(f"  Sequences: {len(seq_motion)} (from {len(motion_data)} individual vectors)")
        return seq_motion, seq_masks, seq_language, seq_labels
```

- [ ] **Step 5: Commit**

```bash
git add gmc_link/dataset.py
git commit -m "feat: add sliding-window sequence generation for temporal transformer training"
```

---

### Task 5: Smoke test sequence data pipeline

**Files:** None (verification only)

- [ ] **Step 1: Test sequence generation on seq 0011**

```bash
conda run -n RMOT python -c "
from gmc_link.dataset import build_training_data
from gmc_link.text_utils import TextEncoder
import numpy as np

encoder = TextEncoder(device='cpu')
result = build_training_data(
    data_root='refer-kitti',
    sequences=['0011'],
    text_encoder=encoder.model,
    use_group_labels=True,
    seq_len=10,
)

seq_motion, seq_masks, seq_language, seq_labels = result
print(f'Sequences: {len(seq_motion)}')
print(f'Motion shape: {seq_motion[0].shape}')  # expect (10, 13)
print(f'Mask shape: {seq_masks[0].shape}')      # expect (11,)
print(f'Lang shape: {seq_language[0].shape}')    # expect (384,)
print(f'Mask example (first): {seq_masks[0]}')   # CLS=False, some may be True

# Verify dimensions
assert seq_motion[0].shape == (10, 13), f'Wrong motion shape: {seq_motion[0].shape}'
assert seq_masks[0].shape == (11,), f'Wrong mask shape: {seq_masks[0].shape}'
assert seq_masks[0][0] == False, 'CLS position must always be valid (False)'

# Check a fully-valid sequence has all-False mask
long_seq_idx = None
for i, m in enumerate(seq_masks):
    if not m.any():  # no padding
        long_seq_idx = i
        break
if long_seq_idx is not None:
    print(f'Found fully-valid sequence at index {long_seq_idx}: mask={seq_masks[long_seq_idx]}')

print('Sequence pipeline test PASSED')
" 2>&1 | tail -15
```

Expected: sequences of shape (10, 13), masks of shape (11,), CLS always valid, more samples than the ~22K individual vectors.

---

### Task 6: Wire up training in train.py

**Files:**
- Modify: `gmc_link/train.py`

- [ ] **Step 1: Update imports**

Add to the imports at the top (around line 27):

```python
from gmc_link.dataset import (
    MotionLanguageDataset, SequenceMotionLanguageDataset,
    collate_fn, sequence_collate_fn,
    build_training_data, compute_extra_dims,
)
```

- [ ] **Step 2: Modify `setup_data` to handle sequences**

Add `seq_len: int = 0` parameter to `setup_data` signature. When `seq_len > 0`, use `SequenceMotionLanguageDataset` and `sequence_collate_fn`. Update the function (around line 86):

```python
def setup_data(
    device: torch.device,
    data_root: str,
    sequences: list,
    batch_size: int,
    use_group_labels: bool = False,
    extra_features: list = None,
    seq_len: int = 0,
) -> Optional[DataLoader]:
```

Replace the `build_training_data` call and dataset creation with:

```python
    result = build_training_data(
        data_root=data_root,
        sequences=sequences,
        text_encoder=encoder.model,
        use_group_labels=use_group_labels,
        extra_features=extra_features,
        seq_len=seq_len,
    )

    if seq_len > 0:
        seq_motion, seq_masks, seq_language, seq_labels = result
        if len(seq_motion) == 0:
            return None
        print(f"Total training sequences: {len(seq_motion)}")
        dataset = SequenceMotionLanguageDataset(seq_motion, seq_masks, seq_language, seq_labels)
        chosen_collate = sequence_collate_fn
    else:
        motion_data, language_data, label_ids = result
        if len(motion_data) == 0:
            return None
        print(f"Total training samples: {len(motion_data)}")
        dataset = MotionLanguageDataset(motion_data, language_data, label_ids)
        chosen_collate = collate_fn
```

And update the DataLoader to use `chosen_collate`:

```python
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=chosen_collate,
        ...
    )
```

- [ ] **Step 3: Modify `setup_model_and_optimizer` to accept architecture**

Add `architecture: str = "mlp"` and `seq_len: int = 10` parameters:

```python
def setup_model_and_optimizer(
    device: torch.device, lang_dim: int, learning_rate: float, epochs: int,
    learnable_temp: bool = False, motion_dim: int = 13,
    architecture: str = "mlp", seq_len: int = 10,
) -> Tuple[...]:
```

Update the model creation:

```python
    model = MotionLanguageAligner(
        motion_dim=motion_dim, lang_dim=lang_dim, embed_dim=256,
        architecture=architecture, seq_len=seq_len,
    ).to(device)
```

- [ ] **Step 4: Modify `train_one_epoch` to handle sequences**

The dataloader may yield 3 items (MLP) or 4 items (transformer). Update the loop (around line 54):

```python
    for batch in dataloader:
        if len(batch) == 4:
            motion_features, padding_masks, language_features, expr_ids = batch
            motion_features = motion_features.to(device)
            padding_masks = padding_masks.to(device)
            language_features = language_features.to(device)
            expr_ids = expr_ids.to(device)
            sim_matrix = model(motion_features, language_features, padding_mask=padding_masks)
        else:
            motion_features, language_features, expr_ids = batch
            motion_features = motion_features.to(device)
            language_features = language_features.to(device)
            expr_ids = expr_ids.to(device)
            sim_matrix = model(motion_features, language_features)
```

Keep the rest of the loss/backward/optimizer logic unchanged.

- [ ] **Step 5: Modify `_run_single_stage` to pass architecture/seq_len**

Add `architecture: str = "mlp"` and `seq_len: int = 10` parameters to the signature.

Pass `seq_len` to `setup_data`:

```python
    dataloader = setup_data(device, data_root, sequences, batch_size,
                            use_group_labels=use_group_labels,
                            extra_features=extra_features,
                            seq_len=seq_len if architecture == "temporal_transformer" else 0)
```

Pass `architecture` and `seq_len` to `setup_model_and_optimizer`:

```python
    model, criterion, optimizer, scheduler = setup_model_and_optimizer(
        device, lang_dim, lr, epochs,
        learnable_temp=learnable_temp, motion_dim=motion_dim,
        architecture=architecture, seq_len=seq_len,
    )
```

Save architecture info in the checkpoint (in the save block):

```python
    save_dict = {
        "model": model.state_dict(),
        "temperature": criterion.temperature,
        "motion_dim": motion_dim,
        "extra_features": extra_features,
        "architecture": architecture,
        "seq_len": seq_len if architecture == "temporal_transformer" else None,
    }
```

- [ ] **Step 6: Add `--architecture` and `--seq-len` CLI args to `main()`**

In the argparse section:

```python
    parser.add_argument("--architecture", default="mlp", choices=["mlp", "temporal_transformer"],
                        help="Motion encoder architecture (default: mlp)")
    parser.add_argument("--seq-len", type=int, default=10,
                        help="Sequence length T for temporal_transformer (default: 10)")
```

Pass them through to `_run_single_stage` calls (both the single-stage and curriculum paths):

```python
    _run_single_stage(
        ...,
        architecture=args.architecture,
        seq_len=args.seq_len,
    )
```

- [ ] **Step 7: Commit**

```bash
git add gmc_link/train.py
git commit -m "feat: wire up temporal transformer training with --architecture flag"
```

---

### Task 7: End-to-end smoke test (2 epochs)

**Files:** None (verification only)

- [ ] **Step 1: Run 2-epoch training with transformer**

```bash
conda run -n RMOT python -m gmc_link.train \
    --split v1 --stage 1 --epochs 2 \
    --architecture temporal_transformer --seq-len 10 \
    --save-path /tmp/test_temporal.pth 2>&1 | tail -20
```

Expected: training starts, shows epoch loss/accuracy for 2 epochs, saves checkpoint. No errors.

- [ ] **Step 2: Verify checkpoint contains architecture metadata**

```bash
conda run -n RMOT python -c "
import torch
ckpt = torch.load('/tmp/test_temporal.pth', map_location='cpu')
print('Keys:', list(ckpt.keys()))
print('Architecture:', ckpt.get('architecture'))
print('Seq len:', ckpt.get('seq_len'))
print('Motion dim:', ckpt.get('motion_dim'))
assert ckpt['architecture'] == 'temporal_transformer'
assert ckpt['seq_len'] == 10
print('Checkpoint metadata OK')
"
```

Expected: architecture=temporal_transformer, seq_len=10

- [ ] **Step 3: Verify MLP training still works (backward compatibility)**

```bash
conda run -n RMOT python -m gmc_link.train \
    --split v1 --stage 1 --epochs 2 \
    --save-path /tmp/test_mlp.pth 2>&1 | tail -10
```

Expected: trains normally with MLP architecture (default). No regressions.

---

### Task 8: Update diagnostics for transformer evaluation

**Files:**
- Modify: `diagnostics/diag_gt_cosine_distributions.py`

- [ ] **Step 1: Update model loading to detect architecture**

In `main()`, after loading the checkpoint (around line 202), update the architecture detection:

```python
    checkpoint = torch.load(args.weights, map_location=device)
    extra_features = None
    architecture = "mlp"
    seq_len = 10
    if isinstance(checkpoint, dict):
        motion_dim = checkpoint.get("motion_dim", 13)
        extra_features = checkpoint.get("extra_features", None)
        architecture = checkpoint.get("architecture", "mlp")
        seq_len = checkpoint.get("seq_len", 10) or 10
    else:
        motion_dim = 13

    model = MotionLanguageAligner(
        motion_dim=motion_dim, lang_dim=384, embed_dim=256,
        architecture=architecture, seq_len=seq_len,
    ).to(device)
```

Update the import at the top to include the new class:

```python
from gmc_link.alignment import MotionLanguageAligner
```

- [ ] **Step 2: Add sequence windowing for transformer evaluation**

After the existing `track_motion_vecs` computation (around line 262), add a branch for transformer architecture. Replace the section from "Precompute motion embeddings" through the `track_emb_slices` computation with:

```python
    # ── Precompute motion embeddings for all tracks ──────────────────
    if architecture == "temporal_transformer":
        print("Encoding motion sequences (transformer)...")
        # Group per-track vectors into sliding windows, then batch-encode
        all_seqs = []
        all_masks = []
        track_emb_slices = {}

        for tid, frame_vecs in track_motion_vecs.items():
            vectors = [v for _, v in frame_vecs]
            n_frames = len(vectors)
            dim = vectors[0].shape[0]
            track_start = len(all_seqs)

            for win_end in range(1, n_frames + 1):
                win_start = max(0, win_end - seq_len)
                window = vectors[win_start:win_end]
                n_valid = len(window)

                if n_valid < seq_len:
                    pad = [np.zeros(dim, dtype=np.float32)] * (seq_len - n_valid)
                    window = pad + list(window)

                all_seqs.append(np.stack(window))

                mask = np.zeros(seq_len + 1, dtype=bool)
                n_padded = seq_len - n_valid
                if n_padded > 0:
                    mask[1:n_padded + 1] = True
                all_masks.append(mask)

            track_emb_slices[tid] = (track_start, len(all_seqs))

        all_motion_embs = None
        if all_seqs:
            seq_tensor = torch.tensor(np.array(all_seqs), dtype=torch.float32).to(device)
            mask_tensor = torch.tensor(np.array(all_masks), dtype=torch.bool).to(device)
            with torch.no_grad():
                all_motion_embs = F.normalize(
                    model.motion_projector(seq_tensor, mask_tensor), p=2, dim=-1
                )
            print(f"  Encoded {len(all_seqs)} motion sequences -> {all_motion_embs.shape}")

    else:
        print("Encoding all motion vectors on GPU...")
        all_vecs_list = []
        vec_to_track = []
        for tid, frame_vecs in track_motion_vecs.items():
            for fid, vec in frame_vecs:
                all_vecs_list.append(vec)
                vec_to_track.append((tid, fid))

        all_motion_embs = None
        if all_vecs_list:
            all_vecs_tensor = torch.tensor(np.array(all_vecs_list), dtype=torch.float32).to(device)
            with torch.no_grad():
                all_motion_embs = F.normalize(model.motion_projector(all_vecs_tensor), p=2, dim=-1)
            print(f"  Encoded {len(all_vecs_list)} motion vectors -> {all_motion_embs.shape}")

        track_emb_slices = {}
        idx = 0
        for tid, frame_vecs in track_motion_vecs.items():
            n = len(frame_vecs)
            track_emb_slices[tid] = (idx, idx + n)
            idx += n
```

Also update the language projection to use `model.lang_projector` directly (around line 349):

```python
        with torch.no_grad():
            lang_proj = F.normalize(model.lang_projector(lang_emb.to(device)), p=2, dim=-1)
```

This should already be correct from the existing code.

- [ ] **Step 3: Commit**

```bash
git add diagnostics/diag_gt_cosine_distributions.py
git commit -m "feat: add temporal transformer support to Layer 3 diagnostic"
```

---

### Task 9: Full training and evaluation

**Files:** None (experiment execution)

- [ ] **Step 1: Train temporal transformer (100 epochs, Stage 1)**

```bash
conda run -n RMOT python -m gmc_link.train \
    --split v1 --stage 1 \
    --architecture temporal_transformer --seq-len 10 \
    --save-path gmc_link_weights_v1train_temporal.pth 2>&1 | tail -15
```

Expected: completes 100 epochs, saves weights.

- [ ] **Step 2: Run Layer 3 diagnostic**

```bash
conda run -n RMOT python diagnostics/diag_gt_cosine_distributions.py \
    --weights gmc_link_weights_v1train_temporal.pth 2>&1 | tail -20
```

Expected: prints per-expression AUC table and aggregate results. Compare against baseline AUC 0.779 and separation +0.195.

- [ ] **Step 3: Record results**

Note the Mean AUC and Mean Separation from the diagnostic output. Success criteria:
- Primary: AUC > 0.800
- Secondary: Separation > +0.250

---

### Task 10: Record results in RESEARCH_NOTES.md

**Files:**
- Modify: `RESEARCH_NOTES.md`

- [ ] **Step 1: Add Exp 32 entry**

After the Exp 31 entry, add:

```markdown
### Exp 32: Temporal Transformer Motion Encoder (1-Layer, T=10)

**Branch:** `exp/temporal-transformer`
**Motivation:** Exp 31 showed that enriching the 13D feature vector couldn't break the AUC ~0.79 ceiling. The bottleneck is the single-frame snapshot — a transformer encoder over T=10 consecutive frames can learn temporal patterns (braking, acceleration, stop-start) that a single-frame MLP cannot represent.

**Architecture:**
- TemporalMotionEncoder: Linear(13→64) → [CLS] token → positional encoding → 1-layer TransformerEncoder (d_model=64, 4 heads, ff=128) → [CLS] output → Linear(64→256) + LayerNorm
- Language projector: unchanged (384→256)
- Training: Stage 1 group-level, 100 epochs, lr=1e-3, V1 split
- Data: sliding-window sequences (T=10, stride=1) from per-track per-frame 13D vectors

**Layer 3 GT cosine diagnostic (seq 0011):**

| Model | Mean Sep | Mean AUC | Δ vs Baseline |
|-------|----------|----------|---------------|
| Baseline MLP (13D) | +0.195 | 0.779 | — |
| Temporal Transformer (T=10) | +X.XXX | X.XXX | +X.XXX |

**Key findings:**
[Fill in based on actual results]

**Weight files:** `gmc_link_weights_v1train_temporal.pth` + `*_curves.png`
```

- [ ] **Step 2: Commit**

```bash
git add RESEARCH_NOTES.md
git commit -m "docs: record Exp 32 temporal transformer results"
```
