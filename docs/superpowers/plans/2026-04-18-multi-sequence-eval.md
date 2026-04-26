# Multi-Sequence Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build tooling that aggregates GT-vs-non-GT cosine ranking AUC across the 3 V1 held-out Refer-KITTI sequences (`0005`, `0011`, `0013`) and retrospectively re-evaluates all 11 prior V1 weight files, producing a single comparison report that shows whether seq 0011's ~0.79 "AUC ceiling" is representative or a seq-specific artifact.

**Architecture:** Three-component pipeline with file-system IPC (no Python imports between components). (1) The existing `diagnostics/diag_gt_cosine_distributions.py` is amended additively to persist raw cosine arrays in its `.npz`. (2) A new shell driver `run_multiseq_eval.sh` calls the diagnostic in a seq-outer/weight-inner loop, renaming each output to a namespaced path in `diagnostics/results/multiseq/`. (3) A new Python aggregator `diagnostics/aggregate_multiseq.py` reads those `.npz` files, computes macro and micro AUC per (weight, expression), and emits per-weight JSON / Markdown / PNG plus one cross-weight comparison Markdown. The aggregator is unit-tested against synthetic `.npz` fixtures and is pure post-processing — re-runnable any time without re-inferencing.

**Tech Stack:** Python 3 (conda env `RMOT`), NumPy, SciPy (`mannwhitneyu`), Matplotlib, pytest for the aggregator. Shell driver is plain bash.

**Python interpreter:** `~/miniconda/envs/RMOT/bin/python` — all commands in this plan use this interpreter. Do not use `python` / `python3` (those map to a no-NumPy base env on this machine).

**Spec:** `docs/superpowers/specs/2026-04-18-multi-sequence-eval-design.md`

---

## File Structure

| File | Responsibility | Status |
|---|---|---|
| `diagnostics/diag_gt_cosine_distributions.py` | Per-seq diagnostic. Additive: persist raw cosine arrays in `.npz`. | Modify (1 hunk, ~10 LoC) |
| `diagnostics/aggregate_multiseq.py` | Aggregator: reads per-(seq, weight) `.npz` files, computes macro + micro AUC, writes JSON/MD/PNG + comparison MD. | New (~280 LoC) |
| `tests/test_aggregate_multiseq.py` | Unit tests for aggregator against synthetic fixtures. | New (~180 LoC) |
| `run_multiseq_eval.sh` | Shell driver: seq-outer / weight-inner loop, renames `.npz` files, then calls aggregator. | New (~50 LoC) |
| `diagnostics/results/multiseq/` | Output directory for all per-(seq, weight) `.npz` files and aggregate artifacts. | New dir |
| `.gitignore` | Ignore `diagnostics/results/multiseq/*.npz` and `*.png`, keep `*.md` / `*.json` tracked. | Modify (+2 lines) |
| `RESEARCH_NOTES.md` | New section interpreting multi-seq results against Exp 30–32 conclusions. | Modify (append after full run) |

No changes to: `gmc_link/*`, training pipeline, other diagnostics, weight files.

---

## Task 1: Branch setup and output directory

**Files:**
- Create: `diagnostics/results/multiseq/.gitkeep`
- Modify: `.gitignore`

- [ ] **Step 1: Create branch off main**

```bash
git fetch origin
git checkout -b exp/multi-seq-eval origin/main
```

Expected: on branch `exp/multi-seq-eval`, working tree clean (apart from any pre-existing untracked files from the parent session).

- [ ] **Step 2: Create output directory with a tracked `.gitkeep`**

```bash
mkdir -p diagnostics/results/multiseq
touch diagnostics/results/multiseq/.gitkeep
```

- [ ] **Step 3: Append ignore rules to `.gitignore`**

Show the existing tail of `.gitignore` with the Read tool first so you append in the right place. Append these two lines at the end of the file:

```
diagnostics/results/multiseq/*.npz
diagnostics/results/multiseq/*.png
```

Rationale: `.npz` and `.png` are regenerable from the diagnostic runs; `.json` and `.md` are the committed artifacts of each retrospective run.

- [ ] **Step 4: Verify `.gitkeep` is tracked and `.npz` is ignored**

```bash
git add .gitignore diagnostics/results/multiseq/.gitkeep
git status --porcelain
```

Expected output contains:
```
M  .gitignore
A  diagnostics/results/multiseq/.gitkeep
```

and no mention of any `.npz` file in that directory.

- [ ] **Step 5: Commit**

```bash
git commit -m "chore: scaffold multiseq results dir and ignore rules

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2: Amend diagnostic to persist raw cosine arrays

**Files:**
- Modify: `diagnostics/diag_gt_cosine_distributions.py:407-489`

The diagnostic already computes `gt_cosines` and `non_gt_cosines` per expression inside the loop (lines 407–419) but only stores summary stats in `per_expr_results`. We add a parallel list that holds the raw arrays, and extend `np.savez` to persist them. Math, logging, and plots are unchanged — this is purely adding saved data.

- [ ] **Step 1: Write the verification script first** (no pytest; this is an integration check against a real `.npz`)

Create `tests/test_diag_npz_schema.py`:

```python
"""Verify the diagnostic persists raw cosine arrays in addition to summary results."""
import numpy as np
from pathlib import Path


def test_npz_contains_raw_cosines():
    npz_path = Path("diagnostics/results/layer3_gt_cosine_0011.npz")
    assert npz_path.exists(), (
        f"{npz_path} not found. Run the diagnostic first: "
        f"~/miniconda/envs/RMOT/bin/python diagnostics/diag_gt_cosine_distributions.py "
        f"--weights gmc_link_weights_v1train_stage1.pth --seq 0011"
    )
    d = np.load(npz_path, allow_pickle=True)

    # Existing key still present
    assert "results" in d.files, "regression: 'results' key missing"
    results = d["results"].tolist()
    assert isinstance(results, list) and len(results) > 0
    assert set(results[0].keys()) >= {
        "sentence", "n_gt", "n_nongt", "gt_mean", "gt_std",
        "nongt_mean", "nongt_std", "separation", "auc",
    }

    # New keys
    assert "gt_cosines_by_expr" in d.files, "missing new key 'gt_cosines_by_expr'"
    assert "nongt_cosines_by_expr" in d.files, "missing new key 'nongt_cosines_by_expr'"

    gt_list = d["gt_cosines_by_expr"]  # object array, same length as results
    nongt_list = d["nongt_cosines_by_expr"]
    assert len(gt_list) == len(results)
    assert len(nongt_list) == len(results)

    # Order and counts must match results[i]
    for i, r in enumerate(results):
        assert len(gt_list[i]) == r["n_gt"], (
            f"expr {i}: raw gt length {len(gt_list[i])} != n_gt {r['n_gt']}"
        )
        assert len(nongt_list[i]) == r["n_nongt"], (
            f"expr {i}: raw nongt length {len(nongt_list[i])} != n_nongt {r['n_nongt']}"
        )
        # Means must match summary within float tolerance
        np.testing.assert_allclose(
            float(np.mean(gt_list[i])), r["gt_mean"], rtol=1e-5, atol=1e-7,
        )
        np.testing.assert_allclose(
            float(np.mean(nongt_list[i])), r["nongt_mean"], rtol=1e-5, atol=1e-7,
        )
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
~/miniconda/envs/RMOT/bin/python -m pytest tests/test_diag_npz_schema.py -v
```

Expected: FAIL. If the existing seq-0011 `.npz` exists (it does, from Exp 30), the test fails with `AssertionError: missing new key 'gt_cosines_by_expr'`. If it doesn't exist, the test fails on the `npz_path.exists()` guard.

- [ ] **Step 3: Read the diagnostic around the per-expression loop**

```bash
```

Use the Read tool on `diagnostics/diag_gt_cosine_distributions.py` lines 392–490 so you see the exact current structure before editing.

- [ ] **Step 4: Add parallel lists collecting raw cosines alongside summary stats**

Locate this block (around line 392):

```python
    per_expr_results = []

    for ei, expr in enumerate(motion_exprs):
```

Change it to:

```python
    per_expr_results = []
    gt_cosines_by_expr = []
    nongt_cosines_by_expr = []

    for ei, expr in enumerate(motion_exprs):
```

Then locate the summary-append block (around line 432):

```python
            per_expr_results.append({
                "sentence": sentence,
                "n_gt": len(gt_cosines),
                "n_nongt": len(non_gt_cosines),
                "gt_mean": float(gt_arr.mean()),
                "gt_std": float(gt_arr.std()),
                "nongt_mean": float(nongt_arr.mean()),
                "nongt_std": float(nongt_arr.std()),
                "separation": float(separation),
                "auc": float(auc),
            })
```

Add the two `append` lines immediately after it, still inside the `if gt_cosines and non_gt_cosines:` branch:

```python
            per_expr_results.append({
                "sentence": sentence,
                "n_gt": len(gt_cosines),
                "n_nongt": len(non_gt_cosines),
                "gt_mean": float(gt_arr.mean()),
                "gt_std": float(gt_arr.std()),
                "nongt_mean": float(nongt_arr.mean()),
                "nongt_std": float(nongt_arr.std()),
                "separation": float(separation),
                "auc": float(auc),
            })
            gt_cosines_by_expr.append(gt_arr.astype(np.float32))
            nongt_cosines_by_expr.append(nongt_arr.astype(np.float32))
```

Invariant: `len(per_expr_results) == len(gt_cosines_by_expr) == len(nongt_cosines_by_expr)`. Because both appends live inside the same conditional and neither branch appends alone, this holds by construction.

- [ ] **Step 5: Extend the `np.savez` call**

Locate (around line 487–489):

```python
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, f"layer3_gt_cosine_{args.seq}.npz")
    np.savez(save_path, results=per_expr_results)
```

Replace with:

```python
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, f"layer3_gt_cosine_{args.seq}.npz")
    np.savez(
        save_path,
        results=per_expr_results,
        gt_cosines_by_expr=np.array(gt_cosines_by_expr, dtype=object),
        nongt_cosines_by_expr=np.array(nongt_cosines_by_expr, dtype=object),
    )
```

The `dtype=object` wrapper is required because inner arrays have different lengths (ragged).

- [ ] **Step 6: Regenerate the seq-0011 `.npz` so the test has something to assert against**

```bash
~/miniconda/envs/RMOT/bin/python diagnostics/diag_gt_cosine_distributions.py \
  --weights gmc_link_weights_v1train_stage1.pth --seq 0011
```

Expected: the script prints its usual per-expression table and "Saved: diagnostics/results/layer3_gt_cosine_0011.npz". Takes ~3–8 minutes (ORB homography is the slow step).

- [ ] **Step 7: Run the verification test**

```bash
~/miniconda/envs/RMOT/bin/python -m pytest tests/test_diag_npz_schema.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add diagnostics/diag_gt_cosine_distributions.py tests/test_diag_npz_schema.py
git commit -m "feat(diag): persist raw cosine arrays in layer3 npz for pooling

Adds gt_cosines_by_expr and nongt_cosines_by_expr to the saved .npz,
parallel to the existing results list. Enables cross-seq micro AUC
pooling in the upcoming multi-seq aggregator without recomputing
cosines. Math, logging, and plots are unchanged.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: Aggregator skeleton — module, CLI shell, AUC core (macro + micro)

**Files:**
- Create: `diagnostics/aggregate_multiseq.py`
- Create: `tests/test_aggregate_multiseq.py`

This task builds the computational core: given a dict of per-(seq, expression) raw cosine arrays, produce macro and micro AUCs. We drive it with a synthetic `.npz` fixture so we do not wait 45 minutes on inference.

- [ ] **Step 1: Write fixture builder and failing tests**

Create `tests/test_aggregate_multiseq.py`:

```python
"""Unit tests for the multi-sequence aggregator.

Uses synthetic .npz fixtures that match the schema produced by
diagnostics/diag_gt_cosine_distributions.py (Task 2). No real inference.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def synthetic_npz_dir(tmp_path: Path) -> Path:
    """Write fake per-(seq, model) .npz files under tmp_path/multiseq/.

    Three seqs, two weights. Deterministic cosines so AUC is predictable.

    Model A: GT scores > non-GT scores on all seqs/exprs (clean positive).
    Model B: GT scores < non-GT scores on seq 0005 only (seq-dependent).
    """
    rng = np.random.default_rng(42)
    out_dir = tmp_path / "multiseq"
    out_dir.mkdir()

    seqs = ["0005", "0011", "0013"]
    sentences = ["moving cars", "parking cars", "cars in left"]

    def _build(model_tag: str, invert_on: set[str]) -> None:
        for s in seqs:
            results = []
            gt_list = []
            nongt_list = []
            for sent in sentences:
                # 20 GT, 80 non-GT per expression per seq — realistic-ish ratio
                base_gt = rng.normal(0.3, 0.1, size=20).astype(np.float32)
                base_nongt = rng.normal(0.1, 0.1, size=80).astype(np.float32)
                if s in invert_on:
                    gt, nongt = base_nongt[:20], base_gt.tolist() + rng.normal(0.1, 0.1, size=60).astype(np.float32).tolist()
                    gt = np.array(gt, dtype=np.float32)
                    nongt = np.array(nongt, dtype=np.float32)
                else:
                    gt, nongt = base_gt, base_nongt
                results.append({
                    "sentence": sent,
                    "n_gt": int(len(gt)),
                    "n_nongt": int(len(nongt)),
                    "gt_mean": float(gt.mean()),
                    "gt_std": float(gt.std()),
                    "nongt_mean": float(nongt.mean()),
                    "nongt_std": float(nongt.std()),
                    "separation": float(gt.mean() - nongt.mean()),
                    "auc": 0.0,  # aggregator recomputes from raw arrays
                })
                gt_list.append(gt)
                nongt_list.append(nongt)
            path = out_dir / f"layer3_{s}_{model_tag}.npz"
            np.savez(
                path,
                results=results,
                gt_cosines_by_expr=np.array(gt_list, dtype=object),
                nongt_cosines_by_expr=np.array(nongt_list, dtype=object),
            )

    _build("model_A", invert_on=set())
    _build("model_B", invert_on={"0005"})
    return out_dir


def test_load_per_seq_expressions(synthetic_npz_dir: Path):
    from diagnostics.aggregate_multiseq import load_per_seq_expressions
    data = load_per_seq_expressions(synthetic_npz_dir, "model_A", ["0005", "0011", "0013"])
    # keys: expression sentence; values: dict seq -> (gt_arr, nongt_arr)
    assert set(data.keys()) == {"moving cars", "parking cars", "cars in left"}
    assert set(data["moving cars"].keys()) == {"0005", "0011", "0013"}
    gt, nongt = data["moving cars"]["0005"]
    assert gt.shape == (20,)
    assert nongt.shape == (80,)


def test_compute_per_seq_auc_high_when_gt_dominates(synthetic_npz_dir: Path):
    from diagnostics.aggregate_multiseq import (
        load_per_seq_expressions, compute_per_seq_auc,
    )
    data = load_per_seq_expressions(synthetic_npz_dir, "model_A", ["0005", "0011", "0013"])
    for sent in data:
        for s in ["0005", "0011", "0013"]:
            gt, nongt = data[sent][s]
            auc = compute_per_seq_auc(gt, nongt)
            assert auc > 0.85, (
                f"model_A {sent}/{s}: GT dominates so AUC should be high; got {auc:.3f}"
            )


def test_compute_per_seq_auc_low_when_inverted(synthetic_npz_dir: Path):
    from diagnostics.aggregate_multiseq import (
        load_per_seq_expressions, compute_per_seq_auc,
    )
    data = load_per_seq_expressions(synthetic_npz_dir, "model_B", ["0005", "0011", "0013"])
    for sent in data:
        gt, nongt = data[sent]["0005"]
        auc = compute_per_seq_auc(gt, nongt)
        assert auc < 0.15, f"model_B inverted on 0005; AUC should be low; got {auc:.3f}"
        # Non-inverted seqs still good
        gt2, nongt2 = data[sent]["0011"]
        auc2 = compute_per_seq_auc(gt2, nongt2)
        assert auc2 > 0.85, f"model_B non-inverted on 0011; got {auc2:.3f}"


def test_macro_aggregation_math(synthetic_npz_dir: Path):
    from diagnostics.aggregate_multiseq import (
        load_per_seq_expressions, aggregate_expression,
    )
    data = load_per_seq_expressions(synthetic_npz_dir, "model_A", ["0005", "0011", "0013"])
    agg = aggregate_expression(data["moving cars"], seqs=["0005", "0011", "0013"])
    # macro = mean of per-seq AUCs
    expected_macro = np.mean([agg["auc_per_seq"][s] for s in ["0005", "0011", "0013"]])
    np.testing.assert_allclose(agg["auc_macro_mean"], expected_macro, rtol=1e-6)
    # std is the sample std across seqs
    expected_std = np.std([agg["auc_per_seq"][s] for s in ["0005", "0011", "0013"]])
    np.testing.assert_allclose(agg["auc_macro_std"], expected_std, rtol=1e-6)


def test_micro_aggregation_pools_before_auc(synthetic_npz_dir: Path):
    """Micro AUC must come from concatenated arrays, not averaged per-seq AUCs."""
    from diagnostics.aggregate_multiseq import (
        load_per_seq_expressions, aggregate_expression, compute_per_seq_auc,
    )
    data = load_per_seq_expressions(synthetic_npz_dir, "model_B", ["0005", "0011", "0013"])
    # On model_B, "moving cars" has one inverted seq (0005) and two clean.
    per_seq = data["moving cars"]
    gt_all = np.concatenate([per_seq[s][0] for s in ["0005", "0011", "0013"]])
    nongt_all = np.concatenate([per_seq[s][1] for s in ["0005", "0011", "0013"]])
    expected_micro = compute_per_seq_auc(gt_all, nongt_all)

    agg = aggregate_expression(per_seq, seqs=["0005", "0011", "0013"])
    np.testing.assert_allclose(agg["auc_micro"], expected_micro, rtol=1e-6)
    # sanity: macro ≠ micro when distributions differ
    assert abs(agg["auc_macro_mean"] - agg["auc_micro"]) > 1e-3


def test_expression_missing_from_seq_is_skipped_in_macro(tmp_path: Path):
    """If an expression has no data in a seq, macro skips that seq; micro uses pooled."""
    from diagnostics.aggregate_multiseq import aggregate_expression
    rng = np.random.default_rng(0)
    per_seq = {
        "0005": (rng.normal(0.3, 0.1, 20).astype(np.float32),
                 rng.normal(0.1, 0.1, 80).astype(np.float32)),
        "0011": (np.array([], dtype=np.float32), np.array([], dtype=np.float32)),
        "0013": (rng.normal(0.3, 0.1, 25).astype(np.float32),
                 rng.normal(0.1, 0.1, 75).astype(np.float32)),
    }
    agg = aggregate_expression(per_seq, seqs=["0005", "0011", "0013"])
    # 0011 has no samples — excluded from macro
    assert "0011" not in agg["auc_per_seq"] or agg["auc_per_seq"]["0011"] is None
    assert agg["gt_count_per_seq"]["0011"] == 0
    # macro computed over 0005 + 0013 only
    valid_aucs = [agg["auc_per_seq"][s] for s in ["0005", "0013"]]
    np.testing.assert_allclose(agg["auc_macro_mean"], np.mean(valid_aucs), rtol=1e-6)
```

- [ ] **Step 2: Run the tests to verify all fail with ImportError**

```bash
~/miniconda/envs/RMOT/bin/python -m pytest tests/test_aggregate_multiseq.py -v
```

Expected: 6 failures, all with `ModuleNotFoundError: No module named 'diagnostics.aggregate_multiseq'` or similar.

- [ ] **Step 3: Create the aggregator module with the core helpers**

Create `diagnostics/aggregate_multiseq.py`:

```python
#!/usr/bin/env python3
"""Multi-sequence aggregator for Layer 3 GT cosine diagnostics.

Reads per-(seq, weight) .npz files produced by
diag_gt_cosine_distributions.py and emits per-weight JSON + Markdown + PNG
aggregates plus a single comparison Markdown across all weights.

See docs/superpowers/specs/2026-04-18-multi-sequence-eval-design.md for the
full design. This module is pure post-processing — it does not run inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import mannwhitneyu


def _npz_path(results_dir: Path, seq: str, model_tag: str) -> Path:
    return results_dir / f"layer3_{seq}_{model_tag}.npz"


def load_per_seq_expressions(
    results_dir: Path,
    model_tag: str,
    seqs: Iterable[str],
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Return {expression: {seq: (gt_cosines, nongt_cosines)}} for one weight.

    Missing (seq, expression) pairs are represented by empty arrays so callers
    can treat them uniformly.
    """
    seqs = list(seqs)
    # First pass: collect the union of expressions across seqs (they should
    # be identical in practice but V1 expression files vary by seq).
    per_seq_raw: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    all_sentences: set[str] = set()
    for s in seqs:
        path = _npz_path(Path(results_dir), s, model_tag)
        if not path.exists():
            per_seq_raw[s] = {}
            continue
        d = np.load(path, allow_pickle=True)
        results = d["results"].tolist()
        gt_list = d["gt_cosines_by_expr"]
        nongt_list = d["nongt_cosines_by_expr"]
        seq_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for i, r in enumerate(results):
            sent = r["sentence"]
            seq_map[sent] = (np.asarray(gt_list[i]), np.asarray(nongt_list[i]))
            all_sentences.add(sent)
        per_seq_raw[s] = seq_map

    # Second pass: pivot to expression-major, filling missing with empty arrays.
    out: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    empty = (np.array([], dtype=np.float32), np.array([], dtype=np.float32))
    for sent in sorted(all_sentences):
        out[sent] = {}
        for s in seqs:
            out[sent][s] = per_seq_raw.get(s, {}).get(sent, empty)
    return out


def compute_per_seq_auc(gt: np.ndarray, nongt: np.ndarray) -> float | None:
    """Ranking AUC: P(score_gt > score_nongt) via Mann-Whitney U.

    Returns None when either pool is empty (AUC undefined).
    """
    if len(gt) == 0 or len(nongt) == 0:
        return None
    try:
        U, _ = mannwhitneyu(gt, nongt, alternative="greater")
    except ValueError:
        return 0.5
    return float(U / (len(gt) * len(nongt)))


def aggregate_expression(
    per_seq: dict[str, tuple[np.ndarray, np.ndarray]],
    seqs: Iterable[str],
) -> dict:
    """Compute macro + micro AUCs and counts for one expression across seqs.

    Macro: mean of per-seq AUCs (seqs with empty pools excluded).
    Micro: AUC of the pooled (concatenated) arrays across all seqs.
    """
    seqs = list(seqs)
    auc_per_seq: dict[str, float | None] = {}
    gt_count_per_seq: dict[str, int] = {}
    nongt_count_per_seq: dict[str, int] = {}
    gt_pooled_parts: list[np.ndarray] = []
    nongt_pooled_parts: list[np.ndarray] = []

    for s in seqs:
        gt, nongt = per_seq[s]
        gt_count_per_seq[s] = int(len(gt))
        nongt_count_per_seq[s] = int(len(nongt))
        auc_per_seq[s] = compute_per_seq_auc(gt, nongt)
        if len(gt) > 0:
            gt_pooled_parts.append(gt)
        if len(nongt) > 0:
            nongt_pooled_parts.append(nongt)

    valid_aucs = [a for a in auc_per_seq.values() if a is not None]
    auc_macro_mean = float(np.mean(valid_aucs)) if valid_aucs else None
    auc_macro_std = float(np.std(valid_aucs)) if len(valid_aucs) >= 2 else None

    gt_pooled = np.concatenate(gt_pooled_parts) if gt_pooled_parts else np.array([])
    nongt_pooled = np.concatenate(nongt_pooled_parts) if nongt_pooled_parts else np.array([])
    auc_micro = compute_per_seq_auc(gt_pooled, nongt_pooled)

    return {
        "auc_per_seq": auc_per_seq,
        "auc_macro_mean": auc_macro_mean,
        "auc_macro_std": auc_macro_std,
        "auc_micro": auc_micro,
        "gt_count_per_seq": gt_count_per_seq,
        "nongt_count_per_seq": nongt_count_per_seq,
    }
```

Add a stub `__main__` guard at the bottom — CLI wiring comes in Task 8:

```python
if __name__ == "__main__":  # pragma: no cover — filled in Task 8
    raise SystemExit("CLI entrypoint not yet implemented; see Task 8 of the plan.")
```

- [ ] **Step 4: Add `__init__.py` so tests can import `diagnostics.aggregate_multiseq`**

```bash
```

Check if `diagnostics/__init__.py` exists first (Read tool). If not:

```bash
touch diagnostics/__init__.py
```

- [ ] **Step 5: Run the tests to verify all pass**

```bash
~/miniconda/envs/RMOT/bin/python -m pytest tests/test_aggregate_multiseq.py -v
```

Expected: 6 passes.

If `test_compute_per_seq_auc_low_when_inverted` specifically fails with something like `auc = 0.5`, re-check the fixture: `model_B` should have GT and non-GT swapped only on seq 0005, and `compute_per_seq_auc` uses `alternative="greater"`. An AUC < 0.15 confirms the inversion survived.

- [ ] **Step 6: Commit**

```bash
git add diagnostics/__init__.py diagnostics/aggregate_multiseq.py tests/test_aggregate_multiseq.py
git commit -m "feat(aggregate): core AUC aggregation across held-out seqs

Adds load_per_seq_expressions, compute_per_seq_auc, and
aggregate_expression. Macro averages per-seq AUCs; micro pools the
raw cosine arrays before computing a single Mann-Whitney U. Covered
by synthetic-fixture unit tests.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: Aggregator — build per-weight record and JSON writer

**Files:**
- Modify: `diagnostics/aggregate_multiseq.py`
- Modify: `tests/test_aggregate_multiseq.py`

- [ ] **Step 1: Add failing tests for the per-weight record builder and JSON writer**

Append to `tests/test_aggregate_multiseq.py`:

```python
def test_build_weight_record_structure(synthetic_npz_dir: Path):
    from diagnostics.aggregate_multiseq import build_weight_record
    rec = build_weight_record(
        results_dir=synthetic_npz_dir,
        model_tag="model_A",
        weights_path="fake.pth",
        seqs=["0005", "0011", "0013"],
    )
    assert rec["model_tag"] == "model_A"
    assert rec["weights_path"] == "fake.pth"
    assert rec["sequences"] == ["0005", "0011", "0013"]
    # All three synthetic expressions present
    assert set(rec["per_expression"].keys()) == {
        "moving cars", "parking cars", "cars in left",
    }
    # Headline: micro mean ≥ macro mean only if distributions match; both
    # should be > 0.85 for the clean model.
    assert rec["headline"]["mean_auc_micro"] > 0.85
    assert rec["headline"]["mean_auc_macro"] > 0.85
    assert rec["headline"]["n_expressions"] == 3
    assert rec["headline"]["std_across_seqs"] is not None


def test_headline_excludes_single_seq_expressions_from_macro(tmp_path: Path):
    """An expression that exists in only 1 seq has no macro std; exclude it from macro headline."""
    from diagnostics.aggregate_multiseq import build_weight_record
    rng = np.random.default_rng(1)
    out_dir = tmp_path / "multiseq"
    out_dir.mkdir()
    # Seq 0005 has both expressions; 0011 and 0013 have only "common".
    for s in ["0005", "0011", "0013"]:
        results = []
        gt_list, nongt_list = [], []
        sentences = ["common", "only_in_0005"] if s == "0005" else ["common"]
        for sent in sentences:
            gt = rng.normal(0.3, 0.1, 20).astype(np.float32)
            nongt = rng.normal(0.1, 0.1, 80).astype(np.float32)
            results.append({
                "sentence": sent,
                "n_gt": 20, "n_nongt": 80,
                "gt_mean": float(gt.mean()), "gt_std": float(gt.std()),
                "nongt_mean": float(nongt.mean()), "nongt_std": float(nongt.std()),
                "separation": float(gt.mean() - nongt.mean()), "auc": 0.0,
            })
            gt_list.append(gt)
            nongt_list.append(nongt)
        np.savez(
            out_dir / f"layer3_{s}_partial.npz",
            results=results,
            gt_cosines_by_expr=np.array(gt_list, dtype=object),
            nongt_cosines_by_expr=np.array(nongt_list, dtype=object),
        )

    rec = build_weight_record(
        results_dir=out_dir, model_tag="partial", weights_path="p.pth",
        seqs=["0005", "0011", "0013"],
    )
    # "only_in_0005" is excluded from macro headline (<2 seqs) but present
    # in per_expression map; and in micro.
    assert "only_in_0005" in rec["per_expression"]
    # macro headline should be computed over 1 expression only ("common")
    assert rec["headline"]["n_expressions_macro"] == 1
    # micro includes both
    assert rec["headline"]["n_expressions"] == 2


def test_write_weight_json_roundtrip(tmp_path: Path, synthetic_npz_dir: Path):
    from diagnostics.aggregate_multiseq import (
        build_weight_record, write_weight_json,
    )
    rec = build_weight_record(
        results_dir=synthetic_npz_dir, model_tag="model_A",
        weights_path="fake.pth", seqs=["0005", "0011", "0013"],
    )
    out = tmp_path / "layer3_multiseq_model_A.json"
    write_weight_json(rec, out)
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded["model_tag"] == "model_A"
    assert loaded["per_expression"]["moving cars"]["auc_per_seq"]["0005"] is not None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
~/miniconda/envs/RMOT/bin/python -m pytest tests/test_aggregate_multiseq.py -v
```

Expected: 3 new failures with `AttributeError: module 'diagnostics.aggregate_multiseq' has no attribute 'build_weight_record'` (and similar for `write_weight_json`).

- [ ] **Step 3: Implement `build_weight_record` and `write_weight_json` in `diagnostics/aggregate_multiseq.py`**

Add below `aggregate_expression`:

```python
def build_weight_record(
    results_dir: Path,
    model_tag: str,
    weights_path: str,
    seqs: Iterable[str],
) -> dict:
    """Compute the full per-weight record (matches spec's JSON schema)."""
    seqs = list(seqs)
    data = load_per_seq_expressions(results_dir, model_tag, seqs)

    per_expression: dict[str, dict] = {}
    for sent, per_seq in data.items():
        per_expression[sent] = aggregate_expression(per_seq, seqs)

    # Headline: mean over expressions. Macro headline uses only expressions
    # defined in ≥2 seqs (std is otherwise meaningless).
    micro_values = [
        v["auc_micro"] for v in per_expression.values() if v["auc_micro"] is not None
    ]
    macro_values_full = []
    for v in per_expression.values():
        n_seqs_with_data = sum(
            1 for a in v["auc_per_seq"].values() if a is not None
        )
        if n_seqs_with_data >= 2 and v["auc_macro_mean"] is not None:
            macro_values_full.append(v["auc_macro_mean"])

    headline = {
        "mean_auc_micro": float(np.mean(micro_values)) if micro_values else None,
        "mean_auc_macro": (
            float(np.mean(macro_values_full)) if macro_values_full else None
        ),
        "std_across_seqs": (
            float(np.std(macro_values_full)) if len(macro_values_full) >= 2 else None
        ),
        "n_expressions": len(micro_values),
        "n_expressions_macro": len(macro_values_full),
    }

    return {
        "model_tag": model_tag,
        "weights_path": weights_path,
        "sequences": seqs,
        "per_expression": per_expression,
        "headline": headline,
    }


def write_weight_json(record: dict, out_path: Path) -> None:
    """Write the per-weight record to disk as pretty JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
~/miniconda/envs/RMOT/bin/python -m pytest tests/test_aggregate_multiseq.py -v
```

Expected: 9 passes total (6 from Task 3, 3 new).

- [ ] **Step 5: Commit**

```bash
git add diagnostics/aggregate_multiseq.py tests/test_aggregate_multiseq.py
git commit -m "feat(aggregate): per-weight record builder and JSON writer

Builds the full per-weight record per the spec's JSON schema. Macro
headline averages only over expressions present in ≥2 seqs so the std
is meaningful; micro headline uses all expressions.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5: Aggregator — per-weight Markdown writer

**Files:**
- Modify: `diagnostics/aggregate_multiseq.py`
- Modify: `tests/test_aggregate_multiseq.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_aggregate_multiseq.py`:

```python
def test_write_weight_markdown_has_expected_sections(
    tmp_path: Path, synthetic_npz_dir: Path,
):
    from diagnostics.aggregate_multiseq import (
        build_weight_record, write_weight_markdown,
    )
    rec = build_weight_record(
        results_dir=synthetic_npz_dir, model_tag="model_A",
        weights_path="fake.pth", seqs=["0005", "0011", "0013"],
    )
    out = tmp_path / "layer3_multiseq_model_A.md"
    # legacy_seq_0011 is a plain number (the legacy single-seq AUC for
    # continuity with Exp 30-32). Passed in by the CLI; for test purposes
    # we pass a sentinel.
    write_weight_markdown(rec, out, legacy_seq_0011_auc=0.779)
    text = out.read_text()
    assert "# Multi-Sequence Eval: model_A" in text
    assert "## What this measures" in text
    assert "AUC = probability" in text  # interpretive gloss baked in
    assert "0.50 = chance" in text
    assert "## Headline" in text
    assert "Mean AUC (micro" in text
    assert "Mean AUC (macro" in text
    assert "Seq-0011 only (legacy" in text
    assert "0.779" in text  # legacy number
    assert "## Per-expression breakdown" in text
    # Table headers
    assert "| Expression | 0005 | 0011 | 0013 | macro μ ± σ | micro | GT counts |" in text
    # Row for one of the expressions
    assert "moving cars" in text
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
~/miniconda/envs/RMOT/bin/python -m pytest tests/test_aggregate_multiseq.py::test_write_weight_markdown_has_expected_sections -v
```

Expected: `AttributeError: ... has no attribute 'write_weight_markdown'`.

- [ ] **Step 3: Implement `write_weight_markdown`**

Append to `diagnostics/aggregate_multiseq.py`:

```python
_AUC_GLOSS = (
    "AUC = probability that a randomly chosen GT-matching track gets a higher "
    "cosine score than a randomly chosen non-matching track, for a given "
    "expression. 0.50 = chance, 1.00 = perfect. < 0.50 means inverted."
)


def _fmt(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "—"
    return f"{x:.{digits}f}"


def write_weight_markdown(
    record: dict,
    out_path: Path,
    legacy_seq_0011_auc: float | None,
) -> None:
    """Write the human-readable per-weight Markdown report."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tag = record["model_tag"]
    seqs = record["sequences"]
    hl = record["headline"]
    per_expr = record["per_expression"]

    lines: list[str] = []
    lines.append(f"# Multi-Sequence Eval: {tag}")
    lines.append("")
    lines.append("## What this measures")
    lines.append(_AUC_GLOSS)
    lines.append("")
    lines.append("## Headline")
    lines.append(
        f"- Mean AUC (micro, pooled across {len(seqs)} seqs): "
        f"**{_fmt(hl['mean_auc_micro'])}** "
        f"(over {hl['n_expressions']} expressions)"
    )
    macro_str = _fmt(hl["mean_auc_macro"])
    std_str = _fmt(hl["std_across_seqs"])
    lines.append(
        f"- Mean AUC (macro, per-seq averaged):     "
        f"**{macro_str}** ± {std_str} "
        f"(over {hl['n_expressions_macro']} expressions in ≥2 seqs)"
    )
    lines.append(
        f"- Seq-0011 only (legacy, for continuity): **{_fmt(legacy_seq_0011_auc)}**"
    )
    lines.append("")
    lines.append("## Per-expression breakdown")
    header = "| Expression | " + " | ".join(seqs) + " | macro μ ± σ | micro | GT counts |"
    lines.append(header)
    sep = "|" + "|".join(["---"] * (len(seqs) + 4)) + "|"
    lines.append(sep)

    def _sort_key(kv):
        sent, agg = kv
        v = agg["auc_micro"]
        return (-v if v is not None else 1.0, sent)

    for sent, agg in sorted(per_expr.items(), key=_sort_key):
        cells = [sent]
        for s in seqs:
            cells.append(_fmt(agg["auc_per_seq"].get(s), digits=3))
        cells.append(
            f"{_fmt(agg['auc_macro_mean'])} ± {_fmt(agg['auc_macro_std'])}"
        )
        cells.append(_fmt(agg["auc_micro"]))
        gt_counts = "/".join(str(agg["gt_count_per_seq"].get(s, 0)) for s in seqs)
        cells.append(gt_counts)
        lines.append("| " + " | ".join(cells) + " |")

    out_path.write_text("\n".join(lines) + "\n")
```

- [ ] **Step 4: Run tests**

```bash
~/miniconda/envs/RMOT/bin/python -m pytest tests/test_aggregate_multiseq.py -v
```

Expected: 10 passes.

- [ ] **Step 5: Commit**

```bash
git add diagnostics/aggregate_multiseq.py tests/test_aggregate_multiseq.py
git commit -m "feat(aggregate): per-weight Markdown report writer

Renders headline metrics with AUC interpretation gloss, legacy
seq-0011 carryover, and sorted per-expression breakdown table.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 6: Aggregator — per-weight PNG box plot

**Files:**
- Modify: `diagnostics/aggregate_multiseq.py`
- Modify: `tests/test_aggregate_multiseq.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_aggregate_multiseq.py`:

```python
def test_write_weight_boxplot_creates_file(
    tmp_path: Path, synthetic_npz_dir: Path,
):
    from diagnostics.aggregate_multiseq import (
        build_weight_record, write_weight_boxplot,
    )
    rec = build_weight_record(
        results_dir=synthetic_npz_dir, model_tag="model_A",
        weights_path="fake.pth", seqs=["0005", "0011", "0013"],
    )
    out = tmp_path / "layer3_multiseq_model_A.png"
    write_weight_boxplot(rec, out)
    assert out.exists()
    assert out.stat().st_size > 1000, "PNG should not be trivially small"
```

- [ ] **Step 2: Run the test to confirm failure**

```bash
~/miniconda/envs/RMOT/bin/python -m pytest tests/test_aggregate_multiseq.py::test_write_weight_boxplot_creates_file -v
```

Expected: `AttributeError`.

- [ ] **Step 3: Implement `write_weight_boxplot`**

Append to `diagnostics/aggregate_multiseq.py` (put the matplotlib import inside the function to avoid forcing matplotlib on every import of the module — unit tests for the core math don't need it):

```python
def write_weight_boxplot(record: dict, out_path: Path) -> None:
    """Render one box per expression, showing per-seq AUC spread.

    Tall boxes = seq-dependent expressions; short boxes = stable findings.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seqs = record["sequences"]
    per_expr = record["per_expression"]

    # Sort expressions by micro AUC descending (matches Markdown order).
    def _sort_key(kv):
        v = kv[1]["auc_micro"]
        return (-v if v is not None else 1.0, kv[0])

    items = sorted(per_expr.items(), key=_sort_key)
    labels = [sent for sent, _ in items]
    aucs_per_expr = [
        [agg["auc_per_seq"].get(s) for s in seqs if agg["auc_per_seq"].get(s) is not None]
        for _, agg in items
    ]
    # Empty lists break boxplot — filter expressions that have no data at all.
    labels = [l for l, d in zip(labels, aucs_per_expr) if len(d) > 0]
    aucs_per_expr = [d for d in aucs_per_expr if len(d) > 0]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(labels))))
    ax.boxplot(
        aucs_per_expr, vert=False, labels=labels, widths=0.6,
        showmeans=True, meanline=True,
    )
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.4, label="chance")
    ax.axvline(0.8, color="green", linestyle=":", alpha=0.4, label="0.80 target")
    ax.set_xlabel("Ranking AUC (0.5 = chance, 1.0 = perfect)")
    ax.set_title(
        f"Multi-seq AUC spread: {record['model_tag']} "
        f"({len(seqs)} held-out seqs: {', '.join(seqs)})"
    )
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 4: Run tests**

```bash
~/miniconda/envs/RMOT/bin/python -m pytest tests/test_aggregate_multiseq.py -v
```

Expected: 11 passes.

- [ ] **Step 5: Commit**

```bash
git add diagnostics/aggregate_multiseq.py tests/test_aggregate_multiseq.py
git commit -m "feat(aggregate): per-weight box plot PNG renderer

One horizontal box per expression showing per-seq AUC spread. Chance
(0.5) and target (0.8) vertical guides baked in.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 7: Aggregator — cross-weight comparison Markdown

**Files:**
- Modify: `diagnostics/aggregate_multiseq.py`
- Modify: `tests/test_aggregate_multiseq.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_aggregate_multiseq.py`:

```python
def test_comparison_report_ranks_models_and_includes_max_gap(
    tmp_path: Path, synthetic_npz_dir: Path,
):
    from diagnostics.aggregate_multiseq import (
        build_weight_record, write_comparison_markdown,
    )
    rec_a = build_weight_record(
        results_dir=synthetic_npz_dir, model_tag="model_A",
        weights_path="a.pth", seqs=["0005", "0011", "0013"],
    )
    rec_b = build_weight_record(
        results_dir=synthetic_npz_dir, model_tag="model_B",
        weights_path="b.pth", seqs=["0005", "0011", "0013"],
    )
    out = tmp_path / "layer3_multiseq_comparison.md"
    write_comparison_markdown([rec_a, rec_b], out, seqs=["0005", "0011", "0013"])
    text = out.read_text()
    # AUC gloss baked in
    assert "AUC =" in text
    # Column headers
    for col in ["model_tag", "mean_auc_micro", "mean_auc_macro",
                "best_seq", "worst_seq", "max_gap"]:
        assert col in text
    # Rows ordered by micro descending: model_A should appear before model_B
    a_idx = text.index("model_A")
    b_idx = text.index("model_B")
    assert a_idx < b_idx, "model_A has higher AUC so it should be listed first"
    # best_seq formatted as "{seq}: {auc:.3f}"
    import re
    assert re.search(r"\d{4}: 0\.\d{3}", text), (
        "best_seq / worst_seq should be formatted as '{seq}: {auc:.3f}'"
    )
```

- [ ] **Step 2: Run the test to confirm failure**

```bash
~/miniconda/envs/RMOT/bin/python -m pytest tests/test_aggregate_multiseq.py::test_comparison_report_ranks_models_and_includes_max_gap -v
```

Expected: `AttributeError: ... has no attribute 'write_comparison_markdown'`.

- [ ] **Step 3: Implement `write_comparison_markdown`**

Append to `diagnostics/aggregate_multiseq.py`:

```python
def _per_seq_mean_auc(record: dict) -> dict[str, float | None]:
    """For each seq, mean across expressions of its per-seq AUC."""
    out: dict[str, list[float]] = {s: [] for s in record["sequences"]}
    for agg in record["per_expression"].values():
        for s, v in agg["auc_per_seq"].items():
            if v is not None:
                out[s].append(v)
    return {
        s: (float(np.mean(vs)) if vs else None) for s, vs in out.items()
    }


def write_comparison_markdown(
    records: list[dict],
    out_path: Path,
    seqs: Iterable[str],
) -> None:
    """Write the cross-weight comparison report, ordered by micro AUC desc."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seqs = list(seqs)

    rows = []
    for rec in records:
        hl = rec["headline"]
        per_seq_mean = _per_seq_mean_auc(rec)
        valid = {s: v for s, v in per_seq_mean.items() if v is not None}
        if valid:
            best_s = max(valid, key=valid.get)
            worst_s = min(valid, key=valid.get)
            max_gap = valid[best_s] - valid[worst_s]
            best_str = f"{best_s}: {valid[best_s]:.3f}"
            worst_str = f"{worst_s}: {valid[worst_s]:.3f}"
        else:
            best_str = worst_str = "—"
            max_gap = None
        rows.append({
            "model_tag": rec["model_tag"],
            "mean_auc_micro": hl["mean_auc_micro"],
            "mean_auc_macro": hl["mean_auc_macro"],
            "std_across_seqs": hl["std_across_seqs"],
            "best_seq": best_str,
            "worst_seq": worst_str,
            "max_gap": max_gap,
        })

    rows.sort(
        key=lambda r: (
            -r["mean_auc_micro"] if r["mean_auc_micro"] is not None else 1.0,
            r["model_tag"],
        )
    )

    lines: list[str] = []
    lines.append("# Multi-Sequence Eval: Comparison Across Weights")
    lines.append("")
    lines.append("## What AUC means here")
    lines.append(_AUC_GLOSS)
    lines.append("")
    lines.append(f"**Held-out sequences:** {', '.join(seqs)}")
    lines.append("")
    lines.append(
        "| model_tag | mean_auc_micro | mean_auc_macro ± std | best_seq | worst_seq | max_gap |"
    )
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        macro_cell = (
            f"{_fmt(r['mean_auc_macro'])} ± {_fmt(r['std_across_seqs'])}"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    r["model_tag"],
                    _fmt(r["mean_auc_micro"]),
                    macro_cell,
                    r["best_seq"],
                    r["worst_seq"],
                    _fmt(r["max_gap"]),
                ]
            )
            + " |"
        )
    out_path.write_text("\n".join(lines) + "\n")
```

- [ ] **Step 4: Run all tests**

```bash
~/miniconda/envs/RMOT/bin/python -m pytest tests/test_aggregate_multiseq.py -v
```

Expected: 12 passes.

- [ ] **Step 5: Commit**

```bash
git add diagnostics/aggregate_multiseq.py tests/test_aggregate_multiseq.py
git commit -m "feat(aggregate): cross-weight comparison Markdown

Ranks all weights by mean micro AUC descending. best_seq and worst_seq
columns formatted as '{seq}: {auc:.3f}'; max_gap surfaces
seq-dependent models.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 8: Aggregator — CLI entrypoint

**Files:**
- Modify: `diagnostics/aggregate_multiseq.py`
- Modify: `tests/test_aggregate_multiseq.py`

- [ ] **Step 1: Write failing test for the end-to-end CLI function**

Append to `tests/test_aggregate_multiseq.py`:

```python
def test_run_produces_all_artifacts(tmp_path: Path, synthetic_npz_dir: Path):
    from diagnostics.aggregate_multiseq import run
    out_dir = tmp_path / "aggregated"
    run(
        results_dir=synthetic_npz_dir,
        output_dir=out_dir,
        weights=[("model_A", "a.pth"), ("model_B", "b.pth")],
        seqs=["0005", "0011", "0013"],
        legacy_seq_0011_auc_by_tag={"model_A": 0.77, "model_B": 0.60},
    )
    # Per-weight artifacts
    for tag in ["model_A", "model_B"]:
        assert (out_dir / f"layer3_multiseq_{tag}.json").exists()
        assert (out_dir / f"layer3_multiseq_{tag}.md").exists()
        assert (out_dir / f"layer3_multiseq_{tag}.png").exists()
    # Comparison artifact
    assert (out_dir / "layer3_multiseq_comparison.md").exists()
```

- [ ] **Step 2: Run test to confirm failure**

```bash
~/miniconda/envs/RMOT/bin/python -m pytest tests/test_aggregate_multiseq.py::test_run_produces_all_artifacts -v
```

Expected: `AttributeError: ... has no attribute 'run'`.

- [ ] **Step 3: Implement `run` and the `argparse` CLI**

Replace the `__main__` stub at the bottom of `diagnostics/aggregate_multiseq.py` with the following. Also add the `run` function just above it:

```python
def run(
    results_dir: Path,
    output_dir: Path,
    weights: list[tuple[str, str]],  # (model_tag, weights_path)
    seqs: Iterable[str],
    legacy_seq_0011_auc_by_tag: dict[str, float | None] | None = None,
) -> None:
    """Build per-weight artifacts + cross-weight comparison, end-to-end."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy = legacy_seq_0011_auc_by_tag or {}
    seqs = list(seqs)

    records: list[dict] = []
    for model_tag, weights_path in weights:
        record = build_weight_record(
            results_dir=results_dir,
            model_tag=model_tag,
            weights_path=weights_path,
            seqs=seqs,
        )
        records.append(record)
        write_weight_json(record, output_dir / f"layer3_multiseq_{model_tag}.json")
        write_weight_markdown(
            record,
            output_dir / f"layer3_multiseq_{model_tag}.md",
            legacy_seq_0011_auc=legacy.get(model_tag),
        )
        write_weight_boxplot(record, output_dir / f"layer3_multiseq_{model_tag}.png")

    write_comparison_markdown(
        records, output_dir / "layer3_multiseq_comparison.md", seqs=seqs,
    )


def _parse_weights_arg(raw: list[str]) -> list[tuple[str, str]]:
    """Each entry is either 'tag=path' or just 'path' (tag derived from stem).

    Stem derivation: strip 'gmc_link_weights_' prefix and '.pth' suffix.
    """
    out: list[tuple[str, str]] = []
    for entry in raw:
        if "=" in entry:
            tag, path = entry.split("=", 1)
        else:
            path = entry
            stem = Path(path).stem
            tag = stem.removeprefix("gmc_link_weights_")
        out.append((tag, path))
    return out


def _parse_legacy_arg(raw: list[str]) -> dict[str, float]:
    """Each entry: 'tag=0.779'."""
    out: dict[str, float] = {}
    for entry in raw:
        tag, val = entry.split("=", 1)
        out[tag] = float(val)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir", required=True, type=Path,
        help="Directory containing layer3_{seq}_{model_tag}.npz files",
    )
    p.add_argument(
        "--output-dir", required=True, type=Path,
        help="Where to write per-weight JSON/MD/PNG and comparison MD",
    )
    p.add_argument(
        "--weights", nargs="+", required=True,
        help="One or more weight entries. Each is 'tag=path' or just 'path'.",
    )
    p.add_argument(
        "--seqs", nargs="+", default=["0005", "0011", "0013"],
        help="Sequences to aggregate (default: 0005 0011 0013)",
    )
    p.add_argument(
        "--legacy-seq-0011", nargs="*", default=[],
        help="Optional: 'tag=auc' entries for the legacy seq-0011 AUC line",
    )
    args = p.parse_args()
    run(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        weights=_parse_weights_arg(args.weights),
        seqs=args.seqs,
        legacy_seq_0011_auc_by_tag=_parse_legacy_arg(args.legacy_seq_0011),
    )
    print(f"Wrote aggregated artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
~/miniconda/envs/RMOT/bin/python -m pytest tests/test_aggregate_multiseq.py -v
```

Expected: 13 passes.

- [ ] **Step 5: Sanity-check the CLI help**

```bash
~/miniconda/envs/RMOT/bin/python diagnostics/aggregate_multiseq.py --help
```

Expected: argparse help text listing `--results-dir`, `--output-dir`, `--weights`, `--seqs`, `--legacy-seq-0011`.

- [ ] **Step 6: Commit**

```bash
git add diagnostics/aggregate_multiseq.py tests/test_aggregate_multiseq.py
git commit -m "feat(aggregate): CLI entrypoint for end-to-end aggregation

run() ties record-building and all three writers together. argparse
CLI accepts tag=path and tag=auc shorthand for weight and legacy
arguments respectively.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 9: Shell driver `run_multiseq_eval.sh`

**Files:**
- Create: `run_multiseq_eval.sh`

- [ ] **Step 1: Create the driver script**

Create `run_multiseq_eval.sh`:

```bash
#!/usr/bin/env bash
# Multi-sequence Layer 3 evaluation driver.
#
# For each (seq, weight), calls diag_gt_cosine_distributions.py, then
# relocates the output .npz into diagnostics/results/multiseq/ with a
# namespaced filename. Finally invokes the aggregator across all results.
#
# Spec: docs/superpowers/specs/2026-04-18-multi-sequence-eval-design.md
# Plan: docs/superpowers/plans/2026-04-18-multi-sequence-eval.md
#
# Usage: bash run_multiseq_eval.sh
#
# Loop order: seq-outer / weight-inner, to preserve OS page-cache warmth on
# each sequence's frames across the 11 weight invocations.

set -euo pipefail

PY=~/miniconda/envs/RMOT/bin/python
DIAG=diagnostics/diag_gt_cosine_distributions.py
RESULTS_DIR=diagnostics/results
MULTISEQ_DIR="${RESULTS_DIR}/multiseq"
mkdir -p "${MULTISEQ_DIR}"

SEQS=(0005 0011 0013)

# tag => weight path. Keep tags in sync with filename stems (strip
# gmc_link_weights_ prefix, .pth suffix) so aggregator can derive them.
declare -a WEIGHTS=(
  "v1train_stage1=gmc_link_weights_v1train_stage1.pth"
  "v1train_F1_speed=gmc_link_weights_v1train_F1_speed.pth"
  "v1train_F2_heading=gmc_link_weights_v1train_F2_heading.pth"
  "v1train_F3_accel=gmc_link_weights_v1train_F3_accel.pth"
  "v1train_F4_ego=gmc_link_weights_v1train_F4_ego.pth"
  "v1train_F5_nbrmean=gmc_link_weights_v1train_F5_nbrmean.pth"
  "v1train_F6_velrank=gmc_link_weights_v1train_F6_velrank.pth"
  "v1train_F7_headdiff=gmc_link_weights_v1train_F7_headdiff.pth"
  "v1train_F8_nndist=gmc_link_weights_v1train_F8_nndist.pth"
  "v1train_F9_density=gmc_link_weights_v1train_F9_density.pth"
  "v1train_temporal=gmc_link_weights_v1train_temporal.pth"
)

# Pre-flight: every weight must exist on disk before we start.
for entry in "${WEIGHTS[@]}"; do
  path="${entry#*=}"
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: weight file not found: ${path}" >&2
    exit 1
  fi
done

# Main loop: seq-outer, weight-inner.
for seq in "${SEQS[@]}"; do
  echo "============================================================"
  echo "Sequence ${seq}"
  echo "============================================================"
  for entry in "${WEIGHTS[@]}"; do
    tag="${entry%%=*}"
    path="${entry#*=}"
    echo "--- ${seq} / ${tag} ---"
    "${PY}" "${DIAG}" --weights "${path}" --seq "${seq}"
    src="${RESULTS_DIR}/layer3_gt_cosine_${seq}.npz"
    dst="${MULTISEQ_DIR}/layer3_${seq}_${tag}.npz"
    if [[ ! -f "${src}" ]]; then
      echo "ERROR: expected ${src} but it was not produced" >&2
      exit 2
    fi
    mv "${src}" "${dst}"
    # Relocate the plot too so the legacy single-seq PNG in results/ stays
    # as a persistent reference from the original experiment rather than
    # being overwritten 11 times.
    src_png="${RESULTS_DIR}/layer3_gt_cosine_${seq}.png"
    if [[ -f "${src_png}" ]]; then
      mv "${src_png}" "${MULTISEQ_DIR}/layer3_${seq}_${tag}.png"
    fi
  done
done

# Build the weights argument for the aggregator.
AGG_WEIGHTS=()
for entry in "${WEIGHTS[@]}"; do
  AGG_WEIGHTS+=("${entry}")
done

# Pass through legacy seq-0011 AUCs so per-weight MDs carry them. These are
# the numbers reported in RESEARCH_NOTES for Exp 30-32; other ablations
# without a known legacy number are omitted (the MD will render "—").
LEGACY=(
  "v1train_stage1=0.779"
  "v1train_temporal=0.770"
)

echo "============================================================"
echo "Aggregating 33 result files → multiseq reports"
echo "============================================================"
"${PY}" diagnostics/aggregate_multiseq.py \
  --results-dir "${MULTISEQ_DIR}" \
  --output-dir "${MULTISEQ_DIR}" \
  --weights "${AGG_WEIGHTS[@]}" \
  --seqs "${SEQS[@]}" \
  --legacy-seq-0011 "${LEGACY[@]}"

echo "Done. See ${MULTISEQ_DIR}/layer3_multiseq_comparison.md"
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x run_multiseq_eval.sh
```

- [ ] **Step 3: Syntax-check the script**

```bash
bash -n run_multiseq_eval.sh && echo "syntax OK"
```

Expected: `syntax OK`.

- [ ] **Step 4: Commit**

```bash
git add run_multiseq_eval.sh
git commit -m "feat: shell driver for multi-seq Layer 3 evaluation

Seq-outer / weight-inner loop across 11 V1 weights × 3 held-out seqs.
Pre-flights weight paths, relocates each .npz into multiseq/ namespace
with model_tag suffix, then calls aggregate_multiseq.py.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 10: End-to-end smoke test (1 weight × 1 seq)

Before burning 15–45 minutes on the full 11 × 3 run, validate the pipeline with a single (weight, seq) pair.

**Files:** No new files. Verifying Task 9's driver on a narrowed input.

- [ ] **Step 1: Run the diagnostic directly for `v1train_stage1` on seq 0005**

(Seq 0011 already has an up-to-date `.npz` from Task 2. Seq 0005 does not, so this exercises a fresh invocation.)

```bash
~/miniconda/envs/RMOT/bin/python diagnostics/diag_gt_cosine_distributions.py \
  --weights gmc_link_weights_v1train_stage1.pth --seq 0005
```

Expected: prints the per-expression table, "Saved: diagnostics/results/layer3_gt_cosine_0005.npz".

- [ ] **Step 2: Relocate the .npz to multiseq namespace**

```bash
mv diagnostics/results/layer3_gt_cosine_0005.npz \
   diagnostics/results/multiseq/layer3_0005_v1train_stage1.npz
mv diagnostics/results/layer3_gt_cosine_0005.png \
   diagnostics/results/multiseq/layer3_0005_v1train_stage1.png
```

Similarly rename the existing seq-0011 results the driver would have produced for this weight:

```bash
~/miniconda/envs/RMOT/bin/python diagnostics/diag_gt_cosine_distributions.py \
  --weights gmc_link_weights_v1train_stage1.pth --seq 0011
mv diagnostics/results/layer3_gt_cosine_0011.npz \
   diagnostics/results/multiseq/layer3_0011_v1train_stage1.npz
mv diagnostics/results/layer3_gt_cosine_0011.png \
   diagnostics/results/multiseq/layer3_0011_v1train_stage1.png
```

And seq 0013:

```bash
~/miniconda/envs/RMOT/bin/python diagnostics/diag_gt_cosine_distributions.py \
  --weights gmc_link_weights_v1train_stage1.pth --seq 0013
mv diagnostics/results/layer3_gt_cosine_0013.npz \
   diagnostics/results/multiseq/layer3_0013_v1train_stage1.npz
mv diagnostics/results/layer3_gt_cosine_0013.png \
   diagnostics/results/multiseq/layer3_0013_v1train_stage1.png
```

- [ ] **Step 3: Run the aggregator on just this one weight**

```bash
~/miniconda/envs/RMOT/bin/python diagnostics/aggregate_multiseq.py \
  --results-dir diagnostics/results/multiseq/ \
  --output-dir diagnostics/results/multiseq/ \
  --weights v1train_stage1=gmc_link_weights_v1train_stage1.pth \
  --seqs 0005 0011 0013 \
  --legacy-seq-0011 v1train_stage1=0.779
```

Expected: prints `Wrote aggregated artifacts to diagnostics/results/multiseq/`.

- [ ] **Step 4: Verify all four artifacts exist and are non-trivial**

```bash
ls -la diagnostics/results/multiseq/layer3_multiseq_v1train_stage1.* \
       diagnostics/results/multiseq/layer3_multiseq_comparison.md
```

Expected: JSON, MD, PNG per weight, plus the comparison MD. Sizes > 1KB each.

- [ ] **Step 5: Eyeball the per-weight MD report**

Read `diagnostics/results/multiseq/layer3_multiseq_v1train_stage1.md` and confirm:
- Headline lists mean micro AUC, mean macro AUC ± std, seq-0011 legacy = 0.779.
- Per-expression table has columns for 0005, 0011, 0013, macro, micro, GT counts.
- A reasonable number of expressions listed (~20+ for the Stage 1 weight).

Read `diagnostics/results/multiseq/layer3_multiseq_comparison.md` and confirm:
- AUC gloss at the top.
- One-row table with `v1train_stage1`.

If anything looks off, fix the aggregator before proceeding to Task 11.

- [ ] **Step 6: Commit the verified smoke-test artifacts (markdowns and JSON only; .npz and .png are ignored)**

```bash
git add diagnostics/results/multiseq/layer3_multiseq_v1train_stage1.json \
        diagnostics/results/multiseq/layer3_multiseq_v1train_stage1.md \
        diagnostics/results/multiseq/layer3_multiseq_comparison.md
git commit -m "chore: smoke-test multiseq aggregator on v1train_stage1 × 3 seqs

Confirms end-to-end pipeline works before the full 11-weight retro run.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 11: Full retrospective run (11 weights × 3 seqs)

**Files:** Overwrites `diagnostics/results/multiseq/*` artifacts from Task 10.

- [ ] **Step 1: Remove smoke-test stand-alone artifacts so the full run writes clean state**

```bash
rm -f diagnostics/results/multiseq/layer3_0005_v1train_stage1.npz \
      diagnostics/results/multiseq/layer3_0011_v1train_stage1.npz \
      diagnostics/results/multiseq/layer3_0013_v1train_stage1.npz \
      diagnostics/results/multiseq/layer3_0005_v1train_stage1.png \
      diagnostics/results/multiseq/layer3_0011_v1train_stage1.png \
      diagnostics/results/multiseq/layer3_0013_v1train_stage1.png \
      diagnostics/results/multiseq/layer3_multiseq_v1train_stage1.json \
      diagnostics/results/multiseq/layer3_multiseq_v1train_stage1.md \
      diagnostics/results/multiseq/layer3_multiseq_comparison.md
```

- [ ] **Step 2: Kick off the full run (long-running, 30–60 min expected)**

```bash
bash run_multiseq_eval.sh 2>&1 | tee diagnostics/results/multiseq/_run_log.txt
```

Run this in the background using Bash's `run_in_background` flag so you are free to observe. Monitor periodically — each (seq, weight) pair takes 2–5 minutes dominated by ORB homography precompute.

Expected (high level):
- 33 "Saved: ..." lines for the diagnostic.
- 33 `mv` operations to `multiseq/`.
- One "Aggregating 33 result files..." line.
- One "Wrote aggregated artifacts to ..." line.

- [ ] **Step 3: Verify artifact counts**

```bash
ls diagnostics/results/multiseq/ | sort > /tmp/multiseq_files.txt
wc -l /tmp/multiseq_files.txt
grep -c '^layer3_0005_' /tmp/multiseq_files.txt
grep -c '^layer3_0011_' /tmp/multiseq_files.txt
grep -c '^layer3_0013_' /tmp/multiseq_files.txt
grep -c '^layer3_multiseq_.*\.json$' /tmp/multiseq_files.txt
grep -c '^layer3_multiseq_.*\.md$' /tmp/multiseq_files.txt
grep -c '^layer3_multiseq_.*\.png$' /tmp/multiseq_files.txt
```

Expected:
- `layer3_0005_*`, `layer3_0011_*`, `layer3_0013_*` each: 11 `.npz` + 11 `.png` = 22 files per seq (66 total).
- `layer3_multiseq_*.json`: 11.
- `layer3_multiseq_*.md`: 11 per-weight + 1 comparison = 12.
- `layer3_multiseq_*.png`: 11 per-weight box plots.

If counts are off, inspect `_run_log.txt` for the first failure before proceeding.

- [ ] **Step 4: Skim the comparison report for sanity**

Read `diagnostics/results/multiseq/layer3_multiseq_comparison.md`. Check that:
- All 11 `model_tag` rows present.
- Rows ordered by `mean_auc_micro` descending.
- No cells contain raw Python `None` literals (they should render as `—`).
- `v1train_stage1`'s micro AUC is in the ~0.75–0.82 range (plausible given the seq-0011 0.779 baseline).

- [ ] **Step 5: Commit the committed artifacts (MD + JSON only)**

```bash
git add diagnostics/results/multiseq/*.md diagnostics/results/multiseq/*.json
git status --porcelain | grep multiseq
```

Verify `.npz` and `.png` files do NOT appear in `git status` (they're gitignored). If any do, fix `.gitignore` first.

```bash
git commit -m "docs(multiseq): full retrospective across 11 V1 weights × 3 seqs

11 per-weight JSON/MD reports plus cross-weight comparison
Markdown. Answers whether seq-0011's ~0.79 AUC ceiling holds under
multi-seq aggregation.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 12: RESEARCH_NOTES.md — interpret multi-seq findings

**Files:**
- Modify: `RESEARCH_NOTES.md`

- [ ] **Step 1: Read the comparison MD and extract the key numbers**

Open `diagnostics/results/multiseq/layer3_multiseq_comparison.md`. Capture:
- `v1train_stage1` mean_auc_micro (call it `M30`) and best/worst seq.
- Any F1–F9 weight with mean_auc_micro > 0.800 OR any single-seq mean > 0.800 (call out violators; otherwise state "none exceeds 0.800 on either metric" per the spec's interpretive threshold).
- `v1train_temporal` mean_auc_micro (call it `M32`). Compute `M30 - M32` vs the ±0.010 spec threshold.
- For each weight, note whether seq-0011 AUC is within ±1 std of macro mean. If ≥70% of weights satisfy this, seq-0011 was representative; otherwise it was an outlier.

- [ ] **Step 2: Read the tail of RESEARCH_NOTES.md**

Use the Read tool on the last ~100 lines so you see the current formatting conventions for experiment entries and know where to append.

- [ ] **Step 3: Append a new "Exp 33: Multi-Sequence Re-Evaluation" section**

Template (fill in actual numbers from Step 1):

```markdown
## Exp 33: Multi-Sequence Re-Evaluation of Exp 30–32

**Date:** 2026-04-18
**Motivation:** Exps 30–32 reported a ~0.79 AUC ceiling from seq 0011 alone. This run aggregates the same weights across all 3 V1 held-out seqs (0005, 0011, 0013) to separate the ceiling's signal from seq-0011 variance.
**Spec:** `docs/superpowers/specs/2026-04-18-multi-sequence-eval-design.md`
**Plan:** `docs/superpowers/plans/2026-04-18-multi-sequence-eval.md`
**Comparison artifact:** `diagnostics/results/multiseq/layer3_multiseq_comparison.md`

### Findings against the spec's interpretive thresholds

1. **Is seq 0011 representative?** For each of the 11 weights, check whether seq-0011's per-expression AUCs are within ±1 std of the macro mean for ≥70% of expressions. Count how many of the 11 weights pass that check. If ≥8/11 → seq 0011 was representative; if <8/11 → seq 0011 was an outlier and Exp 30–32 conclusions need revisiting. <Fill: X/11 weights pass; conclusion = representative / outlier>.
2. **Is feature enrichment truly dead (Exp 31)?** Check every F1–F9 weight against *both* its micro AUC and its per-seq mean AUCs. <Fill: "no F1–F9 weight exceeds 0.800 on either metric" → confirmed dead. If any single-seq mean crosses 0.800, list the (weight, seq) pair(s) that reopen the feature.>
3. **Is Exp 32 a genuine regression?** Baseline micro AUC M30=<fill>, Temporal micro AUC M32=<fill>, gap=(M30−M32)=<fill>. If gap ≥ 0.010 → genuine regression. If |gap| < 0.010 → noise-level, the temporal result is inconclusive rather than negative. <Fill conclusion.>

### Headline numbers

| Model | Mean micro AUC | Mean macro AUC ± std | Seq-0011 legacy | Max gap |
|---|---|---|---|---|
| v1train_stage1 | ... | ... | 0.779 | ... |
| v1train_temporal | ... | ... | 0.770 | ... |
| ... (top 3 and bottom 3 F1–F9) | | | | |

### Protocol change

Per the spec, the **next 3 experiments** report both legacy seq-0011 AUC *and* multi-seq micro AUC. After that transition window, multi-seq micro becomes canonical and legacy seq-0011 reporting is dropped.
```

- [ ] **Step 4: Commit**

```bash
git add RESEARCH_NOTES.md
git commit -m "docs: Exp 33 multi-seq retrospective on Exp 30-32 conclusions

Records the headline comparison across all 11 V1 weights and applies
the spec's three interpretive thresholds to decide whether seq-0011
was representative, feature enrichment is truly dead, and the
temporal transformer regression is genuine.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Final Review

After all 12 tasks are complete, use `superpowers:finishing-a-development-branch` or run an independent `code-reviewer` agent over the full diff from `exp/multi-seq-eval` vs `main` to confirm:

- All spec sections have been implemented.
- No stray debug prints, leftover `pytest.skip`, or gitignored artifacts committed.
- Aggregator is idempotent: re-running `bash run_multiseq_eval.sh` on a fully-populated `multiseq/` directory reproduces bitwise-identical JSONs.

Then merge `exp/multi-seq-eval` into `main` and proceed to the next experimental direction per the spec's "Future Scaling" section.
