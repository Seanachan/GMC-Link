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
            seq_map[sent] = (
                np.asarray(gt_list[i], dtype=np.float32),
                np.asarray(nongt_list[i], dtype=np.float32),
            )
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
    # defined in >=2 seqs (std is otherwise meaningless).
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


if __name__ == "__main__":  # pragma: no cover — filled in Task 8
    raise SystemExit("CLI entrypoint not yet implemented; see Task 8 of the plan.")
