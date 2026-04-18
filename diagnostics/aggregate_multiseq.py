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


if __name__ == "__main__":  # pragma: no cover — filled in Task 8
    raise SystemExit("CLI entrypoint not yet implemented; see Task 8 of the plan.")
