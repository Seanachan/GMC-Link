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
    out_dir = tmp_path / "multiseq"
    out_dir.mkdir()

    seqs = ["0005", "0011", "0013"]
    sentences = ["moving cars", "parking cars", "cars in left"]

    def _build(model_tag: str, invert_on: set[str], seed: int) -> None:
        rng = np.random.default_rng(seed)
        for s in seqs:
            results = []
            gt_list = []
            nongt_list = []
            for sent in sentences:
                # 20 GT, 80 non-GT per expression per seq — realistic-ish ratio
                base_gt = rng.normal(0.3, 0.1, size=20).astype(np.float32)
                base_nongt = rng.normal(0.1, 0.1, size=80).astype(np.float32)
                if s in invert_on:
                    gt, nongt = base_nongt[:20], base_gt.tolist() + rng.normal(0.3, 0.1, size=60).astype(np.float32).tolist()
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

    _build("model_A", invert_on=set(), seed=42)
    _build("model_B", invert_on={"0005"}, seed=44)
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
    # Headline: micro mean >= macro mean only if distributions match; both
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
