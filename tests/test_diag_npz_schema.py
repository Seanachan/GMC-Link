"""Verify the diagnostic persists raw cosine arrays in addition to summary results.

This is an integration test: it inspects a real .npz produced by the diagnostic
script. It is skipped on a clean checkout where no diagnostic has been run yet.
"""
import numpy as np
import pytest
from pathlib import Path


_CANDIDATE_NPZ_PATHS = [
    Path("diagnostics/results/layer3_gt_cosine_0011.npz"),
    Path("diagnostics/results/multiseq/layer3_0011_v1train_stage1.npz"),
]


def test_npz_contains_raw_cosines():
    npz_path = next((p for p in _CANDIDATE_NPZ_PATHS if p.exists()), None)
    if npz_path is None:
        pytest.skip(
            "No diagnostic .npz on disk. Run the diagnostic first, e.g.: "
            "~/miniconda/envs/RMOT/bin/python diagnostics/diag_gt_cosine_distributions.py "
            "--weights gmc_link_weights_v1train_stage1.pth --seq 0011"
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
