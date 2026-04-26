"""Tests that omf_stats is wired into the dataset feature registry.

Spec §3: 13D + OMF (15D) = 28D total. The full per-frame dense-flow pipeline
lives in the Stage B runner (Task 11); this file only covers registry + cache
key behaviour so the feature survives integration without silently becoming
a no-op.
"""
import os

from gmc_link.dataset import (
    EXTRA_FEATURE_DIMS,
    _build_cache_key,
    compute_extra_dims,
)


def test_omf_stats_feature_is_registered():
    assert "omf_stats" in EXTRA_FEATURE_DIMS
    assert EXTRA_FEATURE_DIMS["omf_stats"] == 15  # 5 stats × 3 scales


def test_28d_vector_size_matches_base_plus_omf():
    """13D base + 15D OMF stats = 28D target dim per spec §3."""
    assert 13 + compute_extra_dims(["omf_stats"]) == 28


def test_cache_key_differs_across_ego_source_even_with_omf_features(tmp_path):
    """OMF outputs depend on the ego-compensated residual, so the cache must
    partition on ego router × feature set jointly."""
    common = dict(
        data_root=str(tmp_path),
        sequences=["0001"],
        frame_gaps=(2, 5, 10),
        frame_shape=(375, 1242),
        use_group_labels=False,
        extra_features=["omf_stats"],
        seq_len=0,
        text_encoder_name="all-MiniLM-L6-v2",
    )
    k_orb, _ = _build_cache_key(**common, ego_router_name="orb")
    k_rec, _ = _build_cache_key(**common, ego_router_name="recoverpose")
    assert k_orb != k_rec


def test_cache_key_differs_between_13d_and_28d_feature_sets(tmp_path):
    """Feature set must partition the cache so 13D and 28D don't collide."""
    common = dict(
        data_root=str(tmp_path),
        sequences=["0001"],
        frame_gaps=(2, 5, 10),
        frame_shape=(375, 1242),
        use_group_labels=False,
        seq_len=0,
        text_encoder_name="all-MiniLM-L6-v2",
    )
    k13, _ = _build_cache_key(**common, extra_features=None)
    k28, _ = _build_cache_key(**common, extra_features=["omf_stats"])
    assert k13 != k28
