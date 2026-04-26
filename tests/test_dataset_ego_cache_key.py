"""Tests that the dataset cache key is ego-router-aware so ORB and recoverPose
cannot silently share a cache entry.

Per spec §3 "Does change" row: new cache keys cache_v1_orb_13d,
cache_v1_recoverpose_13d, etc. This is critical to avoid cross-contamination
between Stage A arms where the 13D feature layout is nominally the same but
the residuals depend on the ego source.
"""
from gmc_link.dataset import _build_cache_key


def test_cache_key_differs_when_ego_router_changes(tmp_path):
    kwargs = dict(
        data_root=str(tmp_path),
        sequences=["0001"],
        frame_gaps=(2, 5, 10),
        frame_shape=(375, 1242),
        use_group_labels=False,
        extra_features=None,
        seq_len=0,
        text_encoder_name="all-MiniLM-L6-v2",
    )
    k_orb, _ = _build_cache_key(**kwargs, ego_router_name="orb")
    k_rec, _ = _build_cache_key(**kwargs, ego_router_name="recoverpose")
    assert k_orb != k_rec, "Cache keys must differ across ego sources"


def test_cache_key_default_ego_matches_explicit_orb(tmp_path):
    kwargs = dict(
        data_root=str(tmp_path),
        sequences=["0001"],
        frame_gaps=(2, 5, 10),
        frame_shape=(375, 1242),
        use_group_labels=False,
        extra_features=None,
        seq_len=0,
        text_encoder_name="all-MiniLM-L6-v2",
    )
    k_default, _ = _build_cache_key(**kwargs)
    k_orb, _ = _build_cache_key(**kwargs, ego_router_name="orb")
    assert k_default == k_orb, "Default kwarg must preserve historic ORB cache hits"


def test_cache_key_records_ego_in_sidecar_for_non_orb(tmp_path):
    _, key_obj = _build_cache_key(
        data_root=str(tmp_path),
        sequences=["0001"],
        frame_gaps=(2, 5, 10),
        frame_shape=(375, 1242),
        use_group_labels=False,
        extra_features=None,
        seq_len=0,
        text_encoder_name="all-MiniLM-L6-v2",
        ego_router_name="recoverpose",
    )
    assert key_obj.get("ego_router") == "recoverpose"


def test_orb_key_omits_ego_field_for_backwards_compat(tmp_path):
    _, key_obj = _build_cache_key(
        data_root=str(tmp_path),
        sequences=["0001"],
        frame_gaps=(2, 5, 10),
        frame_shape=(375, 1242),
        use_group_labels=False,
        extra_features=None,
        seq_len=0,
        text_encoder_name="all-MiniLM-L6-v2",
        ego_router_name="orb",
    )
    assert "ego_router" not in key_obj, (
        "Historic ORB caches must remain valid; key field only appears for non-ORB backends"
    )
