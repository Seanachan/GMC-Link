"""Tests that GMCLinkManager accepts an injected EgoRouter and defaults to ORB."""
from gmc_link.ego.ego_router import OrbEgoRouter
from gmc_link.ego.recoverpose_ego import RecoverPoseEgoRouter
from gmc_link.manager import GMCLinkManager


def test_manager_defaults_to_orb_router():
    manager = GMCLinkManager()
    assert isinstance(manager.ego_engine, OrbEgoRouter)


def test_manager_accepts_injected_recoverpose_router():
    router = RecoverPoseEgoRouter(max_features=500)
    manager = GMCLinkManager(ego_router=router)
    assert manager.ego_engine is router


def test_manager_accepts_ego_router_by_name_string():
    manager = GMCLinkManager(ego_router="recoverpose")
    assert isinstance(manager.ego_engine, RecoverPoseEgoRouter)


def test_manager_unknown_router_name_raises():
    import pytest

    with pytest.raises(KeyError):
        GMCLinkManager(ego_router="nonexistent")
