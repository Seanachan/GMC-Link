"""Pluggable ego-motion estimator registry.

Defines the structural contract every ego backend must satisfy and a name-keyed
registry so GMCLinkManager can swap backends (ORB 2D homography, recoverPose
6DoF, SparseMFE if weights are ever released) without touching residual-velocity
math. Contract matches the existing ORBHomographyEngine signature so the
historic backend is its own adapter.
"""
from __future__ import annotations

from typing import Dict, Protocol, Tuple, Type

import numpy as np

from gmc_link.core import ORBHomographyEngine


class EgoRouter(Protocol):
    """Structural protocol for ego-motion estimators.

    Implementations must return a 3x3 homography that maps points from
    `prev_frame` to `curr_frame` coordinates, plus a 2-vector of per-axis
    background-residual magnitudes in pixels (median RANSAC inlier residuals).
    """

    def estimate_homography(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        prev_bboxes=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        ...


OrbEgoRouter = ORBHomographyEngine


_REGISTRY: Dict[str, Type] = {}


def register_ego_router(name: str, cls: Type) -> None:
    _REGISTRY[name] = cls


def make_ego_router(name: str, **kwargs) -> EgoRouter:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown ego router '{name}'. Known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)


def available_ego_routers() -> Tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


register_ego_router("orb", OrbEgoRouter)
