"""Pluggable ego-motion estimator backends for GMC-Link.

Importing this package eagerly registers the built-in routers so that
`make_ego_router("orb" | "recoverpose")` resolves without the caller needing
to know which submodule to pull in.
"""

from gmc_link.ego.ego_router import (  # noqa: F401
    EgoRouter,
    OrbEgoRouter,
    available_ego_routers,
    make_ego_router,
    register_ego_router,
)
from gmc_link.ego import recoverpose_ego  # noqa: F401 — side-effect registration
