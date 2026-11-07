"""Python interface to the C++ minimum-snap B-spline trajectory engine.

Two planning paradigms:

Unconstrained (free space, closed form, C = SE @ Q)::

    import numpy as np
    from min_snap import MinSnapClamped, TrajectorySampler

    solver = MinSnapClamped(num_segments=10)
    C = solver.solve_unconstrained(p0=[0, 0, 0], v0=[0, 0, 0], a0=[0, 0, 0],
                                   pf=[10, 5, -2], vf=[0, 0, 0], af=[0, 0, 0], dt=0.5)
    traj = TrajectorySampler(C, solver.scaled_knots(0.5), solver.degree)

SFC-constrained (obstacle avoidance through Safe Flight Corridors)::

    from min_snap import SparseVoxelGrid, ObstacleBox, FrontEndConfig, TrajectoryPlanner

    grid = SparseVoxelGrid(resolution=0.5, inflation_radius=0.5)
    grid.update_from_obstacles([ObstacleBox([4, -2, -6], [6, 2, 0])])
    planner = TrajectoryPlanner(FrontEndConfig(voxel_resolution=0.5), grid)
    result = planner.plan_mission([0, 0, -2], [12, 0, -2])

Coordinates are NED (z is negative-up); see the README.
"""

from ._core import (
    Corridor,
    FrontEndConfig,
    MinSnapEvalClamped,
    MinSnapEvalNatural,
    ObstacleBox,
    PlanningResult,
    SparseVoxelGrid,
    TrajectoryPlanner,
    TrajectorySampler,
)

# Shorter, pythonic aliases.
MinSnapClamped = MinSnapEvalClamped
MinSnapNatural = MinSnapEvalNatural

try:
    from importlib.metadata import PackageNotFoundError, version as _version

    try:
        __version__ = _version("min-snap")
    except PackageNotFoundError:  # imported from a plain CMake build, not pip-installed
        __version__ = "0.0.0+local"
except ImportError:  # pragma: no cover
    __version__ = "0.0.0+local"

__all__ = [
    "Corridor",
    "FrontEndConfig",
    "MinSnapClamped",
    "MinSnapEvalClamped",
    "MinSnapEvalNatural",
    "MinSnapNatural",
    "ObstacleBox",
    "PlanningResult",
    "SparseVoxelGrid",
    "TrajectoryPlanner",
    "TrajectorySampler",
]
