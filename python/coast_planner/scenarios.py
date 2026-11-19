"""Test environments for the planner: obstacles, start and goal, and planner settings.

The demo (``coast-planner-demo --env NAME``) plans through these. Two are built in:

    simple  two pillars between start and goal
    hard    a weave through three staggered rows of pillars, then a window in a wall

To design your own, export a built-in one as a starting point, edit it, and run it::

    coast-planner-demo --env hard --export my_env.json
    coast-planner-demo --env my_env.json

or build a ``Scenario`` in Python with ``pillar``, ``wall`` and ``wall_with_window``. The file
format is described in ``Scenario.to_dict``. Coordinates are NED: x North, y East, z Down, so
an altitude of 3 m is ``z = -3`` and the ground is ``z = 0``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from ._core import FrontEndConfig, ObstacleBox, SparseVoxelGrid, TrajectoryPlanner

__all__ = ["Scenario", "BUILTIN", "load", "pillar", "wall", "wall_with_window", "box_clearance"]


def pillar(north, east, half_width=0.5, height=7.0):
    """A square pillar standing on the ground, centred at (north, east)."""
    return ObstacleBox([north - half_width, east - half_width, -height],
                       [north + half_width, east + half_width, 0.0])


def wall(north, east_min, east_max, height=8.0, thickness=0.5):
    """A wall across the North axis at ``north``, from East ``east_min`` to ``east_max``."""
    return ObstacleBox([north - thickness / 2, east_min, -height],
                       [north + thickness / 2, east_max, 0.0])


def wall_with_window(north, half_span, window_east, window_alt, height=8.0, thickness=0.5):
    """A wall across the North axis with a rectangular window; returns four boxes.

    ``window_east = (e0, e1)`` and ``window_alt = (a0, a1)`` (altitudes, metres above ground)
    give the opening. The wall spans East from ``-half_span`` to ``half_span``.
    """
    (e0, e1), (a0, a1) = window_east, window_alt
    x0, x1 = north - thickness / 2, north + thickness / 2
    return [ObstacleBox([x0, -half_span, -height], [x1, e0, 0.0]),  # west of the window
            ObstacleBox([x0, e1, -height], [x1, half_span, 0.0]),   # east of the window
            ObstacleBox([x0, e0, -a0], [x1, e1, 0.0]),              # below it
            ObstacleBox([x0, e0, -height], [x1, e1, -a1])]          # above it


def box_clearance(points, boxes):
    """Smallest distance (m) from any of ``points`` (N, 3) to any box; 0 inside a box."""
    P = np.asarray(points, dtype=float).reshape(-1, 3)
    best = np.inf
    for b in boxes:
        lo, hi = np.asarray(b.min, float), np.asarray(b.max, float)
        d = np.linalg.norm(np.maximum(np.maximum(lo - P, P - hi), 0.0), axis=1)
        best = min(best, float(d.min()))
    return best


@dataclass
class Scenario:
    """An environment to plan through. Coordinates are NED (z = -altitude)."""

    name: str
    start: list
    goal: list
    obstacles: list = field(default_factory=list)  # ObstacleBox
    description: str = ""
    drone_radius: float = 0.49
    v_max: float = 3.0
    a_max: float = 2.0
    voxel_resolution: float = 0.5  # map cell size (m)
    config: dict = field(default_factory=dict)  # other FrontEndConfig fields

    def front_end_config(self):
        """The planner settings. Unlike the library default, the planning horizon is off
        (``planning_horizon=0``), so A* searches all the way to the goal: an environment tests
        whether the planner can get from start to goal, and how far ahead to plan is the
        integrator's choice. Set ``planning_horizon`` in ``config`` to test with one."""
        kw = dict(drone_physical_radius=self.drone_radius,
                  min_altitude=self.drone_radius + 0.1, planning_horizon=0.0)
        kw.update(self.config)
        kw["voxel_resolution"] = self.voxel_resolution
        return FrontEndConfig(**kw)

    def grid(self):
        cfg = self.front_end_config()
        grid = SparseVoxelGrid(resolution=cfg.voxel_resolution, inflation_radius=self.drone_radius)
        grid.update_from_obstacles(self.obstacles)
        return grid

    def plan(self, grid=None):
        """Plan from start to goal; returns ``(result, grid)``."""
        grid = grid if grid is not None else self.grid()
        planner = TrajectoryPlanner(self.front_end_config(), grid, v_max=self.v_max,
                                    a_max=self.a_max)
        return planner.plan_mission(self.start, self.goal), grid

    def to_dict(self):
        """The JSON file format. Every key but ``start`` and ``goal`` is optional::

            {
              "name": "my_env",
              "description": "what this tests",
              "start": [0, 0, -2],            # NED, metres: z = -altitude
              "goal": [15, 0, -2],
              "obstacles": [                  # axis-aligned boxes; "note" is ignored
                {"min": [2.5, -1.7, -7], "max": [3.5, -0.7, 0], "note": "pillar"}
              ],
              "drone_radius": 0.49,           # obstacles are inflated by this
              "v_max": 3.0, "a_max": 2.0,     # per-axis limits (m/s, m/s^2)
              "voxel_resolution": 0.5,        # map cell size (m)
              "config": {"sfc_width": 5.0}    # any other FrontEndConfig field
            }
        """
        return {
            "name": self.name,
            "description": self.description,
            "start": [float(v) for v in self.start],
            "goal": [float(v) for v in self.goal],
            "obstacles": [{"min": [float(v) for v in b.min], "max": [float(v) for v in b.max]}
                          for b in self.obstacles],
            "drone_radius": self.drone_radius,
            "v_max": self.v_max,
            "a_max": self.a_max,
            "voxel_resolution": self.voxel_resolution,
            "config": dict(self.config),
        }

    @classmethod
    def from_dict(cls, d, name="custom"):
        unknown = set(d) - {"name", "description", "start", "goal", "obstacles", "drone_radius",
                            "v_max", "a_max", "voxel_resolution", "config"}
        if unknown:
            raise ValueError(f"unknown scenario keys: {sorted(unknown)}")
        for key in ("start", "goal"):
            if key not in d or len(d[key]) != 3:
                raise ValueError(f"scenario needs '{key}' as [x, y, z] (NED)")
        for key in ("voxel_resolution", "drone_physical_radius"):
            if key in d.get("config", {}):
                top = "drone_radius" if key == "drone_physical_radius" else key
                raise ValueError(f"set '{top}' at the top level of the file, not in 'config'")
        obstacles = []
        for i, o in enumerate(d.get("obstacles", [])):
            lo, hi = o.get("min"), o.get("max")
            if lo is None or hi is None or len(lo) != 3 or len(hi) != 3:
                raise ValueError(f"obstacle {i} needs 'min' and 'max' as [x, y, z]")
            if any(a >= b for a, b in zip(lo, hi)):
                raise ValueError(f"obstacle {i}: every 'min' coordinate must be below 'max'")
            obstacles.append(ObstacleBox(lo, hi))
        s = cls(name=d.get("name", name), start=list(d["start"]), goal=list(d["goal"]),
                obstacles=obstacles, description=d.get("description", ""),
                config=dict(d.get("config", {})))
        for key in ("drone_radius", "v_max", "a_max", "voxel_resolution"):
            if key in d:
                setattr(s, key, float(d[key]))
        s.front_end_config()  # unknown config fields raise here, not mid-plan
        return s

    def save(self, path):
        """Write the JSON file format, one obstacle per line so it is easy to edit by hand."""
        d = self.to_dict()
        obstacles = d.pop("obstacles")
        lines = [f"  {json.dumps(k)}: {json.dumps(v)}" for k, v in d.items()]
        rows = ",\n".join(f"    {json.dumps(o)}" for o in obstacles)
        lines.insert(4, f'  "obstacles": [\n{rows}\n  ]' if obstacles else '  "obstacles": []')
        with open(path, "w", encoding="utf-8") as f:
            f.write("{\n" + ",\n".join(lines) + "\n}\n")


def _simple():
    return Scenario(
        name="simple",
        description="Two pillars between start and goal.",
        start=[0.0, 0.0, -3.0], goal=[11.0, 0.0, -3.0],
        obstacles=[pillar(5.0, 0.0, height=7.0), pillar(8.0, 2.0, height=7.0)])


def _hard():
    # Three staggered rows of pillars 3.5 m apart, so no straight line gets through, then a
    # wall with a 3.5 m-wide window. Narrower windows (about 2.8 m and below) or fatter pillars
    # make this infeasible for the planner; that edge is a good place to start experimenting.
    rows = []
    for i, north in enumerate((3.0, 6.0, 9.0)):
        offset = 0.0 if i % 2 == 0 else 1.75
        rows += [pillar(north, offset + 3.5 * k) for k in range(-2, 3)
                 if abs(offset + 3.5 * k) <= 6]
    return Scenario(
        name="hard",
        description="Weave through three staggered rows of pillars, then fly through a window.",
        start=[0.0, 0.0, -2.0], goal=[16.0, 0.0, -2.0],
        obstacles=rows + wall_with_window(13.0, 6.0, window_east=(-1.75, 1.75),
                                          window_alt=(0.75, 3.25)))


# Built-in scenarios by name. Each call returns a fresh Scenario, so editing one is safe.
BUILTIN = {"simple": _simple, "hard": _hard}


def load(name_or_path):
    """A built-in scenario by name, or a scenario from a JSON file."""
    if name_or_path in BUILTIN:
        return BUILTIN[name_or_path]()
    if not str(name_or_path).lower().endswith(".json"):
        raise ValueError(f"unknown scenario {name_or_path!r}: use one of {sorted(BUILTIN)} "
                         "or a path to a .json file")
    with open(name_or_path, encoding="utf-8") as f:
        d = json.load(f)
    return Scenario.from_dict(d, name=os.path.splitext(os.path.basename(name_or_path))[0])
