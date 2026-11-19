# Python guide

## Unconstrained minimum snap (free space)

The minimum-snap control points for given start/end position, velocity and acceleration have a
closed form, `C = SE @ Q`: one small matrix product, no optimiser. Both clamped and natural
B-splines are available. `solve_unconstrained` builds `SE` for you. The two solvers order its
columns differently, and both work in unit-knot time, so `dt` (knot spacing in seconds) rescales
the derivatives.

```python
import numpy as np
from coast_planner import MinSnapClamped, MinSnapNatural, TrajectorySampler

dt = 0.5                                   # seconds per knot interval
solver = MinSnapClamped(num_segments=10)   # or MinSnapNatural(10); degree defaults to 4

C = solver.solve_unconstrained(
    p0=[0, 0, -2], v0=[1, 0, 0], a0=[0, 0, 0],      # start state (NED)
    pf=[10, 5, -4], vf=[0, 0, 0], af=[0, 0, 0],     # end state
    dt=dt)                                          # -> (3, num_control_points)

traj = TrajectorySampler(C, solver.scaled_knots(dt), solver.degree)
print(traj.duration)                       # 5.0 s  (= num_segments * dt)
print(traj.position(traj.end_time))        # [10. 5. -4.]
samples = traj.sample(np.linspace(traj.start_time, traj.end_time, 100))  # (100, 9): pos|vel|acc

# Raw access, if you want to build SE yourself (native column order, unit-knot time):
Q = solver.get_Q_matrix()                  # (6, N)
SE = MinSnapClamped.build_SE([0, 0, -2], [1, 0, 0], [0, 0, 0], [10, 5, -4], [0, 0, 0], [0, 0, 0], dt=dt)
assert np.allclose(SE @ Q, C)
```

## Obstacle avoidance with Safe Flight Corridors

```python
import numpy as np
from coast_planner import SparseVoxelGrid, ObstacleBox, FrontEndConfig, TrajectoryPlanner

drone_radius = 0.49
grid = SparseVoxelGrid(resolution=0.5, inflation_radius=drone_radius)
grid.update_from_obstacles([ObstacleBox(min=[4.5, -0.5, -7.0], max=[5.5, 0.5, 0.0])])  # a pillar
# or from a sensor:  grid.update_from_scan(sensor_origin, hits)   # hits: (N, 3) array

cfg = FrontEndConfig(voxel_resolution=0.5, drone_physical_radius=drone_radius,
                     map_bounds=[100.0, 100.0, 15.0], min_altitude=drone_radius + 0.1,
                     planning_horizon=12.0)
planner = TrajectoryPlanner(cfg, grid, v_max=3.0, a_max=2.0)

result = planner.plan_mission(start_pos=[0, 0, -3], end_pos=[10, 0, -3],
                              start_vel=[0, 0, 0], start_acc=[0, 0, 0])
if result.success:
    traj = result.sampler()                # de Boor over result.control_points / knots / degree
    print(result.total_time_ms, result.solve_attempts, len(result.corridors))
    print(traj.position(traj.end_time))    # ~[10. 0. -3.]
    pts = result.control_points_xyz        # (N, 3); result.control_points is the flat array
```

`FrontEndConfig` accepts any field as a keyword (unknown names raise). Every array a binding
returns is a copy. The grid is shared with the planner (`shared_ptr`), so keep updating it
between plans and build a fresh planner per replan, as the ROS node does. Calls hold the GIL,
which serialises map updates against planning across Python threads (the engine itself is not
thread-safe).

## Visualization (optional)

`coast_planner.plot` draws trajectories and plans with matplotlib, for checking results by eye.
It is an optional extra, and nothing is plotted unless you call it:

```bash
pip install "coast-planner[viz]"          # or, from a clone: pip install -e ".[viz]"
```

```python
import matplotlib.pyplot as plt
from coast_planner.plot import plot_trajectory, plot_profiles, plot_plan

plot_trajectory(C, solver.scaled_knots(dt), solver.degree)  # free space: unconstrained solve, no map
plot_trajectory(C, solver.scaled_knots(dt), solver.degree, obstacles=boxes)  # ...drawn among obstacles
plot_profiles(C, solver.scaled_knots(dt), solver.degree)    # position / velocity / acceleration vs time
plot_plan(result, obstacles=boxes, goal=[10, 0, -3])        # planner result: corridors, obstacles, path
plot_plan(result, grid=grid, view="top")                    # plan view; draws the grid's occupied voxels
plt.show()
```

Each function returns the matplotlib Axes or Figure (`ax.figure.savefig("plan.png")`) and
never calls `plt.show()` itself, so it is safe in scripts and headless jobs. Plots display
NED as East-North-Up (altitude = −z); pass `frame="raw"` to plot x, y, z unchanged.
`coast-planner-demo` (or `python/examples/plot_demo.py` in a clone) renders all of them; see
[Test environments and the demo](#test-environments-and-the-demo).

## Test environments and the demo

`coast-planner-demo` plans through a test environment, plots the result, and checks it. Two
environments are built in:

| `--env` | What it tests |
|---|---|
| `simple` (default) | Two pillars between start and goal. |
| `hard` | A weave through three staggered rows of pillars, then a 3.5 m window in a wall. |

```bash
coast-planner-demo --env hard               # plan through the hard environment
coast-planner-demo --env hard --mode sfc    # only the corridor-constrained plan
coast-planner-demo --env hard --mode free   # only the unconstrained solve, drawn with the obstacles
coast-planner-demo --env hard --voxel 0.25  # the same environment on a finer map
coast-planner-demo --env hard --verbose     # also print the planner's own log
coast-planner-demo --env hard --save out    # write PNGs to ./out instead of opening windows
```

`--mode free` shows the closed-form minimum-snap solve from start to goal with no corridors.
It ignores the obstacles, so it usually flies straight through them, which is why the
corridors exist. `--mode sfc` runs `TrajectoryPlanner`. The default, `both`, shows both.

The demo plans all the way from start to goal: the planning horizon is off
(`planning_horizon = 0`). An environment tests whether the planner can weave through the
obstacles from A to B; how far ahead to plan on each replan is a choice for whoever
integrates the planner. To test with a horizon anyway, set `planning_horizon` in `config`.

It prints the planning time, split into the front end (A* and corridors) and the optimiser.
A route of 15 m or less through moderate clutter should plan in roughly 5–10 ms (the `hard`
environment takes about 7 ms at 0.5 m voxels), and the demo adds a note when one takes
longer. Then it classifies the result:

| Outcome | Meaning | Exit status |
|---|---|---|
| Checks passed | Reached the goal, at least the drone radius from every obstacle box. | 0 |
| Checks failed | The planner reported success, but the trajectory comes closer than the drone radius to an obstacle or stops short of the goal. **Likely a planner bug**: please report it with the environment file. | 1 |
| No trajectory, optimiser refused | A* found a route, but the optimiser found no feasible trajectory through its corridors. The route can be genuinely too tight to fly within the limits, so this is **not necessarily a bug**. `--verbose` shows which constraints blocked it. | 2 |
| No trajectory, no route | A* found no route to the goal. | 2 |
| Invalid environment | The start or goal is within the drone radius of an obstacle. | 2 |

Clearance is measured against the obstacle boxes themselves, not the voxel map, because the
map is deliberately conservative (see below).

### Designing your own environment

Export a built-in environment as a starting point, edit the JSON, and run it:

```bash
coast-planner-demo --env hard --export my_env.json
coast-planner-demo --env my_env.json
```

```json
{
  "name": "my_env",
  "description": "what this environment tests",
  "start": [0, 0, -2],
  "goal": [16, 0, -2],
  "obstacles": [
    {"min": [2.5, -0.5, -7], "max": [3.5, 0.5, 0], "note": "a pillar 7 m tall"}
  ],
  "drone_radius": 0.49,
  "v_max": 3.0,
  "a_max": 2.0,
  "voxel_resolution": 0.5,
  "config": {"sfc_width": 5.0}
}
```

- **Coordinates are NED**: x North, y East, z Down, in metres. The ground is `z = 0` and an
  altitude of 2 m is `z = -2`, so an obstacle standing on the ground has `max` z = 0 and a
  negative `min` z (its top).
- **Obstacles are axis-aligned boxes**, given by opposite corners `min` and `max` (every `min`
  coordinate below the matching `max`). Build pillars, walls and windows out of several boxes.
  `note` is ignored; use it to label boxes.
- Only `start` and `goal` are required. `voxel_resolution` is the map cell size; `--voxel`
  overrides it from the command line. `config` takes any other `FrontEndConfig` field (see
  [configuration.md](configuration.md)), and misspelled fields are rejected.

Things worth knowing while tinkering:

- **The planner sees obstacles as voxels.** Each box is inflated by `drone_radius` and rounded
  outward to whole cells of `voxel_resolution` (0.5 m by default). A box that grows by 5 cm
  can therefore add a whole row of cells, and close a gap that looked wide enough. In the
  `hard` environment, widening the pillars from 1.0 m to 1.1 m makes every variant
  infeasible.
- **Try more than one voxel size.** An environment that plans at one resolution can fail, or
  plan worse, at another, so `--voxel` is a quick robustness check. Finer maps also cost much
  more front-end time.
- **A gap must be about `2 × drone_radius` plus a cell or more wider than the obstacles
  beside it.** Two pillars 1 m apart with a 0.49 m drone leave almost nothing after inflation.
- **Openings approached at an angle are hard.** A window that is offset sideways or vertically
  from the approach, or a gap under an overhang, often fails where the same opening lined up
  with the path succeeds.

The same environments are available from Python, with helpers for common shapes:

```python
from coast_planner.scenarios import Scenario, load, pillar, wall, wall_with_window

sc = load("hard")                         # or load("my_env.json")
sc.obstacles.append(pillar(14.5, 1.0))    # add a pillar behind the window
sc.voxel_resolution = 0.4                 # and try a different map resolution
result, grid = sc.plan()
print(result.success, result.reached_goal)
sc.save("hard_plus_pillar.json")          # then: coast-planner-demo --env hard_plus_pillar.json

custom = Scenario(name="corridor", start=[0, 0, -2], goal=[12, 0, -2],
                  obstacles=[wall(6.0, -6.0, -1.5), wall(6.0, 1.5, 6.0)])  # a 3 m gap
```

## API reference

The bindings cover the two planning workflows end to end, not every C++ function:

| Python | C++ | Notes |
|---|---|---|
| `MinSnapClamped`, `MinSnapNatural` | `MinSnapEvalClamped`, `MinSnapEvalNatural` | `solve_unconstrained(...)`, `build_SE`, `get_Q_matrix`, `get_W`, `get_knots`, `scaled_knots`, `update_segments`. The full long names are also exported. |
| `SparseVoxelGrid`, `ObstacleBox` | `mapping::SparseVoxelGrid`, `mapping::ObstacleBox` | Map updates and state queries. Queries take **world points** (metres), not cell indices. |
| `FrontEndConfig` | `FrontEndConfig` | Every field; also accepts keyword arguments. |
| `TrajectoryPlanner` | `TrajectoryPlanner` | Constructor, `plan_mission`, `set_previous_path`. |
| `PlanningResult`, `Corridor` | `PlanningResult`, `sfc::SFCResult` | Plus `control_points_xyz` and `sampler()`. |
| `TrajectorySampler` | `bspline::TrajectorySampler` | Plus a vectorised `sample(times)`. |
| `coast_planner.plot` | — | Optional matplotlib plots: `plot_trajectory`, `plot_profiles`, `plot_plan` (needs the `viz` extra). |
| `coast_planner.scenarios` | — | Test environments: `Scenario`, `load`, `BUILTIN`, the `pillar` / `wall` / `wall_with_window` helpers and `box_clearance`. |

Not bound: the front-end and optimisation internals (`FrontEndSFC`, `astar_sfc`,
`matrix_builder`, `optimize`, MINVO builders), `set_front_end_override`, and the
`trajectory_metrics.hpp` helpers. Use C++ for those, or add them to
`cpp/src/python_bindings.cpp`.
