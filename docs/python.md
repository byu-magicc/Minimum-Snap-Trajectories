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
plot_profiles(C, solver.scaled_knots(dt), solver.degree)    # position / velocity / acceleration vs time
plot_plan(result, obstacles=boxes, goal=[10, 0, -3])        # planner result: corridors, obstacles, path
plot_plan(result, grid=grid, view="top")                    # plan view; draws the grid's occupied voxels
plt.show()
```

Each function returns the matplotlib Axes or Figure (`ax.figure.savefig("plan.png")`) and
never calls `plt.show()` itself, so it is safe in scripts and headless jobs. Plots display
NED as East-North-Up (altitude = −z); pass `frame="raw"` to plot x, y, z unchanged.
`python/examples/plot_demo.py` renders all of them.

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

Not bound: the front-end and optimisation internals (`FrontEndSFC`, `astar_sfc`,
`matrix_builder`, `optimize`, MINVO builders), `set_front_end_override`, and the
`trajectory_metrics.hpp` helpers. Use C++ for those, or add them to
`cpp/src/python_bindings.cpp`.
