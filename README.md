# Minimum-Snap-Trajectories

[![CI](https://github.com/byu-magicc/Minimum-Snap-Trajectories/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/byu-magicc/Minimum-Snap-Trajectories/actions/workflows/ci.yml)

Real-time minimum-snap trajectory generation for multirotors flying through cluttered,
partially-unknown environments.

Given a start state, a goal, and a stream of LiDAR returns, the planner produces a
collision-free B-spline trajectory — typically in **5–20 ms**, fast enough to re-plan at
10 Hz. It is a self-contained C++17 library with no ROS dependency; the ROS integration
lives in whatever system consumes it.

---

## Contents

- [How it works](#how-it-works)
- [Coordinate conventions](#coordinate-conventions--read-this-first)
- [Building](#building)
- [Quick start](#quick-start)
- [API reference](#api-reference)
- [Configuration](#configuration)
- [Integrating into your own system](#integrating-into-your-own-system)
- [Benchmarking](#benchmarking)
- [Testing](#testing)
- [Gotchas](#gotchas)
- [Repository layout](#repository-layout)
- [Known limitations](#known-limitations)
- [Citing](#citing)
- [License](#license)

---

## How it works

```
LiDAR returns + sensor origin
        │
        ▼
SparseVoxelGrid                     three-state map: FREE / OCCUPIED / UNKNOWN
        │                           accumulated across scans, raycast-cleared
        ▼
FrontEndSFC::get_corridors_astar
        │
        ├─▶ A* over the voxel grid   occupied = blocked, unknown = extra cost
        │                            bounded by planning_horizon
        ├─▶ line-of-sight smoother   collapses the cell path to few waypoints
        └─▶ Safe Flight Corridors    one oriented box per path segment, each of its
                                     six faces shrunk independently against obstacles
        │
        ▼
OSQP quadratic program              minimum-snap B-spline control points, constrained
  + MINVO kinodynamic bounds        to stay inside the corridors, velocity/accel limited
        │
        ▼
PlanningResult                      control points + knot vector + corridors + timings
```

**The three-state map** is what lets the planner distinguish *"measured empty"* from
*"never looked at"*. Each beam is walked from the sensor origin with Amanatides–Woo: cells
along the way become `FREE` (and any stale occupancy there is cleared), the return point
becomes `OCCUPIED`, everything else stays `UNKNOWN`. Observations accumulate across scans
inside a bounded window around the vehicle.

**Corridor containment without integer variables.** For a degree-*d* B-spline, each curve
span is the convex hull of *d+1* consecutive control points. Control points are allocated
into *exclusive* pools (constrained to one corridor) separated by *bridge* pools of exactly
*p* points constrained to the **intersection** of two adjacent corridors. Every window of
*p+1* consecutive control points therefore lies wholly within at least one corridor, so
every curve segment is contained — in continuous time, with no mixed-integer assignment.
(That is the design. There are known exceptions; see [Known limitations](#known-limitations).)

---

## Coordinate conventions — read this first

**Everything is NED.** North-East-Down, meters.

- **`z` is negative-up.** An altitude of 5 m is `z = -5.0`. This trips up almost everyone,
  and has historically caused real bugs in this codebase (a geofence that demanded `z >= 0`
  made every solve infeasible; test fixtures written positive-up made A* refuse to plan).
- **The world is centred on the origin.** `map_bounds` is a *full extent*, so the reachable
  region is `±map_bounds/2` on each axis. With the default `(100, 100, 15)` that is
  x,y ∈ [-50, +50] and z ∈ [-7.5, +7.5] — i.e. only **7.5 m of usable altitude**.
- **`min_altitude` is positive-up** (meters above ground) and converted internally. It is
  the one field that does not follow the sign convention, because it reads more naturally.

If the planner mysteriously refuses to produce corridors, check these first.

---

## Building

### Dependencies

| Dependency | Version | Notes |
|---|---|---|
| CMake | ≥ 3.16 | |
| C/C++ compiler | C++17 | Linux: GCC or Clang. macOS: Xcode Command Line Tools (`xcode-select --install`). Windows: Visual Studio 2019+ or its Build Tools, with the "Desktop development with C++" workload |
| Eigen3 | 3.3+ | **optional to install**: used if found, otherwise 3.4.0 is downloaded |
| OSQP | 1.x | the QP solver. **Optional to install**: used if found, otherwise v1.0.0 is downloaded and linked statically |
| GoogleTest | any | only for the test suite; used if found, otherwise 1.15.2 is downloaded |
| nanoflann | bundled | vendored at `cpp/deps/nanoflann.hpp`, nothing to install |
| pybind11 | ≥ 2.12 | Python bindings only. pip installs it automatically; a CMake build uses an installed one or downloads 2.13.6. **2.11 (Ubuntu 24.04's `pybind11-dev`) is rejected** — it predates NumPy 2 and segfaults on any array argument. |
| NumPy | any | Python bindings only |

Nothing beyond CMake, a C/C++ compiler and git has to be installed. When Eigen or OSQP is
missing, the configure step downloads it into the build tree, which needs network access the
first time. Set `-DMIN_SNAP_BUNDLED_DEPS=ON` to always use the downloaded copies, even when
system ones exist. `pip install` does this, so the Python package is self-contained. A system
install is still worthwhile if another project in the same binary uses Eigen (roscopter does),
because two Eigen versions in one binary is undefined behaviour. In that case install them and
leave the option off. To install OSQP system-wide from source:

```bash
git clone --recursive https://github.com/osqp/osqp
cd osqp && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) && sudo make install && sudo ldconfig
```

### Build

```bash
cd cpp && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

The same build on any OS, including Windows, where there is no `make`:

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --config Release --parallel
```

CI builds and tests this on Linux, macOS and Windows on every pull request.

Produces:

| Artifact | What it is |
|---|---|
| `libmin_snap_engine.a` | The planner itself — every `src/*.cpp` compiled into one **static** library. This is what you link against; the code ends up inside your binary. |
| `bspline_tests` | GoogleTest suite (`ctest`). |
| `minimal_planner_loop` | The runnable example. |

It is **static** deliberately. As a shared library it tended to exist in two places at once
— this build tree, and the copy a consumer installs after pulling the repo in via
`add_subdirectory()` — and the loader would silently pick whichever it found first. A stale
static archive fails at *link* time with a clear undefined-symbol error instead; a stale
shared one failed at *run* time with garbage. Nothing needs installing or finding at runtime.

### Python package

The engine ships as the `min_snap` Python package. From a clone:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .            # builds cpp/ with scikit-build-core, installs min_snap in editable mode
python -c "import min_snap; print(min_snap.__version__)"
```

A venv is recommended (on Ubuntu the system Python refuses `pip install`). Works on Linux,
macOS and Windows with Python 3.9 or newer; CI runs `pip install` + the test suite on all three
with Python 3.9 and 3.13. To use it from another project, install into that project's
environment instead:

```bash
pip install "min-snap @ git+https://github.com/byu-magicc/Minimum-Snap-Trajectories.git"
```

Nothing needs to be installed system-wide besides a C/C++ compiler and git. pip fetches
pybind11, scikit-build-core, CMake/Ninja if needed, and NumPy. The build downloads Eigen and
OSQP and links OSQP statically, so the installed module has no runtime library dependencies.
It also skips the GoogleTest suite and examples. The compiled build is
kept under `build/` so re-running `pip install -e .` after a C++ change is incremental.
**Python edits under `python/min_snap/` are live; C++ edits need `pip install -e .` again.**

Without pip, a plain CMake build writes the extension module straight into the package:

```bash
cd cpp/build && cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON && make -j$(nproc) min_snap_bindings
cd ../.. && PYTHONPATH=python python -c "import min_snap"
```

Don't mix the two in one environment: a `_core*.so` left in `python/min_snap/` by a CMake
build can shadow the pip-installed one when `python/` is on `sys.path`.

### CMake options

| Option | Default | Purpose |
|---|---|---|
| `MIN_SNAP_BUNDLED_DEPS` | `OFF` | Always download Eigen 3.4.0 / OSQP v1.0.0 instead of using system copies. With `OFF`, only missing ones are downloaded. pip turns it on. |
| `MIN_SNAP_EIGEN_SAFETY` | `ON` | Turn Eigen bounds assertions into exceptions even in release builds. **Leave this on** — see [Gotchas](#gotchas). |
| `BUILD_EXAMPLES` | `ON` | Build `examples/`. Kept on by default so CI notices if an example stops compiling. |
| `BUILD_TESTS` | `ON` | Build the GoogleTest suite (GoogleTest is downloaded if not installed). pip builds turn it off. |
| `BUILD_PYTHON_BINDINGS` | `OFF` | Build the `min_snap._core` pybind11 module. Off by default so `add_subdirectory()` consumers do not need Python headers; `pip install` turns it on. |

---

## Quick start

### Python — unconstrained minimum snap (free space)

The minimum-snap control points for given start/end position, velocity and acceleration have a
closed form, `C = SE @ Q`: one small matrix product, no optimiser. Both clamped and natural
B-splines are available. `solve_unconstrained` builds `SE` for you. The two solvers order its
columns differently, and both work in unit-knot time, so `dt` (knot spacing in seconds) rescales
the derivatives.

```python
import numpy as np
from min_snap import MinSnapClamped, MinSnapNatural, TrajectorySampler

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

### Python — SFC-constrained planning (obstacle avoidance)

```python
import numpy as np
from min_snap import SparseVoxelGrid, ObstacleBox, FrontEndConfig, TrajectoryPlanner

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

### C++

```cpp
#include "trajectory_planner.hpp"
#include "voxel_grid.hpp"

using namespace trajectory_planner;

// 1. Create the map. Inflation radius should be your vehicle's circumscribed radius --
//    corridors are guaranteed clear by exactly this margin.
const double voxel_resolution = 0.5;
const double drone_radius     = 0.49;
auto grid = std::make_shared<mapping::SparseVoxelGrid>(voxel_resolution, drone_radius);

// 2. Feed it sensor data. Pass the BEAM ORIGIN, not just the returns -- that is what lets
//    the map record which space was measured empty.
std::vector<Eigen::Vector3d> hits = /* LiDAR returns, world frame, NED meters */;
Eigen::Vector3d sensor_origin     = /* vehicle position, NED meters */;
grid->update_from_scan(sensor_origin, hits);

// 3. Configure the planner.
FrontEndConfig cfg;
cfg.voxel_resolution      = voxel_resolution;
cfg.drone_physical_radius = drone_radius;
cfg.map_bounds            = Eigen::Vector3d(100.0, 100.0, 15.0);
cfg.min_altitude          = drone_radius + 0.1;
cfg.planning_horizon      = 10.0;

// 4. Plan. Construct fresh each cycle so it picks up the latest map.
TrajectoryPlanner planner(cfg, grid, /*v_max=*/3.0, /*a_max=*/2.0);

auto result = planner.plan_mission(
    Eigen::Vector3d(0.0, 0.0, -3.0),   // start position  (3 m altitude)
    Eigen::Vector3d(20.0, 5.0, -3.0),  // goal
    Eigen::Vector3d(1.0, 0.0, 0.0),    // start velocity
    Eigen::Vector3d::Zero());          // start acceleration

if (result.success) {
    // result.control_points : flat [x0,y0,z0, x1,y1,z1, ...]
    // result.knots          : knot vector, unit-spaced (1 segment == 1 second)
    // result.degree         : spline degree (4)
    // Evaluate with de Boor -- see "Integrating into your own system".
}
```

**Always check `result.success`.** A refusal is normal and correct: goals inside obstacles,
goals too close to stop at from the current speed, or states outside the kinematic envelope
all produce `success == false`. Hold position and try again next cycle.

---

## API reference

### Python (`import min_snap`)

The bindings cover the two planning workflows end to end, not every C++ function:

| Python | C++ | Notes |
|---|---|---|
| `MinSnapClamped`, `MinSnapNatural` | `MinSnapEvalClamped`, `MinSnapEvalNatural` | `solve_unconstrained(...)`, `build_SE`, `get_Q_matrix`, `get_W`, `get_knots`, `scaled_knots`, `update_segments`. The full long names are also exported. |
| `SparseVoxelGrid`, `ObstacleBox` | `mapping::SparseVoxelGrid`, `mapping::ObstacleBox` | Map updates and state queries. Queries take **world points** (metres), not cell indices. |
| `FrontEndConfig` | `FrontEndConfig` | Every field; also accepts keyword arguments. |
| `TrajectoryPlanner` | `TrajectoryPlanner` | Constructor, `plan_mission`, `set_previous_path`. |
| `PlanningResult`, `Corridor` | `PlanningResult`, `sfc::SFCResult` | Plus `control_points_xyz` and `sampler()`. |
| `TrajectorySampler` | `bspline::TrajectorySampler` | Plus a vectorised `sample(times)`. |

Not bound: the front-end and optimisation internals (`FrontEndSFC`, `astar_sfc`,
`matrix_builder`, `optimize`, MINVO builders), `set_front_end_override`, and the
`trajectory_metrics.hpp` helpers. Use C++ for those, or add them to
`cpp/src/python_bindings.cpp`.

### `mapping::SparseVoxelGrid`

The occupancy map. Thread-compatible but **not** thread-safe — serialise access, or run
sensor updates and planning on the same executor thread.

| Method | Purpose |
|---|---|
| `update_from_scan(origin, hits)` | Primary entry point. Ray-marches each beam: FREE along the way, OCCUPIED at the return, clears stale occupancy. |
| `update_from_obstacles(boxes)` | Marks OCCUPIED only, everything else stays UNKNOWN. For synthetic maps and tests — carries no free-space information. |
| `set_accumulate(on, window_radius_m)` | Accumulation across scans, default **on**, 25 m window. See [Gotchas](#gotchas) — turning this off is not merely "less memory". |
| `set_frontier_radius(r)` / `set_frontier_focus(centre, r)` | Scope frontier extraction. Cost scales with the cube of the radius. |
| `is_free / is_unknown / is_occupied_inflated / is_occupied_raw` | O(1) cell queries. |
| `is_occupied_radius(pt, r)` | KD-tree query — is anything solid within `r` of this point? Useful for validating a buffered trajectory. |
| `get_unknown_frontier()` | UNKNOWN cells adjacent to FREE. Computed lazily and cached. **Not** fed to the corridor builder by default (`set_frontier_blocks_corridors(true)` turns that back on); unknown space is handled by A*'s `unknown_traversal_cost` instead. |

### `TrajectoryPlanner`

```cpp
TrajectoryPlanner(const FrontEndConfig& config,
                  std::shared_ptr<mapping::SparseVoxelGrid> grid,
                  double v_max = 3.0, double a_max = 2.0);

PlanningResult plan_mission(const Eigen::Vector3d& start_pos,
                            const Eigen::Vector3d& end_pos,
                            const Eigen::Vector3d& start_vel = Eigen::Vector3d::Zero(),
                            const Eigen::Vector3d& start_acc = Eigen::Vector3d::Zero());
```

Construct it fresh each planning cycle — it holds borrowed views into the grid, and
rebuilding is cheap (< 1 ms). The grid must outlive the planner; the `shared_ptr` guarantees
that.

### `PlanningResult`

| Field | Meaning |
|---|---|
| `success` | A usable trajectory was produced. **Check this.** |
| `reached_goal` | The path reached the *actual* goal, not just the planning horizon. |
| `control_points` | Flat `[x0,y0,z0, x1,y1,z1, ...]`. |
| `knots`, `degree` | Knot vector and spline degree, for evaluation. |
| `corridors` | The SFCs, each with `A_mat`/`b_vec` (as `Ax ≤ b`) and 8 vertices for visualisation. |
| `total_control_points` | Control-point count. Duration in seconds is `total_control_points - degree`. |
| `sfc_time_ms`, `opt_time_ms`, `matrix_time_ms`, `overhead_ms`, `total_time_ms` | Per-stage timings. `opt_time_ms` is summed across retries. |
| `solve_attempts` | > 1 means the trajectory was re-sized and re-solved. |
| `d_free_ahead`, `terminal_speed` | Measured-free run-out past the last corridor, and the end speed the braking bound allows from it. |

---

## Configuration

All fields of `FrontEndConfig`, with defaults:

### Geometry
| Field | Default | Notes |
|---|---|---|
| `voxel_resolution` | 0.5 | Map cell size (m). Drives cost and precision everywhere. |
| `drone_physical_radius` | 0.5 | Circumscribed vehicle radius. Corridors are clear by this margin — **your controller's tracking error must stay below it** or the guarantee does not transfer to the vehicle. |
| `map_bounds` | (100, 100, 15) | Full extent; usable region is ±half on each axis. |
| `min_altitude` | 0.3 | meters above ground, positive-up. |

### Corridors
| Field | Default | Notes |
|---|---|---|
| `sfc_width` / `sfc_height` | 5.0 | Lateral corridor extent is capped at `sfc_width - 2*drone_radius`. Raise this to let corridors expand into open space. |
| `sfc_start_ext` / `sfc_end_ext` | 5.0 | How far each corridor extends behind/ahead of its segment. Large values make neighbouring corridors overlap heavily; small values tighten the QP and can make it harder to solve. |
| `max_sfc_count` | -1 | Cap on corridors per plan (-1 = uncapped). A safety valve on constraint growth, not the way to bound the horizon. |

### Search
| Field | Default | Notes |
|---|---|---|
| `time_budget_factor` | 1.20 | Slack on the initial flight-time budget. The ideal straight-line time is systematically short once the trajectory must also curve and stop, so the first solve fails and the retry loop lengthens it — and added control points **are** flight time, permanently. Measured over 12 scenarios: 1.00 → 30% first-attempt success, 8.5 ms mean solve; 1.20 → 60% and 5.8 ms. |
| `astar_max_nodes` | 3000 | Deterministic cap on A* expansions. Keeps planning reproducible; see [Gotchas](#gotchas). |
| `astar_max_time_ms` | 25.0 | Wall-clock backstop for pathological searches only; the node cap is what normally binds. |
| `planning_horizon` | 12.0 | How far ahead A* searches (m). **The single most influential parameter** — see [Gotchas](#gotchas). |
| `unknown_traversal_cost` | 0.5 | Extra cost per step through UNKNOWN. `0` treats unknown as free (unsafe); negative hard-blocks it (cannot route around corners). Keep `1 + unknown_traversal_cost + astar_clearance_cost < 2.0`, or A* stops preferring the goal and floods outward (the old default of 2.0 did exactly that). |
| `sensor_range` | -1.0 | Search radius cap (m), -1 disables. |
| `fov_vertical_min/max` | -7 / 52 | Parsed and passed but **no longer gate anything** — unknown-space cost subsumed the old FOV gate. |

### Trajectory
| Field | Default | Notes |
|---|---|---|
| `degree` | 4 | Spline degree. Minimum snap requires ≥ 4. |
| `spline_type` | "clamped" | `"clamped"` or `"natural"`. |
| `aircraft_type` | "multi-rotor" | `"multi-rotor"` or `"fixed-wing"` (different corridor and allocation strategies). |
| `v_max`, `a_max` | 3.0 / 2.0 | Kinematic limits. `TrajectoryPlanner` overwrites these from its own constructor arguments, so set them there. Enforced **per axis** by default; see `enforce_norm_limits`. |
| `enforce_norm_limits` | false | `false`: each of `abs(v_x)`, `abs(v_y)`, `abs(v_z)` is `≤ v_max`, so speed can reach `√3·v_max` (likewise for acceleration). `true`: shrinks the box so speed really is `≤ v_max`, at the cost of more refusals on long paths. |
| `max_solve_time_ms` | 60.0 | Wall-clock cap on the OSQP retry loop. A failing solve burns the full iteration budget, so this keeps a doomed plan from overrunning the replan period. 0 disables. |

---

## Integrating into your own system

The library deliberately stops at "here are the control points". Wiring it into a vehicle
means supplying four things:

**1. A sensor callback.** Convert your point cloud to world-frame NED, and pass the beam
origin alongside the returns:

```cpp
grid->update_from_scan(vehicle_position_ned, hits_ned);
```

**2. Evaluate the spline.** Use `bspline_eval.hpp` (header-only, Eigen-only):

```cpp
#include "bspline_eval.hpp"

bspline::TrajectorySampler traj(result.control_points, result.knots, result.degree);
if (traj.valid()) {
    for (double t = traj.start_time(); t <= traj.end_time(); t += 0.02) {
        Eigen::Vector3d p = traj.position(t);
        Eigen::Vector3d v = traj.velocity(t);
        Eigen::Vector3d a = traj.acceleration(t);
    }
}
```

`TrajectorySampler` builds the velocity and acceleration curves once on construction, so
sampling at 50 Hz is cheap. Knot time is seconds — do not rescale.

**3. A replanning policy.** Deciding *when* to re-plan is your call. A workable policy:
re-plan at 10 Hz, but skip when the buffered trajectory is still long, still collision-free
against the latest map, and younger than ~0.5 s. Without an age limit the vehicle flies
open-loop and ignores newly revealed space; without a skip condition it stutters.

**4. Continuity between plans.** Do not plan from the vehicle's current state — plan from a
*stitch point* a few hundred milliseconds into the already-buffered trajectory, passing that
point's position, velocity and acceleration as the boundary conditions. Otherwise every
re-plan introduces a discontinuity.

**A complete worked example of all four**, runnable and ROS-free, is
`examples/minimal_planner_loop.cpp` (~200 lines, fake sensor and vehicle):

```bash
cd cpp
g++ -std=c++17 -O2 examples/minimal_planner_loop.cpp \
    -Iinclude -I/usr/include/eigen3 -I/usr/local/include \
    -Lbuild -lmin_snap_engine -L/usr/local/lib -losqp -o /tmp/minimal_loop
/tmp/minimal_loop
```

Or just build it as part of the project — it is a CMake target (`BUILD_EXAMPLES=ON` by
default), so `make` produces `build/minimal_planner_loop` and CI notices if it stops
compiling.

For the full production picture — buffering, dispatch rates, takeoff, visualisation, failure
handling — see `minimum_snap_manager.cpp` in the
[roscopter](https://github.com/rosflight/roscopter) integration.

---

## Benchmarking

`include/trajectory_metrics.hpp` (header-only) scores a planned trajectory:

```cpp
auto m = metrics::evaluate(result.control_points, result.knots, result.degree);
// m.duration, m.path_length, m.tortuosity, m.mean_speed, m.max_speed, m.max_accel,
// m.jerk_rms, m.control_effort, m.thrust_integral
```

- **`jerk_rms`** is the metric most likely to line up with other planners' published figures.
  It is `sqrt(mean(||jerk||^2))` over **uniform time** samples — an implementation sampling
  uniformly in arc length, or reporting per-axis, will get a different number for the same
  curve. Check before quoting a comparison.
- **`control_effort`** = `∫||a||² dt`, the usual control cost.
- **`thrust_integral`** = `∫||a − g|| dt`. An energy *proxy*, not joules: real electrical
  power goes roughly as thrust^1.5, so this understates aggressive segments. Fine for A/B.

`metrics::RunningStats` accumulates min/max/mean over a stream — feed it `total_time_ms` per
replan. A 10 Hz loop is broken by the tail, not the average, so the max is the number to
watch.

In the ROS integration these are published per replan on `computation_times`
(`replan_count`, `min_total_ms`, `max_total_ms`, `mean_total_ms`), and summarised to the log
on each waypoint arrival.

## Testing

### Python bindings

```bash
pip install -e ".[test]"
pytest python/tests                    # or: python python/tests/test_bindings.py
```

Checks the unconstrained solvers by evaluating the splines: boundary states reproduced at
several knot spacings, and no boundary-preserving perturbation lowers the snap cost. It also
plans past obstacles, both from `update_from_obstacles` and from a raycast scan. In a shell with
ROS sourced, run `env -u PYTHONPATH pytest ...`: ROS's `launch_testing` pytest plugin otherwise
loads and fails on a missing `yaml`.

### Unit tests

```bash
cmake --build cpp/build --config Release --target bspline_tests --parallel
ctest --test-dir cpp/build -C Release --output-on-failure
```

The engine is statically linked, so there is no shared-library path to get wrong.

### Stress / edge-case sweep

`tools/stress_planner.cpp` exercises the degenerate regimes: zero-length goals, goals inside
obstacles, starts inside obstacles, velocities above `v_max`, empty maps, horizons from
0.5 m to 50 m, and short-but-bending paths. Build it **without** `-DNDEBUG` so Eigen's
bounds assertions stay live, and with sanitizers:

```bash
cd cpp
g++ -std=c++17 -O0 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
    tools/stress_planner.cpp \
    src/voxel_grid.cpp src/astar_sfc.cpp src/front_end.cpp src/dynamic_sfc.cpp \
    src/static_sfc.cpp src/trajectory_planner.cpp src/optimize.cpp \
    src/min_snap_clamped.cpp src/min_snap_natural.cpp src/matrix_builder.cpp \
    -Iinclude -I/usr/include/eigen3 -I/usr/local/include/osqp -L/usr/local/lib -losqp \
    -o /tmp/stress_planner

export LD_LIBRARY_PATH=/usr/local/lib ASAN_OPTIONS=detect_leaks=0
N=$(/tmp/stress_planner --count)
for i in $(seq 0 $((N-1))); do
  /tmp/stress_planner $i > /dev/null 2>&1 || echo "FAILED case $i"
done
```

Sources are listed explicitly because `src/*.cpp` would pull in `python_bindings.cpp`. One
case per process so a crash isolates rather than ending the sweep.

**Run this after any change to `planning_horizon`, `max_sfc_count`, or time allocation** —
those push the planner into the small-problem regime, which is where the sharp edges are.

---

## Gotchas

**Release builds delete Eigen's bounds checks.** `-DNDEBUG` makes `eigen_assert` a no-op, so
an out-of-range `.block()` does not fault — it corrupts the heap and the process dies later
with something unrelated. `include/eigen_safety.hpp` is force-included into every
translation unit and makes those assertions **throw** instead, at no measurable cost. Keep
`MIN_SNAP_EIGEN_SAFETY=ON`.

**Accumulation is required, not an optimisation.** The beam pattern is angular, so past
roughly `voxel_resolution / beam_angle` (~9.5 m for 3° beams at 0.5 m voxels) neighbouring
beams are further apart than a voxel and a *single* scan's free space degenerates into a fan
of thin lines with gaps between them. The FREE/UNKNOWN boundary is then meaningless.
Accumulating across scans as the vehicle moves is what fills those gaps.
`set_accumulate(false)` exists for tests.

**`planning_horizon` dominates everything.** Cost grows sharply and non-linearly with it,
because unknown space is traversable and a distant goal makes A* fan out. Measured on one
scene: 10 m → 2 corridors, 11 control points, ~6 ms total; 20 m → 4 corridors, 32 control
points, ~65 ms. Longer horizons also produce more convoluted paths. Start short.

**Duration is tied to control-point count.** Knots are unit-spaced, so flight time in
seconds is `total_control_points - degree`. Adding control points for resolution also adds
flight time; there is currently no way to decouple them.

**`v_max` is a per-axis limit unless you ask otherwise.** With the default
`enforce_norm_limits = false` a diagonal trajectory can legitimately fly at up to
`√3 · v_max`. Set it to `true` if `v_max` must be a true speed limit.

**Planning is deterministic — keep it that way.** A*'s budget is a node count
(`astar_max_nodes`), not a wall-clock timer, so identical inputs give identical trajectories.
This matters more than it sounds: when the budget was time-based, the same scenario reached
the goal on some runs and stranded the vehicle on others, purely from CPU scheduling jitter.
A wall-clock backstop (`astar_max_time_ms`) still exists for pathological cases and logs
loudly when it fires.

**A detour longer than `planning_horizon` can strand the vehicle.** The horizon-projected
goal points into the obstacle and the search never sees the way around. Raising the horizon
costs solve time super-linearly, so this is a genuine trade rather than a tuning oversight.

**Rebuild consumers after changing this library.** It is statically linked, so a consumer
built against an older archive keeps the old code until it is relinked. That failure is at
least loud — an undefined symbol at link time — rather than the silent runtime corruption
the previous shared-library setup produced. `ldd <binary> | grep min_snap` should return
nothing; if it returns a path, something is still linking it dynamically.

**Corridors guarantee the *reference trajectory* is clear**, by exactly
`drone_physical_radius`. The vehicle is only actually safe while its tracking error stays
below that margin. Measure it.

---

## Repository layout

```
LICENSE                 BSD 3-Clause
.github/workflows/      CI: C++ and Python builds/tests on Linux, macOS, Windows
cpp/                    the C++ engine — this is the production artifact
  include/              public headers
  src/                  implementation (python_bindings.cpp is the pybind11 module)
  tests/                GoogleTest suite
  examples/             minimal_planner_loop.cpp
  tools/                stress_planner, replay/repro tools, monte_carlo (needs roscopter)
  deps/                 vendored nanoflann
  cmake/                MinSnapDeps.cmake: finds or downloads Eigen + OSQP
pyproject.toml          pip packaging (scikit-build-core -> cpp/CMakeLists.txt)
python/min_snap/        the Python package: bindings to the C++ engine
python/tests/           binding tests
python/                 (rest) original Python prototype the C++ is a port of
eVTOL_BSplines/         general B-spline tooling for eVTOL applications
rrt-astar_mavsim/       RRT + SFC + B-spline optimisation experiments
data/                   benchmark outputs
```

The rest of the Python code under `python/` (everything outside `min_snap/` and `tests/`) predates the C++ port and remains a useful
reference for the underlying maths — several C++ files are direct ports and say so in their
header comments. It is **not** kept in sync with the C++ engine.

```bash
pip install -r requirements.txt   # for the Python side
```

---

## Known limitations

Honest list of rough edges, current as of this writing:

- **No prebuilt wheels yet.** `pip install` compiles from source (about a minute, and it needs a
  compiler and network). The build is already self-contained, so publishing to PyPI is now
  just a matter of building wheels with cibuildwheel.
- **Every plan ends at a full stop.** The terminal boundary condition pins zero velocity and
  acceleration. With a goal beyond `planning_horizon` that stop lands at the horizon rather
  than the goal, which costs average speed.
- **Corridor containment has known exceptions.** (1) Control points pinned by a start
  velocity that points out of the first corridor are not corridor-constrained, so the opening
  span can leave it (measured: 2.3 cm). (2) Some allocations leave a *(degree+1)*-window of
  control points in no single corridor (measured: curve 4.5 mm outside the union). Obstacle
  clearance held in every case observed, but the corridor chain is not a strict certificate.
  `python/tests/test_bindings.py` reports the excursion per plan.
- **Wide obstacles dead ahead can be infeasible.** A 3 m-wide pillar directly on the path
  fails to plan: A* hugs it and the flanking corridors come out ~0.1 m wide.
- **Some geometries are genuinely infeasible** and the QP correctly refuses, taking tens of
  milliseconds to do so. Refusals are handled (the caller holds position), but they are not
  rare.
- **`is_occupied_radius` and the KD-tree cover real obstacles only**, not the unknown
  frontier — deliberate, so trajectory validation is not tripped by unmeasured space.
- **Not thread-safe.** Serialise map updates against planning.

---

## Citing

If you use this software in research, please cite it. Citation metadata is in
[`CITATION.cff`](CITATION.cff); GitHub's "Cite this repository" button generates BibTeX and APA
from it.

<!-- TODO (authors, #15): add the BibTeX for the ACC paper / arXiv preprint here, and the JOSS
     paper after acceptance. -->

---

## License

BSD 3-Clause; see [`LICENSE`](LICENSE). Copyright (c) 2026, BYU MAGICC Lab.
