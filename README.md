# Minimum-Snap-Trajectories

Real-time minimum-snap trajectory generation for multirotors flying through cluttered,
partially-unknown environments.

Given a start state, a goal, and a stream of LiDAR returns, the planner produces a
collision-free B-spline trajectory — typically in **5–20 ms**, fast enough to re-plan at
10 Hz. It is a self-contained C++17 library with no ROS dependency; the ROS integration
lives in whatever system consumes it.

---

## Contents

- [How it works](#how-it-works)
- [Coordinate conventions](#coordinate-conventions-read-this-first)
- [Building](#building)
- [Quick start](#quick-start)
- [API reference](#api-reference)
- [Configuration](#configuration)
- [Integrating into your own system](#integrating-into-your-own-system)
- [Testing](#testing)
- [Gotchas](#gotchas)
- [Repository layout](#repository-layout)
- [Known limitations](#known-limitations)

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
| C++ compiler | C++17 | |
| Eigen3 | 3.3+ | `apt install libeigen3-dev` |
| OSQP | 1.0+ | the QP solver; must be installed (see below) |
| GoogleTest | any | `apt install libgtest-dev` — only for the test suite |
| nanoflann | bundled | vendored at `cpp/deps/nanoflann.hpp`, nothing to install |
| pybind11 | fetched | auto-downloaded if Python bindings are enabled |

OSQP is not usually packaged; build it from source:

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

### CMake options

| Option | Default | Purpose |
|---|---|---|
| `MIN_SNAP_EIGEN_SAFETY` | `ON` | Turn Eigen bounds assertions into exceptions even in release builds. **Leave this on** — see [Gotchas](#gotchas). |
| `BUILD_EXAMPLES` | `ON` | Build `examples/`. Kept on by default so CI notices if an example stops compiling. |
| `BUILD_PYTHON_BINDINGS` | `OFF` | The pybind11 module. Off because it does not currently compile — see [Known limitations](#known-limitations). |

---

## Quick start

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
| `get_unknown_frontier()` | UNKNOWN cells adjacent to FREE — the boundary corridors must not cross. Computed lazily and cached. |

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
| `astar_max_nodes` | 6000 | Deterministic cap on A* expansions. Keeps planning reproducible; see [Gotchas](#gotchas). |
| `planning_horizon` | 10.0 | How far ahead A* searches (m). **The single most influential parameter** — see [Gotchas](#gotchas). |
| `unknown_traversal_cost` | 2.0 | Extra cost multiplier per step through UNKNOWN. `0` treats unknown as free (unsafe); negative hard-blocks it (cannot route around corners). |
| `sensor_range` | -1.0 | Search radius cap (m), -1 disables. |
| `fov_vertical_min/max` | -7 / 52 | Parsed and passed but **no longer gate anything** — unknown-space cost subsumed the old FOV gate. |

### Trajectory
| Field | Default | Notes |
|---|---|---|
| `degree` | 4 | Spline degree. Minimum snap requires ≥ 4. |
| `spline_type` | "clamped" | `"clamped"` or `"natural"`. |
| `aircraft_type` | "multi-rotor" | `"multi-rotor"` or `"fixed-wing"` (different corridor and allocation strategies). |
| `v_max`, `a_max` | 3.0 / 2.0 | Kinematic limits. `TrajectoryPlanner` overwrites these from its own constructor arguments, so set them there. |

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

### Unit tests

```bash
cd cpp/build
make bspline_tests -j$(nproc)
ctest --output-on-failure
```

(The engine is statically linked, so there is no shared-library path to get wrong. If you
are working from an older checkout where it was built `SHARED`, you may still need
`LD_LIBRARY_PATH=$PWD` to avoid picking up a stale copy from elsewhere.)

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
    -Iinclude -I/usr/include/eigen3 -I/usr/local/include -L/usr/local/lib -losqp \
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

**Planning is deterministic — keep it that way.** A*'s budget is a node count
(`astar_max_nodes`), not a wall-clock timer, so identical inputs give identical trajectories.
This matters more than it sounds: when the budget was time-based, the same scenario reached
the goal on some runs and stranded the vehicle on others, purely from CPU scheduling jitter.
A wall-clock backstop still exists for pathological cases and logs loudly when it fires.

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
cpp/                    the C++ engine — this is the production artifact
  include/              public headers
  src/                  implementation
  tests/                GoogleTest suite
  tools/                stress_planner.cpp
  deps/                 vendored nanoflann
python/                 original Python prototype the C++ is a port of
eVTOL_BSplines/         general B-spline tooling for eVTOL applications
rrt-astar_mavsim/       RRT + SFC + B-spline optimisation experiments
data/                   benchmark outputs
```

The Python implementation under `python/` predates the C++ port and remains a useful
reference for the underlying maths — several C++ files are direct ports and say so in their
header comments. It is **not** kept in sync with the C++ engine.

```bash
pip install -r requirements.txt   # for the Python side
```

---

## Known limitations

Honest list of rough edges, current as of this writing:

- **Python bindings do not compile.** `src/python_bindings.cpp` still calls the old
  `TrajectoryPlanner(config, obstacles, ...)` signature; the constructor now takes a
  `shared_ptr<SparseVoxelGrid>`. Build with `-DBUILD_PYTHON_BINDINGS=OFF` until updated.
- **Every plan ends at a full stop.** The terminal boundary condition pins zero velocity and
  acceleration. With a goal beyond `planning_horizon` that stop lands at the horizon rather
  than the goal, which costs average speed.
- **Some geometries are genuinely infeasible** and the QP correctly refuses, taking tens of
  milliseconds to do so. Refusals are handled (the caller holds position), but they are not
  rare.
- **`is_occupied_radius` and the KD-tree cover real obstacles only**, not the unknown
  frontier — deliberate, so trajectory validation is not tripped by unmeasured space.
- **Not thread-safe.** Serialise map updates against planning.
