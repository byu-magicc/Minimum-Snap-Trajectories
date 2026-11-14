# C++ guide

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

## API: `mapping::SparseVoxelGrid`

The occupancy map. Thread-compatible but **not** thread-safe — serialise access, or run
sensor updates and planning on the same executor thread.

| Method | Purpose |
|---|---|
| `update_from_scan(origin, hits)` | Primary entry point. Ray-marches each beam: FREE along the way, OCCUPIED at the return, clears stale occupancy. |
| `update_from_obstacles(boxes)` | Marks OCCUPIED only, everything else stays UNKNOWN. For synthetic maps and tests — carries no free-space information. |
| `set_accumulate(on, window_radius_m)` | Accumulation across scans, default **on**, 25 m window. See [Gotchas](limitations.md#gotchas) — turning this off is not merely "less memory". |
| `set_frontier_radius(r)` / `set_frontier_focus(centre, r)` | Scope frontier extraction. Cost scales with the cube of the radius. |
| `is_free / is_unknown / is_occupied_inflated / is_occupied_raw` | O(1) cell queries. |
| `is_occupied_radius(pt, r)` | KD-tree query — is anything solid within `r` of this point? Useful for validating a buffered trajectory. |
| `get_unknown_frontier()` | UNKNOWN cells adjacent to FREE. Computed lazily and cached. **Not** fed to the corridor builder by default (`set_frontier_blocks_corridors(true)` turns that back on); unknown space is handled by A*'s `unknown_traversal_cost` instead. |

## API: `TrajectoryPlanner`

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

## API: `PlanningResult`

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
