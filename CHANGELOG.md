# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Until 1.0.0, minor versions may
include breaking changes; they are called out under **Changed** or **Removed**.

Each pull request adds its entry under **Unreleased**. Cutting a release turns that section
into a version (see [`RELEASING.md`](RELEASING.md)).

## [Unreleased]

The first release, 0.1.0, will be cut from this section.

### Added

- **Unconstrained minimum-snap solvers** for clamped and natural uniform B-splines
  (`MinSnapEvalClamped`, `MinSnapEvalNatural`). The optimal control points are a closed-form
  linear map of the boundary states, `C = SE · Q`, so a free-space solve is one matrix product.
- **SFC-constrained mission planner** (`TrajectoryPlanner::plan_mission`):
  - a three-state (FREE / OCCUPIED / UNKNOWN) sparse voxel map (`SparseVoxelGrid`), raycast
    with Amanatides–Woo and accumulated across scans;
  - an A* front end in which unknown space is traversable at a cost, bounded by
    `planning_horizon`;
  - Safe Flight Corridors built along the smoothed path;
  - an OSQP minimum-snap QP with MINVO kinodynamic bounds and corridor containment through
    exclusive and bridge control-point pools;
  - whole-path time allocation, a time-stretch retry loop, and a wall-clock solve budget
    (`max_solve_time_ms`).
- **Header-only helpers:** `bspline_eval.hpp` (de Boor `TrajectorySampler`) and
  `trajectory_metrics.hpp` (duration, jerk RMS, control effort and related scores).
- **Safety net for release builds:** `eigen_safety.hpp` is force-included so Eigen bounds
  assertions throw instead of corrupting memory (`MIN_SNAP_EIGEN_SAFETY`, on by default).
- **Tools:** `minimal_planner_loop` example, stress sweep (`stress_planner`), QP-failure replay
  tools, and probes for walls and `v_max`.
- **Python package `min_snap`** (pybind11), installable with `pip install`. It covers the
  unconstrained solvers, the voxel grid, `FrontEndConfig`, `TrajectoryPlanner`,
  `PlanningResult` (with its corridors) and `TrajectorySampler`. A test suite checks boundary
  states and minimum-snap optimality by evaluating the splines. (#2)
- **Continuous integration** on Linux, macOS and Windows: C++ build, tests and example, plus
  `pip install` and pytest on Python 3.9 and 3.13. (#3, #7)
- **Dependency bundling:** Eigen 3.4.0, OSQP v1.0.0 and GoogleTest 1.15.2 are downloaded when
  not installed. `MIN_SNAP_BUNDLED_DEPS=ON` always downloads them, and pip builds use it, so
  the Python package needs only a compiler and git. (#7)
- BSD 3-Clause license.
- Contributing guide, code of conduct (Contributor Covenant 2.1), and issue and
  pull-request templates. (#24)

### Changed

- `TrajectoryPlanner` takes a `std::shared_ptr<mapping::SparseVoxelGrid>` rather than a list
  of obstacle boxes. The Python bindings were rebuilt for this API. (#2)
- `optimize.cpp` includes `<osqp.h>` instead of `<osqp/osqp.h>`, so OSQP installed outside a
  default include path (or the bundled copy) works. (#7)
- README brought up to date: corrected defaults, Python API reference, cross-platform
  build commands, known limitations. (#8)

### Fixed

- The default CMake build no longer fails on machines without roscopter: `monte_carlo`, which
  needs a roscopter header, is skipped when that header is not found. (#3)
- Windows (MSVC) build: force-include with `/FI`, `_USE_MATH_DEFINES`, `/bigobj`, `/utf-8`,
  and missing `<chrono>` includes in two test files. (#7)
- Python bindings segfaulted on NumPy array arguments when built with pybind11 < 2.12, which
  predates NumPy 2; pybind11 ≥ 2.12 is now required. (#2)
- `TrajectoryPlanner` unit tests no longer depend on CI runner speed. (#25)
- The README documented `unknown_traversal_cost = 2.0`, a value that makes A* flood outward;
  the real default is 0.5. Also corrected `astar_max_nodes` (3000) and `planning_horizon`
  (12.0). (#8)

[Unreleased]: https://github.com/byu-magicc/Minimum-Snap-Trajectories/commits/main
