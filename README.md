# COAST: Corridor-Optimized Aerially-Safe Trajectories

[![CI](https://github.com/byu-magicc/Minimum-Snap-Trajectories/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/byu-magicc/Minimum-Snap-Trajectories/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/coast-planner)](https://pypi.org/project/coast-planner/)
[![Python](https://img.shields.io/pypi/pyversions/coast-planner)](https://pypi.org/project/coast-planner/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)](https://github.com/byu-magicc/Minimum-Snap-Trajectories/blob/main/LICENSE)

COAST is a real-time trajectory planner for multirotors flying through cluttered,
partially-unknown environments. Given a start state, a goal and LiDAR returns, it produces a
smooth, snap-minimised B-spline trajectory that is kept inside Safe Flight Corridors as a
**hard constraint**: fast enough to re-plan on board at 10 Hz.

Online planners usually trade one of these for the other: penalty-based methods are fast but
give no collision guarantee, while corridor-based methods are safe but slow. COAST keeps both:

- **Corridors as hard constraints, without integer variables.** An A\* front end over a
  three-state (free / occupied / unknown) voxel map builds a chain of rectangular-prism Safe
  Flight Corridors that resize to the local obstacles. The B-spline's convex-hull and
  finite-support properties keep the curve inside the corridors by construction.
- **Non-conservative kinodynamic limits.** Velocity and acceleration bounds are enforced on
  MINVO control points, which bound the curve far more tightly than B-spline control points.
- **Snap minimisation as one sparse QP.** A precomputed penalty matrix makes the objective
  quadratic in the control points, solved with OSQP.
- **Fast.** Median QP solve **0.32 ms** and mean end-to-end replan **3.1 ms** across 1,000
  randomised forests at 5% obstacle density (103,687 solves). In closed-loop ROSflight
  simulation, **zero** commanded setpoints violated the per-axis velocity or acceleration
  limits (Intel i7-7700; see the COAST paper).
- **C++17 library with Python bindings**, no ROS dependency, on Linux, macOS and Windows.

Known limitations, including the exceptions to corridor containment, are listed in
[docs/limitations.md](https://github.com/byu-magicc/Minimum-Snap-Trajectories/blob/main/docs/limitations.md).

## Installation

```bash
pip install coast-planner            # Python 3.9–3.14 on Linux, macOS and Windows
pip install "coast-planner[viz]"     # optional: plotting with matplotlib
```

Prebuilt wheels are provided, so no compiler is needed. Installing into a virtual environment
is recommended:

| | Create | Activate |
|---|---|---|
| Linux / macOS | `python3 -m venv .venv` | `source .venv/bin/activate` |
| Windows (PowerShell) | `py -m venv .venv` | `.venv\Scripts\Activate.ps1` |

On Windows, if activation is blocked ("running scripts is disabled"), run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.

### From source (to modify the code)

Building from source needs a C++17 compiler and git. Everything else (Eigen, OSQP,
pybind11) is downloaded automatically.

| OS | Install the compiler |
|---|---|
| Ubuntu / Debian | `sudo apt install build-essential git` |
| macOS | `xcode-select --install` |
| Windows | [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/), workload "Desktop development with C++", plus [Git](https://git-scm.com/download/win) |

```bash
git clone https://github.com/byu-magicc/Minimum-Snap-Trajectories.git
cd Minimum-Snap-Trajectories
pip install -e ".[viz,test]"         # editable: Python edits apply immediately; re-run after C++ edits
```

To use the C++ library directly (CMake), see
[docs/building.md](https://github.com/byu-magicc/Minimum-Snap-Trajectories/blob/main/docs/building.md).

## Example

Plan around an obstacle and plot the result. Coordinates are **NED** (x north, y east,
z down), so an altitude of 3 m is `z = -3`.

```python
from coast_planner import SparseVoxelGrid, ObstacleBox, FrontEndConfig, TrajectoryPlanner
from coast_planner.plot import plot_plan
import matplotlib.pyplot as plt

radius = 0.49                                        # vehicle radius (m)
grid = SparseVoxelGrid(resolution=0.5, inflation_radius=radius)
pillar = ObstacleBox(min=[4.5, -0.5, -7.0], max=[5.5, 0.5, 0.0])
grid.update_from_obstacles([pillar])                 # or grid.update_from_scan(origin, lidar_points)

cfg = FrontEndConfig(voxel_resolution=0.5, drone_physical_radius=radius, min_altitude=radius + 0.1)
planner = TrajectoryPlanner(cfg, grid, v_max=3.0, a_max=2.0)
result = planner.plan_mission(start_pos=[0, 0, -3], end_pos=[10, 0, -3])

print(result)                                        # success, corridors, timing
traj = result.sampler()                              # position / velocity / acceleration at any time t
plot_plan(result, obstacles=[pillar], goal=[10, 0, -3])
plt.show()
```

More examples, including the closed-form unconstrained solver, are in
[`python/examples/`](https://github.com/byu-magicc/Minimum-Snap-Trajectories/tree/main/python/examples)
and the [Python guide](https://github.com/byu-magicc/Minimum-Snap-Trajectories/blob/main/docs/python.md).

## Use with ROS 2 and ROSflight

For a complete flight stack (LiDAR processing, 10 Hz replanning, setpoint dispatch and
trajectory tracking), see the ROSflight integration used in the COAST paper's experiments:

- [COAST_roscopter](https://github.com/apd915/COAST_roscopter): the ROScopter multirotor
  autopilot running COAST for real-time obstacle avoidance
- [COAST_rosflight_ros_pkgs](https://github.com/apd915/COAST_rosflight_ros_pkgs): ROSflight
  packages with the simulated forest benchmark worlds and sensor setup

## Documentation

- [How COAST works](https://github.com/byu-magicc/Minimum-Snap-Trajectories/blob/main/docs/how-it-works.md): pipeline, corridor containment, coordinate conventions
- [Python guide](https://github.com/byu-magicc/Minimum-Snap-Trajectories/blob/main/docs/python.md) and [C++ guide](https://github.com/byu-magicc/Minimum-Snap-Trajectories/blob/main/docs/cpp.md): usage, API reference, integrating into your own system
- [Configuration](https://github.com/byu-magicc/Minimum-Snap-Trajectories/blob/main/docs/configuration.md): every `FrontEndConfig` parameter
- [Building and development](https://github.com/byu-magicc/Minimum-Snap-Trajectories/blob/main/docs/building.md): CMake options, tests, benchmarks
- [Gotchas and known limitations](https://github.com/byu-magicc/Minimum-Snap-Trajectories/blob/main/docs/limitations.md)
- [Changelog](https://github.com/byu-magicc/Minimum-Snap-Trajectories/blob/main/CHANGELOG.md)

## Running the tests

```bash
pip install -e ".[test]" && pytest python/tests          # Python
cmake -S cpp -B cpp/build && cmake --build cpp/build --config Release && ctest --test-dir cpp/build -C Release   # C++
```

## Contributing and support

Questions, bug reports and pull requests are welcome through
[GitHub issues](https://github.com/byu-magicc/Minimum-Snap-Trajectories/issues).

## Citing

If you use COAST in research, please cite it. Citation metadata, including authors and ORCIDs,
is in [`CITATION.cff`](https://github.com/byu-magicc/Minimum-Snap-Trajectories/blob/main/CITATION.cff);
GitHub's **Cite this repository** button generates BibTeX and APA from it.

<!-- TODO (authors, #15): add the BibTeX for the ACC paper / arXiv preprint here, and the JOSS
     paper after acceptance. -->

## License

BSD 3-Clause; see [`LICENSE`](https://github.com/byu-magicc/Minimum-Snap-Trajectories/blob/main/LICENSE).
Copyright (c) 2026, BYU MAGICC Lab.
