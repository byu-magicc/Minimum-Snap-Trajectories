# Building, testing and development

## Dependencies

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

## Build

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

## Python package

The engine ships as the `coast_planner` Python package. From a clone:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .            # builds cpp/ with scikit-build-core, installs coast_planner in editable mode
python -c "import coast_planner; print(coast_planner.__version__)"
```

A venv is recommended (on Ubuntu the system Python refuses `pip install`). Works on Linux,
macOS and Windows with Python 3.9 or newer; CI runs `pip install` + the test suite on all three
with Python 3.9 and 3.13. To use it from another project, install the release from PyPI into
that project's environment (prebuilt wheels, no compiler needed), or the latest development
version straight from GitHub:

```bash
pip install coast-planner                                                                  # release
pip install "coast-planner @ git+https://github.com/byu-magicc/Minimum-Snap-Trajectories.git"  # development
```

Nothing needs to be installed system-wide besides a C/C++ compiler and git. pip fetches
pybind11, scikit-build-core, CMake/Ninja if needed, and NumPy. The build downloads Eigen and
OSQP and links OSQP statically, so the installed module has no runtime library dependencies.
It also skips the GoogleTest suite and examples. The compiled build is
kept under `build/` so re-running `pip install -e .` after a C++ change is incremental.
**Python edits under `python/coast_planner/` are live; C++ edits need `pip install -e .` again.**

Without pip, a plain CMake build writes the extension module straight into the package:

```bash
cd cpp/build && cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON && make -j$(nproc) min_snap_bindings
cd ../.. && PYTHONPATH=python python -c "import coast_planner"
```

Don't mix the two in one environment: a `_core*.so` left in `python/coast_planner/` by a CMake
build can shadow the pip-installed one when `python/` is on `sys.path`.

## CMake options

| Option | Default | Purpose |
|---|---|---|
| `MIN_SNAP_BUNDLED_DEPS` | `OFF` | Always download Eigen 3.4.0 / OSQP v1.0.0 instead of using system copies. With `OFF`, only missing ones are downloaded. pip turns it on. |
| `MIN_SNAP_EIGEN_SAFETY` | `ON` | Turn Eigen bounds assertions into exceptions even in release builds. **Leave this on** — see [Gotchas](limitations.md#gotchas). |
| `BUILD_EXAMPLES` | `ON` | Build `examples/`. Kept on by default so CI notices if an example stops compiling. |
| `BUILD_TESTS` | `ON` | Build the GoogleTest suite (GoogleTest is downloaded if not installed). pip builds turn it off. |
| `BUILD_PYTHON_BINDINGS` | `OFF` | Build the `coast_planner._core` pybind11 module. Off by default so `add_subdirectory()` consumers do not need Python headers; `pip install` turns it on. |

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

## Repository layout

```
LICENSE                 BSD 3-Clause
CHANGELOG.md            notable changes per release (Keep a Changelog)
RELEASING.md            how to cut a release
.github/workflows/      CI: C++ and Python builds/tests on Linux, macOS, Windows
docs/                   detailed documentation (linked from the README)
cpp/                    the C++ engine — this is the production artifact
  include/              public headers
  src/                  implementation (python_bindings.cpp is the pybind11 module)
  tests/                GoogleTest suite
  examples/             minimal_planner_loop.cpp
  tools/                stress_planner, replay/repro tools, monte_carlo (needs roscopter)
  deps/                 vendored nanoflann
  cmake/                MinSnapDeps.cmake: finds or downloads Eigen + OSQP
pyproject.toml          pip packaging (scikit-build-core -> cpp/CMakeLists.txt)
python/coast_planner/        the Python package: bindings to the C++ engine
python/tests/           binding tests
python/examples/        benchmark.py (timing), plot_demo.py (visualization)
```

The original pure-Python prototype that the C++ engine was ported from, and the `data/`
benchmark figures, are no longer on `main`. They are preserved on the
[`legacy/python-prototype`](https://github.com/byu-magicc/Minimum-Snap-Trajectories/tree/legacy/python-prototype)
branch and the `prototype-archive` tag (`git checkout prototype-archive`). Several C++ files
are direct ports and say so in their header comments; the prototype is **not** kept in sync
with the C++ engine.
