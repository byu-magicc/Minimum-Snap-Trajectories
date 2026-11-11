# Contributing

Thanks for your interest in Minimum-Snap-Trajectories. Bug reports, questions, documentation
fixes and code contributions are all welcome.

## Getting help and reporting problems

- **Questions and usage help:** open an issue using the *Question* template.
- **Bugs:** open an issue using the *Bug report* template. Include the platform, how you
  installed (pip or CMake), and a minimal script or program that reproduces the problem. For
  planning failures, include the start, goal, map (obstacle boxes or scan) and `FrontEndConfig`.
- **Feature requests:** open an issue using the *Feature request* template, describing the
  use case before the proposed solution.
- **Security issues:** please do not open a public issue; use GitHub's private vulnerability
  reporting on this repository (*Security* tab → *Report a vulnerability*).

The project is maintained by the [BYU MAGICC Lab](https://github.com/byu-magicc). Maintainers
aim to respond to new issues and pull requests within two weeks. This is a research project,
so support is best-effort.

## Development setup

You need CMake ≥ 3.16, a C++17 compiler and git. Eigen, OSQP and GoogleTest are downloaded
automatically if they are not installed (see the README's *Building* section).

```bash
git clone https://github.com/byu-magicc/Minimum-Snap-Trajectories.git
cd Minimum-Snap-Trajectories

# C++ engine and its tests
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --config Release --parallel
ctest --test-dir cpp/build -C Release --output-on-failure

# Python package (editable) and its tests
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[test]"
pytest python/tests
```

Python edits under `python/coast_planner/` take effect immediately; after changing C++ code, re-run
`pip install -e .` (the rebuild is incremental).

## Making a change

1. Open an issue first for anything beyond a small fix, so the approach can be agreed before
   you spend time on it.
2. Create a branch from `main`. Keep each pull request focused on one change.
3. Add or update tests:
   - C++: GoogleTest, one file per module under test, in `cpp/tests/test_*.cpp`
     (add new files to `bspline_tests` in `cpp/CMakeLists.txt`).
   - Python bindings: `python/tests/test_bindings.py`.
4. Update the README when behaviour, configuration defaults or the public API change.
5. Make sure CI passes. It builds and tests the C++ engine and the Python package on Linux,
   macOS and Windows.
6. Describe *why* the change is needed in the pull request, and link the issue it addresses
   (`Fixes #123` closes it on merge).

## Conventions

- **C++17.** Engine code lives in the `trajectory_planner` namespace (sub-namespaces
  `mapping`, `sfc`, `path_finding`, `bspline`). Document public methods with `/** ... */`
  comments that explain intent, not just mechanics.
- **Coordinates are NED** (z is negative-up). `FrontEndConfig::min_altitude` is the one
  positive-up field. Double-check signs in anything that touches altitude.
- **Keep the Eigen safety net on.** Release builds would otherwise delete Eigen's bounds
  checks; `MIN_SNAP_EIGEN_SAFETY` turns them into exceptions. Don't disable it outside
  benchmarking.
- **Small problems are the under-tested regime.** After changing the planning horizon,
  corridor count or time allocation, run the stress sweep in `cpp/tools/stress_planner.cpp`
  (see the README's *Testing* section).
- **Claims need numbers.** Performance or behaviour changes should come with a measurement
  (before and after), not just a description.

## Code of conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). By participating you agree to
abide by it.
