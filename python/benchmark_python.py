"""Benchmark the C++ engine through the `min_snap` Python bindings.

    pip install -e .            (from the repo root), then
    python python/benchmark_python.py

1. Unconstrained minimum snap (clamped and natural): Python-side cost of one C = SE @ Q solve.
2. SFC-constrained planning on the floating-blocks scene from
   cpp/tests/test_trajectory_planner.cpp (FloatingBlocksBenchmark), so the numbers can be read
   against the C++ test's. Coordinates are NED: z is negative-up.
"""

import contextlib
import os
import statistics
import time

import numpy as np

from min_snap import (FrontEndConfig, MinSnapClamped, MinSnapNatural, ObstacleBox,
                      SparseVoxelGrid, TrajectoryPlanner)


@contextlib.contextmanager
def quiet_stdout():
    """The C++ planner prints a banner per plan; silence it at the file-descriptor level."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(1)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


def summarize(label, samples_ms):
    samples_ms = sorted(samples_ms)
    p = lambda q: samples_ms[min(len(samples_ms) - 1, int(q * len(samples_ms)))]  # noqa: E731
    print(f"  {label:<28} median {statistics.median(samples_ms):8.4f} ms   "
          f"p90 {p(0.9):8.4f}   max {samples_ms[-1]:8.4f}   (n={len(samples_ms)})")


def bench_unconstrained(n=2000, num_segments=15):
    print(f"\nUnconstrained minimum snap ({num_segments} segments, degree 4)")
    rng = np.random.default_rng(0)
    for name, cls in [("clamped", MinSnapClamped), ("natural", MinSnapNatural)]:
        t0 = time.perf_counter()
        solver = cls(num_segments)
        build_ms = (time.perf_counter() - t0) * 1e3
        times = []
        for _ in range(n):
            st = rng.uniform(-10, 10, size=(6, 3))
            t0 = time.perf_counter()
            solver.solve_unconstrained(*st, dt=0.5)
            times.append((time.perf_counter() - t0) * 1e3)
        print(f"  {name}: Q build {build_ms:.3f} ms (once per segment count)")
        summarize(f"{name} solve_unconstrained", times)


def floating_blocks():
    obstacles = []
    num_blocks, block_width = 4, 10.0
    x_inc, y_inc, z_inc = 100.0 / num_blocks, 100.0 / num_blocks, 15.0 / num_blocks
    for i in range(num_blocks):
        for j in range(num_blocks):
            for k in range(num_blocks):
                c = np.array([x_inc / 2 + i * x_inc - 50.0,
                              y_inc / 2 + j * y_inc - 50.0,
                              -(z_inc / 2 + k * z_inc) * 0.5])
                obstacles.append(ObstacleBox(c - block_width / 2, c + block_width / 2))
    return obstacles


def bench_sfc(n=50):
    print("\nSFC-constrained planning (floating blocks, fixed-wing allocation)")
    cfg = FrontEndConfig(voxel_resolution=0.5, drone_physical_radius=0.5, sfc_height=5.0, sfc_width=5.0,
                         sfc_start_ext=5.0, sfc_end_ext=5.0, spline_type="clamped",
                         aircraft_type="fixed-wing", degree=4, map_bounds=[100.0, 100.0, 15.0])
    grid = SparseVoxelGrid(cfg.voxel_resolution, cfg.drone_physical_radius)
    t0 = time.perf_counter()
    grid.update_from_obstacles(floating_blocks())
    print(f"  map build: {(time.perf_counter() - t0) * 1e3:.2f} ms "
          f"({grid.num_occupied} occupied / {grid.num_inflated} inflated voxels)")

    start, goal = np.array([-30.0, -30.0, -4.0]), np.array([30.0, 30.0, -6.0])
    wall, total, sfc, opt, ok = [], [], [], [], 0
    result = None
    for _ in range(n):
        t0 = time.perf_counter()
        with quiet_stdout():
            result = TrajectoryPlanner(cfg, grid, 3.0, 2.0).plan_mission(start, goal)
        wall.append((time.perf_counter() - t0) * 1e3)
        total.append(result.total_time_ms)
        sfc.append(result.sfc_time_ms)
        opt.append(result.opt_time_ms)
        ok += result.success
    print(f"  success {ok}/{n};  last: {result}")
    summarize("Python wall time", wall)
    summarize("C++ total_time_ms", total)
    summarize("  sfc_time_ms", sfc)
    summarize("  opt_time_ms", opt)


if __name__ == "__main__":
    bench_unconstrained()
    bench_sfc()
