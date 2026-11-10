"""Visualise an unconstrained solve and an obstacle-avoiding plan.

    pip install "min-snap[viz]"
    python python/examples/plot_demo.py            # opens windows
    python python/examples/plot_demo.py --save out  # writes PNGs to ./out instead

Coordinates are NED; plots display East-North-Up (altitude = -z).
"""

import argparse
import contextlib
import os

import numpy as np

from min_snap import (FrontEndConfig, MinSnapClamped, ObstacleBox, SparseVoxelGrid,
                      TrajectoryPlanner)
from min_snap.plot import plot_plan, plot_profiles, plot_trajectory


@contextlib.contextmanager
def quiet_stdout():
    """The C++ planner prints a progress banner; hide it for a tidy demo."""
    devnull, saved = os.open(os.devnull, os.O_WRONLY), os.dup(1)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--save", metavar="DIR", help="write PNGs here instead of showing windows")
    args = ap.parse_args()

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. Free space, no map: the closed-form minimum-snap solve.
    dt = 0.5
    solver = MinSnapClamped(num_segments=10)
    z = np.zeros(3)
    C = solver.solve_unconstrained(p0=[0, 0, -2], v0=[2, 0, 0], a0=z,
                                   pf=[10, 6, -5], vf=[0, 1, 0], af=z, dt=dt)
    knots = solver.scaled_knots(dt)
    ax1 = plot_trajectory(C, knots, solver.degree)
    ax1.set_title("Unconstrained minimum snap (clamped, 10 segments)")
    fig2 = plot_profiles(C, knots, solver.degree, title="Unconstrained solve: derivatives")

    # 2. Obstacle avoidance: two pillars between start and goal.
    radius = 0.49
    boxes = [ObstacleBox([4.5, -0.5, -7.0], [5.5, 0.5, 0.0]),
             ObstacleBox([7.5, 1.5, -7.0], [8.5, 2.5, 0.0])]
    grid = SparseVoxelGrid(resolution=0.5, inflation_radius=radius)
    grid.update_from_obstacles(boxes)
    cfg = FrontEndConfig(voxel_resolution=0.5, drone_physical_radius=radius,
                         min_altitude=radius + 0.1, planning_horizon=12.0)
    start, goal = [0.0, 0.0, -3.0], [11.0, 0.0, -3.0]
    with quiet_stdout():
        result = TrajectoryPlanner(cfg, grid, v_max=3.0, a_max=2.0).plan_mission(start, goal)
    print(result)
    ax3 = plot_plan(result, obstacles=boxes, start=start, goal=goal)
    ax4 = plot_plan(result, obstacles=boxes, start=start, goal=goal, view="top")
    fig5 = plot_profiles(result, v_max=3.0, a_max=2.0,
                         title="Planned trajectory: derivatives (limits are per axis)")

    figs = {"unconstrained_3d": ax1.figure, "unconstrained_profiles": fig2,
            "plan_3d": ax3.figure, "plan_top": ax4.figure, "plan_profiles": fig5}
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        for name, fig in figs.items():
            path = os.path.join(args.save, f"{name}.png")
            fig.savefig(path, dpi=110, bbox_inches="tight")
            print("wrote", path)
    else:
        plt.show()


if __name__ == "__main__":
    main()
