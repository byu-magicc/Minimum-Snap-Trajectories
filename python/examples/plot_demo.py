"""Visualise an unconstrained solve and an obstacle-avoiding plan.

    pip install "coast-planner[viz]"
    python python/examples/plot_demo.py            # opens windows
    python python/examples/plot_demo.py --save out  # writes PNGs to ./out instead

If plot windows cannot be opened (no Tk on this Python, no display over SSH), the demo says
why and saves the PNGs to ./coast_demo_plots instead.

Coordinates are NED; plots display East-North-Up (altitude = -z).
"""

import argparse
import contextlib
import os

import numpy as np

from coast_planner import (FrontEndConfig, MinSnapClamped, ObstacleBox, SparseVoxelGrid,
                      TrajectoryPlanner)
from coast_planner.plot import plot_plan, plot_profiles, plot_trajectory

FALLBACK_DIR = "coast_demo_plots"
# Backends that render to files only; with one of these, plt.show() silently does nothing.
_NON_INTERACTIVE = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}


def can_open_windows():
    """True if matplotlib can open plot windows here; otherwise explain why and return False.

    Two common failure modes are caught. Either matplotlib already chose a file-only backend
    (Linux without python3-tk, or no display), so plt.show() would silently do nothing. Or the
    window toolkit is installed but broken: on Windows, a Python whose Tcl/Tk files are
    missing raises "Can't find a usable init.tcl" when the first window is created.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    backend = matplotlib.get_backend()
    reason = None
    if backend.lower() in _NON_INTERACTIVE:
        reason = f"matplotlib is using the file-only '{backend}' backend"
    else:
        try:
            plt.close(plt.figure())  # creating a figure opens the toolkit; fails fast if broken
        except Exception as e:  # noqa: BLE001 - any failure here means "no windows"
            reason = f"{type(e).__name__}: {str(e).splitlines()[0]}"
    if reason is None:
        return True
    print(f"Cannot open plot windows ({reason}).\n"
          f"Saving the plots as PNG files in ./{FALLBACK_DIR} instead.\n"
          "To get windows: on Ubuntu/Debian `sudo apt install python3-tk`; on Windows, re-run\n"
          "the python.org installer, choose Modify and enable \"tcl/tk and IDLE\" (and don't run\n"
          "inside an active conda environment); over SSH, view the PNG files instead.\n")
    plt.close("all")
    plt.switch_backend("Agg")
    return False


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
    elif not can_open_windows():
        args.save = FALLBACK_DIR
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
