"""Optional matplotlib visualisation for coast_planner trajectories and plans.

Install the extra first::

    pip install "coast-planner[viz]"

Nothing here is called by the planner: plots are only made when you ask for them, and no
function calls ``plt.show()``. Each returns the matplotlib Axes or Figure, so you can save it
(``ax.figure.savefig("plan.png")``), restyle it, or show it yourself.

Coordinates. The engine works in NED (x = North, y = East, z = Down). Plotted as-is that is a
mirror image, so by default (``frame="ned"``) points are displayed East-North-Up: horizontal
axes East and North, vertical axis altitude = -z. Pass ``frame="raw"`` to plot x, y, z
unchanged.

Functions:
    plot_trajectory  path of any spline: an unconstrained solve, a TrajectorySampler, a plan
    plot_profiles    position / velocity / acceleration against time
    plot_plan        a planner result with its corridors, obstacles and start/goal
"""

from __future__ import annotations

import numpy as np

from ._core import PlanningResult, TrajectorySampler

__all__ = ["plot_trajectory", "plot_profiles", "plot_plan"]

_FRAMES = ("ned", "raw")


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------

def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - exercised only without matplotlib
        raise ImportError(
            'coast_planner.plot needs matplotlib. Install it with: pip install "coast-planner[viz]"'
        ) from e
    return plt


def _check_frame(frame):
    if frame not in _FRAMES:
        raise ValueError(f"frame must be one of {_FRAMES}, got {frame!r}")


def _disp(points, frame):
    """(N, 3) world points -> (N, 3) display coordinates."""
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    if frame == "raw":
        return p
    return np.column_stack([p[:, 1], p[:, 0], -p[:, 2]])  # East, North, Up


def _labels(frame):
    if frame == "raw":
        return "x (m)", "y (m)", "z (m)"
    return "East, y (m)", "North, x (m)", "Altitude, -z (m)"


def _as_sampler(traj, knots, degree):
    """Accept a TrajectorySampler, a PlanningResult, or (control_points, knots, degree)."""
    if isinstance(traj, TrajectorySampler):
        return traj, None
    if isinstance(traj, PlanningResult):
        return traj.sampler(), traj.control_points_xyz
    if knots is None or degree is None:
        raise TypeError(
            "pass a TrajectorySampler or PlanningResult, or control points with knots= and degree="
        )
    cps = np.asarray(traj, dtype=float)
    sampler = TrajectorySampler(cps, np.asarray(knots, dtype=float), int(degree))
    if cps.ndim == 2 and cps.shape[0] == 3 and cps.shape[1] != 3:
        cps = cps.T  # (3, N) as returned by solve_unconstrained
    return sampler, cps.reshape(-1, 3)


def _new_axes(view, ax):
    if ax is not None:
        return ax
    plt = _plt()
    fig = plt.figure(figsize=(7, 6))
    if view == "3d":
        return fig.add_subplot(projection="3d")
    if view == "top":
        return fig.add_subplot()
    raise ValueError(f"view must be '3d' or 'top', got {view!r}")


def _is_3d(ax):
    return getattr(ax, "name", "") == "3d"


def _equal_aspect(ax, pts):
    """True proportions (1 m is the same length on every axis), fitted to the plotted points."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 3 if _is_3d(ax) else 2)
    if pts.size == 0:
        return
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    if _is_3d(ax):
        # Fit each axis to its own range and shape the box to match, so a flat scene is not
        # squeezed into a tall cube; keep a minimum range so a flat path still has some depth.
        span = np.maximum(hi - lo, 0.15 * float((hi - lo).max()) + 1e-9)
        pad = 0.05 * span
        lo, hi = (lo + hi) / 2 - span / 2 - pad, (lo + hi) / 2 + span / 2 + pad
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(tuple(hi - lo))
    else:
        centre, half = (lo + hi) / 2, max(float((hi - lo).max()) / 2, 0.5) * 1.05
        ax.set_xlim(centre[0] - half, centre[0] + half)
        ax.set_ylim(centre[1] - half, centre[1] + half)
        ax.set_aspect("equal", adjustable="box")


def _convex_hull_2d(points):
    """Monotone-chain convex hull; returns the hull vertices in order."""
    pts = sorted(set(map(tuple, np.round(np.asarray(points, dtype=float), 9))))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


# Box edges for 8 corners ordered as Corridor.vertices / _box_corners: 0-3 bottom, 4-7 top.
_BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]
_BOX_FACES = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]


def _box_corners(box):
    (x0, y0, z0), (x1, y1, z1) = np.asarray(box.min, float), np.asarray(box.max, float)
    return np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                     [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])


# ----------------------------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------------------------

def plot_trajectory(traj, knots=None, degree=None, *, ax=None, view="3d", frame="ned",
                    samples=400, color_by_speed=True, show_control_points=True, label=None,
                    colorbar=True):
    """Plot the path of a spline.

    Args:
        traj: a ``TrajectorySampler``, a ``PlanningResult``, or control points (``(3, N)`` as
            returned by ``solve_unconstrained``, ``(N, 3)``, or flat), in which case ``knots``
            and ``degree`` are required.
        knots, degree: for raw control points, e.g. ``solver.scaled_knots(dt), solver.degree``.
        ax: existing Axes to draw into (3-D for ``view="3d"``); a new figure otherwise.
        view: ``"3d"`` or ``"top"`` (plan view).
        frame: ``"ned"`` (display East-North-Up) or ``"raw"``.
        samples: number of time samples along the curve.
        color_by_speed: colour the path by speed, with a colour bar.
        show_control_points: draw the control polygon (not available for a bare sampler).

    Returns:
        The matplotlib Axes.
    """
    _check_frame(frame)
    sampler, cps = _as_sampler(traj, knots, degree)
    ax = _new_axes(view, ax)
    if not sampler.valid:
        ax.set_title("no trajectory (spline not evaluable)")
        return ax

    plt = _plt()
    t = np.linspace(sampler.start_time, sampler.end_time, int(samples))
    S = sampler.sample(t)
    P = _disp(S[:, :3], frame)
    speed = np.linalg.norm(S[:, 3:6], axis=1)
    three_d = _is_3d(ax)
    XY = P if three_d else P[:, :2]

    if color_by_speed and len(XY) > 1:
        segs = np.stack([XY[:-1], XY[1:]], axis=1)
        if three_d:
            from mpl_toolkits.mplot3d.art3d import Line3DCollection as Coll
        else:
            from matplotlib.collections import LineCollection as Coll
        lc = Coll(segs, cmap="viridis", linewidths=2.5, label=label)
        lc.set_array(0.5 * (speed[:-1] + speed[1:]))
        ax.add_collection(lc)
        if colorbar:
            plt.colorbar(lc, ax=ax, shrink=0.7, pad=0.1, label="speed (m/s)")
    else:
        ax.plot(*XY.T, lw=2.5, label=label)

    if show_control_points and cps is not None and len(cps):
        C = _disp(cps, frame)
        C = C if three_d else C[:, :2]
        ax.plot(*C.T, "o--", color="0.55", ms=3.5, lw=0.8, label="control points")

    ax.scatter(*XY[0], color="tab:green", s=40, zorder=5, label="start")
    ax.scatter(*XY[-1], color="tab:red", marker="*", s=90, zorder=5, label="end")

    xl, yl, zl = _labels(frame)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    if three_d:
        ax.set_zlabel(zl)
    _equal_aspect(ax, XY)
    return ax


def plot_profiles(traj, knots=None, degree=None, *, samples=400, v_max=None, a_max=None,
                  title=None, fig=None):
    """Plot position, velocity and acceleration against time.

    Velocity and acceleration show each axis plus the norm. ``v_max`` / ``a_max`` draw the
    per-axis limits as dashed lines; by default the planner enforces limits per axis, so the
    norm may legitimately exceed them (see ``FrontEndConfig.enforce_norm_limits``).
    Components are in the world frame as given (NED for the planner).

    Returns:
        The matplotlib Figure (three stacked Axes).
    """
    sampler, _ = _as_sampler(traj, knots, degree)
    plt = _plt()
    if fig is None:
        fig = plt.figure(figsize=(8, 7), constrained_layout=True)
    if title:
        fig.suptitle(title)
    axes = fig.subplots(3, 1, sharex=True)
    if not sampler.valid:
        axes[0].set_title("no trajectory (spline not evaluable)")
        return fig

    t = np.linspace(sampler.start_time, sampler.end_time, int(samples))
    S = sampler.sample(t)
    names = ("x", "y", "z")
    for k, (ax, title, block) in enumerate(zip(axes, ("position (m)", "velocity (m/s)",
                                                       "acceleration (m/s²)"), (0, 3, 6))):
        for i in range(3):
            ax.plot(t, S[:, block + i], label=names[i])
        if k > 0:
            ax.plot(t, np.linalg.norm(S[:, block:block + 3], axis=1), "k", lw=1.2, label="norm")
        limit = (None, v_max, a_max)[k]
        if limit is not None:
            for sgn in (1, -1):
                ax.axhline(sgn * limit, color="0.4", ls="--", lw=0.8)
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, ncol=4)
    axes[-1].set_xlabel("time (s)")
    return fig


def plot_plan(result, *, grid=None, obstacles=None, start=None, goal=None, ax=None,
              view="3d", frame="ned", show_corridors=True, show_control_points=False,
              max_voxels=20000, color_by_speed=True):
    """Plot a planner result in context.

    Args:
        result: the ``PlanningResult`` from ``TrajectoryPlanner.plan_mission``.
        grid: optional ``SparseVoxelGrid``; its OCCUPIED cells are drawn as points
            (randomly subsampled above ``max_voxels``).
        obstacles: optional list of ``ObstacleBox`` to draw as boxes.
        start, goal: optional points to mark (e.g. the goal passed to ``plan_mission``).
        view: ``"3d"`` or ``"top"``.
        show_corridors: draw the Safe Flight Corridors (wireframes in 3-D, outlines in top view).

    Returns:
        The matplotlib Axes. A failed plan still shows corridors, obstacles and markers, with
        the failure in the title.
    """
    _check_frame(frame)
    ax = _new_axes(view, ax)
    three_d = _is_3d(ax)
    dim = 3 if three_d else 2
    extent = []  # display points used for the axis limits

    if obstacles:
        for box in obstacles:
            V = _disp(_box_corners(box), frame)
            extent.append(V[:, :dim])
            if three_d:
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                ax.add_collection3d(Poly3DCollection([V[list(f)] for f in _BOX_FACES],
                                                     facecolor="0.35", edgecolor="0.2",
                                                     alpha=0.35, linewidths=0.5))
            else:
                H = _convex_hull_2d(V[:, :2])
                ax.fill(*H.T, color="0.35", alpha=0.45, lw=0.5, ec="0.2")

    if grid is not None:
        idx = np.asarray(grid.occupied_indices(), dtype=float)
        if len(idx):
            if len(idx) > max_voxels:
                idx = idx[np.random.default_rng(0).choice(len(idx), max_voxels, replace=False)]
            V = _disp((idx + 0.5) * grid.resolution, frame)
            ax.scatter(*V[:, :dim].T, s=3, color="0.3", alpha=0.35, label="occupied voxels")
            extent.append(V[:, :dim])

    if show_corridors:
        for i, c in enumerate(result.corridors):
            V = _disp(c.vertices, frame)
            if three_d:
                for a, b in _BOX_EDGES:
                    ax.plot(*np.vstack([V[a], V[b]]).T, color="tab:blue", lw=0.8, alpha=0.6,
                            label="corridors" if (i == 0 and a == 0 and b == 1) else None)
            else:
                H = _convex_hull_2d(V[:, :2])
                ax.fill(*np.vstack([H, H[:1]]).T, color="tab:blue", alpha=0.08)
                ax.plot(*np.vstack([H, H[:1]]).T, color="tab:blue", lw=0.8, alpha=0.7,
                        label="corridors" if i == 0 else None)
            extent.append(V[:, :dim])

    sampler = result.sampler()
    if result.success and sampler.valid:
        plot_trajectory(result, ax=ax, frame=frame, color_by_speed=color_by_speed,
                        show_control_points=show_control_points, label="trajectory")
        t = np.linspace(sampler.start_time, sampler.end_time, 200)
        extent.append(_disp(sampler.sample(t)[:, :3], frame)[:, :dim])
    else:
        ax.set_title("plan failed: no trajectory")

    for point, style in ((start, dict(color="tab:green", marker="o", s=60, label="start")),
                         (goal, dict(color="tab:red", marker="*", s=140, label="goal"))):
        if point is not None:
            P = _disp(point, frame)[0, :dim]
            ax.scatter(*P, zorder=6, **style)
            extent.append(P[None, :])

    xl, yl, zl = _labels(frame)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    if three_d:
        ax.set_zlabel(zl)
    if extent:
        _equal_aspect(ax, np.vstack(extent))
    if result.success:
        ax.set_title(f"plan: {len(result.corridors)} corridors, "
                     f"{result.total_time_ms:.1f} ms, reached goal: {result.reached_goal}")
    return ax
