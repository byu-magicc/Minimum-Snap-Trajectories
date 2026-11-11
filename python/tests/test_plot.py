"""Tests for coast_planner.plot (rendered headless with the Agg backend)."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from coast_planner import (FrontEndConfig, MinSnapClamped, MinSnapNatural, ObstacleBox,  # noqa: E402
                      SparseVoxelGrid, TrajectoryPlanner, TrajectorySampler)
from coast_planner.plot import _disp, plot_plan, plot_profiles, plot_trajectory  # noqa: E402

DRONE_R = 0.49


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def unconstrained(cls=MinSnapClamped, dt=0.5):
    s = cls(10)
    z = np.zeros(3)
    C = s.solve_unconstrained([0, 0, -2], [1, 0, 0], z, [10, 5, -4], z, z, dt=dt)
    return C, s.scaled_knots(dt), s.degree


def pillar_plan():
    boxes = [ObstacleBox([4.5, -0.5, -7.0], [5.5, 0.5, 0.0])]
    grid = SparseVoxelGrid(0.5, DRONE_R)
    grid.update_from_obstacles(boxes)
    cfg = FrontEndConfig(voxel_resolution=0.5, drone_physical_radius=DRONE_R,
                         min_altitude=DRONE_R + 0.1, planning_horizon=12.0)
    res = TrajectoryPlanner(cfg, grid, 3.0, 2.0).plan_mission([0, 0, -3], [10, 0, -3])
    assert res.success
    return res, grid, boxes


def test_ned_display_frame():
    # NED (North, East, Down) -> displayed (East, North, Up)
    assert np.allclose(_disp([1.0, 2.0, -3.0], "ned"), [[2.0, 1.0, 3.0]])
    assert np.allclose(_disp([1.0, 2.0, -3.0], "raw"), [[1.0, 2.0, -3.0]])


@pytest.mark.parametrize("cls", [MinSnapClamped, MinSnapNatural])
@pytest.mark.parametrize("view", ["3d", "top"])
def test_plot_trajectory_unconstrained(cls, view, tmp_path):
    C, knots, degree = unconstrained(cls)
    ax = plot_trajectory(C, knots, degree, view=view)
    assert (ax.name == "3d") == (view == "3d")
    # endpoints are drawn where the boundary states put them: start (0,0,-2) -> ENU (0,0,2)
    starts = [c for c in ax.collections if c.get_label() == "start"]
    assert starts and np.allclose(np.asarray(starts[0].get_offsets())[0][:2], [0.0, 0.0])
    out = tmp_path / f"traj_{view}.png"
    ax.figure.savefig(out)
    assert out.stat().st_size > 1000


def test_plot_trajectory_inputs_agree():
    C, knots, degree = unconstrained()
    sampler = TrajectorySampler(C, knots, degree)
    for traj in (sampler, C, C.T, C.T.ravel()):
        kw = {} if isinstance(traj, TrajectorySampler) else dict(knots=knots, degree=degree)
        plot_trajectory(traj, view="top", color_by_speed=False, **kw)
    with pytest.raises(TypeError):
        plot_trajectory(C)  # raw control points need knots and degree
    with pytest.raises(ValueError):
        plot_trajectory(sampler, frame="enu")
    with pytest.raises(ValueError):
        plot_trajectory(sampler, view="side")


def test_plot_profiles():
    C, knots, degree = unconstrained()
    fig = plot_profiles(C, knots, degree, v_max=3.0, a_max=2.0)
    axes = fig.get_axes()
    assert len(axes) == 3
    # position x starts at p0.x = 0 and ends at pf.x = 10
    x_line = axes[0].get_lines()[0].get_ydata()
    assert abs(x_line[0]) < 1e-6 and abs(x_line[-1] - 10.0) < 1e-6


@pytest.mark.parametrize("view", ["3d", "top"])
def test_plot_plan(view, tmp_path):
    res, grid, boxes = pillar_plan()
    ax = plot_plan(res, grid=grid, obstacles=boxes, start=[0, 0, -3], goal=[10, 0, -3], view=view)
    labels = {h.get_label() for h in ax.get_legend_handles_labels()[0]}
    assert {"corridors", "goal", "start"} <= labels
    assert len(res.corridors) >= 1
    out = tmp_path / f"plan_{view}.png"
    ax.figure.savefig(out)
    assert out.stat().st_size > 1000


def test_corridor_vertices_match_constraints():
    res, _, _ = pillar_plan()
    for c in res.corridors:
        V = c.vertices
        assert V.shape == (8, 3)
        # every corner lies on or inside the corridor's own A x <= b box planes
        # (the corridor may additionally be clipped by map-bound rows, which the OBB ignores)
        assert np.max(c.A[:6] @ V.T - c.b[:6, None]) < 1e-6
        # vertex i+4 sits on the opposite z-face of vertex i
        assert np.allclose(V[4:] - V[:4], np.tile(V[4] - V[0], (4, 1)))


def test_plot_plan_failed_result():
    grid = SparseVoxelGrid(0.5, DRONE_R)
    grid.update_from_obstacles([ObstacleBox([-1, -1, -7], [1, 1, 0])])
    cfg = FrontEndConfig(voxel_resolution=0.5, drone_physical_radius=DRONE_R)
    res = TrajectoryPlanner(cfg, grid).plan_mission([0, 0, -3], [10, 0, -3])  # starts inside
    ax = plot_plan(res, grid=grid)
    if not res.success:
        assert "failed" in ax.get_title()
