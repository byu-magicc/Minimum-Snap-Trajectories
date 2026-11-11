"""Verification suite for the coast_planner Python bindings.

Run either way:
    pytest python/tests                       # after `pip install -e .`
    python python/tests/test_bindings.py      # same, without pytest
    PYTHONPATH=python python python/tests/test_bindings.py   # after a plain CMake build
                                                              # with -DBUILD_PYTHON_BINDINGS=ON

Unconstrained tests check the solver output by EVALUATING the spline (de Boor), not just by
checking shapes: the boundary states must be reproduced exactly, and the solution must be the
minimum-snap one (any boundary-preserving perturbation raises the snap cost).

SFC tests check that a plan succeeds, starts where asked, reaches the goal, stays out of the
obstacles and respects v_max. Coordinates are NED: z is negative-up.
"""

import math
import sys

import numpy as np

import coast_planner
from coast_planner import (
    FrontEndConfig,
    MinSnapClamped,
    MinSnapEvalClamped,
    MinSnapEvalNatural,
    MinSnapNatural,
    ObstacleBox,
    PlanningResult,
    SparseVoxelGrid,
    TrajectoryPlanner,
    TrajectorySampler,
)

RNG = np.random.default_rng(1234)
SOLVERS = [("clamped", MinSnapEvalClamped), ("natural", MinSnapEvalNatural)]


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------

def random_states(dim=3):
    return dict(
        p0=RNG.uniform(-10, 10, dim), v0=RNG.uniform(-2, 2, dim), a0=RNG.uniform(-1, 1, dim),
        pf=RNG.uniform(-10, 10, dim), vf=RNG.uniform(-2, 2, dim), af=RNG.uniform(-1, 1, dim),
    )


def assert_close(actual, expected, tol, what):
    actual, expected = np.asarray(actual), np.asarray(expected)
    err = np.max(np.abs(actual - expected))
    assert err <= tol, f"{what}: max error {err:.3e} > {tol:.1e}\n  got {actual}\n  want {expected}"


def snap_cost(C, W):
    """Sum over dimensions of c_d^T W c_d, for C of shape (D, N)."""
    return float(np.einsum("dn,nm,dm->", C, W, C))


def box_distance(p, box_min, box_max):
    """Euclidean distance from p to an axis-aligned box (0 inside)."""
    d = np.maximum(np.maximum(box_min - p, 0.0), p - box_max)
    return float(np.linalg.norm(d))


# ----------------------------------------------------------------------------------------------
# 1. Unconstrained minimum snap
# ----------------------------------------------------------------------------------------------

def test_aliases_and_exports():
    assert MinSnapClamped is MinSnapEvalClamped
    assert MinSnapNatural is MinSnapEvalNatural
    for name in coast_planner.__all__:
        assert hasattr(coast_planner, name), name


def test_solver_structure():
    for name, cls in SOLVERS:
        for M, d in [(4, 4), (10, 4), (12, 5)]:
            s = cls(M, d)
            N = M + d
            assert s.num_segments == M and s.degree == d and s.num_control_points == N, name
            assert s.get_Q_matrix().shape == (6, N), name
            assert s.get_W().shape == (N, N), name
            assert s.get_knots().shape == (N + d + 1,), name
            assert_close(s.get_W(), s.get_W().T, 1e-9, f"{name} W symmetric")
        s = cls(6)
        s.update_segments(9)
        assert s.num_control_points == 9 + 4 and s.get_Q_matrix().shape == (6, 13), name

    # 3 segments satisfies the 6 boundary constraints but not the degree-4 snap stencil (needs
    # 4); the engine rejects it and the binding surfaces that as ValueError.
    try:
        MinSnapEvalClamped(3)
        raise AssertionError("3 segments accepted")
    except ValueError:
        pass

    # natural keeps its original camelCase API too
    n = MinSnapEvalNatural(8)
    assert np.array_equal(n.getQMatrix(), n.get_Q_matrix())
    assert np.array_equal(n.getKnots(), n.get_knots())
    assert np.array_equal(n.getW(), n.get_W())
    n.updateSegments(10)
    assert n.num_segments == 10

    # clamped knots are clamped, natural knots are uniform from -degree
    k = MinSnapEvalClamped(5).get_knots()
    assert np.all(k[:5] == 0) and np.all(k[-5:] == 5)
    assert_close(MinSnapEvalNatural(5).get_knots(), np.arange(-4, 10), 1e-12, "natural knots")


def test_boundary_conditions_reproduced():
    """Evaluating the solved spline must hit (p, v, a) at both ends, at any knot spacing."""
    for name, cls in SOLVERS:
        for M in (4, 10, 25):
            s = cls(M)
            for dt in (1.0, 0.5, 0.2):
                st = random_states()
                C = s.solve_unconstrained(**st, dt=dt)
                assert C.shape == (3, s.num_control_points), name
                traj = TrajectorySampler(C, s.scaled_knots(dt), s.degree)
                assert traj.valid
                assert_close(traj.duration, M * dt, 1e-9, f"{name} duration")
                t0, tf = traj.start_time, traj.end_time
                tag = f"{name} M={M} dt={dt}"
                assert_close(traj.position(t0), st["p0"], 1e-6, f"{tag} p0")
                assert_close(traj.velocity(t0), st["v0"], 1e-6, f"{tag} v0")
                assert_close(traj.acceleration(t0), st["a0"], 1e-6, f"{tag} a0")
                assert_close(traj.position(tf), st["pf"], 1e-6, f"{tag} pf")
                assert_close(traj.velocity(tf), st["vf"], 1e-6, f"{tag} vf")
                assert_close(traj.acceleration(tf), st["af"], 1e-6, f"{tag} af")


def test_se_overload_matches_state_overload():
    for name, cls in SOLVERS:
        s = cls(10)
        st = random_states()
        SE = cls.build_SE(**st, dt=0.5)
        assert SE.shape == (3, 6)
        assert_close(s.solve_unconstrained(SE), s.solve_unconstrained(**st, dt=0.5), 1e-12, name)
        assert_close(s.solve_unconstrained(SE), SE @ s.get_Q_matrix(), 1e-12, name)

    # The two solvers' native column orders differ; build_SE owns that.
    st = dict(p0=[1, 1, 1], v0=[2, 2, 2], a0=[3, 3, 3], pf=[4, 4, 4], vf=[5, 5, 5], af=[6, 6, 6])
    assert list(MinSnapEvalClamped.build_SE(**st)[0]) == [1, 2, 3, 6, 5, 4]
    assert list(MinSnapEvalNatural.build_SE(**st)[0]) == [1, 2, 3, 4, 5, 6]


def test_solution_is_minimum_snap():
    """Any perturbation preserving the boundary states must not lower the snap cost."""
    for name, cls in SOLVERS:
        s = cls(12)
        C = s.solve_unconstrained(**random_states())
        W = s.get_W()
        # Boundary states are linear in C: SE = C @ B (B is N x 6). Perturbations Z with
        # Z @ B = 0 leave them unchanged -> Z rows live in the left null space of B.
        B = s.get_B_combined()
        _, sv, Vt = np.linalg.svd(B.T)
        null = Vt[np.sum(sv > 1e-10):]           # (N - 6, N)
        assert null.shape[0] == s.num_control_points - 6, name
        J0 = snap_cost(C, W)
        for _ in range(20):
            Z = RNG.normal(size=(3, null.shape[0])) @ null
            assert np.max(np.abs(Z @ B)) < 1e-8
            for eps in (1e-3, 1e-1):
                assert snap_cost(C + eps * Z, W) >= J0 - 1e-9 * max(1.0, J0), name


def test_rest_to_rest_is_straight_and_symmetric():
    for name, cls in SOLVERS:
        s = cls(10)
        z = np.zeros(3)
        p0, pf = np.array([0.0, 0.0, -2.0]), np.array([10.0, 4.0, -5.0])
        C = s.solve_unconstrained(p0, z, z, pf, z, z)
        traj = TrajectorySampler(C, s.get_knots(), s.degree)
        ts = np.linspace(traj.start_time, traj.end_time, 101)
        X = traj.sample(ts)[:, :3]
        # every sample lies on the segment p0 -> pf
        u = (pf - p0) / np.linalg.norm(pf - p0)
        off = (X - p0) - np.outer((X - p0) @ u, u)
        assert np.max(np.linalg.norm(off, axis=1)) < 1e-6, name
        # and the midpoint is the geometric midpoint (time-symmetric problem)
        assert_close(traj.position(0.5 * (ts[0] + ts[-1])), 0.5 * (p0 + pf), 1e-6, name)


def test_other_dimensions():
    for name, cls in SOLVERS:
        s = cls(8)
        st2 = random_states(dim=2)
        C2 = s.solve_unconstrained(**st2)
        assert C2.shape == (2, s.num_control_points), name
        st1 = random_states(dim=1)
        assert s.solve_unconstrained(**st1).shape == (1, s.num_control_points), name


def test_bad_arguments_raise():
    for name, cls in SOLVERS:
        for bad in [lambda: cls(2), lambda: cls(10, 3), lambda: cls(10).solve_unconstrained(np.zeros((3, 5))),
                    lambda: cls(10).solve_unconstrained([0, 0, 0], [0, 0], [0, 0, 0], [1, 1, 1], [0, 0, 0], [0, 0, 0]),
                    lambda: cls(10).solve_unconstrained(*([np.zeros(3)] * 6), dt=0.0),
                    lambda: cls(10).update_segments(1)]:
            try:
                bad()
            except (ValueError, TypeError):
                continue
            raise AssertionError(f"{name}: expected an exception")


# ----------------------------------------------------------------------------------------------
# 2. Map
# ----------------------------------------------------------------------------------------------

def test_obstacle_box_and_grid_from_obstacles():
    box = ObstacleBox([1.0, 2.0, -3.0], np.array([2.0, 3.0, -1.0]))
    assert_close(box.min, [1, 2, -3], 0, "box.min")
    assert_close(box.max_pt, [2, 3, -1], 0, "box.max_pt")
    box.max = [2.5, 3.0, -1.0]
    assert box.max[0] == 2.5
    try:
        ObstacleBox([0, 0, 0], [-1, 1, 1])
        raise AssertionError("inverted box accepted")
    except ValueError:
        pass

    g = SparseVoxelGrid(resolution=0.5, inflation_radius=0.5)
    assert g.resolution == 0.5 and g.inflation_radius == 0.5
    g.update_from_obstacles([box])
    assert g.num_occupied > 0 and g.num_inflated > g.num_occupied
    assert g.is_occupied([1.5, 2.5, -2.0])
    assert not g.is_occupied([10.0, 10.0, -2.0])
    assert g.is_unknown([10.0, 10.0, -2.0])  # a box list says nothing about free space
    assert g.num_free == 0
    idx = g.occupied_indices()
    assert idx.shape == (g.num_occupied, 3) and idx.dtype.kind == "i"


def wall_scan(origin, wall_x, n_az=61, n_el=21):
    """Synthetic LiDAR: beams fanned toward +x, each returning where it hits the plane x=wall_x."""
    hits = []
    for az in np.radians(np.linspace(-30, 30, n_az)):
        for el in np.radians(np.linspace(-20, 20, n_el)):
            d = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), -math.sin(el)])
            s = (wall_x - origin[0]) / d[0]
            hits.append(origin + s * d)
    return np.array(hits)


def test_update_from_scan_three_state():
    origin = np.array([0.0, 0.0, -3.0])
    hits = wall_scan(origin, wall_x=6.0)
    for as_list in (False, True):
        g = SparseVoxelGrid(resolution=0.5)
        g.update_from_scan(origin, hits.tolist() if as_list else hits)
        assert g.num_free > 0 and g.num_occupied > 0
        assert g.is_free([3.0, 0.0, -3.0]), "cells along a beam must be FREE"
        assert g.is_occupied([6.0 + 1e-3, 0.0, -3.0]), "beam end cell must be OCCUPIED"
        assert g.is_unknown([10.0, 0.0, -3.0]), "behind the wall must stay UNKNOWN"
        assert g.is_unknown([-5.0, 0.0, -3.0]), "behind the sensor must stay UNKNOWN"
    # empty scans and flat arrays are accepted
    g.update_from_scan(origin, np.empty((0, 3)))
    g.update_from_scan(origin, [])
    g.update_from_scan(origin, hits[:4].ravel())
    g.set_accumulate(False)
    assert not g.accumulating
    try:
        g.update_from_scan(origin, np.zeros((4, 2)))
        raise AssertionError("bad hit shape accepted")
    except ValueError:
        pass


# ----------------------------------------------------------------------------------------------
# 3. Front end config
# ----------------------------------------------------------------------------------------------

def test_front_end_config():
    cfg = FrontEndConfig()
    assert cfg.spline_type == "clamped" and cfg.aircraft_type == "multi-rotor" and cfg.degree == 4
    cfg.v_max = 4.5
    cfg.map_bounds = [50, 60, 20]
    assert cfg.v_max == 4.5
    assert_close(cfg.map_bounds, [50, 60, 20], 0, "map_bounds")
    cfg2 = FrontEndConfig(voxel_resolution=0.25, planning_horizon=8.0, spline_type="natural",
                          map_bounds=np.array([10.0, 10.0, 10.0]))
    assert cfg2.voxel_resolution == 0.25 and cfg2.planning_horizon == 8.0
    assert cfg2.spline_type == "natural" and cfg2.map_bounds[2] == 10.0
    for field in ["drone_physical_radius", "sfc_height", "sfc_width", "sfc_start_ext", "sfc_end_ext",
                  "min_altitude", "max_altitude", "a_max", "unknown_traversal_cost", "max_sfc_count",
                  "max_solve_time_ms", "time_budget_factor"]:
        assert hasattr(cfg, field), field
    try:
        FrontEndConfig(not_a_field=1)
        raise AssertionError("unknown kwarg accepted")
    except AttributeError:
        pass
    try:
        cfg.degree = "four"
        raise AssertionError("bad type accepted")
    except TypeError:
        pass


# ----------------------------------------------------------------------------------------------
# 4. SFC-constrained planning
# ----------------------------------------------------------------------------------------------

DRONE_R = 0.49
VOXEL = 0.5
V_MAX, A_MAX = 3.0, 2.0


def planner_config(**overrides):
    kw = dict(voxel_resolution=VOXEL, drone_physical_radius=DRONE_R,
              map_bounds=[100.0, 100.0, 15.0], min_altitude=DRONE_R + 0.1,
              planning_horizon=12.0, v_max=V_MAX, a_max=A_MAX)
    kw.update(overrides)
    return FrontEndConfig(**kw)


def check_plan(res, start, goal, boxes, label, norm_limits=False):
    assert isinstance(res, PlanningResult)
    assert res.success, f"{label}: plan failed ({res})"
    assert res.reached_goal, f"{label}: did not reach goal ({res})"
    n = res.total_control_points
    assert res.control_points.shape == (3 * n,) and res.control_points_xyz.shape == (n, 3)
    assert res.knots.shape == (n + res.degree + 1,)
    assert len(res.corridors) >= 1 and res.solve_attempts >= 1
    assert res.total_time_ms > 0 and res.opt_time_ms >= 0 and res.sfc_time_ms >= 0

    traj = res.sampler()
    assert traj.valid and traj.duration > 0
    ts = np.linspace(traj.start_time, traj.end_time, 400)
    S = traj.sample(ts)
    pos, vel = S[:, :3], S[:, 3:6]
    # The planner's start state comes out of OSQP, so it is exact only to solver tolerance.
    assert_close(pos[0], start, 1e-3, f"{label} start")
    assert np.linalg.norm(pos[-1] - goal) < 0.5, f"{label}: ends {pos[-1]}, goal {goal}"
    clearance = min(box_distance(p, b.min, b.max) for p in pos for b in boxes)
    assert clearance > 0.0, f"{label}: trajectory enters an obstacle"
    speed = np.max(np.linalg.norm(vel, axis=1))
    if norm_limits:
        # enforce_norm_limits=True: v_max bounds the actual speed
        assert speed <= V_MAX * 1.02, f"{label}: peak speed {speed:.2f} > v_max"
    else:
        # default: v_max is a PER-AXIS box (|v_i| <= v_max), so speed may reach sqrt(3)*v_max
        axis = np.max(np.abs(vel))
        assert axis <= V_MAX * 1.02, f"{label}: peak axis velocity {axis:.2f} > v_max"
    # Corridor excursion: how far the curve leaves the union of corridors. Reported, not
    # asserted -- it is a property of the engine, not of the bindings. Known non-zero cases:
    # control points pinned by a start velocity that points out of the first corridor are
    # not corridor-constrained, and some allocations leave a (degree+1)-window of control
    # points in no single corridor. The hard safety check is the obstacle clearance above.
    excursion = max(min(float(np.max(c.A @ p - c.b)) for c in res.corridors) for p in pos)
    print(f"  {label}: clearance {clearance:.2f} m, max corridor excursion {max(excursion, 0.0) * 100:.1f} cm")
    return clearance, speed, traj


def test_sfc_planning_from_obstacles():
    start, goal = np.array([0.0, 0.0, -3.0]), np.array([10.0, 0.0, -3.0])
    boxes = [ObstacleBox([4.5, -0.5, -7.0], [5.5, 0.5, 0.0])]  # floor-to-ceiling pillar, in the way
    grid = SparseVoxelGrid(VOXEL, DRONE_R)
    grid.update_from_obstacles(boxes)
    planner = TrajectoryPlanner(planner_config(), grid, v_max=V_MAX, a_max=A_MAX)
    res = planner.plan_mission(start, goal)
    clearance, speed, traj = check_plan(res, start, goal, boxes, "obstacles")
    # a straight line would go through the pillar, so the plan must have detoured
    lateral = np.max(np.abs(traj.sample(np.linspace(traj.start_time, traj.end_time, 200))[:, 1]))
    assert lateral > 0.5 + DRONE_R, f"no detour around the pillar (max |y| {lateral:.2f} m)"
    print(f"  obstacles: {res}  clearance={clearance:.2f} m  peak speed={speed:.2f} m/s  "
          f"duration={traj.duration:.2f} s")

    # with a non-zero start velocity / acceleration (stitched replan)
    v0, a0 = np.array([1.0, 0.0, 0.0]), np.array([0.2, 0.0, 0.0])
    res2 = TrajectoryPlanner(planner_config(), grid, V_MAX, A_MAX).plan_mission(start, goal, v0, a0)
    _, _, traj2 = check_plan(res2, start, goal, boxes, "obstacles+v0")
    assert_close(traj2.velocity(traj2.start_time), v0, 1e-3, "start velocity")

    # the grid is shared: dropping the planner leaves Python's grid usable, and vice versa
    del planner
    assert grid.num_occupied > 0
    p = TrajectoryPlanner(planner_config(), SparseVoxelGrid(VOXEL, DRONE_R))
    assert p.plan_mission(start, goal).success  # grid owned only by the planner now


def test_sfc_planning_from_scan():
    """Map built from a raycast LiDAR scan, so FREE / OCCUPIED / UNKNOWN are all in play:
    two pillars, the far one partly shadowed by the near one."""
    origin = np.array([0.0, 0.0, -3.0])
    goal = np.array([10.0, 0.0, -3.0])
    boxes = [ObstacleBox([4.5, -0.5, -7.0], [5.5, 0.5, 0.0]),
             ObstacleBox([7.5, 1.0, -7.0], [8.5, 2.0, 0.0])]

    def hit(o, d, max_range=20.0):
        for s in np.arange(0.1, max_range, 0.05):
            q = o + s * d
            if any(np.all(q >= b.min) and np.all(q <= b.max) for b in boxes):
                return q
        return None

    hits = []
    for az in np.radians(np.arange(-60, 61, 2.0)):
        for el in np.radians(np.arange(-15, 16, 3.0)):
            d = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), -math.sin(el)])
            q = hit(origin, d)
            if q is not None:
                hits.append(q)
    hits = np.array(hits)

    grid = SparseVoxelGrid(VOXEL, DRONE_R)
    grid.update_from_scan(origin, hits)
    assert grid.num_free > 0 and grid.num_occupied > 0
    res = TrajectoryPlanner(planner_config(), grid, V_MAX, A_MAX).plan_mission(origin, goal)
    clearance, speed, traj = check_plan(res, origin, goal, boxes, "scan")
    print(f"  scan:      {res}  clearance={clearance:.2f} m  peak speed={speed:.2f} m/s  "
          f"duration={traj.duration:.2f} s")


def test_free_space_plan_matches_goal():
    grid = SparseVoxelGrid(VOXEL, DRONE_R)
    start, goal = np.array([0.0, 0.0, -2.0]), np.array([6.0, 3.0, -4.0])
    res = TrajectoryPlanner(planner_config(), grid).plan_mission(start, goal)
    far = [ObstacleBox([90, 90, -1], [91, 91, 0])]
    check_plan(res, start, goal, far, "free space")

    # enforce_norm_limits=True makes v_max a bound on SPEED rather than per axis
    res = TrajectoryPlanner(planner_config(enforce_norm_limits=True), grid).plan_mission(start, goal)
    check_plan(res, start, goal, far, "free space+norm", norm_limits=True)


# ----------------------------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed  (coast_planner {coast_planner.__version__}, "
          f"from {coast_planner.__file__})")
    sys.exit(1 if failed else 0)
