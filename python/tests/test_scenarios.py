"""Tests for coast_planner.scenarios and the demo's environment options."""

import json

import numpy as np
import pytest

from coast_planner import ObstacleBox
from coast_planner.scenarios import (BUILTIN, Scenario, box_clearance, load, pillar,
                                     wall_with_window)


def path_points(result, n=3000):
    s = result.sampler()
    return s.sample(np.linspace(s.start_time, s.end_time, n))[:, :3]


@pytest.mark.parametrize("name", sorted(BUILTIN))
def test_builtin_scenario_plans_to_goal_with_clearance(name):
    # Every bundled environment must be solvable: the demo is useless if its own map fails.
    sc = load(name)
    result, _ = sc.plan()
    assert result.success and result.reached_goal
    P = path_points(result)
    assert np.allclose(P[-1], sc.goal, atol=1e-3)
    assert box_clearance(P, sc.obstacles) >= sc.drone_radius - 0.01


def test_hard_is_harder_than_simple():
    simple, hard = (load(n).plan()[0] for n in ("simple", "hard"))
    assert len(load("hard").obstacles) > len(load("simple").obstacles)
    assert hard.total_control_points > simple.total_control_points  # longer, twistier flight


def test_json_round_trip(tmp_path):
    sc = load("hard")
    path = tmp_path / "env.json"
    sc.save(path)
    back = load(str(path))
    assert back.name == "hard"  # the name stored in the file wins over the file name
    assert back.start == sc.start and back.goal == sc.goal and back.config == sc.config
    assert len(back.obstacles) == len(sc.obstacles)
    for a, b in zip(back.obstacles, sc.obstacles):
        assert np.allclose(a.min, b.min) and np.allclose(a.max, b.max)


def test_minimal_json_uses_defaults(tmp_path):
    path = tmp_path / "mini.json"
    path.write_text(json.dumps({
        "start": [0, 0, -3], "goal": [10, 0, -3],
        "obstacles": [{"min": [4.5, -0.5, -7], "max": [5.5, 0.5, 0], "note": "a pillar"}]}))
    sc = load(str(path))
    assert sc.drone_radius == 0.49 and sc.v_max == 3.0
    result, _ = sc.plan()
    assert result.success


@pytest.mark.parametrize("bad, match", [
    ({"goal": [1, 0, -2]}, "start"),
    ({"start": [0, 0], "goal": [1, 0, -2]}, "start"),
    ({"start": [0, 0, -2], "goal": [1, 0, -2], "obstacles": [{"min": [0, 0, 0]}]}, "obstacle 0"),
    ({"start": [0, 0, -2], "goal": [1, 0, -2],
      "obstacles": [{"min": [1, 0, -1], "max": [0, 1, 0]}]}, "below"),
    ({"start": [0, 0, -2], "goal": [1, 0, -2], "speed": 3}, "unknown scenario keys"),
])
def test_invalid_scenarios_are_rejected(bad, match):
    with pytest.raises(ValueError, match=match):
        Scenario.from_dict(bad)


def test_unknown_config_field_is_rejected():
    with pytest.raises(Exception):
        Scenario.from_dict({"start": [0, 0, -2], "goal": [1, 0, -2],
                            "config": {"planing_horizon": 10}})


def test_unknown_name_is_rejected():
    with pytest.raises(ValueError, match="unknown scenario"):
        load("medium")


def test_helpers_build_expected_boxes():
    p = pillar(3.0, -1.0, half_width=0.5, height=6.0)
    assert np.allclose(p.min, [2.5, -1.5, -6.0]) and np.allclose(p.max, [3.5, -0.5, 0.0])
    boxes = wall_with_window(10.0, 5.0, window_east=(-1.0, 1.0), window_alt=(1.0, 3.0))
    assert len(boxes) == 4
    # the window centre is clear of every wall piece; the wall either side of it is not
    assert box_clearance([10.0, 0.0, -2.0], boxes) > 0.9
    assert box_clearance([10.0, 3.0, -2.0], boxes) == 0.0


def test_box_clearance():
    box = ObstacleBox([0, 0, 0], [1, 1, 1])
    assert box_clearance([[2, 0.5, 0.5]], [box]) == pytest.approx(1.0)
    assert box_clearance([[0.5, 0.5, 0.5]], [box]) == 0.0


# --- the demo ---------------------------------------------------------------------------------

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


@pytest.mark.parametrize("env", sorted(BUILTIN))
@pytest.mark.parametrize("mode, expected", [
    ("both", {"unconstrained_3d", "unconstrained_profiles",
              "plan_3d", "plan_top", "plan_profiles"}),
    ("sfc", {"plan_3d", "plan_top", "plan_profiles"}),
    ("free", {"unconstrained_3d", "unconstrained_profiles"}),
])
def test_demo_modes(env, mode, expected, tmp_path, capsys):
    from coast_planner._demo import main
    main(["--env", env, "--mode", mode, "--save", str(tmp_path)])
    assert {p.stem for p in tmp_path.glob("*.png")} == expected
    if mode != "free":
        assert "Checks passed" in capsys.readouterr().out


def test_demo_export_then_run(tmp_path):
    from coast_planner._demo import main
    path = tmp_path / "mine.json"
    main(["--env", "simple", "--export", str(path)])
    assert not list(tmp_path.glob("*.png"))  # export only writes the file
    main(["--env", str(path), "--mode", "sfc", "--save", str(tmp_path / "out")])
    assert len(list((tmp_path / "out").glob("*.png"))) == 3


def test_demo_reports_failed_checks(tmp_path, capsys):
    # Start inside a pillar: the demo must say so and exit non-zero, not crash.
    from coast_planner._demo import main
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"start": [5, 0, -3], "goal": [10, 0, -3],
                                "obstacles": [{"min": [4.5, -0.5, -7], "max": [5.5, 0.5, 0]}]}))
    with pytest.raises(SystemExit) as exit_info:
        main(["--env", str(path), "--mode", "sfc", "--save", str(tmp_path / "out")])
    assert exit_info.value.code == 1
    assert "start" in capsys.readouterr().out
