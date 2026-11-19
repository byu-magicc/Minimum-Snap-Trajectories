"""Plan through a test environment and plot the result.

    pip install "coast-planner[viz]"
    coast-planner-demo                      # the simple environment; opens windows
    coast-planner-demo --env hard           # weave through pillars, then a window
    coast-planner-demo --mode free          # only the unconstrained (no-corridor) solve
    coast-planner-demo --env hard --voxel 0.25   # the same environment on a finer map
    coast-planner-demo --save out           # writes PNGs to ./out instead

Design your own environment by exporting a built-in one, editing the JSON, and running it:

    coast-planner-demo --env hard --export my_env.json
    coast-planner-demo --env my_env.json

(`python python/examples/plot_demo.py` in a clone of the repository runs the same demo.)

The demo plans all the way to the goal (no planning horizon) and reports how long it took.
It then checks the result and exits with status 0 if the plan reached the goal with the
drone radius clear of every obstacle, 1 if the planner reported success but the trajectory
is wrong (likely a bug), and 2 if there is no trajectory: no route, an invalid environment,
or a route the optimiser found infeasible (not necessarily a bug). If plot windows cannot be opened (no Tk on this Python, no
display over SSH), the demo says why and saves the PNGs to ./coast_demo_plots instead.

Coordinates are NED; plots display East-North-Up (altitude = -z).
"""

import argparse
import contextlib
import os

import numpy as np

from coast_planner import MinSnapClamped, TrajectorySampler
from coast_planner.plot import plot_plan, plot_profiles, plot_trajectory
from coast_planner.scenarios import BUILTIN, box_clearance, load

FALLBACK_DIR = "coast_demo_plots"
# Backends that render to files only; with one of these, plt.show() silently does nothing.
_NON_INTERACTIVE = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
# Clearance shortfall tolerated before the demo calls it a violation (m). The corridor chain
# has known millimetre-scale excursions; see docs/limitations.md.
_CLEARANCE_TOL = 0.01
# A route this short through moderate clutter should plan within about this time.
_FAST_ROUTE_M, _FAST_ROUTE_MS = 15.0, 10.0


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


def _samples(sampler, n=2000):
    return sampler.sample(np.linspace(sampler.start_time, sampler.end_time, n))[:, :3]


def check_endpoints(scenario):
    """Problems with the environment itself: a start or goal inside an obstacle's inflation."""
    problems = []
    for label, p in (("start", scenario.start), ("goal", scenario.goal)):
        clearance = box_clearance(p, scenario.obstacles) if scenario.obstacles else np.inf
        if clearance < scenario.drone_radius:
            problems.append(f"the {label} {list(p)} is within the drone radius "
                            f"({scenario.drone_radius} m) of an obstacle")
    return problems


def assess(scenario, result):
    """Classify a plan; returns ``(outcome, messages)``.

    ``outcome`` is one of
        ``"ok"``         reached the goal with the drone radius clear of every obstacle;
        ``"violation"``  the planner reported success but the trajectory is wrong (too close
                         to an obstacle, or short of the goal): likely a planner bug;
        ``"refused"``    A* found a route but the optimiser found no feasible trajectory
                         through its corridors. The route itself can be genuinely infeasible
                         (too tight to fly within the limits), so this is not necessarily a bug;
        ``"no_route"``   the front end found no route to the goal.

    Clearance is measured against the obstacle boxes themselves, not the voxel map: the map
    rounds each box outward to whole cells, so it is conservative by up to one voxel.
    """
    if not result.success:
        if not result.corridors:
            return "no_route", ["the front end found no route from start to goal"]
        return "refused", [f"A* found a route ({len(result.corridors)} corridors), but the "
                           f"optimiser found no feasible trajectory through it after "
                           f"{result.solve_attempts} attempt(s). The route may be genuinely too "
                           "tight to fly; --verbose shows which constraints blocked it"]
    problems = []
    P = _samples(result.sampler())
    if not result.reached_goal:
        gap = float(np.linalg.norm(P[-1] - np.asarray(scenario.goal, float)))
        problems.append(f"the plan stops {gap:.2f} m short of the goal (reached_goal=False)")
    if scenario.obstacles:
        clearance = box_clearance(P, scenario.obstacles)
        if clearance < scenario.drone_radius - _CLEARANCE_TOL:
            problems.append(f"the trajectory passes {clearance:.3f} m from an obstacle, inside "
                            f"the drone radius ({scenario.drone_radius} m)")
    return ("violation", problems) if problems else ("ok", [])


def report_timing(scenario, result):
    """Print where the planning time went, and flag a short route that planned slowly."""
    print(f"Planning time {result.total_time_ms:.1f} ms: front end (A* + corridors) "
          f"{result.sfc_time_ms:.1f} ms, optimiser {result.opt_time_ms:.1f} ms over "
          f"{result.solve_attempts} attempt(s).")
    dist = float(np.linalg.norm(np.subtract(scenario.goal, scenario.start)))
    if dist <= _FAST_ROUTE_M and result.total_time_ms > _FAST_ROUTE_MS:
        print(f"Note: slower than the ~{_FAST_ROUTE_MS:.0f} ms expected for a {dist:.0f} m route "
              "through moderate clutter.")


def free_space_figures(scenario):
    """The closed-form minimum-snap solve from start to goal, ignoring obstacles."""
    start, goal = np.asarray(scenario.start, float), np.asarray(scenario.goal, float)
    # Average half of v_max, so the peak speed of a rest-to-rest minimum-snap path (about twice
    # the average) stays near v_max; at least 4 s so short hops stay smooth.
    duration = max(4.0, 2.0 * float(np.linalg.norm(goal - start)) / scenario.v_max)
    solver = MinSnapClamped(num_segments=10)
    dt = duration / 10
    z = np.zeros(3)
    C = solver.solve_unconstrained(p0=start, v0=z, a0=z, pf=goal, vf=z, af=z, dt=dt)
    knots = solver.scaled_knots(dt)

    path = _samples(TrajectorySampler(C, knots, solver.degree))
    clearance = box_clearance(path, scenario.obstacles) if scenario.obstacles else np.inf
    if clearance < scenario.drone_radius:
        verdict = (f"comes within {clearance:.2f} m of an obstacle (drone radius "
                   f"{scenario.drone_radius} m): the corridors are what keep the plan clear")
    else:
        verdict = f"stays {clearance:.2f} m from every obstacle"
    print(f"Free-space solve (no corridors, ignores obstacles): {verdict}.")

    ax = plot_trajectory(C, knots, solver.degree, obstacles=scenario.obstacles)
    ax.set_title(f"{scenario.name}: unconstrained minimum snap (no corridors)")
    fig = plot_profiles(C, knots, solver.degree,
                        title=f"{scenario.name}: unconstrained solve, derivatives")
    return {"unconstrained_3d": ax.figure, "unconstrained_profiles": fig}


def plan_figures(scenario, verbose=False):
    """Plan through the corridors, assess and plot it. Returns (figures, outcome, messages)."""
    problems = check_endpoints(scenario)
    if problems:
        return {}, "invalid", problems
    with contextlib.nullcontext() if verbose else quiet_stdout():
        result, _ = scenario.plan()
    print(result)
    report_timing(scenario, result)
    outcome, messages = assess(scenario, result)

    kw = dict(obstacles=scenario.obstacles, start=scenario.start, goal=scenario.goal)
    ax3 = plot_plan(result, **kw)
    ax4 = plot_plan(result, view="top", **kw)
    figs = {"plan_3d": ax3.figure, "plan_top": ax4.figure}
    if result.success:
        figs["plan_profiles"] = plot_profiles(
            result, v_max=scenario.v_max, a_max=scenario.a_max,
            title=f"{scenario.name}: planned trajectory, derivatives (limits are per axis)")
    return figs, outcome, messages


# What the demo prints for each outcome, and its exit status.
_OUTCOMES = {
    "ok": ("Checks passed: reached the goal with the drone radius clear of every obstacle.", 0),
    "violation": ("CHECKS FAILED (likely a planner bug; please report it with the environment "
                  "file, saved with --export for a built-in one):", 1),
    "refused": ("No trajectory (not necessarily a bug):", 2),
    "no_route": ("No trajectory:", 2),
    "invalid": ("Invalid environment:", 2),
}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="coast-planner-demo", description=__doc__.splitlines()[0])
    ap.add_argument("--env", default="simple", metavar="NAME_OR_FILE",
                    help=f"built-in environment ({', '.join(BUILTIN)}) or a .json file "
                         "(default: simple)")
    ap.add_argument("--mode", choices=("both", "sfc", "free"), default="both",
                    help="sfc: plan through Safe Flight Corridors; free: the unconstrained "
                         "minimum-snap solve, which ignores obstacles; both (default)")
    ap.add_argument("--save", metavar="DIR", help="write PNGs here instead of showing windows")
    ap.add_argument("--export", metavar="FILE",
                    help="write the environment to a JSON file to edit, then exit")
    ap.add_argument("--voxel", type=float, metavar="METRES",
                    help="map cell size, overriding the environment's voxel_resolution (0.5)")
    ap.add_argument("--verbose", action="store_true", help="show the planner's own log")
    args = ap.parse_args(argv)

    try:
        scenario = load(args.env)
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise SystemExit(f"Cannot load environment {args.env!r}: {e}")
    if args.voxel is not None:
        if args.voxel <= 0:
            raise SystemExit("--voxel must be positive")
        scenario.voxel_resolution = args.voxel
    if args.export:
        scenario.save(args.export)
        print(f"Wrote {args.export}. Edit it, then run:  coast-planner-demo --env {args.export}")
        return

    try:
        import matplotlib
    except ImportError:
        raise SystemExit('The demo plots with matplotlib. Install it with:  pip install "coast-planner[viz]"')
    if args.save:
        matplotlib.use("Agg")
    elif not can_open_windows():
        args.save = FALLBACK_DIR
    import matplotlib.pyplot as plt

    print(f"Environment '{scenario.name}': {scenario.description or 'no description'} "
          f"({len(scenario.obstacles)} obstacles, {scenario.voxel_resolution} m voxels)")
    figs, outcome = {}, "ok"
    if args.mode in ("free", "both"):
        figs.update(free_space_figures(scenario))
    if args.mode in ("sfc", "both"):
        plan_figs, outcome, messages = plan_figures(scenario, verbose=args.verbose)
        figs.update(plan_figs)
        headline, _ = _OUTCOMES[outcome]
        print("\n" + headline + "".join(f"\n  - {m}" for m in messages))

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        for name, fig in figs.items():
            path = os.path.join(args.save, f"{name}.png")
            fig.savefig(path, dpi=110, bbox_inches="tight")
            print("wrote", path)
    else:
        plt.show()
    status = _OUTCOMES[outcome][1]
    if status:
        raise SystemExit(status)


if __name__ == "__main__":
    main()
