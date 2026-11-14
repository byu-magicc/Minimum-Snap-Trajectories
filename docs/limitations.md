# Gotchas and known limitations

## Gotchas

**Release builds delete Eigen's bounds checks.** `-DNDEBUG` makes `eigen_assert` a no-op, so
an out-of-range `.block()` does not fault — it corrupts the heap and the process dies later
with something unrelated. `include/eigen_safety.hpp` is force-included into every
translation unit and makes those assertions **throw** instead, at no measurable cost. Keep
`MIN_SNAP_EIGEN_SAFETY=ON`.

**Accumulation is required, not an optimisation.** The beam pattern is angular, so past
roughly `voxel_resolution / beam_angle` (~9.5 m for 3° beams at 0.5 m voxels) neighbouring
beams are further apart than a voxel and a *single* scan's free space degenerates into a fan
of thin lines with gaps between them. The FREE/UNKNOWN boundary is then meaningless.
Accumulating across scans as the vehicle moves is what fills those gaps.
`set_accumulate(false)` exists for tests.

**`planning_horizon` dominates everything.** Cost grows sharply and non-linearly with it,
because unknown space is traversable and a distant goal makes A* fan out. Measured on one
scene: 10 m → 2 corridors, 11 control points, ~6 ms total; 20 m → 4 corridors, 32 control
points, ~65 ms. Longer horizons also produce more convoluted paths. Start short.

**Duration is tied to control-point count.** Knots are unit-spaced, so flight time in
seconds is `total_control_points - degree`. Adding control points for resolution also adds
flight time; there is currently no way to decouple them.

**`v_max` is a per-axis limit unless you ask otherwise.** With the default
`enforce_norm_limits = false` a diagonal trajectory can legitimately fly at up to
`√3 · v_max`. Set it to `true` if `v_max` must be a true speed limit.

**Planning is deterministic — keep it that way.** A*'s budget is a node count
(`astar_max_nodes`), not a wall-clock timer, so identical inputs give identical trajectories.
This matters more than it sounds: when the budget was time-based, the same scenario reached
the goal on some runs and stranded the vehicle on others, purely from CPU scheduling jitter.
A wall-clock backstop (`astar_max_time_ms`) still exists for pathological cases and logs
loudly when it fires.

**A detour longer than `planning_horizon` can strand the vehicle.** The horizon-projected
goal points into the obstacle and the search never sees the way around. Raising the horizon
costs solve time super-linearly, so this is a genuine trade rather than a tuning oversight.

**Rebuild consumers after changing this library.** It is statically linked, so a consumer
built against an older archive keeps the old code until it is relinked. That failure is at
least loud — an undefined symbol at link time — rather than the silent runtime corruption
the previous shared-library setup produced. `ldd <binary> | grep min_snap_engine` should return
nothing; if it returns a path, something is still linking it dynamically.

**Corridors guarantee the *reference trajectory* is clear**, by exactly
`drone_physical_radius`. The vehicle is only actually safe while its tracking error stays
below that margin. Measure it.

## Known limitations

Honest list of rough edges, current as of this writing:

- **No prebuilt wheels yet.** `pip install` compiles from source (about a minute, and it needs a
  compiler and network). The build is already self-contained, so publishing to PyPI is now
  just a matter of building wheels with cibuildwheel.
- **Every plan ends at a full stop.** The terminal boundary condition pins zero velocity and
  acceleration. With a goal beyond `planning_horizon` that stop lands at the horizon rather
  than the goal, which costs average speed.
- **Corridor containment has known exceptions.** (1) Control points pinned by a start
  velocity that points out of the first corridor are not corridor-constrained, so the opening
  span can leave it (measured: 2.3 cm). (2) Some allocations leave a *(degree+1)*-window of
  control points in no single corridor (measured: curve 4.5 mm outside the union). Obstacle
  clearance held in every case observed, but the corridor chain is not a strict certificate.
  `python/tests/test_bindings.py` reports the excursion per plan.
- **Wide obstacles dead ahead can be infeasible.** A 3 m-wide pillar directly on the path
  fails to plan: A* hugs it and the flanking corridors come out ~0.1 m wide.
- **Some geometries are genuinely infeasible** and the QP correctly refuses, taking tens of
  milliseconds to do so. Refusals are handled (the caller holds position), but they are not
  rare.
- **`is_occupied_radius` and the KD-tree cover real obstacles only**, not the unknown
  frontier — deliberate, so trajectory validation is not tripped by unmeasured space.
- **Not thread-safe.** Serialise map updates against planning.
