# Configuration (`FrontEndConfig`)

All fields of `FrontEndConfig`, with defaults:

## Geometry
| Field | Default | Notes |
|---|---|---|
| `voxel_resolution` | 0.5 | Map cell size (m). Drives cost and precision everywhere. |
| `drone_physical_radius` | 0.5 | Circumscribed vehicle radius. Corridors are clear by this margin — **your controller's tracking error must stay below it** or the guarantee does not transfer to the vehicle. |
| `map_bounds` | (100, 100, 15) | Full extent; usable region is ±half on each axis. |
| `min_altitude` | 0.3 | meters above ground, positive-up. |

## Corridors
| Field | Default | Notes |
|---|---|---|
| `sfc_width` / `sfc_height` | 5.0 | Lateral corridor extent is capped at `sfc_width - 2*drone_radius`. Raise this to let corridors expand into open space. |
| `sfc_start_ext` / `sfc_end_ext` | 5.0 | How far each corridor extends behind/ahead of its segment. Large values make neighbouring corridors overlap heavily; small values tighten the QP and can make it harder to solve. |
| `max_sfc_count` | -1 | Cap on corridors per plan (-1 = uncapped). A safety valve on constraint growth, not the way to bound the horizon. |

## Search
| Field | Default | Notes |
|---|---|---|
| `time_budget_factor` | 1.20 | Slack on the initial flight-time budget. The ideal straight-line time is systematically short once the trajectory must also curve and stop, so the first solve fails and the retry loop lengthens it — and added control points **are** flight time, permanently. Measured over 12 scenarios: 1.00 → 30% first-attempt success, 8.5 ms mean solve; 1.20 → 60% and 5.8 ms. |
| `astar_max_nodes` | 3000 | Deterministic cap on A* expansions. Keeps planning reproducible; see [Gotchas](limitations.md#gotchas). |
| `astar_max_time_ms` | 25.0 | Wall-clock backstop for pathological searches only; the node cap is what normally binds. |
| `planning_horizon` | 12.0 | How far ahead A* searches (m). **The single most influential parameter** — see [Gotchas](limitations.md#gotchas). |
| `unknown_traversal_cost` | 0.5 | Extra cost per step through UNKNOWN. `0` treats unknown as free (unsafe); negative hard-blocks it (cannot route around corners). Keep `1 + unknown_traversal_cost + astar_clearance_cost < 2.0`, or A* stops preferring the goal and floods outward (the old default of 2.0 did exactly that). |
| `sensor_range` | -1.0 | Search radius cap (m), -1 disables. |
| `fov_vertical_min/max` | -7 / 52 | Parsed and passed but **no longer gate anything** — unknown-space cost subsumed the old FOV gate. |

## Trajectory
| Field | Default | Notes |
|---|---|---|
| `degree` | 4 | Spline degree. Minimum snap requires ≥ 4. |
| `spline_type` | "clamped" | `"clamped"` or `"natural"`. |
| `aircraft_type` | "multi-rotor" | `"multi-rotor"` or `"fixed-wing"` (different corridor and allocation strategies). |
| `v_max`, `a_max` | 3.0 / 2.0 | Kinematic limits. `TrajectoryPlanner` overwrites these from its own constructor arguments, so set them there. Enforced **per axis** by default; see `enforce_norm_limits`. |
| `enforce_norm_limits` | false | `false`: each of `abs(v_x)`, `abs(v_y)`, `abs(v_z)` is `≤ v_max`, so speed can reach `√3·v_max` (likewise for acceleration). `true`: shrinks the box so speed really is `≤ v_max`, at the cost of more refusals on long paths. |
| `max_solve_time_ms` | 60.0 | Wall-clock cap on the OSQP retry loop. A failing solve burns the full iteration budget, so this keeps a doomed plan from overrunning the replan period. 0 disables. |
