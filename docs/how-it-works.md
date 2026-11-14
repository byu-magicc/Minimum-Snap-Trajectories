# How COAST works

## Pipeline

```
LiDAR returns + sensor origin
        │
        ▼
SparseVoxelGrid                     three-state map: FREE / OCCUPIED / UNKNOWN
        │                           accumulated across scans, raycast-cleared
        ▼
FrontEndSFC::get_corridors_astar
        │
        ├─▶ A* over the voxel grid   occupied = blocked, unknown = extra cost
        │                            bounded by planning_horizon
        ├─▶ line-of-sight smoother   collapses the cell path to few waypoints
        └─▶ Safe Flight Corridors    one oriented box per path segment, each of its
                                     six faces shrunk independently against obstacles
        │
        ▼
OSQP quadratic program              minimum-snap B-spline control points, constrained
  + MINVO kinodynamic bounds        to stay inside the corridors, velocity/accel limited
        │
        ▼
PlanningResult                      control points + knot vector + corridors + timings
```

**The three-state map** is what lets the planner distinguish *"measured empty"* from
*"never looked at"*. Each beam is walked from the sensor origin with Amanatides–Woo: cells
along the way become `FREE` (and any stale occupancy there is cleared), the return point
becomes `OCCUPIED`, everything else stays `UNKNOWN`. Observations accumulate across scans
inside a bounded window around the vehicle.

**Corridor containment without integer variables.** For a degree-*d* B-spline, each curve
span is the convex hull of *d+1* consecutive control points. Control points are allocated
into *exclusive* pools (constrained to one corridor) separated by *bridge* pools of exactly
*p* points constrained to the **intersection** of two adjacent corridors. Every window of
*p+1* consecutive control points therefore lies wholly within at least one corridor, so
every curve segment is contained — in continuous time, with no mixed-integer assignment.
(That is the design. There are known exceptions; see [Known limitations](limitations.md#known-limitations).)

## Coordinate conventions — read this first

**Everything is NED.** North-East-Down, meters.

- **`z` is negative-up.** An altitude of 5 m is `z = -5.0`. This trips up almost everyone,
  and has historically caused real bugs in this codebase (a geofence that demanded `z >= 0`
  made every solve infeasible; test fixtures written positive-up made A* refuse to plan).
- **The world is centred on the origin.** `map_bounds` is a *full extent*, so the reachable
  region is `±map_bounds/2` on each axis. With the default `(100, 100, 15)` that is
  x,y ∈ [-50, +50] and z ∈ [-7.5, +7.5] — i.e. only **7.5 m of usable altitude**.
- **`min_altitude` is positive-up** (meters above ground) and converted internally. It is
  the one field that does not follow the sign convention, because it reads more naturally.

If the planner mysteriously refuses to produce corridors, check these first.
