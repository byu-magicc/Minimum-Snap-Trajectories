#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <vector>
#include <string>
#include <memory>
#include "voxel_grid.hpp"
#include "ring_buffer.hpp"
#include "astar_sfc.hpp"
#include "dynamic_sfc.hpp"
#include "static_sfc.hpp"

namespace trajectory_planner {

// ==========================================
// Constraint Pool (Port of front_end.py allocation_data)
// ==========================================
struct ConstraintPool {
    int pts;                       // Number of control points in this pool
    std::vector<int> sfc_indices;  // Which SFC(s) constrain this pool
    std::string type;              // "exclusive" or "bridge"
};

// ==========================================
// FrontEnd Configuration
// ==========================================
struct FrontEndConfig {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double voxel_resolution = 0.5;
    double drone_physical_radius = 0.5;
    double sfc_height = 5.0;
    double sfc_width = 5.0;
    double sfc_start_ext = 5.0;
    double sfc_end_ext = 5.0;
    std::string spline_type = "clamped";
    std::string aircraft_type = "multi-rotor";
    int degree = 4;
    Eigen::Vector3d map_bounds = Eigen::Vector3d(100.0, 100.0, 15.0);
    double sensor_range = -1.0;   // meters, -1 = disabled (full map visibility)
    double fov_vertical_min = -7.0; // degrees, bottom edge of FOV
    double fov_vertical_max = 52.0; // degrees, top edge of FOV
    double min_altitude = 0.3;    // meters above ground (positive-up, converted to NED internally)
    // Altitude CEILING, metres above ground (positive-up, converted to NED internally).
    // <= 0 disables, falling back to the map_bounds-derived geofence.
    //
    // The counterpart to min_altitude, and it was missing: the only thing bounding the search
    // upward was map_bounds.z/2, a map-SIZING number doing safety-critical duty by accident. At
    // map_bounds.z = 15 that is a 7.5 m ceiling, and 6 m trunks inflate to 6.69 m -- leaving a
    // 0.81 m window that A* duly found. Measured: the path climbed from 4.14 m to 7.24 m, crossed
    // above the canopy, and dived back at 77 degrees from horizontal. That is a legal path, so
    // A* reported success, but the corridor chain (14 deg -> 0 -> 45 -> 77) turns far too sharply
    // for consecutive boxes to share enough volume, and the QP came back primal-infeasible on the
    // corridor block while boundary and velocity sat at ~1e-5.
    //
    // SANDO carries the same bound explicitly as z_max (6.0 m in sando.yaml, exactly its static
    // forest's trunk height) and applies it to both the graph search and the ellipsoid
    // decomposition. Setting it to the trunk height makes overflight structurally impossible
    // rather than merely unattractive, which is what the benchmark intends -- the trunks are
    // sized against the cruise altitude specifically so that horizontal avoidance is what is
    // being measured.
    double max_altitude = -1.0;
    /// Maximum number of SFCs per replan (receding-horizon truncation). -1 = disabled.
    /// Flight runs 3 (params/forest_benchmark.yaml). SWEPT 2/3/4 on 2026-09-28 and 3 is a genuine
    /// optimum -- both neighbours are worse, for different reasons:
    ///
    ///   n=150 synthetic, density 0.10, horizon 12   R_succ  collided  ms90   OSQP p99
    ///                                       cap 2    70%*      0%*   44.44      -
    ///                                       cap 3    66.7%   27.3%   19.07    12.47   <-- best
    ///                                       cap 4    55.3%   40.0%   32.22    23.01
    ///   (*cap 2 row is the n=10 forest run; its speed was 7.6% below cap 3 with ms90 +47%)
    ///
    /// Capping LOWER does not reclaim bridge points. Single-corridor share is ~51% at every cap,
    /// so the cap never reshapes the typical plan -- it only constrains the minority of routes
    /// that genuinely need more corridors, and squeezing those into fewer makes each span more
    /// turning, so they come out narrow, the QP struggles, and retries burn time.
    /// Allowing MORE is worse still: more junctions means more constraint rows and a longer,
    /// more complex committed chain, which costs 11 points of R_succ and 13 of collisions.
    ///
    /// BEWARE n=10. The 10-environment forest sweep ranked cap 4 ABOVE cap 3 (90% vs 80% R_succ);
    /// at n=150 the ordering reverses. One environment is worth 10 points there, so never settle
    /// a success comparison on the forest set alone.
    ///
    /// Corridor count is NOT what limits speed -- see planning_horizon above. No cap moved median
    /// control points off 8, i.e. every arm stayed on the duration floor.
    int max_sfc_count = -1;
    /// Minimum ground distance the truncated corridor chain must cover, in metres. 0 disables.
    ///
    /// max_sfc_count counts CORRIDORS, not distance, and that is an absorbing failure when
    /// segments are short. Threading a gap makes A* emit one-voxel segments, so three corridors
    /// can span under 2 m -- and because allocate_dynamic_control_points zeroes a corridor's
    /// exclusive pool whenever its segment is shorter than (sfc_start_ext + sfc_end_ext), every
    /// pool zeroes out and the plan becomes nothing but junction bridges. The chain then ENDS
    /// essentially where the vehicle already is, the QP solves start-to-end happily, reports
    /// success, and commands a hover. The vehicle does not move, so the next cycle produces the
    /// same chain: a permanent stall with `result: true` at 10 Hz.
    ///
    /// Measured in flight, hard s0 at v_max 8: T_trav 1662 s, v_avg 0.11 m/s, 14502 replans with
    /// the route unchanged 99.53% of the time and median route divergence 0.00 deg.
    ///
    /// So the chain is grown PAST max_sfc_count until it covers this distance, bounded by
    /// max_sfc_count_hard so QP size still cannot run away.
    /// Whether the A* geofence is extended to CONTAIN the vehicle when it sits outside the
    /// altitude band. true = plan from wherever the vehicle is (it can recover); false = the
    /// band is absolute and a vehicle outside it cannot plan at all.
    ///
    /// This is an A/B switch, not a preference. Extending the bounds fixed a hard lock-up (the
    /// vehicle climbed above max_altitude and planning stopped dead), but it also let A* CRUISE
    /// above the canopy: trunks are 6 m and max_altitude is 6 m, so above the ceiling the forest
    /// is empty and a straight run to the goal exists. Measured on hard s0: start 7.5 m -> whole
    /// chain at 6.25-7.50 m; start 8.5 m -> ONE corridor over the entire forest; start <= 6.5 m
    /// -> chain stays in the forest.
    ///
    /// Set false to recover the pre-fix behaviour exactly, at the cost of the lock-up returning.
    bool geofence_contains_vehicle = true;
    double min_chain_length = 0.0;
    /// Absolute ceiling on corridor count once min_chain_length is in play. 0 => 3x max_sfc_count.
    int max_sfc_count_hard = 0;


    // Kinematic limits used by the control-point/time allocator. TrajectoryPlanner
    // stamps these from its own v_max/a_max so the allocator and the MINVO
    // kinodynamic constraints are always sized against the same limits.
    double v_max = 3.0;
    double a_max = 2.0;
    // Extra cost multiplier A* pays per step through UNKNOWN (never-observed) space.
    // 0 = unknown is as cheap as free (unsafe); negative = hard-block it, restoring the old
    // behaviour where the search simply could not route past a corner.
    // 2.0 was the historical default and is TOO HIGH: it lets a detour through mapped space win
    // whenever it is under (1 + c) = 3x the direct route, which in forward flight pushes the
    // search sideways and backwards into already-mapped space (observed: A* retreating rather
    // than advancing). It also breaks the guidance invariant once any clearance term is added --
    // 1 + 2.0 + heat_weight exceeds astar_heuristic_weight. 0.5 keeps a real preference for
    // measured-free space (a 1.5x detour still wins) and leaves room for the heat field.
    double unknown_traversal_cost = 0.5;
    // Extra cost A* pays per step for edges leaving a cell adjacent to an inflated obstacle.
    //
    // MUST satisfy  1 + unknown_traversal_cost + astar_clearance_cost  <  2.0, the weight on the
    // heuristic (new_h = dist * 2.0 in astar_sfc.cpp). A* prefers a step toward the goal only
    // when the step's cost multiplier is below that weight: stepping closer costs m and pays
    // back w, so f moves by (m - w). Once m >= w the search stops preferring the goal and floods
    // outward instead.
    //
    // This defaulted to 1.0 and was never passed from here, so it was always on. With
    // unknown_traversal_cost at its old 2.0 the multiplier reached 4.0 against a weight of 2.0.
    // In the sparse shakeout world the toll's six-neighbour probe almost never fired and it went
    // unnoticed; in a forest at 10-20% obstacle area with 0.79 m inflation nearly every free
    // cell touches an inflated obstacle, so it fired everywhere. Symptom: A* burned its whole
    // 4000-node budget to return a 15-cell (7 m) partial path against a 15 m horizon, which then
    // capped the trajectory at the allocator's 8-control-point floor -- 4 s for 7 m, ~1.75 m/s,
    // regardless of v_max. The vehicle also drifted backwards, because retreating scored better
    // than advancing.
    //
    // 0.0 disables it. Raise it only alongside the heuristic weight.
    double astar_clearance_cost = 0.0;
    // --- Proximity ("heat") field ---------------------------------------------------------
    // Width, m, of the halo beyond the inflated boundary over which a graded proximity cost
    // decays from 1 to 0. <= 0 disables the field and costs nothing to maintain.
    //
    // This is the soft half of obstacle avoidance. The inflated set is binary -- flyable or not
    // -- so a search using it alone has no reason to prefer the middle of a gap over grazing its
    // edge. Symptom: paths hug inflated boundaries and wedge in places a slightly wider route
    // would have cleared.
    double heat_radius = 1.0;
    // Weight on the heat term in A*'s step cost.
    //
    // BOUND BY THE SAME INVARIANT AS astar_clearance_cost, and it is the whole reason the earlier
    // attempt failed:
    //     1 + unknown_traversal_cost + astar_clearance_cost + heat_weight  <  astar_heuristic_weight
    // A* prefers a step toward the goal only while the step multiplier is under the heuristic
    // weight. Raise heat_weight and you must raise astar_heuristic_weight with it.
    double heat_weight = 0.6;
    // Weight on the A* heuristic. >1 is weighted A*: faster and greedier, at the cost of
    // optimality. It also sets the CEILING for every step-cost term above, so it is raised here
    // from the historical hardcoded 2.0 to leave room for the heat field.
    double astar_heuristic_weight = 3.0;
    // Penalty per step for traversing an INFLATED cell, instead of blocking it. 0 = hard block.
    //
    // SANDO's use_soft_cost_obstacles (obstacle_soft_cost 5.0 against a unit step). Used here
    // only as a SECOND pass, when the hard search fails to reach the goal: a soft search cannot
    // fail, so it recovers the route's shape, and truncating it at the first blocked cell gives
    // a reachable sub-goal far better than the dead-end stub a failed hard search returns.
    //
    // Must stay large enough that any free detour wins outright -- it should decide the route
    // only where no free route exists at all.
    double astar_soft_obstacle_cost = 5.0;
    // Smoothed path segments shorter than this many VOXELS are merged away, where line-of-sight
    // permits. 0 disables.
    //
    // The smoother's look-ahead loop re-anchors on the cell before each line-of-sight failure, so
    // around a tight corner it emits consecutive cells -- segments one voxel long. Such a segment
    // makes a degenerate corridor spine: it is diagonal (SAT corner term 0.354 m instead of
    // 0.25 m) and the corridor box still extends the full sfc_end_ext past it, so the box runs
    // ~8.6x the length of the segment it represents and is crushed by obstacles belonging to its
    // neighbours. Measured result: a lateral bound of exactly 0, a primal-infeasible QP that time
    // -stretching cannot fix, and a permanent stall.
    //
    // 2.0 catches the one-voxel diagonal (1.41 voxels) while leaving 3-voxel segments alone --
    // a measured 1.5 m corridor came back at +/-0.25 m, tight but solvable.
    double astar_min_segment_voxels = 2.0;

    /// Drop waypoints whose two legs point the same way. Redundant by construction, so unlike
    /// the length-based merge this needs no line-of-sight test and is never refused.
    /// MEASURED HARMFUL, so default OFF. Dropping waypoints whose two legs point the same way
    /// is provably safe and shortens the chain, but over 40 paired Monte-Carlo trials (identical
    /// forests and endpoints both arms) it took goal-reach 70.0% -> 57.5% and stalls
    /// 30.0% -> 42.5%. It was written to fix a corridor that collapsed to zero width and did not
    /// fix it. Kept switchable rather than deleted so the result can be re-checked.
    bool astar_collinear_merge = false;

    /// false hands obstacle_margin back to the corridor (wider boxes, bare-hull guarantee).
    /// See AsymmetricBoxBuilder. Default true = the safe policy.
    bool sfc_clamp_relief = true;

    /// Repair corridors whose narrowest lateral half-width falls below this, by merging the
    /// offending segment into a neighbour. 0 disables (default, pending measurement).
    ///
    /// A zero-width corridor is NOT unsafe -- the box is already inset by hull +
    /// obstacle_margin, so every point in it, wall included, keeps the CENTRE OF MASS clear.
    /// It is INFEASIBLE: every control point must lie inside the corridor (convex-hull
    /// property), so a collapsed axis pins them all to a plane and the spline cannot curve on
    /// it. Fine for straight-line travel; impossible the moment the pinned start velocity
    /// points off-axis or the next corridor has a different axis.
    /// MEASURED: 0.15 m. Over 25 paired Monte-Carlo forests at 20% density, v5:
    ///     off     68% goal-reach, 32% stall, 51.5% of corridors with a collapsed axis
    ///     0.05    76% / 24%
    ///     0.15    92% /  8%   <-- best, and p90 plan time FELL 31.7 -> 26.1 ms
    ///     0.30    84% / 16%   (over-merging makes longer segments that collapse in turn)
    double sfc_min_usable_width = 0.15;

    /// Cap on repair attempts per plan. Each one rebuilds the chain, so this bounds the cost.
    int sfc_max_repairs = 3;
    // --- Path hysteresis -------------------------------------------------------------------
    // Fractional discount on the A* step cost for cells on the PREVIOUS committed route.
    // 0 disables.
    //
    // At a junction where two or three routes cost within a few percent of each other, the
    // argmin flips on tiny changes in vehicle position or map contents. Replanning at 10 Hz then
    // re-decides from scratch every cycle and the vehicle dithers between branches instead of
    // committing -- observed in the forest roughly a third of the way along, oscillating until
    // lateral drift happened to break the tie.
    //
    // A discount makes continuity cheap without making it mandatory: at 0.20 an alternative
    // route must be more than 20% cheaper to win, which ties never are and a genuinely better
    // route usually is. It only ever LOWERS the step multiplier, so it cannot break the
    // guidance invariant that bounds unknown_traversal_cost and heat_weight.
    double path_hysteresis_bonus = 0.20;
    // How far either side of the previous route still counts as "on it", in voxels. 1 keeps the
    // tube one cell wide either side, so the discount survives the small lateral shifts that
    // come from replanning at a slightly different vehicle position.
    int path_hysteresis_radius_cells = 1;
    // How far ahead A* is allowed to search, in metres. A goal beyond this is projected
    // onto the horizon and searched to instead. Bounding the SEARCH (rather than searching
    // to a distant goal and then discarding most of the result via max_sfc_count) is what
    // keeps expansion cost flat while still handing the optimiser a full-length corridor
    // chain. <= 0 disables, restoring search-to-goal.
    /// Receding-horizon goal projection, metres. This is a SPEED parameter as much as a search
    /// bound: at unit knots the snap stencil forces >= degree*2 control points, so every plan is
    /// >= 4 s however short it is, and speed is then arc/4s. Raising the horizon puts more arc
    /// under that floor.
    ///
    /// MEASURED 2026-09-28, tools/monte_carlo --timed on the sim's own forest_medium files
    /// (realised speed = metres flown / seconds elapsed, velocity carried forward at 10 Hz):
    ///
    ///     horizon   realised   R_succ   collided   1-corridor   first-attempt
    ///       10 m      1.56        80%       0%         68.6%        84.8%
    ///       12 m      2.40        80%       0%         51.4%        88.3%   <-- use this
    ///       15 m      2.56        60%      20%         44.4%        89.1%
    ///       20 m      2.59        30%      50%         23.5%        81.9%
    ///
    /// READ THE SAFETY COLUMNS, NOT THE SPEED COLUMN. Speed rises monotonically all the way to
    /// 20 m, which is why this looks like it wants to be large. It does not. Past 12 m you buy
    /// 7% speed for 20 points of R_succ, then 1% more for another 30 -- and 20 m exceeds
    /// map_window_radius (15 m), so the search projects into volume the map has already pruned
    /// and routes optimistically through space that was never measured. 12 m is +54% over 10 m
    /// at IDENTICAL success and zero collisions, with first-attempt solves IMPROVING.
    ///
    /// CONFIRMED IN FLIGHT (medium s1, v_max 5): v_avg 2.23 -> 2.85 m/s (+28%), T_trav
    /// 50.88 -> 40.02 s (-21%) over the same 114 m route, FAIL -> PASS, and worst-cycle planning
    /// time FELL 27.45 -> 25.46 ms. Median control points stayed at 8 -- still the floor, still
    /// 4 s per plan -- while arc per plan grew 8.3 -> 11.4 m. That is the whole mechanism: the
    /// floor is irreducible, so buy speed by putting more distance under it.
    ///
    /// Any increase past map_window_radius is planning into pruned space; raise that first.
    double planning_horizon = 12.0;

    /// Reinterpret the unit-spaced knot vector over the KINEMATIC time budget instead of
    /// 1 second per control point, decoupling trajectory duration from control-point count.
    ///
    /// Default OFF. This was implemented and reverted once before (see the measured record in
    /// trajectory_planner.cpp, "WHY THIS TRAJECTORY IS SLOWER THAN THE KINEMATICS ALLOW"):
    /// it fixes single-corridor plans cleanly but made velocity binding everywhere on a
    /// multi-corridor chain, which went 1 attempt / 1.3 ms -> 6 attempts / 63 ms. It is behind a
    /// flag so the regime can be measured at scale rather than on one logged chain.
    ///
    /// Unlike that attempt, the snap Hessian is deliberately NOT scaled by h^-7. The QP's linear
    /// term is identically zero, so a positive scalar on the objective cannot move the argmin --
    /// leaving the unit-knot Hessian is exact, and avoids handing OSQP a 1e5-range rescale.
    bool knot_scale = false;
    // Deterministic cap on A* node expansions. This, not a wall-clock timer, is what bounds
    // the search -- a time-based budget makes the planner produce different trajectories on
    // identical inputs depending on CPU load. ~6000 nodes is roughly 10 ms on the reference
    // machine. <= 0 disables (search to completion).
    // 6000 was buying nothing measurable: over 204 goals, 3000 / 6000 / 12000 all solved 196
    // and reached the goal 152 times; only 1500 lost any (150). Meanwhile an exhausted search
    // is the single largest front-end cost -- a budget-exhausting expansion measured 11-15 ms
    // on a small map and ~95 ms in flight, where the accumulated map makes each expansion far
    // more cache-hostile. Halving the budget halves that worst case for free.
    int astar_max_nodes = 3000;
    // Wall-clock backstop on the A* search, ms. This was hardcoded to 100.0 -- the ENTIRE replan
    // period -- so it could never protect the 10 Hz loop it was supposed to guard: a single
    // search was permitted to consume the whole budget before it fired. Node count remains the
    // primary, deterministic limit (see astar_sfc.cpp); this only bounds the pathological case
    // where per-node cost has grown with map density. Keep it well inside the replan period.
    double astar_max_time_ms = 25.0;
    // Multiplier on the initial flight-time budget.
    //
    // The budget max(L/v_max, 2*sqrt(L/a_max)) is the time for an ideal straight-line
    // accelerate-to-cruise-decelerate profile. A real trajectory also has to curve, and the
    // terminal condition pins it to a full stop -- so the ideal figure is systematically a
    // little short, the first solve comes back infeasible, and the retry loop lengthens it.
    // Those retries are pure cost: they add solve time AND permanently inflate the duration,
    // because control-point count IS flight time. Budgeting slightly generously up front
    // trades a marginally slower ideal for far fewer retries.
    // 1.20 chosen from a 12-scenario sweep: it doubles the first-attempt success rate
    // (30% -> 60%) and cuts mean solve time ~32% (8.5 -> 5.8 ms) without losing any
    // scenario that 1.00 could solve. Higher values (1.30+) push first-try to 78% and are
    // faster still, but started failing a climb case outright -- a larger initial problem
    // interacts with the retry loop's stall detection. Retuning against your own
    // environment is worthwhile; the sweep harness is tools/ + the bench in the notes.
    double time_budget_factor = 1.20;
    // Fraction of v_max demanded as the TERMINAL speed of a horizon-truncated plan.
    //
    // The end state is a hard equality while the MINVO kinodynamic block is an inequality at
    // v_max, so a terminal speed close to v_max leaves almost no headroom for the MINVO bound's
    // conservatism -- it bounds the control polygon, not the curve, so it reads high. At 0.9 the
    // margin is 10%, and a single straight corridor was measured infeasible with the boundary and
    // velocity blocks fighting while the corridor block was nearly clean.
    //
    // Note this cannot be probed by sweeping v_max: the terminal speed is a FRACTION of it, so
    // both sides of the constraint move together and the ratio is unchanged. It has to be varied
    // on its own, which is why it is a parameter rather than a constant.
    double terminal_speed_fraction = 0.90;
    // Wall-clock ceiling, in ms, on the whole OSQP retry cycle for one replan (summed over
    // attempts). A doomed solve burns a full max_iter per attempt before OSQP can certify
    // infeasibility, so an unlucky replan could run 4 attempts and ~270 ms against a 100 ms
    // period. Missing a replan is cheap -- the committed trajectory keeps flying and the next
    // cycle retries against a fresher map -- but overrunning the period starves the 50 Hz
    // dispatch loop. 0 disables the bound.
    double max_solve_time_ms = 60.0;
    // Worst constraint violation, in the constraints' own units (m, m/s, m/s^2), at which a
    // NON-CONVERGED OSQP iterate is still accepted as flyable. OSQP needs both primal and dual
    // residuals inside tolerance, so it can stop at max_iter holding a point whose constraints
    // are satisfied to ~1e-4 purely because optimality has not settled -- discarding that costs
    // a whole replan and can leave the vehicle hovering. Never applied to a PRIMAL_INFEASIBLE
    // verdict, whose iterate is a diverging ray rather than a nearly feasible point.
    // 1e-2 is negligible against the margins actually in play: corridors already carry the
    // drone-radius inflation (~0.49 m) and v_max is 3 m/s. 0 disables.
    double qp_accept_violation = 1e-2;
    // The MINVO kinodynamic constraints are applied as a PER-AXIS box: |v_x|,|v_y|,|v_z| <= v_max.
    // That does NOT bound the speed -- it permits ||v|| up to v_max*sqrt(3) (5.20 m/s at
    // v_max = 3), and likewise ||a|| up to a_max*sqrt(3). Anything claiming feasibility with
    // respect to v_max/a_max as SPEED is therefore overclaiming.
    //
    // Set true to scale the box by 1/sqrt(3), i.e. inscribe it in the sphere, which makes
    // ||v|| <= v_max and ||a|| <= a_max true guarantees. It is conservative: the box is the
    // largest one that fits inside the sphere, so axis-aligned motion loses the most. The time
    // allocator is left alone -- it already reasons in norms along a 1-D speed profile, so only
    // the QP box is inconsistent with it.
    bool enforce_norm_limits = false;
};

/**
 * Flight time the kinematics call for over `path_length`, in seconds.
 *
 * max(L/v_max, 2*sqrt(L/a_max)) is the ideal accelerate-cruise-decelerate duration; the factor
 * pads it (see FrontEndConfig::time_budget_factor). Shared between the control-point allocator
 * and the knot time-scaling in TrajectoryPlanner, which must agree or the trajectory is either
 * stretched or over-constrained.
 */
inline double kinematic_time_budget(double path_length, double v_max, double a_max, double factor)
{
    const double v = std::max(v_max, 1e-3);
    const double a = std::max(a_max, 1e-3);
    return std::max(factor, 1e-3)
         * std::max(path_length / v, 2.0 * std::sqrt(std::max(path_length, 0.0) / a));
}

// ==========================================
// FrontEnd Result
// ==========================================
struct FrontEndResult {
    std::vector<sfc::SFCResult> corridors;
    std::vector<ConstraintPool> allocation_pools;    // multi-rotor
    std::vector<int> allocation_list;                // fixed-wing (num_pts_list)
    sfc::StandaloneWaypointsSFC waypoints_smooth;
    sfc::StandaloneWaypointsSFC waypoints_not_smooth;
    bool reached_goal;
    Eigen::Vector3d actual_end_pos;
};

// ==========================================
// FrontEndSFC (Port of front_end.py)
// ==========================================
class FrontEndSFC {
public:
    explicit FrontEndSFC(const FrontEndConfig& config,
                         std::shared_ptr<mapping::SparseVoxelGrid> grid);

    /**
     * Main entry point: runs A* → smoothing → SFC generation → control point allocation.
     */
    FrontEndResult get_corridors_astar(const Eigen::Vector3d& start_pos,
                                       const Eigen::Vector3d& end_pos);

    /**
     * Compiles the unified constraint pools into A_sfc / b_sfc matrices for OSQP.
     */
    static void compile_system_constraints(
        const std::vector<sfc::SFCResult>& corridors,
        const std::vector<ConstraintPool>& pools,
        Eigen::MatrixXd& A_sfc_out, Eigen::VectorXd& b_sfc_out, int& total_points_out);

    const FrontEndConfig& config() const { return config_; }

    /**
     * Build the control-point allocation for a corridor chain.
     *
     * Normally done inside get_corridors_astar. Exposed so a failing case can be rebuilt offline
     * from logged corridor geometry and replayed through the back end -- the search and the map
     * are not needed to reproduce a QP failure, and requiring them makes each hypothesis cost a
     * full flight instead of a millisecond.
     */
    std::vector<ConstraintPool> allocate_for(const std::vector<sfc::SFCResult>& corridors) const {
        return allocate_dynamic_control_points(corridors);
    }

    /**
     * Supply the previously committed route, in WORLD coordinates, for path hysteresis.
     *
     * Rasterised into voxels here rather than in the caller, because only the front end knows
     * the grid resolution. An empty vector clears it. Must be called on every replan: the
     * planner is rebuilt each cycle, so nothing persists on its own.
     */
    void set_previous_path(const std::vector<Eigen::Vector3d>& world_points);

private:
    /**
     * Multi-rotor: Sequential pairwise allocator with bridge/exclusive logic.
     */
    std::vector<ConstraintPool> allocate_dynamic_control_points(
        const std::vector<sfc::SFCResult>& corridors) const;

    /**
     * Fixed-wing: Apex-centric allocator with momentum shedding.
     */
    std::vector<int> allocate_dynamic_control_points_fixed_wing(
        const std::vector<sfc::SFCResult>& corridors) const;

    FrontEndConfig config_;

    // Map data
    std::shared_ptr<mapping::SparseVoxelGrid> discrete_grid_;
    std::unique_ptr<mapping::RingBufferGrid> ring_buffer_;   // unused; see the constructor
    /// Absolute vertical geofence, in voxel indices. Altitude is a real world limit, so unlike
    /// the horizontal bounds these do NOT follow the vehicle.
    int astar_z_min_ = 0;
    int astar_z_max_ = 0;

    // SFC generators
    sfc::AsymmetricSFCManager sfc_manager_;
    sfc::StaticSFCManager static_sfc_manager_;

    // A* planner (uses astar_sfc.hpp's existing VoxelGridData + planner)
    // astar_grid_ must outlive astar_planner_, which stores a reference to it.
    /// Previous route rasterised to cells, borrowed by astar_planner_ during the search.
    path_finding::VoxelSet prev_path_cells_;
    path_finding::VoxelGridData astar_grid_;
    std::unique_ptr<path_finding::AStarSFCPlanner> astar_planner_;
};

} // namespace trajectory_planner
