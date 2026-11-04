#include "trajectory_planner.hpp"
#include <iostream>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <cmath>

namespace trajectory_planner {

// ==========================================
// Helper: Sparse Kronecker Product  kron(A, I_n)
// Specialized for the common case where B = Identity(n).
// Result has dims (A.rows()*n, A.cols()*n).
// ==========================================
static Eigen::SparseMatrix<double> kron_with_identity(
    const Eigen::SparseMatrix<double>& A, int n)
{
    int rows_out = static_cast<int>(A.rows()) * n;
    int cols_out = static_cast<int>(A.cols()) * n;

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(A.nonZeros() * n);

    for (int k = 0; k < A.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            int row_a = static_cast<int>(it.row());
            int col_a = static_cast<int>(it.col());
            double val = it.value();
            for (int d = 0; d < n; ++d) {
                triplets.emplace_back(row_a * n + d, col_a * n + d, val);
            }
        }
    }

    Eigen::SparseMatrix<double> result(rows_out, cols_out);
    result.setFromTriplets(triplets.begin(), triplets.end());
    return result;
}

// ==========================================
// Helper: Dense → Sparse kron with identity
// ==========================================
static Eigen::SparseMatrix<double> kron_dense_with_identity(
    const Eigen::MatrixXd& A, int n)
{
    Eigen::SparseMatrix<double> A_sparse = A.sparseView();
    return kron_with_identity(A_sparse, n);
}

// ==========================================
// Constructor
// ==========================================
TrajectoryPlanner::TrajectoryPlanner(
    const FrontEndConfig& config,
    std::shared_ptr<mapping::SparseVoxelGrid> grid,
    double v_max, double a_max)
    : config_(config), v_max_(v_max), a_max_(a_max) {
    // Keep the front-end allocator's kinematic limits in sync with the limits the
    // MINVO constraints are actually built against, otherwise the allocator sizes
    // the trajectory duration for one speed and the QP enforces another.
    config_.v_max = v_max;
    max_solve_time_ms_ = config_.max_solve_time_ms;
    config_.a_max = a_max;
    front_end_ = std::make_unique<FrontEndSFC>(config_, std::move(grid));
}

// ==========================================
// plan_mission
// ==========================================
PlanningResult TrajectoryPlanner::plan_mission(
    const Eigen::Vector3d& start_pos, const Eigen::Vector3d& end_pos,
    const Eigen::Vector3d& start_vel, const Eigen::Vector3d& start_acc) {

    std::cout << "--- Starting Trajectory Planning Mission ---" << std::endl;
    auto mission_start = std::chrono::high_resolution_clock::now();

    // Reported on PlanningResult; see the braking bound below for why these two are worth
    // having. Function-scoped because the bound is computed inside a nested block.
    double measured_d_free = 0.0, measured_v_end = 0.0;

    // =========================================================
    // PHASE 1: Front-End (A* + SFC Generation)
    // =========================================================
    std::cout << "[Phase 1] Generating Safe Flight Corridors..." << std::endl;
    auto sfc_start = std::chrono::high_resolution_clock::now();
    // An override replays a logged corridor chain instead of searching; see
    // set_front_end_override(). Inert unless explicitly set.
    auto fe_result = fe_override_ ? *fe_override_
                                  : front_end_->get_corridors_astar(start_pos, end_pos);
    auto sfc_end = std::chrono::high_resolution_clock::now();
    double sfc_ms = std::chrono::duration<double, std::milli>(sfc_end - sfc_start).count();

    if (fe_result.corridors.empty()) {
        std::cout << "[Error] No valid path found." << std::endl;
        return {Eigen::VectorXd(), Eigen::VectorXd(), fe_result.corridors, config_.degree, 0, false, false, sfc_ms, 0, 0, sfc_ms, 0.0, 0};
    }

    // =========================================================
    // Build SE Boundary Matrix (3 x 6)
    // S = [start_pos, start_vel, start_acc]  (3 x 3)
    // =========================================================
    Eigen::Vector3d actual_end_pos = fe_result.actual_end_pos;
    
    Eigen::MatrixXd S(3, 3);
    S.col(0) = start_pos;
    S.col(1) = start_vel;
    S.col(2) = start_acc;

    Eigen::MatrixXd SE(3, 6);
    if (config_.spline_type == "natural") {
        // E_standard = [end_pos, zeros, zeros]
        Eigen::MatrixXd E(3, 3);
        E.col(0) = actual_end_pos;
        E.col(1) = Eigen::Vector3d::Zero();
        E.col(2) = Eigen::Vector3d::Zero();
        SE << S, E;
    } else {
        // Clamped: E = [end_accel, end_velocity, end_position].
        //
        // TERMINAL VELOCITY, AND WHY IT IS BOUNDED THE WAY IT IS
        // -----------------------------------------------------
        // Pinning a full stop at every horizon is safe but slow: with a goal beyond
        // planning_horizon the vehicle decelerates to rest on every replan and never builds
        // speed. It is also the dominant cause of first-attempt infeasibility -- the
        // boundary block is implicated in essentially every failed solve.
        //
        // Carrying speed through the horizon is only safe if a full stop remains possible
        // inside space we have actually MEASURED. Otherwise a replan that fails to arrive
        // leaves the vehicle moving toward the end of its committed trajectory with nothing
        // planned beyond it, in space that was never observed. So:
        //
        //     v_end <= sqrt(2 * a_max * d_free_ahead)
        //
        // d_free_ahead is the last corridor's forward extent past the trajectory endpoint.
        // Corridors are obstacle-free AND stop at the FREE/UNKNOWN frontier, so that
        // distance is measured-empty by construction. The bound therefore self-limits: with
        // an obstacle or unexplored space close ahead it collapses to zero and we recover
        // the old stop-at-horizon behaviour exactly when it is needed.
        //
        // The guarantee is only real if the stop is actually executable, so the consumer
        // must be able to brake within d_free_ahead if replanning stalls -- see the braking
        // tail in minimum_snap_manager.cpp. Changing one without the other breaks the
        // argument.
        Eigen::Vector3d end_vel = Eigen::Vector3d::Zero();
        if (!fe_result.reached_goal && !fe_result.corridors.empty()) {
            const auto& last = fe_result.corridors.back();
            const double d_free_ahead =
                std::max(0.0, last.bounds(0) - last.getDistancePrimaryToSecondary());

            // Keep margin: spend only part of the measured-free run-out on braking, so the
            // stop completes with room rather than exactly at the frontier.
            constexpr double kBrakeMargin = 0.6;
            const double v_safe = std::sqrt(2.0 * a_max_ * kBrakeMargin * d_free_ahead);

            // Back the terminal speed off the limit instead of pinning it AT the limit.
            //
            // end_vel is a hard EQUALITY in the boundary block, while the MINVO block imposes the
            // per-axis INEQUALITY |v_i| <= v_max. Setting the magnitude to exactly v_max makes the
            // two coincide whenever the corridor axis is near-axis-aligned -- a (0,0)->(105,0)
            // route gives ux ~ (1,0,0), so the pinned end velocity lands exactly on the north
            // velocity face. The QP is then feasible with ZERO slack, which OSQP converges to only
            // slowly: it cannot certify a point on the boundary, so it exhausts max_iter and the
            // retry loop stretches time on a problem that was never over-constrained.
            //
            // Observed in a forest benchmark: five consecutive attempts, every one ending in
            // "maximum iterations reached", with the VELOCITY block the only one violated
            // (2 rows of 48) and the primal residual falling monotonically 0.0709 -> 0.0496 ->
            // 0.0436 -> 0.0254 -> 0.0147 without ever clearing tolerance. Corridors and
            // acceleration were clean throughout -- the geometry was fine, the terminal condition
            // was simply sitting on a constraint face.
            //
            // 10% of headroom costs almost nothing in travel time (the NEXT replan re-accelerates
            // from this state anyway) and gives the solver an interior point to converge to.
            const double v_end_mag =
                std::min(v_safe, config_.terminal_speed_fraction * v_max_);
            if (v_end_mag > 1e-3) end_vel = last.ux * v_end_mag;
            measured_d_free = d_free_ahead;
        } else if (!fe_result.corridors.empty()) {
            const auto& last = fe_result.corridors.back();
            measured_d_free =
                std::max(0.0, last.bounds(0) - last.getDistancePrimaryToSecondary());
        }
        measured_v_end = end_vel.norm();

        Eigen::MatrixXd E(3, 3);
        E.col(0) = Eigen::Vector3d::Zero();   // terminal acceleration stays zero
        E.col(1) = end_vel;
        E.col(2) = actual_end_pos;
        SE << S, E;
    }

    // =========================================================
    // Time-Stretching Retry Loop
    // =========================================================
    // Knot scale, fixed on the FIRST attempt. The retry loop buys flight time by adding a
    // control point per pool; if h were recomputed each attempt it would renormalise that time
    // straight back out and every retry would be a no-op.
    double knot_h = 1.0;
    bool knot_h_fixed = false;

    int max_stretches = 5;
    int stretch_count = 0;
    Eigen::VectorXd optimal_cp;
    Eigen::VectorXd current_knots;
    // Accumulated across ALL attempts. These used to be plain assignments, so a run that
    // retried reported only the LAST attempt's solve time while TOTAL covered every
    // attempt -- the difference silently piled up in "Matrix & System Overhead".
    double opt_ms = 0.0;
    double matrix_ms = 0.0;
    int attempts = 0;
    double attempt_prim_res = 0.0;
    QpFailureInfo qp_failure;
    double prev_prim_res = std::numeric_limits<double>::infinity();
    bool solve_success = false;
    int total_control_points = 0;

    // =========================================================
    // SOLVE IN A LOCAL FRAME
    // =========================================================
    // The QP is written in WORLD coordinates, so every corridor half-space carries an offset the
    // size of the vehicle's distance from the origin while the quantity that actually matters --
    // the corridor half-width -- is under a metre. Past roughly 45 m the ratio is bad enough that
    // OSQP declares primal infeasibility on a problem that is comfortably feasible.
    //
    // Measured with tools/repro_forest_stall: one identical corridor (real bounds, real axis,
    // 0.84 m half-width) solves at every origin up to 46 m, needs a retry at 47 m, and fails at
    // 48 m and beyond. Nothing about the geometry changes -- only where it sits.
    //
    // That is the whole "stalls halfway" behaviour: the benchmark is 105 m long, so the vehicle
    // crosses the threshold near its midpoint and every replan from there on fails, whatever the
    // corridors look like. It also explains why widening corridors, lengthening the time budget,
    // lowering the terminal velocity and loosening the acceptance tolerance all changed nothing.
    //
    // Shifting the problem so the start is the origin removes the offset entirely: the corridor
    // constraint A*p <= b becomes A*p' <= b - A*origin, and the boundary conditions shift with it.
    // Velocities and accelerations are translation-invariant and must NOT be touched. Control
    // points come back into world coordinates after the solve.
    const Eigen::Vector3d frame_origin = start_pos;

    SE.col(0) -= frame_origin;                                     // start position
    if (config_.spline_type == "natural") {
        SE.col(3) -= frame_origin;                                 // E = [end_pos, 0, 0]
    } else {
        SE.col(5) -= frame_origin;                                 // E = [end_acc, end_vel, end_pos]
    }

    // Make mutable copies for stretching
    auto corridors = fe_result.corridors;
    for (auto& c : corridors) {
        // A*(p' + origin) <= b  <=>  A*p' <= b - A*origin
        c.b_vec -= c.A_mat * frame_origin;
        c.primaryPosition -= frame_origin;
        c.secondaryPosition -= frame_origin;
    }
    auto allocation_pools = fe_result.allocation_pools;
    auto allocation_list = fe_result.allocation_list;

    while (stretch_count <= max_stretches) {
        std::cout << "\n--- Optimization Attempt " << (stretch_count + 1) << " ---" << std::endl;

        // Time-stretching fallback: add control points on retry
        if (stretch_count > 0) {
            if (config_.aircraft_type == "fixed-wing") {
                for (auto& pts : allocation_list) pts += 1;
            } else {
                for (auto& pool : allocation_pools) pool.pts += 1;
            }
        }

        // =========================================================
        // PHASE 2: Build Constraint Matrices
        // =========================================================
        attempts++;
        // Sentinel: only a real solve overwrites this. Distinguishes "the solver ran and
        // stalled" from "we never got as far as the solver".
        attempt_prim_res = std::numeric_limits<double>::quiet_NaN();
        auto matrix_start = std::chrono::high_resolution_clock::now();
        Eigen::MatrixXd A_sfc;
        Eigen::VectorXd b_sfc;

        if (config_.aircraft_type == "fixed-wing") {
            total_control_points = 0;
            for (int pts : allocation_list) total_control_points += pts;
            std::tie(A_sfc, b_sfc) = build_overlap_constraints_fixed_wing(
                corridors, allocation_list, total_control_points);
        } else {
            // Use front_end compile_system_constraints to get raw A_sfc/b_sfc
            // then apply map boundary injection via build_overlap_constraints
            std::tie(A_sfc, b_sfc) = build_overlap_constraints(
                corridors, allocation_pools, 0 /* computed internally */);
            // Recompute total from pools
            total_control_points = 0;
            for (const auto& pool : allocation_pools) total_control_points += pool.pts;
        }

        auto matrix_end = std::chrono::high_resolution_clock::now();
        matrix_ms += std::chrono::duration<double, std::milli>(matrix_end - matrix_start).count();

        std::cout << "[Phase 2] Translating " << corridors.size()
                  << " SFCs for " << total_control_points << " points..." << std::endl;

        // =========================================================
        // PHASE 3: Build Optimizer Matrices + Call OSQP
        // =========================================================
        auto opt_start = std::chrono::high_resolution_clock::now();

        int num_segments = total_control_points - config_.degree;

        if (config_.knot_scale && !knot_h_fixed && num_segments > 0) {
            double total_len = 0.0;
            for (const auto& c : corridors) total_len += c.getDistancePrimaryToSecondary();
            const double T_kin = kinematic_time_budget(
                total_len, v_max_, a_max_, config_.time_budget_factor);
            // Only ever COMPRESS. h > 1 would slow the trajectory below today's behaviour, which
            // is never the intent here, and the duration floor means it binds in one direction.
            if (T_kin > 0.0)
                knot_h = std::min(1.0, T_kin / static_cast<double>(num_segments));
            knot_h_fixed = true;
            std::cout << "[knot_scale] h = " << knot_h << "  (" << num_segments
                      << " unit segments -> " << (num_segments * knot_h) << " s for "
                      << total_len << " m)" << std::endl;
        }
        const double h = knot_h;

        // --- WHY THIS TRAJECTORY IS SLOWER THAN THE KINEMATICS ALLOW (measured, unfixed) -------
        // The knot vector is unit-spaced, so duration in seconds EQUALS num_segments. That welds
        // flight time to control-point COUNT, and the count is set by corridor geometry rather
        // than by kinematics: every junction contributes a bridge pool of `degree` (4) points.
        // Those 4 are geometrically required -- by the convex-hull property, `degree` points must
        // lie in BOTH corridors for the curve to cross without leaving either -- so the count
        // cannot simply be cut.
        //
        // Measured in the forest: 8 corridors over a 15 m horizon allocated 36 control points, 28
        // of them bridges, giving a 32 s trajectory for 15 m -- 0.47 m/s against a 5 m/s limit,
        // where the kinematics asked for 3.6 s. Roughly 9x too slow. Replanning at 10 Hz hides it
        // (only each trajectory's fast opening is ever flown); anything that holds a plan longer
        // exposes it, which is why the anti-stutter gate cost 31% travel time and path hysteresis
        // at 0.30 collapsed average speed to 1.02 m/s.
        //
        // Rescaling the knots to the kinematic budget (with MINVO bounds scaled by h and h^2) WAS
        // implemented and reverted. It fixes the single-corridor case cleanly -- 2.71 s, 3.33 m/s,
        // first-attempt solve in 0.54 ms at time_budget_factor 1.5 -- but compressing a
        // multi-corridor chain makes velocity binding everywhere and the QP falls apart: the
        // logged 3-corridor chain went from 1 attempt / 1.3 ms to 6 attempts / 63 ms, relaxing
        // all the way back to unit knots anyway. Since flight is almost always 4-12 corridors,
        // that is a net loss. A real fix needs the QP to handle a velocity-binding problem at
        // scale -- non-uniform knots per segment, or fewer/longer corridors -- not a global
        // rescale. See tools/repro_forest_stall.cpp, which reproduces both regimes offline.
        //
        // RE-TESTED 2026-09-28 at scale, behind FrontEndConfig::knot_scale (still default OFF).
        // The retest dropped the h^-7 Hessian scaling, on the correct observation that q_osqp is
        // identically zero so a positive scalar on the objective cannot move the argmin. That
        // changes nothing, and could not have: feasibility depends only on (A, l, u), and h < 1
        // TIGHTENS the velocity rows (h) and acceleration rows (h^2). The verdict stands, now on
        // 1000+ plans instead of one logged chain:
        //
        //                        synthetic 20% density      the sim's medium forests
        //                        OFF  ->  ON                OFF  ->  ON
        //   R_succ                9.0% ->  1.0%             40.0% ->  0.0%
        //   stall                 6.0% -> 64.0%              0.0% -> 80.0%
        //   first-attempt        65.9% -> 39.0%             77.8% -> 67.4%
        //   OSQP p90 (ms)         4.80 -> 19.48              3.36 ->  9.20
        //   commanded speed     1.95  ->  1.80 m/s          2.47  ->  2.41 m/s
        //
        // Note the last row: speed did not improve even in the 50%-single-corridor forest regime,
        // the one this is supposed to fix. Compression tightens the velocity rows -> first-attempt
        // solves fail -> the retry loop adds a control point per pool -> unit knots ARE flight
        // time, so the added points hand back exactly the duration h removed. It is self-cancelling
        // on speed and a large net loss on reliability. Do not enable it in flight.

        // Row extents of each stacked constraint block, so a failure can be attributed to a
        // named requirement instead of just a residual. Declared out here to survive into
        // the catch.
        int rows_eq = 0, rows_sfc = 0, rows_vel = 0, rows_accel = 0;
        Eigen::VectorXd l_all, u_all;
        Eigen::SparseMatrix<double> A_all;

        try {
            Eigen::MatrixXd W, B_combined, D_vel, D_accel;
            Eigen::MatrixXd SE_qp;

            // Boundary state, expressed in the SCALED parameter. dP/dt = (1/h) dp/ds, so pinning a
            // real velocity v needs dp/ds = h*v, and an acceleration needs h^2*a. Positions are
            // unaffected. SE = [start_pos, start_vel, start_acc, end_acc, end_vel, end_pos].
            Eigen::MatrixXd SE_h = SE;
            if (h != 1.0) {
                SE_h.col(1) *= h;      SE_h.col(4) *= h;
                SE_h.col(2) *= h * h;  SE_h.col(3) *= h * h;
            }

            if (config_.spline_type == "natural") {
                MinSnapEvalNatural optimizer(num_segments, config_.degree);
                D_vel = optimizer.getFastCascadedDMatrix(num_segments, config_.degree, 1).transpose();
                D_accel = optimizer.getFastCascadedDMatrix(num_segments, config_.degree, 2).transpose();
                W = optimizer.getW();
                B_combined = optimizer.getBCombined();
                SE_qp = SE_h;
                current_knots = optimizer.getKnots();
            } else {
                MinSnapEvalClamped optimizer(num_segments, config_.degree);
                D_vel = optimizer.get_fast_cascaded_D_matrix(num_segments, config_.degree, 1);
                D_accel = optimizer.get_fast_cascaded_D_matrix(num_segments, config_.degree, 2);
                Eigen::MatrixXd B_d3 = optimizer.get_B_d3_matrix(config_.degree);
                W = optimizer.get_W();
                B_combined = optimizer.get_B_combined();
                SE_qp = SE_h * B_d3;
                current_knots = optimizer.get_knots();
            }

            if (h != 1.0) current_knots *= h;   // de Boor and the 50 Hz dispatch see real time

            int N = static_cast<int>(W.rows()); // == total_control_points
            Eigen::MatrixXd A_eq = B_combined.transpose(); // N x 6

            // ==========================================
            // 1. OBJECTIVE MATRICES (P and q)
            // ==========================================
            // P_3D = 2 * kron(W, I_3) — expand 1D penalty to 3D
            Eigen::SparseMatrix<double> P_osqp_full = 2.0 * kron_dense_with_identity(W, 3);
            Eigen::SparseMatrix<double> P_osqp = P_osqp_full.triangularView<Eigen::Upper>();
            Eigen::VectorXd q_osqp = Eigen::VectorXd::Zero(3 * N);

            // ==========================================
            // 2. EQUALITY CONSTRAINTS (Boundary Conditions)
            // ==========================================
            // A_eq_3D = kron(A_eq, I_3)
            Eigen::SparseMatrix<double> A_eq_3D = kron_dense_with_identity(A_eq, 3);

            // Flatten SE: SE is (3, 6), SE.T is (6, 3), flatten row-major = interleave
            // Python: SE.T.flatten() means column-major flatten of SE.T
            // SE.T shape (6, 3): row 0 = [SE(0,0), SE(1,0), SE(2,0)] = [sx, sy, sz]
            // flatten() = [SE(0,0), SE(1,0), SE(2,0), SE(0,1), SE(1,1), SE(2,1), ...]
            Eigen::MatrixXd SE_qp_T = SE_qp.transpose(); // (6, 3)
            Eigen::VectorXd l_eq(18), u_eq(18);
            for (int r = 0; r < 6; ++r) {
                for (int c = 0; c < 3; ++c) {
                    l_eq(r * 3 + c) = SE_qp_T(r, c);
                    u_eq(r * 3 + c) = SE_qp_T(r, c);
                }
            }

            // ==========================================
            // 3. GEOMETRIC CONSTRAINTS (Safe Flight Corridors)
            // ==========================================
            Eigen::SparseMatrix<double> A_sfc_sparse = A_sfc.sparseView();
            // Must be OSQP's finite sentinel, NOT IEEE -inf (see kQpInfinity docs).
            Eigen::VectorXd l_sfc = Eigen::VectorXd::Constant(b_sfc.size(), -kQpInfinity);
            Eigen::VectorXd u_sfc = b_sfc;

            // ==========================================
            // 4. KINODYNAMIC CONSTRAINTS (MINVO)
            // ==========================================
            std::cout << "[Phase 3] Building MINVO kinodynamic constraints..." << std::endl;

            Eigen::SparseMatrix<double> A_vel_1D = minvo::build_minvo_kinodynamic_matrix(
                D_vel, config_.degree - 1, config_.spline_type);
            Eigen::SparseMatrix<double> A_accel_1D = minvo::build_minvo_kinodynamic_matrix(
                D_accel, config_.degree - 2, config_.spline_type);

            Eigen::SparseMatrix<double> A_vel_3D = kron_with_identity(A_vel_1D, 3);
            Eigen::SparseMatrix<double> A_accel_3D = kron_with_identity(A_accel_1D, 3);

            // Per-axis box. See FrontEndConfig::enforce_norm_limits: as written this bounds each
            // AXIS, so it permits ||v|| up to v_max*sqrt(3), not v_max.
            const double axis_scale =
                config_.enforce_norm_limits ? (1.0 / std::sqrt(3.0)) : 1.0;
            // |dP/dt| <= v_axis becomes |dp/ds| <= h*v_axis; acceleration picks up h^2. This is
            // the block the earlier attempt recorded as "velocity binding everywhere": h < 1
            // tightens these rows, it does not relax them.
            const double v_axis = v_max_ * axis_scale * h;
            const double a_axis = a_max_ * axis_scale * h * h;
            Eigen::VectorXd l_vel = Eigen::VectorXd::Constant(A_vel_3D.rows(), -v_axis);
            Eigen::VectorXd u_vel = Eigen::VectorXd::Constant(A_vel_3D.rows(), v_axis);
            Eigen::VectorXd l_accel = Eigen::VectorXd::Constant(A_accel_3D.rows(), -a_axis);
            Eigen::VectorXd u_accel = Eigen::VectorXd::Constant(A_accel_3D.rows(), a_axis);

            // ==========================================
            // 5. ASSEMBLE THE MASTER MATRICES
            // ==========================================
            // A_osqp = vstack([A_eq_3D, A_sfc_sparse, A_vel_3D, A_accel_3D])
            int total_rows = static_cast<int>(A_eq_3D.rows() + A_sfc_sparse.rows() + A_vel_3D.rows() + A_accel_3D.rows());
            int total_cols = 3 * N;

            std::vector<Eigen::Triplet<double>> a_triplets;
            a_triplets.reserve(A_eq_3D.nonZeros() + A_sfc_sparse.nonZeros() + A_vel_3D.nonZeros() + A_accel_3D.nonZeros());

            int row_offset = 0;

            // Block 1: A_eq
            for (int k = 0; k < A_eq_3D.outerSize(); ++k)
                for (Eigen::SparseMatrix<double>::InnerIterator it(A_eq_3D, k); it; ++it)
                    a_triplets.emplace_back(static_cast<int>(it.row()) + row_offset, static_cast<int>(it.col()), it.value());
            row_offset += static_cast<int>(A_eq_3D.rows());

            // Block 2: A_sfc
            for (int k = 0; k < A_sfc_sparse.outerSize(); ++k)
                for (Eigen::SparseMatrix<double>::InnerIterator it(A_sfc_sparse, k); it; ++it)
                    a_triplets.emplace_back(static_cast<int>(it.row()) + row_offset, static_cast<int>(it.col()), it.value());
            row_offset += static_cast<int>(A_sfc_sparse.rows());

            // Block 3: A_vel
            for (int k = 0; k < A_vel_3D.outerSize(); ++k)
                for (Eigen::SparseMatrix<double>::InnerIterator it(A_vel_3D, k); it; ++it)
                    a_triplets.emplace_back(static_cast<int>(it.row()) + row_offset, static_cast<int>(it.col()), it.value());
            row_offset += static_cast<int>(A_vel_3D.rows());

            // Block 4: A_accel
            for (int k = 0; k < A_accel_3D.outerSize(); ++k)
                for (Eigen::SparseMatrix<double>::InnerIterator it(A_accel_3D, k); it; ++it)
                    a_triplets.emplace_back(static_cast<int>(it.row()) + row_offset, static_cast<int>(it.col()), it.value());

            Eigen::SparseMatrix<double> A_osqp(total_rows, total_cols);
            A_osqp.setFromTriplets(a_triplets.begin(), a_triplets.end());

            // l_osqp = hstack([l_eq, l_sfc, l_vel, l_accel])
            Eigen::VectorXd l_osqp(total_rows);
            l_osqp << l_eq, l_sfc, l_vel, l_accel;

            Eigen::VectorXd u_osqp(total_rows);
            u_osqp << u_eq, u_sfc, u_vel, u_accel;

            // ==========================================
            // 6. EXECUTE OSQP
            // ==========================================
            std::cout << "[Phase 3] Running OSQP Solver (" << total_rows << " constraints, "
                      << total_cols << " variables)..." << std::endl;

            rows_eq = static_cast<int>(A_eq_3D.rows());
            rows_sfc = static_cast<int>(A_sfc_sparse.rows());
            rows_vel = static_cast<int>(A_vel_3D.rows());
            rows_accel = static_cast<int>(A_accel_3D.rows());
            l_all = l_osqp; u_all = u_osqp; A_all = A_osqp;

            optimal_cp = run_qp_solver(P_osqp, q_osqp, A_osqp, l_osqp, u_osqp,
                                       &attempt_prim_res, &qp_failure,
                                       // Per-solve cap. The cycle budget below bounds the SUM
                                       // across attempts, but it is checked between attempts and
                                       // so cannot interrupt one long solve; this does. Thirds
                                       // it so ~3 full-length attempts fit before the cycle bail,
                                       // bounding a worst-case replan at roughly (budget + one
                                       // solve) rather than (budget + one unbounded solve).
                                       max_solve_time_ms_ > 0.0
                                           ? max_solve_time_ms_ / 3000.0 : 0.0,
                                       config_.qp_accept_violation);

            auto opt_end = std::chrono::high_resolution_clock::now();
            double this_opt_ms = std::chrono::duration<double, std::milli>(opt_end - opt_start).count();
            opt_ms += this_opt_ms;
            solve_success = true;

            std::cout << "[Phase 3] Optimization Successful in " << this_opt_ms << " ms!" << std::endl;
            break; // Success — exit the retry loop

        } catch (const std::exception& e) {
            auto opt_end = std::chrono::high_resolution_clock::now();
            opt_ms += std::chrono::duration<double, std::milli>(opt_end - opt_start).count();

            std::cout << "[Phase 4] Solver failed (Kinematically impossible): " << e.what() << std::endl;

            // --- WHAT problem was it asked to solve? ------------------------------------
            // The blame breakdown below names the constraint BLOCK, which is most often
            // "boundary(start/end state)" -- but that block is 18 rows pinning a start state
            // the vehicle is already in against an end state it must arrive at stopped, and
            // knowing it is implicated says nothing about WHICH of those, or why. The same
            // 18/18 signature is produced by a start velocity too high to shed in the allotted
            // time, by a goal outside its own corridor, and by simply not budgeting enough
            // time -- and the remedies are entirely different. Print the geometry once per
            // replan (first attempt only; the retry loop re-solves the same problem with more
            // time and would otherwise repeat this).
            if (attempts == 1) {
                const Eigen::Vector3d goal_delta = actual_end_pos - end_pos;
                std::cout << "[Phase 4] problem: start [" << start_pos.transpose() << "]"
                          << " |v0| " << start_vel.norm() << " |a0| " << start_acc.norm()
                          << " -> goal [" << actual_end_pos.transpose() << "]"
                          << " (dist " << (actual_end_pos - start_pos).norm() << " m";
                if (goal_delta.norm() > 1e-6) {
                    // A* stopped short: horizon truncation, an exhausted node budget, or an
                    // unreachable goal. The QP is then being asked to stop at a waypoint that
                    // is NOT the one the mission wants.
                    std::cout << ", TRUNCATED " << goal_delta.norm() << " m short of the "
                              << "requested goal [" << end_pos.transpose() << "]";
                }
                std::cout << ")" << std::endl;

                // num_segments is the flight time in unit knots (1 segment = 1 s), so this is
                // literally the seconds allotted versus the distance and the speed limit.
                const double dist = (actual_end_pos - start_pos).norm();
                std::cout << "[Phase 4] budget: " << num_segments << " s allotted for "
                          << dist << " m (v_max " << v_max_ << " => "
                          << (v_max_ > 0.0 ? dist / v_max_ : 0.0) << " s at full speed), "
                          << total_control_points << " control points" << std::endl;

                // A goal outside its own corridor is infeasible no matter how much time it is
                // given, and shows up as boundary + corridor rows together.
                // Per-corridor detail. "narrowest half-width 0 m" alone names neither WHICH
                // corridor collapsed nor on WHICH axis, and those imply different causes:
                //   - lateral (y) collapse -> the axis is running flush against inflated cells,
                //     either because the smoother shortcut it there (sfc_smoother calls
                //     is_line_of_sight_clear with margin_cells = 0, so a shortcut may graze
                //     inflation) or because the corridor is measured against the INFLATED set
                //     rather than raw obstacles, which double-counts the clearance already in it.
                //   - vertical (z) collapse -> something is constraining altitude that should
                //     not be, since the forest environments disable the ground plane.
                // r_y/r_z below is the SAT corner term half_res*L1(u), which is 0.25 m on an
                // axis-aligned corridor and 0.433 m on a full diagonal -- pure OBB overhead that
                // is subtracted from every measured clearance, so a diagonal corridor starts
                // 0.18 m worse off than an aligned one through the same gap.
                double narrowest = std::numeric_limits<double>::infinity();
                for (const auto& c : corridors) {
                    for (int b = 2; b < 6; ++b) {   // lateral and vertical half-extents
                        narrowest = std::min(narrowest, std::abs(c.bounds(b)));
                    }
                }
                const double half_res = config_.voxel_resolution / 2.0;
                std::cout << "[Phase 4] corridor bounds (half-extents, m):" << std::endl;
                for (std::size_t ci = 0; ci < corridors.size(); ++ci) {
                    const auto& c = corridors[ci];
                    const double r_y = half_res * c.uy.cwiseAbs().sum();
                    const double r_z = half_res * c.uz.cwiseAbs().sum();
                    const bool degenerate =
                        std::min(std::min(std::abs(c.bounds(2)), std::abs(c.bounds(3))),
                                 std::min(std::abs(c.bounds(4)), std::abs(c.bounds(5)))) < 1e-6;
                    std::cout << "            [" << ci << "] len "
                              << c.getDistancePrimaryToSecondary()
                              << "  fwd/back " << c.bounds(0) << "/" << c.bounds(1)
                              << "  y +/- " << c.bounds(2) << "/" << c.bounds(3)
                              << "  z +/- " << c.bounds(4) << "/" << c.bounds(5)
                              << "  | ux [" << c.ux.transpose() << "]"
                              << " r_y " << r_y << " r_z " << r_z
                              << (degenerate ? "   <== DEGENERATE" : "") << std::endl;
                }
                std::cout << "[Phase 4] corridors: " << corridors.size()
                          << ", narrowest half-width " << narrowest << " m"
                          << " (drone radius " << config_.drone_physical_radius << " m)"
                          << std::endl;
            }

            // --- WHICH constraint block is impossible? ---------------------------------
            // Blame is assigned from OSQP's primal-infeasibility certificate where one
            // exists (its nonzeros are precisely the rows forming the contradiction), and
            // otherwise from per-row violation of the last iterate. Knowing that it is, say,
            // the velocity block rather than the corridors is the difference between
            // "loosen v_max / lengthen the trajectory" and "the corridor geometry is wrong".
            if (rows_eq + rows_sfc + rows_vel + rows_accel > 0) {
                const char* names[4] = {"boundary(start/end state)", "corridors(SFC+map)",
                                        "velocity(MINVO)", "acceleration(MINVO)"};
                const int starts[4] = {0, rows_eq, rows_eq + rows_sfc,
                                       rows_eq + rows_sfc + rows_vel};
                const int counts[4] = {rows_eq, rows_sfc, rows_vel, rows_accel};

                double mass[4] = {0, 0, 0, 0};
                int hits[4] = {0, 0, 0, 0};
                double worst[4] = {0, 0, 0, 0};

                const bool have_cert = qp_failure.prim_inf_cert.size() ==
                                       static_cast<long>(rows_eq + rows_sfc + rows_vel + rows_accel);
                Eigen::VectorXd Ax;
                if (!have_cert && qp_failure.x.size() == A_all.cols()) Ax = A_all * qp_failure.x;

                for (int b = 0; b < 4; ++b) {
                    for (int r = starts[b]; r < starts[b] + counts[b]; ++r) {
                        if (have_cert) {
                            const double c = std::abs(qp_failure.prim_inf_cert(r));
                            if (c > 1e-9) { ++hits[b]; mass[b] += c; worst[b] = std::max(worst[b], c); }
                        } else if (Ax.size() > r) {
                            const double v = std::max(Ax(r) - u_all(r), l_all(r) - Ax(r));
                            if (v > 1e-6) { ++hits[b]; mass[b] += v; worst[b] = std::max(worst[b], v); }
                        }
                    }
                }
                std::cout << "[Phase 4] blame ("
                          << (have_cert ? "infeasibility certificate" : "violation of last iterate")
                          << "):" << std::endl;
                for (int b = 0; b < 4; ++b) {
                    if (counts[b] == 0) continue;
                    std::cout << "            " << names[b] << ": " << hits[b] << "/" << counts[b]
                              << " rows, total " << mass[b] << ", worst " << worst[b] << std::endl;
                }
            }

            // Give up when stretching stops buying anything. Each retry adds a point per
            // pool, so the QP gets bigger and slower every time; if the primal residual is
            // no longer improving, the obstruction is geometric rather than a shortage of
            // flight time and further attempts only burn milliseconds. Observed in practice:
            // residual 0.578 -> 0.287 -> 0.287 -> ... while the solve cost kept climbing,
            // turning one failed plan into hundreds of milliseconds.
            // If the attempt failed BEFORE reaching OSQP (e.g. the trajectory was too short
            // for the snap stencil), there is no residual to compare -- and stretching is
            // exactly the right remedy, so keep going rather than bailing on a stale value.
            const bool solver_ran = !std::isnan(attempt_prim_res);
            const bool improving = !solver_ran || (attempt_prim_res < prev_prim_res * 0.9);
            if (solver_ran) prev_prim_res = attempt_prim_res;
            if (!improving) {
                std::cout << "[Phase 4] Residual stalled at " << attempt_prim_res
                          << "; time-stretching cannot fix this. Giving up early." << std::endl;
                break;
            }

            // Hard latency bound on the retry cycle.
            //
            // The stall check above only fires when the residual STOPS improving. A residual
            // that keeps creeping down by more than 10% a go passes it every time, so a doomed
            // plan can still run the full stretch budget -- and every one of those attempts
            // costs a whole max_iter of OSQP, because a solve that is heading for infeasibility
            // spends its entire iteration budget before it can certify that. Measured in sim:
            // 4 attempts, 268 ms of OSQP inside a 289 ms replan, against a 100 ms period.
            //
            // A missed replan is cheap and already handled -- replan_loop keeps flying the
            // committed trajectory and tries again in 100 ms. A replan that overruns the period
            // is not: it starves the 50 Hz dispatch loop. So bound the cycle by wall time and
            // let the next one, with a fresher map, have a go.
            if (max_solve_time_ms_ > 0.0 && opt_ms >= max_solve_time_ms_) {
                std::cout << "[Phase 4] Solve budget exhausted (" << opt_ms << " ms over "
                          << attempts << " attempt(s), budget " << max_solve_time_ms_
                          << " ms). Abandoning this replan." << std::endl;
                break;
            }

            if (stretch_count < max_stretches) {
                std::cout << "[Phase 4] Stretching time allocation (+1 point per pool)..." << std::endl;
            } else {
                std::cout << "[Error] Max stretching attempts reached." << std::endl;
            }
            stretch_count++;
        }
    }

    auto mission_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(mission_end - mission_start).count();
    double overhead_ms = total_ms - sfc_ms - opt_ms - matrix_ms;

    std::cout << "\n=================================================="
              << "\n          TRAJECTORY PLANNER BENCHMARKS"
              << "\n=================================================="
              << "\nSFC Generation (Front-End):   " << sfc_ms << " ms"
              << "\nPath Generation (Back-End):   " << opt_ms << " ms  (" << attempts << " attempt(s))"
              << "\nConstraint Matrix Build:      " << matrix_ms << " ms"
              << "\nSystem Overhead:              " << overhead_ms << " ms"
              << "\n--------------------------------------------------"
              << "\nTOTAL PLANNING TIME:          " << total_ms << " ms"
              << "\n==================================================" << std::endl;

    // Back to world coordinates. The control points are flat [x0,y0,z0, x1,y1,z1, ...]; the
    // corridors are returned for visualisation and the downstream collision re-check, both of
    // which are world-frame consumers.
    for (int i = 0; i + 2 < optimal_cp.size(); i += 3) {
        optimal_cp.segment<3>(i) += frame_origin;
    }
    for (auto& c : corridors) {
        c.b_vec += c.A_mat * frame_origin;
        c.primaryPosition += frame_origin;
        c.secondaryPosition += frame_origin;
    }

    return {optimal_cp, current_knots, corridors, config_.degree, total_control_points, solve_success, fe_result.reached_goal, sfc_ms, opt_ms, overhead_ms, total_ms, matrix_ms, attempts, measured_d_free, measured_v_end};
}

// ==========================================
// build_overlap_constraints (Multi-Rotor)
// Port of Python's _build_overlap_constraints()
// ==========================================
std::pair<Eigen::MatrixXd, Eigen::VectorXd> TrajectoryPlanner::build_overlap_constraints(
    const std::vector<sfc::SFCResult>& corridors,
    const std::vector<ConstraintPool>& pools,
    int /* total_num_points */) const
{
    int num_dimensions = 3;

    // Compute total control points from pools
    int total_pts = 0;
    for (const auto& pool : pools) total_pts += pool.pts;

    // Map boundary hyperplanes: [+x, -x, +y, -y, +z, -z]
    //
    // The geofence must match the one A* searches in (front_end.cpp builds the A*
    // bounds as +/- map_bounds/2 about the origin), and it must be expressed in NED,
    // where z is NEGATIVE-up. The previous form (b = [X, 0, Y, 0, Z, 0]) encoded a
    // positive-orthant box, which imposes x>=0, y>=0 and -- fatally -- z>=0, i.e. it
    // required every control point to sit at or below ground level. That made the QP
    // primal infeasible on every replan flown at a positive altitude.
    double half_north = config_.map_bounds.x() / 2.0;
    double half_east = config_.map_bounds.y() / 2.0;
    double half_alt = config_.map_bounds.z() / 2.0;

    Eigen::MatrixXd A_map(6, 3);
    A_map << 1.0,  0.0,  0.0,
            -1.0,  0.0,  0.0,
             0.0,  1.0,  0.0,
             0.0, -1.0,  0.0,
             0.0,  0.0,  1.0,
             0.0,  0.0, -1.0;
    Eigen::VectorXd b_map(6);
    b_map << half_north, half_north, half_east, half_east, half_alt, half_alt;

    std::vector<Eigen::MatrixXd> A_rows;
    std::vector<Eigen::VectorXd> b_rows;
    int global_cp_index = 0;

    for (const auto& pool : pools) {
        // Combine A/b from all SFCs in this pool + map boundaries
        std::vector<Eigen::MatrixXd> A_parts;
        std::vector<Eigen::VectorXd> b_parts;

        for (int sfc_idx : pool.sfc_indices) {
            A_parts.push_back(corridors[sfc_idx].A_mat);
            b_parts.push_back(corridors[sfc_idx].b_vec);
        }

        // Append map boundaries -- but only when they could actually bind.
        //
        // These 6 rows were previously stamped onto EVERY control point, doubling the
        // inequality block. In practice a control point is already confined to its corridor,
        // and corridors sit nowhere near the +/-50 m geofence, so the rows are inactive
        // almost always. Redundant inactive constraints are not free: they enlarge the KKT
        // system and add dual variables with nothing to pin them down, which is precisely the
        // degeneracy that makes ADMM crawl toward its iteration limit. Include them only when
        // a corner of some corridor in this pool actually approaches the boundary.
        constexpr double kMapMargin = 5.0;   // metres of slack before we bother constraining
        bool map_rows_can_bind = false;
        for (int sfc_idx : pool.sfc_indices) {
            const Eigen::Matrix<double, 3, 8> verts = corridors[sfc_idx].getAllVertices_3D();
            for (int v = 0; v < 8; ++v) {
                if (std::abs(verts(0, v)) > half_north - kMapMargin ||
                    std::abs(verts(1, v)) > half_east  - kMapMargin ||
                    std::abs(verts(2, v)) > half_alt   - kMapMargin) {
                    map_rows_can_bind = true;
                    break;
                }
            }
            if (map_rows_can_bind) break;
        }
        if (map_rows_can_bind) {
            A_parts.push_back(A_map);
            b_parts.push_back(b_map);
        }

        // Combine into single A_pool/b_pool
        int total_ineq = 0;
        for (const auto& a : A_parts) total_ineq += static_cast<int>(a.rows());

        Eigen::MatrixXd A_pool(total_ineq, num_dimensions);
        Eigen::VectorXd b_pool(total_ineq);
        int row = 0;
        for (size_t k = 0; k < A_parts.size(); ++k) {
            int r = static_cast<int>(A_parts[k].rows());
            A_pool.block(row, 0, r, num_dimensions) = A_parts[k];
            b_pool.segment(row, r) = b_parts[k];
            row += r;
        }

        // Virtual runway logic for natural splines
        if (config_.spline_type == "natural") {
            // Check if this pool touches the first or last SFC
            bool is_first = false, is_last = false;
            for (int idx : pool.sfc_indices) {
                if (idx == 0) is_first = true;
                if (idx == static_cast<int>(corridors.size()) - 1) is_last = true;
            }

            if (is_first || is_last) {
                // The map hyperplanes were appended last, so they are exactly the final
                // 6 rows of the pool. Relaxing them by index avoids having to pattern-match
                // on the b-values (which is brittle whenever the geofence convention changes).
                double runway_length = 30.0;
                for (int k = static_cast<int>(b_pool.size()) - 6; k < static_cast<int>(b_pool.size()); ++k) {
                    b_pool(k) += runway_length;
                }
            }
        }

        // Lock control points inside the overlapping volume.
        //
        // The first 3 and last 3 control points are skipped: for a clamped B-spline the
        // boundary equality constraints (start position/velocity/acceleration, and the
        // terminal condition) determine them UNIQUELY, so they are not free variables the
        // optimizer can move. Adding inequality rows on top of an already-determined
        // variable cannot change the solution -- it can only render the QP primal
        // infeasible whenever the vehicle's current state sits outside the corridor, which
        // is exactly what happens when replanning at speed near an obstacle.
        // (FrontEndSFC::compile_system_constraints applies the same relaxation.)
        for (int j = 0; j < pool.pts; ++j) {
            if (global_cp_index < 3 || global_cp_index >= total_pts - 3) {
                global_cp_index++;
                continue;
            }

            Eigen::MatrixXd A_padded = Eigen::MatrixXd::Zero(total_ineq, total_pts * num_dimensions);
            int col_start = global_cp_index * num_dimensions;
            A_padded.block(0, col_start, total_ineq, num_dimensions) = A_pool;

            A_rows.push_back(A_padded);
            b_rows.push_back(b_pool);
            global_cp_index++;
        }
    }

    // Stack into final matrices
    int total_rows = 0;
    for (const auto& a : A_rows) total_rows += static_cast<int>(a.rows());

    Eigen::MatrixXd A_sfc_out(total_rows, total_pts * num_dimensions);
    Eigen::VectorXd b_sfc_out(total_rows);

    int r = 0;
    for (size_t i = 0; i < A_rows.size(); ++i) {
        int nr = static_cast<int>(A_rows[i].rows());
        A_sfc_out.block(r, 0, nr, A_rows[i].cols()) = A_rows[i];
        b_sfc_out.segment(r, nr) = b_rows[i];
        r += nr;
    }

    return {A_sfc_out, b_sfc_out};
}

// ==========================================
// build_overlap_constraints_fixed_wing
// Port of Python's _build_overlap_constraints_fixed_wing()
// ==========================================
std::pair<Eigen::MatrixXd, Eigen::VectorXd> TrajectoryPlanner::build_overlap_constraints_fixed_wing(
    const std::vector<sfc::SFCResult>& corridors,
    const std::vector<int>& num_pts_list,
    int total_num_points) const
{
    int num_dimensions = 3;
    int degree = config_.degree;

    // NED-symmetric geofence, matching the A* bounds (see build_overlap_constraints).
    double half_north = config_.map_bounds.x() / 2.0;
    double half_east = config_.map_bounds.y() / 2.0;
    double half_alt = config_.map_bounds.z() / 2.0;

    // Map boundary hyperplanes
    Eigen::MatrixXd A_map(6, 3);
    A_map << 1.0,  0.0,  0.0,
            -1.0,  0.0,  0.0,
             0.0,  1.0,  0.0,
             0.0, -1.0,  0.0,
             0.0,  0.0,  1.0,
             0.0,  0.0, -1.0;
    Eigen::VectorXd b_map(6);
    b_map << half_north, half_north, half_east, half_east, half_alt, half_alt;

    std::vector<Eigen::MatrixXd> A_ineq_list;
    std::vector<Eigen::VectorXd> b_ineq_list;
    int start_idx = 0;

    for (size_t i = 0; i < corridors.size(); ++i) {
        // Intersect SFC with map bounding box
        int sfc_rows = static_cast<int>(corridors[i].A_mat.rows());
        int total_ineq = sfc_rows + 6;

        Eigen::MatrixXd A_mat(total_ineq, 3);
        A_mat.topRows(sfc_rows) = corridors[i].A_mat;
        A_mat.bottomRows(6) = A_map;

        Eigen::VectorXd b_vec(total_ineq);
        b_vec.head(sfc_rows) = corridors[i].b_vec;
        b_vec.tail(6) = b_map;

        // Virtual runway logic for natural splines
        if (config_.spline_type == "natural") {
            if (i == 0 || i == corridors.size() - 1) {
                // Map hyperplanes are the final 6 rows (appended after the SFC rows).
                double runway_length = 30.0;
                for (int k = total_ineq - 6; k < total_ineq; ++k) {
                    b_vec(k) += runway_length;
                }
            }
        }

        int num_pts_in_box = num_pts_list[i];

        for (int j = 0; j < num_pts_in_box; ++j) {
            int global_cp_index = start_idx + j;
            Eigen::MatrixXd A_padded = Eigen::MatrixXd::Zero(total_ineq, total_num_points * num_dimensions);
            int col_start = global_cp_index * num_dimensions;
            A_padded.block(0, col_start, total_ineq, num_dimensions) = A_mat;

            A_ineq_list.push_back(A_padded);
            b_ineq_list.push_back(b_vec);
        }

        start_idx += (num_pts_in_box - degree);
    }

    // Stack
    int total_rows = 0;
    for (const auto& a : A_ineq_list) total_rows += static_cast<int>(a.rows());

    Eigen::MatrixXd A_sfc_total(total_rows, total_num_points * num_dimensions);
    Eigen::VectorXd b_sfc_total(total_rows);

    int r = 0;
    for (size_t i = 0; i < A_ineq_list.size(); ++i) {
        int nr = static_cast<int>(A_ineq_list[i].rows());
        A_sfc_total.block(r, 0, nr, A_ineq_list[i].cols()) = A_ineq_list[i];
        b_sfc_total.segment(r, nr) = b_ineq_list[i];
        r += nr;
    }

    return {A_sfc_total, b_sfc_total};
}

} // namespace trajectory_planner
