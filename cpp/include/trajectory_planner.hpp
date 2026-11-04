#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <string>
#include <memory>
#include <optional>
#include "front_end.hpp"
#include "min_snap_clamped.hpp"
#include "min_snap_natural.hpp"
#include "optimize.hpp"
#include "minvo_builder.hpp"

namespace trajectory_planner {

// ==========================================
// Planning Result
// ==========================================
struct PlanningResult {
    Eigen::VectorXd control_points;     // Flat [x0,y0,z0, x1,y1,z1, ...]
    Eigen::VectorXd knots;
    std::vector<sfc::SFCResult> corridors;
    int degree;
    int total_control_points;
    bool success;
    bool reached_goal;

    // Timing metrics (milliseconds)
    double sfc_time_ms;
    double opt_time_ms;      // summed over ALL solve attempts
    double overhead_ms;
    double total_time_ms;
    double matrix_time_ms = 0.0;  // constraint-matrix assembly, summed over attempts
    int solve_attempts = 0;       // >1 means the trajectory was re-sized and re-solved

    /// Measured-free run-out past the end of the last corridor, and the terminal speed the
    /// braking bound allows from it: v = min(sqrt(2*a_max*0.6*d_free_ahead), fraction*v_max).
    ///
    /// Exposed because it is the suspected cause of "slow on straights, quick through turns".
    /// The last corridor's forward extension is clipped by whatever lies ahead of it. Pointed
    /// down a straight at the next trunk, the run-out collapses and the plan brakes to rest;
    /// pointed into the gap a turn is heading for, it is unobstructed and the plan carries
    /// speed into the next replan. Both are correct -- the vehicle must be able to stop in
    /// space it has measured -- but the asymmetry is invisible without these two numbers.
    double d_free_ahead = 0.0;    // m
    double terminal_speed = 0.0;  // m/s, magnitude of the pinned end velocity
};

// ==========================================
// TrajectoryPlanner (Port of trajectory_planner.py)
// ==========================================
class TrajectoryPlanner {
public:
    TrajectoryPlanner(const FrontEndConfig& config,
                      std::shared_ptr<mapping::SparseVoxelGrid> grid,
                      double v_max = 3.0, double a_max = 2.0);

    /**
     * Main entry point: Front-End → Matrix Build → OSQP → Control Points.
     */
    /**
     * Replace the front end's output with a fixed result.
     *
     * plan_mission then skips the search entirely and optimises the supplied corridor chain. This
     * exists so a QP failure observed in flight can be replayed from its logged geometry -- start
     * state, corridor bounds and axes fully determine the problem, and none of the map, the
     * search or the vehicle is needed to reproduce it. Reset with std::nullopt.
     *
     * Never set this in flight: it pins the plan to a chain that will not track the moving map.
     */
    void set_front_end_override(std::optional<FrontEndResult> r) { fe_override_ = std::move(r); }

    /**
     * Supply the previously committed route for path hysteresis. Forwarded to the front end.
     * Call every replan -- this object is rebuilt each cycle and keeps no state.
     */
    void set_previous_path(const std::vector<Eigen::Vector3d>& world_points) {
        front_end_->set_previous_path(world_points);
    }

    /** Access to the front end, for building an override's allocation pools. */
    FrontEndSFC& front_end() { return *front_end_; }

    PlanningResult plan_mission(const Eigen::Vector3d& start_pos,
                                const Eigen::Vector3d& end_pos,
                                const Eigen::Vector3d& start_vel = Eigen::Vector3d::Zero(),
                                const Eigen::Vector3d& start_acc = Eigen::Vector3d::Zero());

private:
    FrontEndConfig config_;
    std::unique_ptr<FrontEndSFC> front_end_;
    /// When set, plan_mission uses this instead of running the search. Offline reproduction only;
    /// see set_front_end_override().
    std::optional<FrontEndResult> fe_override_;
    double v_max_;
    double a_max_;
    // Wall-clock ceiling on the OSQP retry cycle, summed over attempts. See the bail in
    // plan_mission(). 0 disables. Taken from FrontEndConfig::max_solve_time_ms.
    double max_solve_time_ms_ = 60.0;

    /**
     * Multi-rotor overlap constraint builder with map boundary injection.
     * Port of Python's _build_overlap_constraints().
     */
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> build_overlap_constraints(
        const std::vector<sfc::SFCResult>& corridors,
        const std::vector<ConstraintPool>& pools,
        int total_num_points) const;

    /**
     * Fixed-wing overlap constraint builder with map boundary injection.
     * Port of Python's _build_overlap_constraints_fixed_wing().
     */
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> build_overlap_constraints_fixed_wing(
        const std::vector<sfc::SFCResult>& corridors,
        const std::vector<int>& num_pts_list,
        int total_num_points) const;
};

} // namespace trajectory_planner
