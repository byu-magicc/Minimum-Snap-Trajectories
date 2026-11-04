// Offline reproduction of the forest-benchmark QP stall, replayed from logged geometry.
//
// WHY THIS EXISTS
// ---------------
// The vehicle stops mid-forest with a VALID corridor chain: A* routes correctly around the
// obstacle, every corridor clears the hull, and the QP still returns primal-infeasible with all
// three constraint blocks participating. Each hypothesis about why was costing a full sim run --
// launch, fly, read logs, change one parameter, repeat -- and several of those hypotheses turned
// out to be wrong. That loop is the bottleneck, not any individual bug.
//
// Start state and corridor geometry fully determine the QP. Neither the map, the search, nor the
// vehicle is needed. This rebuilds the chain from the numbers the planner already prints and
// replays it through the real back end, so a hypothesis costs a millisecond.
//
// THE CAPTURED CASE (from [Phase 4] output, forest_medium, v_max 5 / a_max 20):
//     start [44.6255 -3.08698 -4.92133]  |v0| 0  |a0| 0
//     goal  [59.75 -2.25 -4.25]          15.16 m, TRUNCATED 45.32 m short of the real goal
//     budget 10 s, 14 control points, 3 corridors
//     narrowest half-width 0.796 m against a 0.490 m hull  -- corridors are NOT the problem
//     blame: boundary 18/18 total 1.256 | corridors 14/138 total 0.876 | velocity 3/120 total 1.127
//
// The chain reconstructs exactly onto the logged goal, which is what confirms the geometry below
// is the real one rather than an approximation of it.
//
// BUILD
//   cd planner_reference/cpp/build && cmake .. && make repro_forest_stall
//   ./repro_forest_stall

#include <cmath>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "dynamic_sfc.hpp"
#include "front_end.hpp"
#include "trajectory_planner.hpp"
#include "voxel_grid.hpp"

using namespace trajectory_planner;

namespace {

// --- The logged chain -----------------------------------------------------------------------
// bounds order is [+x_fwd, -x_back, +y, -y, +z, -z], matching SFCResult::bounds.
struct LoggedCorridor {
  double len;
  Eigen::Vector3d ux;
  Eigen::Matrix<double, 6, 1> bounds;
};

std::vector<LoggedCorridor> logged_chain()
{
  std::vector<LoggedCorridor> c(3);
  c[0].len = 4.66045;
  c[0].ux = Eigen::Vector3d(0.885008, -0.464122, 0.0367618);
  c[0].bounds << 5.66065, 5.0, 0.795784, 0.795784, 0.919378, 0.919378;

  c[1].len = 2.5;
  c[1].ux = Eigen::Vector3d(1.0, 0.0, 0.0);
  c[1].bounds << 4.94049, 0.940491, 0.940491, 0.940491, 1.44049, 1.44049;

  c[2].len = 9.02774;
  c[2].ux = Eigen::Vector3d(0.941543, 0.332309, 0.0553849);
  c[2].bounds << 14.0277, 0.967416, 0.843034, 0.843034, 0.913226, 0.913226;
  return c;
}

const Eigen::Vector3d kStart(44.6255, -3.08698, -4.92133);

/// Rebuild the corridor chain as real SFCResults, walking the logged axes end to end.
std::vector<sfc::SFCResult> build_corridors(const std::vector<LoggedCorridor>& logged)
{
  std::vector<sfc::SFCResult> out;
  Eigen::Vector3d p = kStart;
  for (const auto& lc : logged) {
    const Eigen::Vector3d q = p + lc.ux * lc.len;

    // Local frame exactly as AsymmetricBoxBuilder derives it, so the box matches the logged one.
    Eigen::Vector3d up(0.0, 0.0, 1.0);
    if (std::abs(lc.ux.dot(up)) > 0.99) { up = Eigen::Vector3d(1.0, 0.0, 0.0); }
    Eigen::Vector3d uy = up.cross(lc.ux).normalized();
    Eigen::Vector3d uz = lc.ux.cross(uy);

    sfc::SFCResult r;
    r.primaryPosition = p;
    r.secondaryPosition = q;
    r.ux = lc.ux;
    r.uy = uy;
    r.uz = uz;
    r.bounds = lc.bounds;
    sfc::OBBAdapters::get_osqp_matrices(p, lc.ux, uy, uz, lc.bounds, r.A_mat, r.b_vec);
    out.push_back(r);
    p = q;
  }
  return out;
}

FrontEndConfig benchmark_config()
{
  FrontEndConfig c;
  c.voxel_resolution = 0.5;
  c.drone_physical_radius = 0.490491;
  c.v_max = 5.0;
  c.a_max = 20.0;
  c.spline_type = "clamped";
  c.degree = 4;
  c.planning_horizon = 15.0;
  c.time_budget_factor = 1.20;
  c.max_solve_time_ms = 60.0;
  c.qp_accept_violation = 0.02;
  c.sfc_width = 5.0;
  c.sfc_height = 5.0;
  c.sfc_start_ext = 5.0;
  c.sfc_end_ext = 5.0;
  return c;
}

struct Outcome {
  bool success;
  int attempts;
  int control_points;
  double opt_ms;
};

/// Replay the chain through the real back end. reached_goal=false reproduces the TRUNCATED case,
/// which is what makes the terminal velocity non-zero.
Outcome replay(const FrontEndConfig& cfg, bool reached_goal)
{
  auto grid = std::make_shared<mapping::SparseVoxelGrid>(cfg.voxel_resolution,
                                                         cfg.drone_physical_radius);
  TrajectoryPlanner planner(cfg, grid, cfg.v_max, cfg.a_max);

  FrontEndResult fe;
  fe.corridors = build_corridors(logged_chain());
  fe.allocation_pools = planner.front_end().allocate_for(fe.corridors);
  fe.reached_goal = reached_goal;
  fe.actual_end_pos = fe.corridors.back().secondaryPosition;
  planner.set_front_end_override(fe);

  const auto r = planner.plan_mission(kStart, fe.actual_end_pos,
                                      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  return {r.success, r.solve_attempts, r.total_control_points, r.opt_time_ms};
}

/// Replay only the corridors named in `idx`, renumbered as a standalone chain.
Outcome replay_subchain(const FrontEndConfig& cfg, const std::vector<int>& idx,
                        bool reached_goal = false)
{
  auto grid = std::make_shared<mapping::SparseVoxelGrid>(cfg.voxel_resolution,
                                                         cfg.drone_physical_radius);
  TrajectoryPlanner planner(cfg, grid, cfg.v_max, cfg.a_max);

  const auto all = build_corridors(logged_chain());
  FrontEndResult fe;
  for (int i : idx) { fe.corridors.push_back(all[i]); }
  fe.allocation_pools = planner.front_end().allocate_for(fe.corridors);
  fe.reached_goal = reached_goal;
  fe.actual_end_pos = fe.corridors.back().secondaryPosition;
  planner.set_front_end_override(fe);

  const auto r = planner.plan_mission(fe.corridors.front().primaryPosition, fe.actual_end_pos,
                                      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  return {r.success, r.solve_attempts, r.total_control_points, r.opt_time_ms};
}

/// Replay one corridor with its LATERAL half-extents scaled, leaving the axis untouched.
Outcome replay_scaled(const FrontEndConfig& cfg, int which, double scale)
{
  auto grid = std::make_shared<mapping::SparseVoxelGrid>(cfg.voxel_resolution,
                                                         cfg.drone_physical_radius);
  TrajectoryPlanner planner(cfg, grid, cfg.v_max, cfg.a_max);

  auto logged = logged_chain();
  for (int b = 2; b < 6; ++b) { logged[which].bounds(b) *= scale; }
  auto all = build_corridors(logged);

  FrontEndResult fe;
  fe.corridors.push_back(all[which]);
  fe.allocation_pools = planner.front_end().allocate_for(fe.corridors);
  fe.reached_goal = false;
  fe.actual_end_pos = fe.corridors.back().secondaryPosition;
  planner.set_front_end_override(fe);

  const auto r = planner.plan_mission(fe.corridors.front().primaryPosition, fe.actual_end_pos,
                                      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  return {r.success, r.solve_attempts, r.total_control_points, r.opt_time_ms};
}

std::string sub_label(const std::vector<int>& idx)
{
  std::string s = "corridors [";
  for (std::size_t i = 0; i < idx.size(); ++i) {
    s += std::to_string(idx[i]);
    if (i + 1 < idx.size()) { s += ","; }
  }
  return s + "]";
}

/// Is p inside the corridor's half-space set?
bool inside(const sfc::SFCResult& c, const Eigen::Vector3d& p)
{
  const Eigen::VectorXd r = c.A_mat * p - c.b_vec;
  return (r.array() <= 1e-9).all();
}

/// Monte-Carlo the shared volume of consecutive boxes over their combined bounding box.
void report_overlaps()
{
  const auto c = build_corridors(logged_chain());
  for (std::size_t i = 0; i + 1 < c.size(); ++i) {
    Eigen::Vector3d lo = c[i].primaryPosition.cwiseMin(c[i + 1].secondaryPosition);
    Eigen::Vector3d hi = c[i].primaryPosition.cwiseMax(c[i + 1].secondaryPosition);
    lo.array() -= 6.0;
    hi.array() += 6.0;
    const Eigen::Vector3d span = hi - lo;
    const int N = 60;
    long both = 0, total = 0;
    for (int a = 0; a < N; ++a) {
      for (int b = 0; b < N; ++b) {
        for (int d = 0; d < N; ++d) {
          const Eigen::Vector3d p(lo.x() + span.x() * a / (N - 1.0),
                                  lo.y() + span.y() * b / (N - 1.0),
                                  lo.z() + span.z() * d / (N - 1.0));
          ++total;
          if (inside(c[i], p) && inside(c[i + 1], p)) { ++both; }
        }
      }
    }
    const double vol = span.prod() * static_cast<double>(both) / total;
    std::printf("  [%zu]-[%zu] shared volume ~ %7.3f m^3   (%.2f%% of the sampled box)\n",
                i, i + 1, vol, 100.0 * both / total);
  }
}

void row(const std::string& label, const Outcome& o)
{
  std::printf("  %-42s %-8s  %2d attempt(s)  %3d cp  %6.2f ms\n",
              label.c_str(), o.success ? "SOLVED" : "FAILED", o.attempts,
              o.control_points, o.opt_ms);
}

}  // namespace

int main()
{
  std::printf("Forest-benchmark stall, replayed from logged geometry\n");
  std::printf("  start [%.3f %.3f %.3f] at rest, 3 corridors, %.2f m to the horizon goal\n\n",
              kStart.x(), kStart.y(), kStart.z(),
              (build_corridors(logged_chain()).back().secondaryPosition - kStart).norm());

  const FrontEndConfig base = benchmark_config();

  std::printf("BASELINE -- as flown\n");
  row("as logged (truncated, terminal v > 0)", replay(base, /*reached_goal=*/false));
  std::printf("\n");

  // Is the terminal velocity the binding term? reached_goal=true pins the end to rest instead.
  std::printf("HYPOTHESIS 1: the terminal velocity is what makes it infeasible\n");
  row("same chain, terminal velocity = 0", replay(base, /*reached_goal=*/true));
  std::printf("\n");

  // The braking bound computes ~10.9 m/s here, so the 0.9*v_max cap always binds and every
  // truncated plan demands 4.5 m/s regardless of geometry. Sweep v_max to move that cap.
  std::printf("HYPOTHESIS 2: the terminal speed is too close to v_max\n");
  std::printf("  NOTE sweeping v_max proves nothing -- the terminal speed is a FRACTION of it, so\n");
  std::printf("  both sides of the velocity constraint move together. Vary the fraction instead.\n");
  for (double f : {0.90, 0.75, 0.60, 0.45, 0.30, 0.15}) {
    FrontEndConfig c = base;
    c.terminal_speed_fraction = f;
    row("terminal = " + std::to_string(f).substr(0, 4) + "*v_max = "
            + std::to_string(f * 5.0).substr(0, 4) + " m/s",
        replay(c, false));
  }
  std::printf("\n");

  // More time means more control points, so more freedom to satisfy the same geometry.
  std::printf("HYPOTHESIS 3: the time budget is too tight\n");
  for (double f : {1.2, 1.5, 2.0, 3.0}) {
    FrontEndConfig c = base;
    c.time_budget_factor = f;
    row("time_budget_factor = " + std::to_string(f).substr(0, 3), replay(c, false));
  }
  std::printf("\n");

  // If a near-feasible iterate is being thrown away, this is nearly free to accept.
  std::printf("HYPOTHESIS 4: a flyable iterate is being rejected\n");
  for (double a : {0.02, 0.05, 0.10, 0.25}) {
    FrontEndConfig c = base;
    c.qp_accept_violation = a;
    row("qp_accept_violation = " + std::to_string(a).substr(0, 4), replay(c, false));
  }
  std::printf("\n");

  // Every parameter sweep above failed, so the chain's GEOMETRY is what is infeasible. Solve
  // sub-chains to find which junction cannot be bridged: a corridor pair that fails on its own
  // localises the problem to that one turn.
  std::printf("HYPOTHESIS 5: one junction in the chain cannot be bridged\n");
  for (const auto& sub : {std::vector<int>{0}, std::vector<int>{1}, std::vector<int>{2},
                          std::vector<int>{0, 1}, std::vector<int>{1, 2},
                          std::vector<int>{0, 1, 2}}) {
    row(sub_label(sub), replay_subchain(base, sub));
  }
  std::printf("\n");

  // How much room do consecutive boxes actually share? A bridge pool needs several control
  // points inside BOTH, and they sit roughly a metre apart along the path.
  std::printf("PAIRWISE OVERLAP (sampled, m^3)\n");
  report_overlaps();
  std::printf("\n");

  // Corridor [2] fails ALONE -- a single straight box with healthy half-widths. Everything
  // external has been eliminated, so probe the box itself.
  std::printf("HYPOTHESIS 6: corridor [2] alone -- what makes one straight box infeasible?\n");
  {
    FrontEndConfig c = base;
    row("[2] as logged, terminal v > 0", replay_subchain(c, {2}, false));
    row("[2] with terminal v = 0", replay_subchain(c, {2}, true));
  }
  for (double scale : {1.0, 1.5, 2.0, 4.0}) {
    row("[2] lateral bounds x" + std::to_string(scale).substr(0, 3),
        replay_scaled(base, 2, scale));
  }
  return 0;
}
