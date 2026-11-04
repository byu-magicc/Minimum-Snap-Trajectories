// Why does the SAME narrow chain fly at v_max 5 and stall at v_max 8?
//
// Observed in flight on forest_hard_s1: the vehicle threads a narrow gap at v_max 5 with only a
// hesitation, and at v_max 8 stops dead at the same place -- with a VALID chain, control points
// allocated, and visibly enough room to pass. So the corridors are not the problem, and neither
// is the search. That leaves the two things v_max actually controls:
//
//   1. THE TIME BUDGET.  T = factor * max(L/v_max, 2*sqrt(L/a_max)).  Raising v_max SHORTENS the
//      budget, and since knots are unit-spaced the budget IS the control-point count. Fewer
//      points = less freedom to bend inside a narrow corridor.
//
//   2. THE APPROACH SPEED.  plan_mission pins the start velocity. For a clamped B-spline with
//      unit knots the first control-point gap is forced: P1 - P0 = v0/degree. A vehicle arriving
//      at 6 m/s forces P1 1.5 m downrange of P0 whether or not the corridor goes that way.
//      Nothing in the QP can relax it -- it is an equality.
//
// Both scale with v_max and neither is visible in the planner's own logs, so this separates them:
// sweep v_max with v0 pinned, then sweep v0 with v_max pinned. Whichever axis moves the outcome
// is the mechanism.
//
// BUILD
//   cd planner_reference/cpp/build && cmake .. && make vmax_probe && ./vmax_probe

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include <iostream>
#include <sstream>

#include <Eigen/Dense>

#include "dynamic_sfc.hpp"
#include "front_end.hpp"
#include "trajectory_planner.hpp"
#include "voxel_grid.hpp"

using namespace trajectory_planner;

namespace {

struct LoggedCorridor {
  double len;
  Eigen::Vector3d ux;
  Eigen::Matrix<double, 6, 1> bounds;
};

// The captured narrow chain: corridor 1 is the tight one (0.94 m half-width, 2.5 m long).
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

std::vector<sfc::SFCResult> build_corridors()
{
  std::vector<sfc::SFCResult> out;
  Eigen::Vector3d p = kStart;
  for (const auto& lc : logged_chain()) {
    const Eigen::Vector3d q = p + lc.ux * lc.len;
    Eigen::Vector3d up(0.0, 0.0, 1.0);
    if (std::abs(lc.ux.dot(up)) > 0.99) { up = Eigen::Vector3d(1.0, 0.0, 0.0); }
    Eigen::Vector3d uy = up.cross(lc.ux).normalized();
    Eigen::Vector3d uz = lc.ux.cross(uy);

    sfc::SFCResult r;
    r.primaryPosition = p;
    r.secondaryPosition = q;
    r.ux = lc.ux; r.uy = uy; r.uz = uz;
    r.bounds = lc.bounds;
    sfc::OBBAdapters::get_osqp_matrices(p, lc.ux, uy, uz, lc.bounds, r.A_mat, r.b_vec);
    out.push_back(r);
    p = q;
  }
  return out;
}

FrontEndConfig cfg_at(double v_max)
{
  FrontEndConfig c;
  c.voxel_resolution = 0.5;
  c.drone_physical_radius = 0.490491;
  c.v_max = v_max;
  c.a_max = 20.0;
  c.spline_type = "clamped";
  c.degree = 4;
  c.planning_horizon = 15.0;
  c.time_budget_factor = 1.20;
  c.max_solve_time_ms = 60.0;
  c.qp_accept_violation = 0.02;
  c.sfc_width = 5.0; c.sfc_height = 5.0;
  c.sfc_start_ext = 5.0; c.sfc_end_ext = 5.0;
  return c;
}

double net_travel(const PlanningResult& r)
{
  if (r.control_points.size() < 6) return 0.0;
  const int n = static_cast<int>(r.control_points.size()) / 3;
  return (r.control_points.segment<3>(3 * (n - 1)) - r.control_points.segment<3>(0)).norm();
}

struct Row {
  bool success; int attempts; int cps; double net; double opt_ms; double budget;
  int pools; int excl; int bridge;
};

/// RAII stdout mute -- the planner logs every phase to std::cout and it buries the tables.
struct Mute {
  std::streambuf* old;
  std::ostringstream sink;
  Mute() : old(std::cout.rdbuf(sink.rdbuf())) {}
  ~Mute() { std::cout.rdbuf(old); }
};

Row run(double v_max, const Eigen::Vector3d& v0, double scale = 1.0)
{
  Mute mute;
  const FrontEndConfig cfg = cfg_at(v_max);
  auto grid = std::make_shared<mapping::SparseVoxelGrid>(cfg.voxel_resolution,
                                                          cfg.drone_physical_radius);
  TrajectoryPlanner planner(cfg, grid, cfg.v_max, cfg.a_max);

  FrontEndResult fe;
  fe.corridors = build_corridors();
  if (scale != 1.0) {
    // Shrink the chain in place: shorten each corridor's axis while leaving the box extensions
    // (bounds(0)/bounds(1)) alone, which is exactly what a tight cluster of trunks produces --
    // short A* segments wrapped in full-size sfc_start_ext/sfc_end_ext.
    Eigen::Vector3d p = kStart;
    for (auto& c : fe.corridors) {
      const double len = (c.secondaryPosition - c.primaryPosition).norm() * scale;
      c.primaryPosition = p;
      c.secondaryPosition = p + c.ux * len;
      sfc::OBBAdapters::get_osqp_matrices(c.primaryPosition, c.ux, c.uy, c.uz,
                                          c.bounds, c.A_mat, c.b_vec);
      p = c.secondaryPosition;
    }
  }
  fe.allocation_pools = planner.front_end().allocate_for(fe.corridors);
  fe.reached_goal = false;
  fe.actual_end_pos = fe.corridors.back().secondaryPosition;
  planner.set_front_end_override(fe);

  double L = 0.0;
  for (const auto& c : fe.corridors) L += c.getDistancePrimaryToSecondary();
  int pts = 0, excl = 0, bridge = 0;
  for (const auto& p : fe.allocation_pools) {
    pts += p.pts;
    if (p.type == "bridge") bridge += p.pts; else excl += p.pts;
  }

  const auto r = planner.plan_mission(fe.corridors.front().primaryPosition, fe.actual_end_pos,
                                      v0, Eigen::Vector3d::Zero());
  return {r.success, r.solve_attempts, r.total_control_points, net_travel(r), r.opt_time_ms,
          kinematic_time_budget(L, v_max, 20.0, 1.20), pts, excl, bridge};
}

}  // namespace

int main()
{
  double L = 0.0;
  for (const auto& c : build_corridors()) L += c.getDistancePrimaryToSecondary();
  std::printf("chain length %.2f m, 3 corridors, narrowest half-width 0.84 m, hull 0.49 m\n", L);
  std::printf("time budget T = 1.20 * max(L/v_max, 2*sqrt(L/20))\n");
  std::printf("  the sqrt term is %.2f s -- v_max stops mattering once L/v_max drops below it\n\n",
              2.0 * std::sqrt(L / 20.0));

  std::printf("=== A. sweep v_max, approach from REST (v0 = 0) ===\n");
  std::printf("  v_max  budget[s]  alloc_pts   QP    cps  attempts  net_travel  opt[ms]\n");
  for (double v : {2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0}) {
    const Row r = run(v, Eigen::Vector3d::Zero());
    std::printf("  %5.1f  %8.2f  %9d   %-4s  %3d  %8d  %9.2f m  %6.2f\n",
                v, r.budget, r.pools, r.success ? "ok" : "FAIL", r.cps, r.attempts,
                r.net, r.opt_ms);
  }

  std::printf("\n=== B. sweep APPROACH SPEED along corridor 0's axis, v_max pinned at 8 ===\n");
  std::printf("   |v0|  forced P1-P0  QP    cps  attempts  net_travel  opt[ms]\n");
  const Eigen::Vector3d ux = logged_chain()[0].ux.normalized();
  for (double s : {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}) {
    const Row r = run(8.0, ux * s);
    std::printf("  %5.1f  %10.2f m  %-4s  %3d  %8d  %9.2f m  %6.2f\n",
                s, s / 4.0, r.success ? "ok" : "FAIL", r.cps, r.attempts, r.net, r.opt_ms);
  }

  std::printf("\n=== C. SHRINK THE CHAIN. Short A* segments keep their full sfc extensions, so\n");
  std::printf("       neighbours subsume them, exclusive points go to 0, and every remaining\n");
  std::printf("       control point is a BRIDGE -- pinned in a corridor OVERLAP. The logged\n");
  std::printf("       stall was 3 corridors / 8 cps, which is exactly all-bridge.\n\n");
  std::printf("  Knots are unit-spaced, so DELIVERED flight time = (cps - degree) seconds.\n");
  std::printf("  Bridges cost `degree` points per junction REGARDLESS of chain length: with 3\n");
  std::printf("  corridors that is a hard floor of 8 points = 4 s of flight, always.\n\n");
  std::printf("  chain[m]  kinematic need[s] |  v_max=8: e/b  cps  dur[s]  net[m]  IMPLIED SPEED\n");
  for (double sc : {1.0, 0.75, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08}) {
    const Row b = run(8.0, Eigen::Vector3d::Zero(), sc);
    const double need = kinematic_time_budget(L * sc, 8.0, 20.0, 1.20);
    const double dur = b.cps - 4;
    std::printf("  %7.2f  %14.2f    |      %2d/%-2d  %3d  %5.1f  %6.2f  %6.2f m/s  (%.1fx too slow)\n",
                L * sc, need, b.excl, b.bridge, b.cps, dur, b.net,
                dur > 0 ? b.net / dur : 0.0, dur / need);
  }
  std::printf("\n  The last column is the over-allocation factor. v_max never appears in it --\n");
  std::printf("  the floor is structural, set by degree*(corridors-1) bridge points.\n");
  return 0;
}
