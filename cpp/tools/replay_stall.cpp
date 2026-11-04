// Replay a flight stall offline, from the map the planner ACTUALLY had.
//
// WHY THIS EXISTS. The planner stalls at a reproducible spot on forest_hard s1 -- it reports
// result:true at 10 Hz with 3 corridors and 8 control points while commanding ~2e-4 m/s, and
// the mission waits forever 33 m from the goal. SEVEN hypotheses were proposed and measured
// wrong: takeoff gate, route-gate self-defence, thrust/tilt coupling, settle-gate regression,
// corridor chain growth, A* geofence, and replanning-from-rest. Every one was tested against an
// offline harness built from GROUND-TRUTH obstacles, and every one died on the same gap: at the
// stall position the vehicle sits outside the trunk field where the LiDAR has swept almost
// nothing, so A* is routing largely through UNKNOWN space. A fully-known map is not the map the
// planner had, and no amount of reasoning from summary statistics substitutes for it.
//
// minimum_snap_manager writes the snapshot when stall_timeout fires. This rebuilds that exact
// map and re-runs the front end and the QP against it, so a hypothesis costs a millisecond
// instead of a sim run.
//
// BUILD
//   cd planner_reference/cpp/build && cmake .. && make replay_stall
//   ./replay_stall /tmp/stall_snapshot.txt
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>

#include "front_end.hpp"
#include "trajectory_planner.hpp"
#include "voxel_grid.hpp"

using namespace trajectory_planner;

namespace {

struct Snapshot
{
  FrontEndConfig cfg;
  double inflation = 0.69;
  double v_max = 8.0, a_max = 20.0, dist_to_goal = 0.0;
  Eigen::Vector3d vehicle{0, 0, 0}, goal{0, 0, 0};
  Eigen::Vector3d last_cmd_pos{0, 0, 0}, last_cmd_vel{0, 0, 0};
  bool have_last_cmd = false;
  std::size_t buffer = 0;
  std::unordered_set<Eigen::Vector3i, mapping::Vector3iHash> occupied, free;
};

bool load(const std::string & path, Snapshot & s)
{
  std::ifstream in(path);
  if (!in) { std::printf("cannot open %s\n", path.c_str()); return false; }

  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ls(line);
    std::string key;
    ls >> key;
    auto vec = [&](Eigen::Vector3d & v) { ls >> v.x() >> v.y() >> v.z(); };

    if (key == "resolution") ls >> s.cfg.voxel_resolution;
    else if (key == "inflation") ls >> s.inflation;
    else if (key == "v_max") ls >> s.v_max;
    else if (key == "a_max") ls >> s.a_max;
    else if (key == "planning_horizon") ls >> s.cfg.planning_horizon;
    else if (key == "max_sfc_count") ls >> s.cfg.max_sfc_count;
    else if (key == "min_chain_length") ls >> s.cfg.min_chain_length;
    else if (key == "max_sfc_count_hard") ls >> s.cfg.max_sfc_count_hard;
    else if (key == "unknown_traversal_cost") ls >> s.cfg.unknown_traversal_cost;
    else if (key == "astar_max_nodes") ls >> s.cfg.astar_max_nodes;
    else if (key == "astar_heuristic_weight") ls >> s.cfg.astar_heuristic_weight;
    else if (key == "min_altitude") ls >> s.cfg.min_altitude;
    else if (key == "max_altitude") ls >> s.cfg.max_altitude;
    else if (key == "drone_radius") ls >> s.cfg.drone_physical_radius;
    else if (key == "geofence_contains_vehicle") {
      int v = 1; ls >> v; s.cfg.geofence_contains_vehicle = (v != 0);
    } else if (key == "dist_to_goal") ls >> s.dist_to_goal;
    else if (key == "vehicle") vec(s.vehicle);
    else if (key == "goal") vec(s.goal);
    else if (key == "last_cmd_pos") { vec(s.last_cmd_pos); s.have_last_cmd = true; }
    else if (key == "last_cmd_vel") vec(s.last_cmd_vel);
    else if (key == "buffer") ls >> s.buffer;
    else if (key == "occupied" || key == "free") {
      std::size_t n = 0; ls >> n;
      auto & dst = (key == "occupied") ? s.occupied : s.free;
      for (std::size_t i = 0; i < n && std::getline(in, line); ++i) {
        std::istringstream cs(line);
        int x, y, z; cs >> x >> y >> z;
        dst.insert(Eigen::Vector3i(x, y, z));
      }
    }
  }
  return true;
}

/// Net displacement of the spline's control polygon -- the quantity the degenerate-plan escape
/// keys on, printed so its threshold can be checked against reality rather than assumed.
double net_travel(const PlanningResult & r)
{
  if (r.control_points.size() < 6) return 0.0;
  const int n = static_cast<int>(r.control_points.size()) / 3;
  return (r.control_points.segment<3>(3 * (n - 1)) - r.control_points.segment<3>(0)).norm();
}

void run_from(const char * label, Snapshot & s, const Eigen::Vector3d & start,
              const Eigen::Vector3d & vel)
{
  auto grid = std::make_shared<mapping::SparseVoxelGrid>(s.cfg.voxel_resolution, s.inflation);
  grid->restore_from_snapshot(s.occupied, s.free);

  FrontEndSFC fe(s.cfg, grid);
  const auto r = fe.get_corridors_astar(start, s.goal);

  double chain = 0.0, smin = 1e18;
  for (const auto & c : r.corridors) {
    const double l = (c.secondaryPosition - c.primaryPosition).norm();
    chain += l; smin = std::min(smin, l);
  }
  int pts = 0;
  for (const auto & p : r.allocation_pools) pts += p.pts;

  TrajectoryPlanner planner(s.cfg, grid, s.v_max, s.a_max);
  const auto plan = planner.plan_mission(start, s.goal, vel, Eigen::Vector3d::Zero());

  // Narrowest LATERAL half-width over the chain. A corridor is only as good as its tightest
  // face, and a zero here is a primal-infeasible QP that no amount of time-stretching fixes.
  double wmin = 1e18;
  for (const auto & c : r.corridors) {
    for (int k = 2; k <= 5; ++k) { wmin = std::min(wmin, c.bounds(k)); }
  }
  std::printf("%-22s corridors %zu  chain %6.2f m  segmin %5.2f  pts %2d  reached %-3s | "
              "narrowest %5.3f m  QP %-4s cps %2d  net %6.2f m\n",
              label, r.corridors.size(), chain, r.corridors.empty() ? 0.0 : smin, pts,
              r.reached_goal ? "yes" : "no", r.corridors.empty() ? 0.0 : wmin,
              plan.success ? "ok" : "FAIL", plan.total_control_points, net_travel(plan));
}

}  // namespace

int main(int argc, char ** argv)
{
  Snapshot s;
  if (!load(argc > 1 ? argv[1] : "/tmp/stall_snapshot.txt", s)) return 1;

  std::printf("map: %zu occupied, %zu free cells at %.2f m, inflation %.3f\n",
              s.occupied.size(), s.free.size(), s.cfg.voxel_resolution, s.inflation);
  std::printf("vehicle [%.2f %.2f %.2f]  goal [%.2f %.2f %.2f]  %.1f m away  buffer %zu\n",
              s.vehicle.x(), s.vehicle.y(), s.vehicle.z(),
              s.goal.x(), s.goal.y(), s.goal.z(), s.dist_to_goal, s.buffer);
  std::printf("v_max %.1f  horizon %.1f  max_sfc %d  min_chain %.1f  unknown_cost %.2f\n\n",
              s.v_max, s.cfg.planning_horizon, s.cfg.max_sfc_count,
              s.cfg.min_chain_length, s.cfg.unknown_traversal_cost);

  // The anchor replan_loop would have used, and the vehicle pose, run separately: they differ,
  // and which one produces the stalled plan is the first thing to settle.
  run_from("vehicle", s, s.vehicle, Eigen::Vector3d::Zero());
  if (s.have_last_cmd) run_from("last command", s, s.last_cmd_pos, s.last_cmd_vel);

  // Does the stall survive the knobs? Each line is one hypothesis, answered in milliseconds.
  std::printf("\n-- sensitivity ------------------------------------------------------------\n");
  const auto base = s.cfg;
  for (double h : {15.0, 30.0, 60.0, -1.0}) {
    s.cfg = base; s.cfg.planning_horizon = h;
    char lbl[48]; std::snprintf(lbl, sizeof(lbl), "horizon %.0f", h);
    run_from(lbl, s, s.vehicle, Eigen::Vector3d::Zero());
  }
  for (double u : {0.5, 2.0, 5.0}) {
    s.cfg = base; s.cfg.unknown_traversal_cost = u;
    char lbl[48]; std::snprintf(lbl, sizeof(lbl), "unknown %.1f", u);
    run_from(lbl, s, s.vehicle, Eigen::Vector3d::Zero());
  }
  // THE CLEARANCE MISMATCH. A* accepts any cell that is not inflated-blocked, i.e. it
  // guarantees only `inflation` = hull + obstacle_margin. The corridor builder must inset by
  // hull PLUS the SAT corner term r = half_res * L1(u), which is 0.25 m on an axis-aligned
  // spine and 0.354 m on a 45-degree one. So the builder needs
  //     hull + half_res*L1(u)   vs   A*'s   hull + obstacle_margin
  // and whenever obstacle_margin < half_res*L1(u) the search can hand over a path it considers
  // legal that the builder cannot wrap a corridor around -- the lateral bound clamps to 0.
  // At margin 0.20 against 0.354 for a diagonal segment, that gap is 0.154 m.
  std::printf("\n--- inflation / obstacle_margin (A* guarantees hull+margin; the builder needs\n");
  std::printf("    hull + half_res*L1(axis) = 0.49+0.25 axis-aligned, 0.49+0.354 diagonal) ---\n");
  const double inf0 = s.inflation;
  for (double margin : {0.20, 0.25, 0.30, 0.36, 0.45}) {
    s.inflation = s.cfg.drone_physical_radius + margin;
    char lbl[64]; std::snprintf(lbl, sizeof lbl, "margin %.2f (inf %.2f)", margin, s.inflation);
    run_from(lbl, s, s.vehicle, Eigen::Vector3d::Zero());
  }
  s.inflation = inf0;

  std::printf("\n--- SFC extension (box length is ext_s + segment + ext_e; a 0.71 m segment\n");
  std::printf("    with the 5 m default gets an 8.5 m box that sweeps unrelated obstacles) ---\n");
  const double e0s = s.cfg.sfc_start_ext, e0e = s.cfg.sfc_end_ext;
  for (double e : {5.0, 3.0, 2.0, 1.0, 0.5}) {
    s.cfg.sfc_start_ext = e; s.cfg.sfc_end_ext = e;
    char lbl[64]; std::snprintf(lbl, sizeof lbl, "sfc_ext %.1f m", e);
    run_from(lbl, s, s.vehicle, Eigen::Vector3d::Zero());
  }
  s.cfg.sfc_start_ext = e0s; s.cfg.sfc_end_ext = e0e;

  for (int n : {4000, 20000}) {
    s.cfg = base; s.cfg.astar_max_nodes = n;
    char lbl[48]; std::snprintf(lbl, sizeof(lbl), "nodes %d", n);
    run_from(lbl, s, s.vehicle, Eigen::Vector3d::Zero());
  }
  s.cfg = base;
  return 0;
}
