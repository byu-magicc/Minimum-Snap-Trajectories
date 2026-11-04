// Offline Monte-Carlo over the real front end + QP. No ROS, no simulator, no estimator.
//
// WHY: every conclusion in this project has been resting on single closed-loop runs of a
// benchmark whose travel time has 2.3x run-to-run spread and a documented bimodal failure mode.
// One run costs ~5 minutes of supervised attention, so a hypothesis costs an afternoon -- and
// six consecutive hypotheses about one stall turned out to be wrong. This runs hundreds of
// randomized scenarios through the SAME planner in seconds, so a hypothesis costs a coffee.
//
// WHAT IT MEASURES, and what it deliberately does not. It reports quantities that are
// properties of the PLANNER: whether a plan is produced, how long it takes, how much clearance
// the corridor actually has, and whether the receding-horizon loop makes progress or wedges.
// It contains no controller and no state estimator, so it cannot and does not report tracking
// error or collisions -- those need the closed-loop sim. That separation is a feature: the GNSS
// random walk in the sim contaminates R_succ, and none of it is here.
//
// THE ROLLOUT is the part that makes stalls visible. A single plan call cannot distinguish "hard
// but fine" from "wedged", because wedging is a property of the LOOP: plan, fly a little, rescan,
// replan. So each trial simulates that loop -- raycast a LiDAR scan, accumulate it into the same
// SparseVoxelGrid flight uses, plan, advance along the result, repeat -- and calls it a stall
// when progress toward the goal dies while the goal is still far away.
//
// BUILD / RUN
//   cd planner_reference/cpp/build && cmake .. && make monte_carlo
//   ./monte_carlo --trials 200 --density 0.20
//   ./monte_carlo --trials 200 --ab collinear     # A/B a feature, same seeds both arms

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

// The directional gate, borrowed from the ROS manager. It is header-only and pulls nothing but
// Eigen and the STL, so the harness can use the SAME code the vehicle flies rather than an
// approximation of it. Without it the rollout replans from scratch every step with no
// hysteresis, which is exactly the flip-flopping the gate exists to suppress -- measured as
// stalls climbing 21.7% -> 68.3% as the commitment distance shrank toward the flight value.
#include "navigation/route_gate.hpp"

#include "front_end.hpp"
#include "trajectory_planner.hpp"
#include "bspline_eval.hpp"
#include "voxel_grid.hpp"

using namespace trajectory_planner;

namespace {

constexpr double kRes = 0.5;
// SANDO Table III, Static Forest column: r_drone 0.1 m, r_margin 0.1 m -> 0.2 m of inflation.
// Their HARDWARE column uses 0.45 / 0.2 -- close to this airframe's 0.490 / 0.20 -- so running
// 0.1 / 0.1 here is what makes the benchmark comparable rather than merely similar.
bool g_timed = false;       // --timed : fly the real spline for one 10 Hz period per replan
bool g_knot_scale = false;   // --knotscale : reinterpret unit knots over the kinematic budget
double g_advance = -1.0;    // --advance : metres the vehicle commits per replan.
                            // Flight replans at 10 Hz and flies ~0.25-0.5 m between scans; the
                            // 5 m default was a speed compromise and DRIVES THROUGH trunks that
                            // appear mid-segment, inflating the collision rate.
int g_sfc_cap = -1;         // --sfc : >0 overrides max_sfc_count
double g_horizon = -1.0;    // --horizon : >0 overrides planning_horizon
double g_a_max = 20.0;   // overridable: the braking bound sqrt(2*a*0.6*d_free) only
                         // limits terminal speed when a is small enough to fall under 0.9*v_max
double g_hull = 0.1;
double g_margin = 0.1;
constexpr double kAlt = -3.0;        // NED cruise altitude
constexpr double kFieldX = 100.0;
constexpr double kFieldY = 40.0;

struct Trunk { double x, y, r; };

/// Load a generated forest yaml (the same file the simulator flies).
///
/// Running the harness on the SAME environments the sim flies is the whole point: offline and
/// closed-loop numbers then refer to one set of worlds, so the difference between them is the
/// estimator and the controller rather than a different draw of trees.
std::vector<Trunk> load_forest(const std::string& path)
{
  std::ifstream in(path);
  std::string line;
  bool in_cyl = false;
  std::vector<double> v;
  while (std::getline(in, line)) {
    if (line.find("cylinders:") != std::string::npos) { in_cyl = true; continue; }
    if (!in_cyl) continue;
    const std::size_t dash = line.find('-');
    if (dash == std::string::npos) { if (!v.empty()) break; else continue; }
    try { v.push_back(std::stod(line.substr(dash + 1))); }
    catch (...) { if (!v.empty()) break; }
  }
  std::vector<Trunk> ts;
  for (std::size_t i = 0; i + 4 < v.size(); i += 5) {   // [x, y, z, radius, half_height]
    ts.push_back({v[i], v[i + 1], v[i + 3]});
  }
  return ts;
}

/// RAII stdout mute. TrajectoryPlanner logs every phase to std::cout; across tens of thousands
/// of plans that is both unreadable and a measurable share of the runtime.
struct Mute {
  std::streambuf * old;
  std::ostringstream sink;
  Mute() : old(std::cout.rdbuf(sink.rdbuf())) {}
  ~Mute() { std::cout.rdbuf(old); }
};

/// Forest matching the benchmark definition: 100x40 m, cylinders r in [1.0,1.5], 6 m tall,
/// difficulty set by the fraction of field area occupied.
std::vector<Trunk> make_forest(std::mt19937 & rng, double density)
{
  std::uniform_real_distribution<double> ux(0.0, kFieldX), uy(-kFieldY / 2, kFieldY / 2);
  std::uniform_real_distribution<double> ur(1.0, 1.5);
  const double target = density * kFieldX * kFieldY;
  std::vector<Trunk> ts;
  double area = 0.0;
  int guard = 0;
  while (area < target && guard++ < 20000) {
    Trunk t{ux(rng), uy(rng), ur(rng)};
    // Keep the start and goal pockets clear so a trial is never unsolvable by construction.
    if (std::hypot(t.x - 0.0, t.y) < 4.0) continue;
    if (std::hypot(t.x - kFieldX, t.y) < 4.0) continue;
    ts.push_back(t);
    area += M_PI * t.r * t.r;
  }
  return ts;
}

/// Livox-style scan: 1.5 deg in both axes, 360 horizontal, as the benchmark configures it.
std::vector<Eigen::Vector3d> scan(const Eigen::Vector3d & o, const std::vector<Trunk> & ts,
                                  double max_range)
{
  std::vector<Eigen::Vector3d> hits;
  hits.reserve(4096);
  for (double el = -20.0; el <= 20.0; el += 3.0) {
    for (double az = -180.0; az < 180.0; az += 1.5) {
      const double e = el * M_PI / 180.0, a = az * M_PI / 180.0;
      const Eigen::Vector3d d(std::cos(e) * std::cos(a), std::cos(e) * std::sin(a), -std::sin(e));
      double best = max_range;
      for (const auto & t : ts) {
        const double ox = o.x() - t.x, oy = o.y() - t.y;
        const double A = d.x() * d.x() + d.y() * d.y();
        if (A < 1e-12) continue;
        const double B = 2.0 * (ox * d.x() + oy * d.y());
        const double C = ox * ox + oy * oy - t.r * t.r;
        const double disc = B * B - 4 * A * C;
        if (disc < 0) continue;
        const double s = (-B - std::sqrt(disc)) / (2 * A);
        if (s > 0.2 && s < best) best = s;
      }
      if (best < max_range - 1e-3) hits.push_back(o + best * d);
    }
  }
  return hits;
}

FrontEndConfig base_config(double v_max, bool collinear, double margin = g_margin,
                          bool clamp_relief = true, double repair_width = -1.0)
{
  FrontEndConfig c;
  c.voxel_resolution = kRes;
  c.drone_physical_radius = g_hull;
  c.v_max = v_max;
  c.a_max = g_a_max;
  c.knot_scale = g_knot_scale;
  c.spline_type = "clamped";
  c.degree = 4;
  // MATCHES params/forest_benchmark.yaml. This was 15.0 while flight ran 10.0, so every offline
  // number described a different planner than the one being flown -- and 15 m is the measurably
  // worse setting (R_succ 60% / 20% collisions vs 80% / 0% at 12 m). Keep these in step.
  c.planning_horizon = 12.0;
  c.time_budget_factor = 1.40;   // matches the flight config; see params/forest_benchmark.yaml
  c.max_solve_time_ms = 60.0;
  c.qp_accept_violation = 0.02;
  c.sfc_width = 5.0;  c.sfc_height = 5.0;
  c.sfc_start_ext = 5.0;  c.sfc_end_ext = 5.0;
  c.max_sfc_count = 3;  c.max_sfc_count_hard = 5;  c.min_chain_length = 4.0;
  c.unknown_traversal_cost = 0.5;
  c.astar_max_nodes = 4000;  c.astar_max_time_ms = 20.0;
  c.astar_heuristic_weight = 3.0;
  c.astar_clearance_cost = 0.0;
  c.heat_radius = 1.0;  c.heat_weight = 0.6;
  c.astar_soft_obstacle_cost = 5.0;
  c.astar_min_segment_voxels = 2.0;
  c.astar_collinear_merge = collinear;
  c.sfc_clamp_relief = clamp_relief;
  // -1 means "leave the FrontEndConfig default alone". An explicit 0.0 default here silently
  // overrode that default and made a whole density sweep run with the repair OFF.
  if (repair_width >= 0.0) { c.sfc_min_usable_width = repair_width; }
  c.min_altitude = 0.59;  c.max_altitude = 6.0;
  c.map_bounds = Eigen::Vector3d(kFieldX * 1.5, kFieldY * 2.0, 15.0);
  // COMMAND-LINE OVERRIDES GO LAST.
  //
  // This function assigns some fields more than once, so an override placed mid-body is silently
  // clobbered by a later literal. That has already invalidated two sweeps in this file -- a
  // density sweep (repair_width) and a horizon sweep (planning_horizon), each of which ran every
  // arm at the same value and produced beautifully consistent, meaningless numbers. Anything
  // driven by a flag belongs here, after the last literal assignment.
  if (g_horizon > 0.0) c.planning_horizon = g_horizon;
  if (g_sfc_cap > 0) c.max_sfc_count = g_sfc_cap;

  return c;
}

struct Trial {
  bool reached = false;
  bool stalled = false;
  /// Hull-radius contact with a trunk anywhere along the traversed path. SANDO's R_succ is
  /// "reaches the goal WITHOUT COLLISION", so a run that arrives having clipped a trunk is a
  /// failure for them and must be one here too.
  bool collided = false;
  int replans = 0, failures = 0;
  double progress = 0.0;        // metres of goal-distance actually closed
  double path_len = 0.0;
  double plan_ms_mean = 0.0, plan_ms_max = 0.0;
  double map_ms_mean = 0.0, map_ms_max = 0.0;   // update_from_scan, the CPU bottleneck
  /// THE CONTRIBUTION, timed alone: OSQP only, no A*, no corridor construction, no map.
  /// PlanningResult::opt_time_ms is summed over ALL solve attempts, so a retried solve shows
  /// its true cost rather than just the last attempt.
  std::vector<double> opt_ms;
  std::vector<double> opt_cps;    ///< control points, paired with opt_ms
  std::vector<double> opt_cons;   ///< constraint rows, paired with opt_ms
  /// Per plan: how straight the corridor chain is (chord / arc, 1.0 = dead straight), the
  /// measured-free run-out past its end, and the terminal speed the braking bound allows.
  /// Together these test whether straight chains are being braked to rest because their last
  /// corridor's forward extension is clipped by the trunk they are pointed at.
  std::vector<double> straightness, d_free, v_end;
  /// Per plan: control points, delivered duration (n_cp - degree, unit knots), arc length,
  /// and the kinematic budget the allocator asked for. Floor binds when duration > budget.
  std::vector<double> n_cp, dur, arc_len, budget;
  /// The speed flight ACTUALLY realises. The vehicle replans at 10 Hz and discards the rest of
  /// each plan, so what it flies is the speed over the first replan period -- not the plan's
  /// mean (chain/dur), which is what `speed_sum` reports and is independent of the rollout step.
  std::vector<double> v_head, v_peak;
  int first_try = 0, solves = 0;   ///< QP solves that succeeded on attempt 1
  double min_width = 1e18;      // narrowest corridor half-width seen
  /// EVERY corridor's half-width, not just the narrowest. min_width is a minimum over the whole
  /// run, so with ~14% of corridors at zero width it saturates at 0 after a few hundred replans
  /// and reports "did any corridor pinch" (always yes) rather than how wide corridors typically
  /// are. A distribution is what belongs in a table.
  std::vector<double> all_widths;
  int sub_hull = 0, corridors = 0;
  int plans = 0, one_corridor = 0;
  int degenerate = 0, escapes = 0;
  int held_degen = 0, force_blocked = 0, no_poly_left = 0;   ///< zero-displacement plans, and how many the escape saved
  double speed_sum = 0.0;   // net travel / delivered duration, per plan
  /// TIMED mode only. speed_sum above is the PLAN's mean (arc/duration) and is independent of
  /// how much of the plan is flown. Flight replans at 10 Hz and discards the rest, so realised
  /// speed is arc actually travelled / seconds actually elapsed. These two are not the same
  /// number and the gap is the thing being diagnosed.
  double flown_m = 0.0, flown_s = 0.0;
};

/// One receding-horizon rollout. This is what makes a wedge visible: a single plan call cannot
/// tell "hard but fine" from "stuck", because being stuck is a property of the loop.
Trial rollout(const std::vector<Trunk> & ts, const FrontEndConfig & cfg,
              const Eigen::Vector3d & start, const Eigen::Vector3d & goal, int max_steps,
              double margin = g_margin, double window = 25.0, bool frontier = false,
              double advance = 5.0, double align = 0.10, bool timed = g_timed)
{
  Trial t;
  Mute mute;
  roscopter::RouteGate::Config gcfg;
  gcfg.reject_angle_deg = 35.0;     // matches params/forest_benchmark.yaml
  gcfg.lookahead_m = 3.0;
  gcfg.max_consecutive_rejects = 10;
  gcfg.alignment_margin = align;
  roscopter::RouteGate gate(gcfg);
  // Two distinct things, which I first conflated:
  //   flown_route -- the CORRIDOR SPINE, which is what the manager hands the gate
  //                  (candidate_route is built from corridor primaryPosition, not control points)
  //   flown_poly  -- the control polygon of the accepted plan, which is what the vehicle flies
  std::vector<Eigen::Vector3d> flown_route;
  std::vector<Eigen::Vector3d> flown_poly;   // TrajectoryPlanner logs every phase to cout; silence it for the sweep
  auto grid = std::make_shared<mapping::SparseVoxelGrid>(kRes, g_hull + margin);
  // The window was hardcoded at 25 m here while flight ran 15 m -- so every earlier sweep
  // measured a configuration the vehicle was not flying. Parametrised.
  grid->set_accumulate(true, window);
  // Unknown-space inflation: SANDO Table III lists this ON for their static forest; it is OFF
  // here. With it on, get_obstacles_in_radius returns the FREE/UNKNOWN frontier alongside real
  // obstacles, so corridors are barred from space the sensor has not swept.
  grid->set_frontier_blocks_corridors(frontier);
  grid->set_heat_radius(cfg.heat_radius);   // BEFORE any scan, or the field stays empty

  Eigen::Vector3d p = start;
  double best_dist = (goal - start).norm();
  int no_progress = 0;
  double ms_sum = 0.0, map_sum = 0.0;

  // Flight anchors each replan on the buffered setpoint ~200 ms ahead, so a plan CONTINUES at
  // the commanded velocity. Planning from rest every cycle (what this harness did) makes every
  // trajectory a standing start, which is the slowest case the planner can be asked for.
  // STALL WINDOW, in committed DISTANCE rather than replan count.
  //
  // The criterion is "N consecutive unproductive cycles", and a cycle covers `advance` metres. A
  // fixed N therefore means a different amount of travel at every advance: 60 m at the 5 m default
  // but only 6 m at 0.5 m, which a lateral detour around one trunk clears WITHOUT closing 0.25 m
  // of straight-line distance to the goal -- so the vehicle is declared stalled while making real
  // progress along its route. Measured on easy/100 forests: stalls went 4% -> 22% between 2 m and
  // 1 m purely from this, which inverted the R_succ ranking and made the advance sweep unreadable.
  //
  // Scaling by 5/advance keeps the window at a constant 60 m, so arms are comparable and the 5 m
  // default is unchanged (12 stays 12). Flight is more patient still: stall_timeout 45 s at 10 Hz
  // is 450 replans.
  int consec_fail = 0;   // consecutive failed replans; flight tolerates 3 before braking
  const int stall_replans =
      std::max(12, static_cast<int>(std::lround(12.0 * 5.0 / std::max(advance, 0.01))));

  Eigen::Vector3d cur_vel = Eigen::Vector3d::Zero();
  Eigen::Vector3d cur_acc = Eigen::Vector3d::Zero();
  constexpr double kStitch = 0.20;   // s, matches stitch_index = 10 at 50 Hz
  constexpr double kCycle  = 0.10;   // s, one 10 Hz replan period

  for (int step = 0; step < max_steps; ++step) {
    {
      // Time the MAP UPDATE separately from planning. It is the larger cost and the one that
      // starves the 50 Hz dispatch loop, but it sits outside the planner's own timing, so a
      // sweep that only reports plan time cannot see it at all.
      const auto m0 = std::chrono::high_resolution_clock::now();
      grid->update_from_scan(p, scan(p, ts, window));
      const double mms = std::chrono::duration<double, std::milli>(
          std::chrono::high_resolution_clock::now() - m0).count();
      map_sum += mms;
      t.map_ms_max = std::max(t.map_ms_max, mms);
    }

    const auto t0 = std::chrono::high_resolution_clock::now();
    TrajectoryPlanner planner(cfg, grid, cfg.v_max, cfg.a_max);
    auto r = planner.plan_mission(p, goal,
                                  timed ? cur_vel : Eigen::Vector3d::Zero(),
                                  timed ? cur_acc : Eigen::Vector3d::Zero());

    // DEGENERATE-PLAN ESCAPE, ported from MinimumSnapManager::replan_loop.
    //
    // A plan that does not move is an absorbing state: the vehicle stays put, so the next cycle
    // rebuilds the same plan. Flight escapes by re-planning with the horizon lifted to
    // min(goal_dist, 3*horizon) and keeping the result only if it travels further. This fires
    // CONSTANTLY in flight -- one medium run logged 240 degenerate plans of which 164 escaped this
    // way -- so a harness without it is not modelling the vehicle. Its absence is why stalls went
    // 4% -> 55% as the advance shrank from 5 m to 0.5 m: at coarse commitment the rollout
    // bulldozes out of traps (and through trunks, hence 13% collisions), at fine commitment it
    // needs the escape the way flight does.
    auto net_travel = [](const trajectory_planner::PlanningResult & pr) {
      if (pr.control_points.size() < 6) return 0.0;
      const int n = static_cast<int>(pr.control_points.size()) / 3;
      return (pr.control_points.segment<3>(3 * (n - 1))
              - pr.control_points.segment<3>(0)).norm();
    };
    {
      constexpr double kDegenerateNetTravel = 1.0;   // matches minimum_snap_manager.cpp
      constexpr double kDegenerateGoalFar   = 5.0;
      const double gd = (goal - p).norm();
      if (gd > kDegenerateGoalFar && net_travel(r) < kDegenerateNetTravel) {
        ++t.degenerate;
        FrontEndConfig esc = cfg;
        esc.planning_horizon = std::min(gd, 3.0 * std::max(1.0, cfg.planning_horizon));
        TrajectoryPlanner escape(esc, grid, esc.v_max, esc.a_max);
        const auto alt = escape.plan_mission(p, goal,
                                             timed ? cur_vel : Eigen::Vector3d::Zero(),
                                             timed ? cur_acc : Eigen::Vector3d::Zero());
        if (alt.success && net_travel(alt) > net_travel(r)) { ++t.escapes; r = alt; }
      }
    }
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    ++t.replans;
    ms_sum += ms;
    t.plan_ms_max = std::max(t.plan_ms_max, ms);

    if (r.success) {
      ++t.plans;
      // A solve time means nothing without the problem size it solved, so carry both.
      ++t.solves;
      if (r.solve_attempts <= 1) ++t.first_try;
      t.opt_ms.push_back(r.opt_time_ms);
      if (!r.corridors.empty()) {
        double arc = 0.0;
        for (const auto & cc : r.corridors) arc += (cc.secondaryPosition - cc.primaryPosition).norm();
        const double chord = (r.corridors.back().secondaryPosition
                              - r.corridors.front().primaryPosition).norm();
        if (arc > 1e-6) {
          t.straightness.push_back(chord / arc);
          t.d_free.push_back(r.d_free_ahead);
          t.v_end.push_back(r.terminal_speed);
          const double ncp = static_cast<double>(r.total_control_points);
          t.n_cp.push_back(ncp);
          t.dur.push_back(ncp - static_cast<double>(r.degree));
          t.arc_len.push_back(arc);
          t.budget.push_back(trajectory_planner::kinematic_time_budget(
              arc, cfg.v_max, cfg.a_max, cfg.time_budget_factor));

          // Head speed over one 10 Hz replan period, and the peak the plan would reach if it
          // were ever flown to completion.
          trajectory_planner::bspline::TrajectorySampler samp(
              r.control_points, r.knots, r.degree);
          if (samp.valid() && samp.duration() > 0.15) {
            const double t0 = samp.start_time();
            t.v_head.push_back(samp.velocity(t0 + 0.10).norm());
            double vp = 0.0;
            for (int k = 0; k <= 40; ++k) {
              vp = std::max(vp, samp.velocity(t0 + samp.duration() * k / 40.0).norm());
            }
            t.v_peak.push_back(vp);
          } else {
            t.v_head.push_back(0.0);
            t.v_peak.push_back(0.0);
          }
        }
      }
      t.opt_cps.push_back(static_cast<double>(r.total_control_points));
      t.opt_cons.push_back(static_cast<double>(r.total_control_points) * 3.0);
      if (r.corridors.size() == 1) ++t.one_corridor;
      // Unit knots: delivered flight time = (control points - degree) seconds. This is the
      // quantity that decides whether the vehicle crawls, and it is floored at degree*2 - degree
      // = 4 s for a single corridor however short that corridor is.
      const double dur = std::max(1.0, static_cast<double>(r.total_control_points) - 4.0);
      double chain = 0.0;
      for (const auto & c : r.corridors) chain += (c.secondaryPosition - c.primaryPosition).norm();
      t.speed_sum += chain / dur;
    }
    for (const auto & c : r.corridors) {
      ++t.corridors;
      double w = 1e18;
      for (int k = 2; k <= 5; ++k) w = std::min(w, c.bounds(k));
      t.min_width = std::min(t.min_width, w);
      t.all_widths.push_back(w);
      if (w < 1e-6) ++t.sub_hull;   // ZERO-WIDTH: infeasible for the QP, not unsafe
    }

    // A FAILED REPLAN MUST NOT FREEZE THE VEHICLE.
    //
    // Flight keeps draining its setpoint buffer through a failed replan and only starts braking
    // after replan_failures_before_brake_ = 3 consecutive failures. Bailing out on the FIRST
    // failure -- what this did -- makes the stall rate a function of replan FREQUENCY rather than
    // of the planner: there are 10x more replans at 0.5 m commitment than at 5 m, hence 10x more
    // frozen cycles. Measured on easy/100 forests: stalls 3% -> 51% from 5 m to 0.5 m, while the
    // real vehicle completes the same difficulty. That was the harness, not the planner.
    //
    // net_travel() returns 0 for a plan with no control points, so these failures were also being
    // counted as "absorbing-state" plans and fed to the escape -- which is why the escape appeared
    // to fire 2532 times and rescue 12. Most of those were failures, not degenerate geometry.
    const bool plan_ok = r.success && r.control_points.size() >= 6;
    if (!plan_ok) {
      ++t.failures;
      ++consec_fail;
      const bool can_hold = flown_poly.size() >= 2 && consec_fail < 3;
      if (!can_hold) {
        ++no_progress;
        if (no_progress >= stall_replans) { t.stalled = true; break; }
        continue;
      }
      // else fall through and keep flying the route already committed to
    } else {
      consec_fail = 0;
    }
    if (plan_ok) {

    // THE DIRECTIONAL GATE, as the manager applies it: a candidate that turns away from the
    // route being flown is declined and the previous route is kept. Forced when there is no
    // current route, which is the manager's "no buffered trajectory" override.
    std::vector<Eigen::Vector3d> cand_route;      // corridor spine, as the manager builds it
    for (const auto & c : r.corridors) cand_route.push_back(c.primaryPosition);
    if (!r.corridors.empty()) cand_route.push_back(r.corridors.back().secondaryPosition);

    std::vector<Eigen::Vector3d> cand_poly;       // control polygon, what the vehicle flies
    {
      const int n_c = static_cast<int>(r.control_points.size()) / 3;
      for (int k = 0; k < n_c; ++k) cand_poly.push_back(r.control_points.segment<3>(3 * k));
    }

    // FORCE-ACCEPT WHEN THE HELD ROUTE HAS GONE STALE.
    //
    // The manager walks its buffered setpoints against the freshly updated map and forces the
    // gate open if any of them is now occupied (is_occupied_radius). Without that check,
    // declining a candidate means flying geometry planned against an OLDER map -- which is
    // exactly what it looked like: collisions rose 26.7% -> 50.0% at 5 m commitment when the
    // gate was added alone. Holding a route is only safe if something notices when the world
    // contradicts it.
    bool force = flown_poly.size() < 2;
    if (!force && !flown_poly.empty()) {
      for (std::size_t k = 0; k < flown_poly.size(); k += 2) {
        if (grid->is_occupied_radius(flown_poly[k], g_hull + g_margin)) { force = true; break; }
      }
    }
    // TRAJECTORY PERSISTENCE, the way the manager's setpoint buffer provides it.
    //
    // Flight stitches each replan 200 ms into the EXISTING buffer and keeps draining it, so a plan
    // that goes nowhere does not stop the vehicle -- it keeps flying the trajectory it already
    // committed to. That is how one medium run absorbed 240 zero-displacement plans without
    // stalling. Replacing the flown polygon with a degenerate candidate instead parks the vehicle,
    // and since it has not moved, the next cycle rebuilds the same degenerate plan: a true
    // absorbing state that flight never enters.
    //
    // Measured on easy/100 forests at 0.5 m commitment, WITHOUT this: 51% stalls, 2532 degenerate
    // plans, R_succ 48% -- while the real vehicle completes the same difficulty. The harness was
    // failing for a reason the vehicle does not have.
    //
    // A degenerate candidate is therefore held exactly like a gate-declined one, EXCEPT when the
    // held geometry is stale against the map -- `force` already covers that, and honouring it here
    // keeps the safety property the gate comment describes.
    const bool degen_now = (goal - p).norm() > 5.0 && net_travel(r) < 1.0;
    if (degen_now && force) ++t.force_blocked;
    const bool cand_degenerate = degen_now && !force;
    if (cand_degenerate) {
      ++t.held_degen;
      // Does the held polygon actually have anything left AHEAD of the vehicle? Holding a route
      // the vehicle has already flown to the end of parks it just as surely as accepting a
      // degenerate one.
      double ahead = 0.0;
      if (flown_poly.size() >= 2) {
        std::size_t k0 = 0; double best = 1e18;
        for (std::size_t k = 0; k < flown_poly.size(); ++k) {
          const double dd = (flown_poly[k] - p).norm();
          if (dd < best) { best = dd; k0 = k; }
        }
        for (std::size_t k = k0 + 1; k < flown_poly.size(); ++k)
          ahead += (flown_poly[k] - flown_poly[k - 1]).norm();
      }
      if (ahead < 0.5) ++t.no_poly_left;
    }
    if (!cand_degenerate && gate.evaluate(flown_route, cand_route, p, goal, force).accept) {
      flown_route = cand_route;
      flown_poly = cand_poly;
    }
    if (flown_poly.size() < 2) { ++no_progress; continue; }

    // ADVANCE ALONG THE CONTROL POLYGON, not straight at its endpoint.
    //
    // The first version of this moved the vehicle directly toward the LAST control point,
    // capped at 5 m. That is a chord, not the trajectory: it cuts across exactly the curvature
    // the planner inserted to avoid a trunk. With collision checking switched on it read as a
    // 100% collision rate that was entirely an artefact of the rollout.
    //
    // Walking the polygon is a fair proxy for the spline: a B-spline lies inside the convex
    // hull of its control points, so the polygon bounds where the trajectory can be, and the
    // corridor chain bounds the polygon.
    // Resume from the closest point on the flown polygon -- when the gate declines, the vehicle
    // keeps flying the plan it already has, partway through, exactly as the manager keeps
    // draining its buffer.
    }   // end if (plan_ok): on a held failure we fly the existing flown_poly unchanged

    // TIMED ROLLOUT. Flies the real spline for one 10 Hz period and then replans, which is what
    // the manager does -- rather than walking the control polygon a fixed DISTANCE with no clock.
    // Only this mode can produce a realised speed; speed_sum is the plan's own mean and would be
    // identical at any advance step.
    if (timed) {
      trajectory_planner::bspline::TrajectorySampler samp(
          r.control_points, r.knots, r.degree);
      if (!samp.valid() || samp.duration() < 1e-3) { ++no_progress; continue; }
      const double t0 = samp.start_time();
      const double tend = std::min(t0 + kCycle, samp.start_time() + samp.duration());
      const int n_s = 8;
      Eigen::Vector3d cursor = samp.position(t0);
      for (int i = 1; i <= n_s && !t.collided; ++i) {
        const Eigen::Vector3d q = samp.position(t0 + (tend - t0) * i / n_s);
        const double seg = (q - cursor).norm();
        t.path_len += seg;
        t.flown_m  += seg;
        for (const auto & tr : ts) {
          if (std::hypot(q.x() - tr.x, q.y() - tr.y) < tr.r + g_hull) { t.collided = true; break; }
        }
        cursor = q;
      }
      t.flown_s += (tend - t0);
      p = cursor;
      cur_vel = samp.velocity(tend);
      cur_acc = samp.acceleration(tend);
      if (t.collided) break;
      const double dd = (goal - p).norm();
      if (dd < best_dist - 0.25) { best_dist = dd; no_progress = 0; }
      else if (++no_progress >= 120) { t.stalled = true; break; }   // 120 cycles = 12 s held
      if (dd < 3.0) { t.reached = true; break; }
      continue;
    }

    std::size_t start_k = 0;
    {
      double best = std::numeric_limits<double>::infinity();
      for (std::size_t k = 0; k < flown_poly.size(); ++k) {
        const double dd = (flown_poly[k] - p).norm();
        if (dd < best) { best = dd; start_k = k; }
      }
    }
    const int n_cp = static_cast<int>(flown_poly.size());
    // HOW FAR THE VEHICLE COMMITS BEFORE RE-SCANNING. This is a fidelity knob, not a planner
    // parameter, and it was chosen to make trials fast rather than realistic. Flight replans at
    // 10 Hz and flies ~0.25-0.5 m between scans; committing 5 m here is 10-20x more, so any
    // trunk that comes into view mid-segment is driven through rather than avoided. If the
    // collision rate falls sharply as this shrinks, the harness was overstating it.
    double budget = advance;
    Eigen::Vector3d cursor = p;
    for (int k = static_cast<int>(start_k) + 1; k < n_cp && budget > 1e-6; ++k) {
      const Eigen::Vector3d cp = flown_poly[k];
      const double leg = (cp - cursor).norm();
      if (leg < 1e-9) continue;
      const Eigen::Vector3d next = (leg <= budget) ? cp : cursor + (cp - cursor) * (budget / leg);

      // Collision over the traversed segment, sampled at 0.1 m -- well under the smallest
      // trunk radius (1.0 m), so a trunk cannot be stepped over.
      const double seg = (next - cursor).norm();
      const int n_s = std::max(1, static_cast<int>(std::ceil(seg / 0.1)));
      for (int i = 0; i <= n_s && !t.collided; ++i) {
        const Eigen::Vector3d q = cursor + (next - cursor) * (static_cast<double>(i) / n_s);
        for (const auto & tr : ts) {
          if (std::hypot(q.x() - tr.x, q.y() - tr.y) < tr.r + g_hull) { t.collided = true; break; }
        }
      }

      t.path_len += seg;
      budget -= seg;
      cursor = next;
      if (t.collided) break;
    }
    // NO TRIMMING. RouteGate::evaluate projects the vehicle onto BOTH routes before comparing
    // (arc_length_of_closest_point), so a stale prefix is already handled. An earlier attempt to
    // trim here was both unnecessary and inverted -- it dropped the whole route and drove
    // goal-reach from 51.7% to 8.3%.
    p = cursor;
    if (t.collided) break;   // a collided run is over; nothing after it is a valid sample

    const double d = (goal - p).norm();
    if (d < best_dist - 0.25) { best_dist = d; no_progress = 0; }
    else if (++no_progress >= stall_replans) { t.stalled = true; break; }
    if (d < 3.0) { t.reached = true; break; }
    if (t.collided) break;   // a collided run is over; nothing after it is a valid sample
  }
  t.progress = (goal - start).norm() - best_dist;
  t.plan_ms_mean = t.replans ? ms_sum / t.replans : 0.0;
  t.map_ms_mean = t.replans ? map_sum / t.replans : 0.0;
  if (t.min_width > 1e17) t.min_width = 0.0;
  return t;
}

struct Agg {
  int n = 0, reached = 0, stalled = 0, collided = 0, succeeded = 0;
  std::vector<double> ms_mean, ms_max, widths, progress_frac, map_mean, map_max;
  std::vector<double> realised;
  std::vector<double> width_all;   // pooled across trials
  long degen = 0, esc = 0, held = 0, fblock = 0, noleft = 0;   // absorbing-state plans and escapes, summed over trials
  std::vector<double> opt_all, cps_all;
  std::vector<std::array<double,9>> geo;   // +v_head,v_peak
  long first_try = 0, solves = 0;
  long replans = 0, failures = 0, corridors = 0, sub_hull = 0;
  std::vector<double> one_frac, speed;

  void add(const Trial & t, double total_dist)
  {
    ++n;
    reached += t.reached;
    stalled += t.stalled;
    collided += t.collided;
    succeeded += (t.reached && !t.collided) ? 1 : 0;
    ms_mean.push_back(t.plan_ms_mean);
    ms_max.push_back(t.plan_ms_max);
    map_mean.push_back(t.map_ms_mean);
    map_max.push_back(t.map_ms_max);
    first_try += t.first_try;
    solves += t.solves;
    for (std::size_t i = 0; i < t.straightness.size(); ++i) {
      geo.push_back({t.straightness[i], t.d_free[i], t.v_end[i],
                     t.n_cp[i], t.dur[i], t.arc_len[i], t.budget[i],
                     i < t.v_head.size() ? t.v_head[i] : 0.0,
                     i < t.v_peak.size() ? t.v_peak[i] : 0.0});
    }
    opt_all.insert(opt_all.end(), t.opt_ms.begin(), t.opt_ms.end());
    cps_all.insert(cps_all.end(), t.opt_cps.begin(), t.opt_cps.end());
    widths.push_back(t.min_width);
    width_all.insert(width_all.end(), t.all_widths.begin(), t.all_widths.end());
    progress_frac.push_back(total_dist > 0 ? t.progress / total_dist : 0.0);
    if (t.plans) {
      one_frac.push_back(static_cast<double>(t.one_corridor) / t.plans);
      speed.push_back(t.speed_sum / t.plans);
      if (t.flown_s > 1e-6) realised.push_back(t.flown_m / t.flown_s);
      degen += t.degenerate; esc += t.escapes;
      held += t.held_degen; fblock += t.force_blocked; noleft += t.no_poly_left;
    }
    replans += t.replans;
    failures += t.failures;
    corridors += t.corridors;
    sub_hull += t.sub_hull;
  }
};

// Mean and sample standard deviation, for a table that has to report spread and not just a
// median. Reported alongside the percentiles because they answer different questions: the median
// says what a typical replan costs, the std says how much a reviewer should trust a single run.
double avg(const std::vector<double> & v)
{
  if (v.empty()) return 0.0;
  double s = 0.0;
  for (double x : v) s += x;
  return s / static_cast<double>(v.size());
}
double sd(const std::vector<double> & v)
{
  if (v.size() < 2) return 0.0;
  const double m = avg(v);
  double s2 = 0.0;
  for (double x : v) s2 += (x - m) * (x - m);
  return std::sqrt(s2 / static_cast<double>(v.size() - 1));
}

double pct(std::vector<double> v, double p)
{
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[static_cast<std::size_t>(p * (v.size() - 1))];
}

std::vector<std::array<double,9>> g_geo;

void report(const char * label, Agg & a)
{
  std::printf("  %-10s  %4d  %5.1f%%  %5.1f%%  %7.1f%%   %6.2f %6.2f   %6.3f %6.3f   %5.1f%%\n",
              label, a.n, 100.0 * a.succeeded / std::max(1, a.n),
              100.0 * a.stalled / std::max(1, a.n),
              100.0 * pct(a.progress_frac, 0.5),
              pct(a.ms_mean, 0.5), pct(a.ms_max, 0.9),
              pct(a.widths, 0.5), pct(a.widths, 0.1),
              100.0 * a.sub_hull / std::max(1L, a.corridors));
  std::printf("  %-10s  %s  goal-reached %5.1f%%   collided %5.1f%%\n",
              "", "         ", 100.0 * a.reached / std::max(1, a.n),
              100.0 * a.collided / std::max(1, a.n));
  std::printf("  %-10s  %s  1-corridor plans %5.1f%%   commanded speed %.2f m/s"
              "   MAP %.1f/%.1f ms (med/p90)   cycle %.1f ms\n",
              "", "         ", 100.0 * pct(a.one_frac, 0.5), pct(a.speed, 0.5),
              pct(a.map_mean, 0.5), pct(a.map_max, 0.9),
              pct(a.map_mean, 0.5) + pct(a.ms_mean, 0.5));
  if (!a.realised.empty()) {
    std::printf("  %-10s  %s  REALISED speed %.2f m/s  (plan mean %.2f -- the gap is the share of\n"
                "  %-10s  %s  each plan that is thrown away at the next replan)\n",
                "", "         ", pct(a.realised, 0.5), pct(a.speed, 0.5), "", "         ");
  }
  if (a.degen) {
    std::printf("  %-10s  %s  ABSORBING-STATE plans %ld, escaped by lifting the horizon %ld (%.0f%%)\n",
                "", "         ", a.degen, a.esc, 100.0 * a.esc / std::max(1L, a.degen));
    std::printf("  %-10s  %s    held instead of accepted %ld | blocked by stale-map force %ld"
                " | held route already exhausted %ld\n",
                "", "         ", a.held, a.fblock, a.noleft);
  }
  std::printf("  %-10s  %s  OPTIMISER ALONE (OSQP, n=%zu)  med %.2f  p90 %.2f  p99 %.2f  max %.2f ms"
              "   at med %.0f control points\n",
              "", "         ", a.opt_all.size(),
              pct(a.opt_all, 0.50), pct(a.opt_all, 0.90), pct(a.opt_all, 0.99),
              a.opt_all.empty() ? 0.0 : *std::max_element(a.opt_all.begin(), a.opt_all.end()),
              pct(a.cps_all, 0.50));
  std::printf("  %-10s  %s  first-attempt solves %5.1f%%\n", "", "         ",
              100.0 * a.first_try / std::max(1L, a.solves));

  // MEAN +/- SD BLOCK. The percentiles above describe the distribution of a single replan; these
  // describe how much a reviewer should trust one run. Both belong in a results table -- a median
  // with no spread hides the 2.3x run-to-run variation measured in closed loop.
  std::printf("  %-10s  %s  mean +/- sd:  OSQP %.3f +/- %.3f ms | replan total %.2f +/- %.2f ms\n",
              "", "         ", avg(a.opt_all), sd(a.opt_all),
              avg(a.ms_mean), sd(a.ms_mean));
  std::printf("  %-10s  %s               map update %.2f +/- %.2f ms | control points %.2f +/- %.2f\n",
              "", "         ", avg(a.map_mean), sd(a.map_mean),
              avg(a.cps_all), sd(a.cps_all));
  if (!a.speed.empty()) {
    std::printf("  %-10s  %s               plan-mean speed %.2f +/- %.2f m/s",
                "", "         ", avg(a.speed), sd(a.speed));
    if (!a.realised.empty())
      std::printf(" | realised %.2f +/- %.2f m/s", avg(a.realised), sd(a.realised));
    std::printf("\n");
  }
  if (!a.width_all.empty()) {
    // Distribution over ALL corridors. The zero-w column above is the share at literally zero:
    // such a corridor cannot contain the vehicle, so the QP is solving a degenerate geometry --
    // a FEASIBILITY pressure, not necessarily a safety failure, since the trajectory tracks the
    // centre of mass and a zero-width corridor with a straight trajectory inside it is flyable.
    std::printf("  %-10s  %s               corridor half-width over %zu corridors:"
                " med %.3f  p10 %.3f  mean %.3f +/- %.3f m\n",
                "", "         ", a.width_all.size(), pct(a.width_all, 0.5),
                pct(a.width_all, 0.10), avg(a.width_all), sd(a.width_all));
  }
  g_geo.insert(g_geo.end(), a.geo.begin(), a.geo.end());
}

}  // namespace

int main(int argc, char ** argv)
{
  int trials = 100, max_steps = 45, seed0 = 0;
  double density = 0.20, v_max = 5.0;
  std::string ab, forest_dir, forest_prefix = "forest_d15";
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--trials") && i + 1 < argc) trials = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--density") && i + 1 < argc) density = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--vmax") && i + 1 < argc) v_max = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--steps") && i + 1 < argc) max_steps = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) seed0 = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--ab") && i + 1 < argc) ab = argv[++i];
    else if (!std::strcmp(argv[i], "--forests") && i + 1 < argc) forest_dir = argv[++i];
    else if (!std::strcmp(argv[i], "--prefix") && i + 1 < argc) forest_prefix = argv[++i];
    else if (!std::strcmp(argv[i], "--hull") && i + 1 < argc) g_hull = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--margin") && i + 1 < argc) g_margin = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--amax") && i + 1 < argc) g_a_max = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--horizon") && i + 1 < argc) g_horizon = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--sfc") && i + 1 < argc) g_sfc_cap = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--advance") && i + 1 < argc) g_advance = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--knotscale")) g_knot_scale = true;
    else if (!std::strcmp(argv[i], "--timed")) { g_timed = true; if (max_steps == 45) max_steps = 1200; }
  }

  std::printf("Offline Monte-Carlo -- real front end + OSQP, no ROS / simulator / estimator.\n");
  std::printf("%d trials, density %.0f%%, v_max %.1f, max_sfc 3\n", trials, 100 * density, v_max);
  std::printf("footprint: r_drone %.2f m + r_margin %.2f m = %.2f m inflation",
              g_hull, g_margin, g_hull + g_margin);
  if (std::abs(g_hull - 0.1) < 1e-9 && std::abs(g_margin - 0.1) < 1e-9) {
    std::printf("   (SANDO Table III, Static Forest)\n");
  } else {
    std::printf("\n");
  }
  if (!ab.empty()) {
    std::printf("A/B on '%s' -- BOTH ARMS SEE IDENTICAL FORESTS AND ENDPOINTS (same seeds),\n"
                "so the difference is the feature and not the draw.\n", ab.c_str());
  }
  std::printf("\n  %-10s  %4s  %6s  %6s  %8s   %6s %6s   %6s %6s   %6s\n",
              "arm", "n", "R_succ", "stall", "progress", "ms50", "ms90", "w50", "w10", "zero-w");
  std::printf("  %s\n", std::string(78, '-').c_str());

  // --- margin / relief sweep --------------------------------------------------------------
  // The corridor keeps (inflation + SAT corner term) clear of obstacles and only gives the
  // margin back when the relief clamp is off. Both knobs move the same quantity from opposite
  // ends, so sweep them together: raising margin widens the no-go region A* must avoid (fewer
  // routes) while relief decides whether the BUILDER gets that width back (wider boxes).
  // --- heat sweep -------------------------------------------------------------------------
  // The margin/relief sweep moves the corridor's FACES but cannot move its AXIS, and the axis
  // is what A* chooses. If corridors are thin because the search hugs obstacle boundaries, only
  // a clearance preference in the SEARCH can fix it -- that is what the heat field is.
  // Bounded by the guidance invariant: 1 + unknown_cost + heat_weight < astar_heuristic_weight.
  // --- degenerate-corridor repair sweep ----------------------------------------------------
  if (ab == "repair") {
    std::printf("\n  A zero-width corridor is INFEASIBLE, not unsafe: the box is already inset by\n");
    std::printf("  hull + margin, so a trajectory on its wall still keeps the centre of mass\n");
    std::printf("  clear. What breaks is the QP -- every control point must lie inside, so a\n");
    std::printf("  collapsed axis forbids curvature. Repair merges the offending segment into a\n");
    std::printf("  neighbour and keeps the merge only if it removes a degeneracy.\n\n");
    std::printf("  %-10s  %4s  %6s  %6s  %8s   %6s %6s   %6s %6s   %6s\n",
                "arm", "n", "R_succ", "stall", "progress", "ms50", "ms90", "w50", "w10", "zero-w");
    std::printf("  %s\n", std::string(78, '-').c_str());
    for (double rw : {0.0, 0.05, 0.15, 0.30}) {
      Agg a;
      for (int i = 0; i < trials; ++i) {
        std::mt19937 rng(static_cast<unsigned>(seed0 + i));
        const auto ts = make_forest(rng, density);
        const Eigen::Vector3d start(0.0, 0.0, kAlt);
        std::uniform_real_distribution<double> gy(-8.0, 8.0);
        const Eigen::Vector3d goal(kFieldX + 5.0, gy(rng), kAlt);
        a.add(rollout(ts, base_config(v_max, false, g_margin, true, rw), start, goal, max_steps),
              (goal - start).norm());
      }
      char lbl[32];
      if (rw <= 0.0) std::snprintf(lbl, sizeof lbl, "off");
      else std::snprintf(lbl, sizeof lbl, "repair %.2f", rw);
      report(lbl, a);
    }
    std::printf("\n  arm = sfc_min_usable_width; a corridor narrower than this is merged into a\n");
    std::printf("  neighbour. zero-w = share of ALL corridors with a collapsed axis.\n");
    return 0;
  }

  // --- unknown-space traversal cost vs COLLISIONS -------------------------------------------
  // A* crosses UNKNOWN cells for unknown_traversal_cost per step rather than refusing them.
  // That is deliberate -- hard-blocking unknown space stops the search routing around a corner
  // -- but it also lets the planner commit into regions it has not seen, and with a small
  // footprint there is little margin for the map being wrong. At 10% density collisions were
  // 27.5% of all runs, the single largest failure mode, so this sweeps the one parameter that
  // prices that risk.
  //
  // BOUNDED BY THE GUIDANCE INVARIANT: 1 + unknown + clearance + heat < astar_heuristic_weight.
  // At heat_weight 0.6 and heuristic weight 3.0 the ceiling is 1.4; past it A* stops preferring
  // the goal and floods outward, which is a worse failure than either collisions or stalls.
  // --- unknown-space inflation vs COLLISIONS -------------------------------------------------
  // Previously measured and reverted -- but that was BEFORE collisions were checked, so
  // "failures" meant stalls only. Collisions are now the dominant failure mode (27.5% at 10%
  // density, 68% at 20%), so the trade has to be re-judged with both halves visible.
  // --- commitment distance: is the harness overstating collisions? --------------------------
  // --- time_budget_factor ---------------------------------------------------------------------
  // Pads the kinematic time budget: T = factor * max(L/v_max, 2*sqrt(L/a_max)). It exists to buy
  // first-attempt QP success (measured 30% -> 60% when it was introduced), and it scales
  // delivered duration DIRECTLY -- so 1.2 -> 1.0 is a guaranteed ~20% on commanded speed if the
  // retries do not eat it back. Retries are the whole question, so first-attempt success and
  // opt_time are reported alongside.
  // --- gate alignment margin vs FLIP-FLOPPING -----------------------------------------------
  // The override that lets a "better aimed" candidate bypass the gate exists so one bad plan
  // cannot self-defend. At a junction with two comparably goal-ward branches it becomes the
  // leak: the vehicle shifts each cycle, which flips the scoring, and the planner alternates.
  // Measured in sim across ten medium seeds -- passing runs used the override 0-43 times, the
  // wedged one used it 213.
  if (ab == "align") {
    std::printf("\n  Higher margin = a candidate must be MEANINGFULLY better aimed to bypass the\n");
    std::printf("  gate, so genuine ties are declined and held. Too high and one bad plan is\n");
    std::printf("  self-defending again, which is what the override was added to prevent.\n\n");
    std::printf("  %-10s  %4s  %6s  %6s  %8s   %6s %6s   %6s %6s   %6s\n",
                "margin", "n", "R_succ", "stall", "progress", "ms50", "ms90", "w50", "w10", "zero-w");
    std::printf("  %s\n", std::string(80, '-').c_str());
    for (double m : {0.05, 0.10, 0.25, 0.50, 1.00}) {
      Agg a;
      for (int i = 0; i < trials; ++i) {
        std::mt19937 rng(static_cast<unsigned>(seed0 + i));
        const auto ts = make_forest(rng, density);
        const Eigen::Vector3d start(0.0, 0.0, kAlt);
        std::uniform_real_distribution<double> gy(-8.0, 8.0);
        const Eigen::Vector3d goal(kFieldX + 5.0, gy(rng), kAlt);
        a.add(rollout(ts, base_config(v_max, false), start, goal, max_steps,
                      g_margin, 25.0, false, 5.0, m),
              (goal - start).norm());
      }
      char lbl[32]; std::snprintf(lbl, sizeof lbl, "%.2f", m);
      report(lbl, a);
    }
    std::printf("\n  NOTE: the harness gate port is unverified -- it scored BELOW the ungated\n");
    std::printf("  baseline even after two fixes -- so read this sweep for the TREND across\n");
    std::printf("  arms, not for absolute R_succ.\n");
    return 0;
  }

  if (ab == "budget") {
    std::printf("\n  lower factor = shorter trajectories = faster, PAID FOR in QP retries.\n");
    std::printf("  Watch first-attempt solves and the OSQP tail, not just commanded speed.\n\n");
    std::printf("  %-10s  %4s  %6s  %6s  %8s   %6s %6s   %6s %6s   %6s\n",
                "factor", "n", "R_succ", "stall", "progress", "ms50", "ms90", "w50", "w10", "zero-w");
    std::printf("  %s\n", std::string(80, '-').c_str());
    for (double f : {1.40, 1.20, 1.10, 1.00, 0.90}) {
      Agg a;
      for (int i = 0; i < trials; ++i) {
        std::mt19937 rng(static_cast<unsigned>(seed0 + i));
        const auto ts = make_forest(rng, density);
        const Eigen::Vector3d start(0.0, 0.0, kAlt);
        std::uniform_real_distribution<double> gy(-8.0, 8.0);
        const Eigen::Vector3d goal(kFieldX + 5.0, gy(rng), kAlt);
        FrontEndConfig c = base_config(v_max, false);
        c.time_budget_factor = f;
        a.add(rollout(ts, c, start, goal, max_steps), (goal - start).norm());
      }
      char lbl[32]; std::snprintf(lbl, sizeof lbl, "%.2f", f);
      report(lbl, a);
    }
    std::printf("\n  A factor below 1.0 asks for less time than the kinematics need, so it should\n");
    std::printf("  fail outright -- included as the control that proves the knob is live.\n");
    return 0;
  }

  if (ab == "advance") {
    std::printf("\n  Flight replans at 10 Hz and flies ~0.25-0.5 m between scans. If collisions\n");
    std::printf("  fall as the commitment shrinks, they are an artefact of the rollout.\n\n");
    std::printf("  %-12s  %4s  %6s  %6s  %8s   %6s %6s   %6s %6s   %6s\n",
                "advance", "n", "R_succ", "stall", "progress", "ms50", "ms90", "w50", "w10", "zero-w");
    std::printf("  %s\n", std::string(80, '-').c_str());
    for (double adv : {5.0, 2.0, 1.0, 0.5}) {
      Agg a;
      for (int i = 0; i < trials; ++i) {
        std::mt19937 rng(static_cast<unsigned>(seed0 + i));
        const auto ts = make_forest(rng, density);
        const Eigen::Vector3d start(0.0, 0.0, kAlt);
        std::uniform_real_distribution<double> gy(-8.0, 8.0);
        const Eigen::Vector3d goal(kFieldX + 5.0, gy(rng), kAlt);
        a.add(rollout(ts, base_config(v_max, false), start, goal,
                      static_cast<int>(max_steps * 5.0 / adv),   // same ground, more steps
                      g_margin, 25.0, false, adv),
              (goal - start).norm());
      }
      char lbl[32]; std::snprintf(lbl, sizeof lbl, "%.1f m", adv);
      report(lbl, a);
    }
    std::printf("\n  max_steps is scaled with the advance so every arm can cover the same ground.\n");
    return 0;
  }

  if (ab == "frontier") {
    std::printf("\n  frontier ON bars corridors from unswept space (SANDO: 'unknown space\n");
    std::printf("  inflation: on'). Watch COLLIDED against stall -- the question is the trade.\n\n");
    std::printf("  %-12s  %4s  %6s  %6s  %8s   %6s %6s   %6s %6s   %6s\n",
                "arm", "n", "R_succ", "stall", "progress", "ms50", "ms90", "w50", "w10", "zero-w");
    std::printf("  %s\n", std::string(80, '-').c_str());
    for (double d : {0.10, 0.20}) {
      for (bool fr : {false, true}) {
        Agg a;
        for (int i = 0; i < trials; ++i) {
          std::mt19937 rng(static_cast<unsigned>(seed0 + i));
          const auto ts = make_forest(rng, d);
          const Eigen::Vector3d start(0.0, 0.0, kAlt);
          std::uniform_real_distribution<double> gy(-8.0, 8.0);
          const Eigen::Vector3d goal(kFieldX + 5.0, gy(rng), kAlt);
          a.add(rollout(ts, base_config(v_max, false), start, goal, max_steps, g_margin, 25.0, fr),
                (goal - start).norm());
        }
        char lbl[32];
        std::snprintf(lbl, sizeof lbl, "d%.0f%% %s", 100 * d, fr ? "ON " : "off");
        report(lbl, a);
      }
    }
    std::printf("\n  A corridor barred from unswept space cannot enclose an unseen trunk, but it\n");
    std::printf("  also collapses wherever the frontier crosses the path -- which is what made\n");
    std::printf("  this a revert the first time it was tried.\n");
    return 0;
  }

  if (ab == "unknown") {
    std::printf("\n  ceiling from the guidance invariant: 1 + u + %.2f < %.1f  ->  u < %.2f\n\n",
                0.6, 3.0, 3.0 - 1.0 - 0.6);
    std::printf("  %-10s  %4s  %6s  %6s  %8s   %6s %6s   %6s %6s   %6s\n",
                "unknown", "n", "R_succ", "stall", "progress", "ms50", "ms90", "w50", "w10", "zero-w");
    std::printf("  %s\n", std::string(78, '-').c_str());
    for (double u : {0.5, 0.8, 1.1, 1.35}) {
      Agg a;
      for (int i = 0; i < trials; ++i) {
        std::mt19937 rng(static_cast<unsigned>(seed0 + i));
        const auto ts = make_forest(rng, density);
        const Eigen::Vector3d start(0.0, 0.0, kAlt);
        std::uniform_real_distribution<double> gy(-8.0, 8.0);
        const Eigen::Vector3d goal(kFieldX + 5.0, gy(rng), kAlt);
        FrontEndConfig c = base_config(v_max, false);
        c.unknown_traversal_cost = u;
        a.add(rollout(ts, c, start, goal, max_steps), (goal - start).norm());
      }
      char lbl[32]; std::snprintf(lbl, sizeof lbl, "u=%.2f", u);
      report(lbl, a);
    }
    std::printf("\n  Read the COLLIDED line, not R_succ alone: a higher cost should trade\n");
    std::printf("  collisions for stalls, and whether that is a good trade is the question.\n");
    return 0;
  }

  if (ab == "heat") {
    std::printf("\n  heat pushes A* off obstacle boundaries; the margin/relief knobs cannot.\n");
    std::printf("  invariant: 1 + %.2f + heat_weight < %.1f  ->  heat_weight < %.2f\n\n",
                0.5, 3.0, 3.0 - 1.0 - 0.5);
    std::printf("  %-10s  %4s  %6s  %6s  %8s   %6s %6s   %6s %6s   %6s\n",
                "arm", "n", "R_succ", "stall", "progress", "ms50", "ms90", "w50", "w10", "zero-w");
    std::printf("  %s\n", std::string(78, '-').c_str());
    for (double hr : {1.0, 2.0, 3.5}) {
      for (double hw : {0.6, 1.4}) {
        Agg a;
        for (int i = 0; i < trials; ++i) {
          std::mt19937 rng(static_cast<unsigned>(seed0 + i));
          const auto ts = make_forest(rng, density);
          const Eigen::Vector3d start(0.0, 0.0, kAlt);
          std::uniform_real_distribution<double> gy(-8.0, 8.0);
          const Eigen::Vector3d goal(kFieldX + 5.0, gy(rng), kAlt);
          FrontEndConfig c = base_config(v_max, false);
          c.heat_radius = hr; c.heat_weight = hw;
          a.add(rollout(ts, c, start, goal, max_steps), (goal - start).norm());
        }
        char lbl[32]; std::snprintf(lbl, sizeof lbl, "r%.1f w%.1f", hr, hw);
        report(lbl, a);
      }
    }
    std::printf("\n  r = heat_radius (m), w = heat_weight. If nothing moves sub-hull, the search\n");
    std::printf("  cannot be steered away from obstacles by this mechanism at these settings.\n");
    return 0;
  }

  // --- density sweep: where does the planner actually work? --------------------------------
  // --- map window vs planning horizon -------------------------------------------------------
  // Two requirements pull against each other:
  //   horizon < window, or the sub-goal is projected into space the map has already pruned
  //                     (measured: the goal cell was `unknown` in 137 of 137 short paths);
  //   horizon LARGE,    or the corridor chain is short, and since delivered duration is floored
  //                     at (control points - degree) seconds while distance is not, a short
  //                     chain is a crawl by arithmetic.
  // Moving the WINDOW satisfies both, at map-update CPU -- which is what this measures.
  if (ab == "window") {
    std::printf("\n  cycle = map update + planning, against a 100 ms replan period. The map\n");
    std::printf("  update is the half that starves the 50 Hz dispatch loop.\n\n");
    std::printf("  %-10s  %4s  %6s  %6s  %8s   %6s %6s   %6s %6s   %6s\n",
                "win/horiz", "n", "R_succ", "stall", "progress", "ms50", "ms90", "w50", "w10", "zero-w");
    std::printf("  %s\n", std::string(78, '-').c_str());
    const double combos[][2] = {{15.0, 10.0}, {15.0, 15.0}, {25.0, 15.0}, {25.0, 20.0}};
    for (const auto & c : combos) {
      const double win = c[0], hor = c[1];
      Agg a;
      for (int i = 0; i < trials; ++i) {
        std::mt19937 rng(static_cast<unsigned>(seed0 + i));
        const auto ts = make_forest(rng, density);
        const Eigen::Vector3d start(0.0, 0.0, kAlt);
        std::uniform_real_distribution<double> gy(-8.0, 8.0);
        const Eigen::Vector3d goal(kFieldX + 5.0, gy(rng), kAlt);
        FrontEndConfig cfg = base_config(v_max, false);
        cfg.planning_horizon = hor;
        a.add(rollout(ts, cfg, start, goal, max_steps, g_margin, win), (goal - start).norm());
      }
      char lbl[32]; std::snprintf(lbl, sizeof lbl, "%.0f/%.0f", win, hor);
      report(lbl, a);
    }
    std::printf("\n  15/10 is the current flight config; 25/15 is the proposal; 15/15 is the\n");
    std::printf("  configuration that put the sub-goal on the map frontier every cycle.\n");
    return 0;
  }

  if (ab == "density") {
    std::printf("\n  Success against forest density. This is the operating envelope: the density\n");
    std::printf("  at which the planner is reliable is a RESULT, not a tuning target.\n\n");
    std::printf("  %-10s  %4s  %6s  %6s  %8s   %6s %6s   %6s %6s   %6s\n",
                "density", "n", "R_succ", "stall", "progress", "ms50", "ms90", "w50", "w10", "zero-w");
    std::printf("  %s\n", std::string(78, '-').c_str());
    for (double d : {0.05, 0.08, 0.10, 0.13, 0.16, 0.20}) {
      Agg a;
      for (int i = 0; i < trials; ++i) {
        std::mt19937 rng(static_cast<unsigned>(seed0 + i));
        const auto ts = make_forest(rng, d);
        const Eigen::Vector3d start(0.0, 0.0, kAlt);
        std::uniform_real_distribution<double> gy(-8.0, 8.0);
        const Eigen::Vector3d goal(kFieldX + 5.0, gy(rng), kAlt);
        a.add(rollout(ts, base_config(v_max, false), start, goal, max_steps),
              (goal - start).norm());
      }
      char lbl[32]; std::snprintf(lbl, sizeof lbl, "%.0f%%", 100 * d);
      report(lbl, a);
    }
    std::printf("\n  Report the CURVE, not one point: a single density invites 'why not the one\n");
    std::printf("  SANDO used?', while a curve answers it and states the envelope outright.\n");
    return 0;
  }

  if (ab == "margin") {
    std::printf("\n  clamp_relief=true keeps obstacle_margin as cover for estimation error;\n");
    std::printf("  false hands it back to the corridor, guaranteeing only the bare hull.\n\n");
    std::printf("  %-10s  %4s  %6s  %6s  %8s   %6s %6s   %6s %6s   %6s\n",
                "arm", "n", "R_succ", "stall", "progress", "ms50", "ms90", "w50", "w10", "zero-w");
    std::printf("  %s\n", std::string(78, '-').c_str());
    for (double margin : {0.10, 0.20, 0.30}) {
      for (bool clamp : {true, false}) {
        Agg a;
        for (int i = 0; i < trials; ++i) {
          std::mt19937 rng(static_cast<unsigned>(seed0 + i));
          const auto ts = make_forest(rng, density);
          const Eigen::Vector3d start(0.0, 0.0, kAlt);
          std::uniform_real_distribution<double> gy(-8.0, 8.0);
          const Eigen::Vector3d goal(kFieldX + 5.0, gy(rng), kAlt);
          a.add(rollout(ts, base_config(v_max, false, margin, clamp), start, goal, max_steps,
                        margin), (goal - start).norm());
        }
        char lbl[32];
        std::snprintf(lbl, sizeof lbl, "m%.2f %s", margin, clamp ? "clmp" : "give");
        report(lbl, a);
      }
    }
    std::printf("\n  m = obstacle_margin; clmp = relief clamped (safe), give = margin returned.\n");
    std::printf("  sub-hull is measured against the %.3f m hull in every arm, so the arms are\n", g_hull);
    std::printf("  directly comparable even though the map inflation differs.\n");
    return 0;
  }

  // --- real environments -------------------------------------------------------------------
  if (!forest_dir.empty()) {
    std::printf("\n  Environments loaded from %s -- the SAME files the simulator flies, so these\n",
                forest_dir.c_str());
    std::printf("  numbers and the closed-loop ones describe one set of worlds.\n\n");
    std::printf("  %-14s  %4s  %6s  %6s  %8s   %6s %6s   %6s\n",
                "environment", "n", "R_succ", "stall", "progress", "ms50", "ms90", "zero-w");
    std::printf("  %s\n", std::string(78, '-').c_str());
    Agg all;
    for (int seed = 0; seed < trials; ++seed) {
      char path[512];
      std::snprintf(path, sizeof path, "%s/%s_s%d.yaml", forest_dir.c_str(),
                    forest_prefix.c_str(), seed);
      const auto ts = load_forest(path);
      if (ts.empty()) { std::printf("  (no trunks in %s -- skipped)\n", path); continue; }
      // THE MISSION'S OWN ENDPOINTS, not a random draw: params/forest_mission.yaml flies
      // (0,0,-3) -> (105,0,-3). Randomising the goal here would make the offline and
      // closed-loop numbers describe different problems on the same trees, which defeats the
      // reason for loading these files at all.
      const Eigen::Vector3d start(0.0, 0.0, kAlt);
      const Eigen::Vector3d goal(105.0, 0.0, kAlt);
      const double adv = (g_advance > 0.0) ? g_advance : 5.0;
      const int steps = (g_advance > 0.0)
                            ? static_cast<int>(max_steps * 5.0 / adv)   // same ground, more steps
                            : max_steps;
      // window 15.0 MATCHES map_window_radius in params/forest_benchmark.yaml. It was 25.0 here,
      // i.e. the harness remembered 10 m more of the world than the vehicle does.
      const Trial t = rollout(ts, base_config(v_max, false), start, goal, steps,
                              g_margin, 15.0, false, adv);
      Agg a;
      a.add(t, (goal - start).norm());
      all.add(t, (goal - start).norm());
      char lbl[32]; std::snprintf(lbl, sizeof lbl, "s%d", seed);
      report(lbl, a);
    }
    std::printf("  %s\n", std::string(78, '-').c_str());
    report("ALL", all);
    return 0;
  }

  Agg on, off;
  for (int i = 0; i < trials; ++i) {
    std::mt19937 rng(static_cast<unsigned>(seed0 + i));
    const auto ts = make_forest(rng, density);
    const Eigen::Vector3d start(0.0, 0.0, kAlt);
    std::uniform_real_distribution<double> gy(-8.0, 8.0);
    const Eigen::Vector3d goal(kFieldX + 5.0, gy(rng), kAlt);
    const double dist = (goal - start).norm();

    on.add(rollout(ts, base_config(v_max, true), start, goal, max_steps), dist);
    if (ab == "collinear") {
      off.add(rollout(ts, base_config(v_max, false), start, goal, max_steps), dist);
    }
  }

  if (ab == "collinear") {
    report("merge ON", on);
    report("merge OFF", off);
  } else {
    report("baseline", on);
  }

  // --- straight vs turning -------------------------------------------------------------------
  if (!g_geo.empty()) {
    std::printf("\n  TERMINAL SPEED BY CHAIN STRAIGHTNESS  (chord/arc; 1.00 = dead straight)\n");
    std::printf("  A plan ends at min(sqrt(2*a_max*0.6*d_free), 0.9*v_max). d_free is the last\n");
    std::printf("  corridor's forward run-out, clipped by whatever lies ahead of it.\n\n");
    std::printf("  %-14s %7s %8s %8s %8s %9s %9s %9s\n", "straightness", "plans",
                "n_cp", "dur s", "v_mean", "v_HEAD", "v_peak", "head/mean");
    std::printf("  %s\n", std::string(50, '-').c_str());
    const double edges[] = {0.0, 0.85, 0.95, 0.99, 1.001};
    const char * lbl[] = {"<0.85 turning", "0.85-0.95", "0.95-0.99", ">0.99 straight"};
    for (int b = 0; b < 4; ++b) {
      std::vector<double> df, ve;
      for (const auto & g : g_geo) {
        if (g[0] >= edges[b] && g[0] < edges[b+1]) { df.push_back(g[1]); ve.push_back(g[2]); }
      }
      std::vector<double> nc, du, ar, bu, sp;
      for (const auto & g : g_geo) {
        if (g[0] >= edges[b] && g[0] < edges[b+1]) {
          nc.push_back(g[3]); du.push_back(g[4]); ar.push_back(g[5]); bu.push_back(g[6]);
          if (g[4] > 1e-6) sp.push_back(g[5] / g[4]);
        }
      }
      if (df.empty()) { std::printf("  %-14s %7d\n", lbl[b], 0); continue; }
      std::vector<double> hd, pk;
      for (const auto & g : g_geo) {
        if (g[0] >= edges[b] && g[0] < edges[b+1]) { hd.push_back(g[7]); pk.push_back(g[8]); }
      }
      const double vm = sp.empty() ? 0.0 : pct(sp, 0.5);
      const double vh = hd.empty() ? 0.0 : pct(hd, 0.5);
      std::printf("  %-14s %7zu %8.1f %8.2f %8.2f %9.2f %9.2f %9.2f\n", lbl[b], df.size(),
                  pct(nc, 0.5), pct(du, 0.5), vm, vh,
                  pk.empty() ? 0.0 : pct(pk, 0.5), vm > 1e-6 ? vh / vm : 0.0);
    }
    // The floor question: how often is delivered duration ABOVE what the kinematics asked?
    {
      long bound = 0, single = 0, pinned = 0; std::vector<double> waste;
      for (const auto & g : g_geo) {
        if (g[4] > g[6] + 1e-6) { ++bound; waste.push_back(g[4] - g[6]); }
        if (g[3] <= 8.5) ++single;
        if (g[2] < 0.1) ++pinned;
      }
      std::printf("\n  DURATION FLOOR: %ld of %zu plans (%.1f%%) were allocated MORE time than the\n",
                  bound, g_geo.size(), 100.0 * bound / std::max<std::size_t>(1, g_geo.size()));
      std::printf("  kinematics asked for (median excess %.2f s). %ld plans (%.1f%%) sit at the\n",
                  waste.empty() ? 0.0 : pct(waste, 0.5), single,
                  100.0 * single / std::max<std::size_t>(1, g_geo.size()));
      std::printf("  8-control-point minimum. Removing the floor can only help where it BINDS.\n");
      std::printf("  PINNED TO REST: %ld of %zu plans (%.1f%%) ended at v_end < 0.1 m/s.\n",
                  pinned, g_geo.size(), 100.0 * pinned / std::max<std::size_t>(1, g_geo.size()));
    }

    std::printf("\n  If v_end collapses toward 0 on the straight rows and sits near 4.5 on the\n");
    std::printf("  turning rows, the braking bound is what slows straight-line flight.\n");
  }

  std::printf("\n  R_succ   = reached the goal AND never came within one hull radius of a trunk,\n");
  std::printf("             matching SANDO's definition of success.\n");
  std::printf("  coll     = share of runs that touched a trunk (shown per arm above)\n");
  std::printf("  stall    = 12 consecutive replans without closing on the goal\n");
  std::printf("  progress = median fraction of the start-to-goal distance actually closed\n");
  std::printf("  ms50/90  = per-replan planning time, median and p90 (budget 100 ms @ 10 Hz)\n");
  std::printf("  w50/w10  = narrowest corridor half-width per trial, median and p10\n");
  std::printf("  sub-hull = share of ALL corridors narrower than the %.3f m hull; such a\n", g_hull);
  std::printf("             corridor cannot contain the vehicle, so the collision-freedom\n");
  std::printf("             argument does not hold for it even when the QP succeeds.\n");
  return 0;
}
