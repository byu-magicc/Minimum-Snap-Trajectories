#include "front_end.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace trajectory_planner {

// ==========================================
// Constructor
// ==========================================
FrontEndSFC::FrontEndSFC(const FrontEndConfig &config,
                         std::shared_ptr<mapping::SparseVoxelGrid> grid)
    : config_(config), discrete_grid_(std::move(grid)),
      sfc_manager_(discrete_grid_, 0.0, 0.0, config.voxel_resolution),
      static_sfc_manager_(config.sfc_height, config.sfc_width) {
  // Reconfigure the asymmetric SFC manager with the correct parameters
  double max_cp_drift =
      (config_.sfc_width / 2.0) - config_.drone_physical_radius;
  sfc_manager_ = sfc::AsymmetricSFCManager(discrete_grid_, config_.drone_physical_radius,
                                max_cp_drift, config_.voxel_resolution, config_.sfc_clamp_relief);

  // The RingBufferGrid that used to be built here was WRITE-ONLY -- nothing
  // ever read it -- while costing a dense (map_bounds/res)^3 allocation plus a
  // full copy of the inflated set into it on EVERY replan, at 10 Hz. Removed.
  // If a dense grid is wanted later, add it back with a reader.

  // Build the A* planner using the existing VoxelGridData interface.
  // astar_grid_ is a member (not a local) because AStarSFCPlanner stores it by
  // const reference, which must stay valid for the lifetime of astar_planner_.
  // Borrow the grid's sets rather than copying them. The planner is
  // reconstructed every replan, so copying tens of thousands of cells here cost
  // milliseconds per cycle and got worse the longer the vehicle flew.
  // discrete_grid_ is a shared_ptr held by this object, so the referenced data
  // outlives astar_planner_.
  astar_grid_.voxel_resolution = config_.voxel_resolution;
  astar_grid_.occupied_voxels_inflated =
      &discrete_grid_->get_occupied_inflated();
  astar_grid_.free_voxels = &discrete_grid_->get_free();
  astar_grid_.continuous_inflated_bounds =
      &discrete_grid_->get_continuous_inflated_bounds();
  astar_grid_.heat = &discrete_grid_->get_heat();

  // The field is derived from occupancy, so the grid owns building it; tell it how wide.
  // NOTE for anyone testing this offline: set_heat_radius only MARKS the grid dirty. The
  // rebuild runs inside update_from_scan, so a harness that loads a map once and never scans
  // again gets an EMPTY heat field and A* silently hugs obstacle boundaries. Flight is fine --
  // scans arrive at 10 Hz -- but it makes offline corridor-width measurements meaningless
  // unless the radius is set BEFORE the map is loaded.
  discrete_grid_->set_heat_radius(config_.heat_radius);

  // Guard the invariant that makes A* work at all. If the worst-case step
  // multiplier reaches the heuristic weight, stepping toward the goal stops
  // reducing f and the search floods outward instead of advancing -- previously
  // observed as a full node budget spent returning a 2-cell path in open air.
  // Warn rather than silently mis-plan.
  {
    const double m_max = 1.0 + std::max(0.0, config_.unknown_traversal_cost) +
                         std::max(0.0, config_.astar_clearance_cost) +
                         std::max(0.0, config_.heat_weight);
    if (m_max >= config_.astar_heuristic_weight) {
      std::cout
          << "[Front-End] WARNING: A* step multiplier " << m_max
          << " >= heuristic weight " << config_.astar_heuristic_weight
          << ". The search will stop preferring the goal and flood. Lower "
             "unknown_traversal_cost / astar_clearance_cost / heat_weight, or "
             "raise "
             "astar_heuristic_weight."
          << std::endl;
    }
  }

  // Initial geofence. The horizontal half-extents are replaced per-search in
  // get_corridors_astar() so the box follows the vehicle; only the VERTICAL
  // limits are a real world constraint (how high/low the vehicle may plan) and
  // stay absolute.
  //
  // These were previously fixed here for the whole mission, centred on the
  // world origin, which turned a search bound into a MISSION bound: at
  // map_bounds 100 m and 0.5 m voxels the box is
  // +/-50 m about (0,0), so a 105 m course hit an invisible wall at its
  // midpoint.
  Eigen::Vector3i bounds_min(
      static_cast<int>(-config_.map_bounds.x() /
                       (2.0 * config_.voxel_resolution)),
      static_cast<int>(-config_.map_bounds.y() /
                       (2.0 * config_.voxel_resolution)),
      static_cast<int>(-config_.map_bounds.z() /
                       (2.0 * config_.voxel_resolution)));
  Eigen::Vector3i bounds_max(
      static_cast<int>(config_.map_bounds.x() /
                       (2.0 * config_.voxel_resolution)),
      static_cast<int>(config_.map_bounds.y() /
                       (2.0 * config_.voxel_resolution)),
      static_cast<int>(config_.map_bounds.z() /
                       (2.0 * config_.voxel_resolution)));

  // NED: altitude a is at z = -a, so the HIGHEST altitude is the MOST NEGATIVE
  // index and belongs in the lower bound. Getting this backwards silently
  // inverts the ceiling into a floor, so keep the sign reasoning here rather
  // than at the call site.
  astar_z_min_ = (config_.max_altitude > 0.0)
                     ? static_cast<int>(std::floor(-config_.max_altitude /
                                                   config_.voxel_resolution))
                     : bounds_min.z();
  astar_z_max_ = bounds_max.z();
  astar_planner_ = std::make_unique<path_finding::AStarSFCPlanner>(
      astar_grid_, bounds_min, bounds_max);
  astar_planner_->set_collinear_merge(config_.astar_collinear_merge);
}

// ==========================================
// get_corridors_astar
// ==========================================
void FrontEndSFC::set_previous_path(
    const std::vector<Eigen::Vector3d> &world_points) {
  prev_path_cells_.clear();
  if (world_points.size() < 2 || config_.path_hysteresis_bonus <= 0.0) {
    astar_planner_->set_previous_path(nullptr, 0.0);
    return;
  }

  const double res = config_.voxel_resolution;
  const int r = std::max(0, config_.path_hysteresis_radius_cells);

  // Walk each segment at half-voxel steps so no cell is skipped, and dilate by
  // r cells so the tube survives the small lateral shift that comes from
  // replanning at a slightly different vehicle position -- without it the
  // discount would miss the new path almost every cycle and the hysteresis
  // would do nothing.
  for (std::size_t i = 1; i < world_points.size(); ++i) {
    const Eigen::Vector3d a = world_points[i - 1];
    const Eigen::Vector3d b = world_points[i];
    const double len = (b - a).norm();
    const int steps =
        std::max(1, static_cast<int>(std::ceil(len / (res * 0.5))));
    for (int k = 0; k <= steps; ++k) {
      const Eigen::Vector3d p = a + (b - a) * (static_cast<double>(k) / steps);
      const Eigen::Vector3i c(static_cast<int>(std::floor(p.x() / res)),
                              static_cast<int>(std::floor(p.y() / res)),
                              static_cast<int>(std::floor(p.z() / res)));
      for (int dx = -r; dx <= r; ++dx) {
        for (int dy = -r; dy <= r; ++dy) {
          for (int dz = -r; dz <= r; ++dz) {
            prev_path_cells_.insert(c + Eigen::Vector3i(dx, dy, dz));
          }
        }
      }
    }
  }

  astar_planner_->set_previous_path(&prev_path_cells_,
                                    config_.path_hysteresis_bonus);
}

FrontEndResult
FrontEndSFC::get_corridors_astar(const Eigen::Vector3d &start_pos,
                                 const Eigen::Vector3d &end_pos) {

  double res = config_.voxel_resolution;

  // 1. Discretize
  Eigen::Vector3i start_idx(static_cast<int>(std::floor(start_pos.x() / res)),
                            static_cast<int>(std::floor(start_pos.y() / res)),
                            static_cast<int>(std::floor(start_pos.z() / res)));
  // Bound the search to a horizon. Expansion count grows steeply with goal
  // distance -- unknown space is traversable now, so a far goal makes A* fan
  // out across a huge volume, and the receding-horizon truncation then throws
  // most of that work away. Projecting the goal onto the horizon keeps the
  // search local and cheap; the next replan carries on from further along.
  Eigen::Vector3d search_goal = end_pos;
  const double goal_dist = (end_pos - start_pos).norm();
  if (config_.planning_horizon > 0.0 && goal_dist > config_.planning_horizon) {
    search_goal = start_pos + (end_pos - start_pos) *
                                  (config_.planning_horizon / goal_dist);
  }

  // Re-centre the horizontal geofence on the vehicle before searching.
  //
  // The bound exists to stop A* wandering unboundedly, not to cap how far the
  // mission may go, so it belongs around the START, not around the world
  // origin. Sized at 3x the planning horizon: comfortably larger than any
  // legitimate detour around a local obstacle, while still bounding the search.
  // The vertical limits stay as constructed -- altitude IS a real world
  // constraint.
  {
    const double pad_m =
        (config_.planning_horizon > 0.0 ? config_.planning_horizon : 20.0) *
        3.0;
    const int pad = static_cast<int>(std::ceil(pad_m / res));

    // The ceiling must never EXCLUDE THE VEHICLE. max_altitude says where we
    // would like to fly; it is not a reason to be unable to plan from where we
    // actually are. If the vehicle is above it, extend the geofence to contain
    // the start so the search can route back DOWN into legal airspace --
    // excluding the start instead leaves A* searching from a cell outside its
    // own bounds, and planning simply stops.
    //
    // This is not hypothetical. Takeoff overshoots the 3 m cruise altitude by
    // ~4.4 m in EVERY run measured (vertical peak_true 4.26-4.91 m across ten
    // flights), putting the vehicle near 7.4 m against a 6.0 m max_altitude --
    // z index -14/-15 against a floor of -12. Symptom: the vehicle climbs away
    // at takeoff and then cannot plan at all.
    //
    // NED: higher altitude is MORE NEGATIVE, so containing the start means
    // taking the min.
    const int z_lo = config_.geofence_contains_vehicle
                         ? std::min(astar_z_min_, start_idx.z())
                         : astar_z_min_;
    const int z_hi = config_.geofence_contains_vehicle
                         ? std::max(astar_z_max_, start_idx.z())
                         : astar_z_max_;
    astar_planner_->set_bounds(
        Eigen::Vector3i(start_idx.x() - pad, start_idx.y() - pad, z_lo),
        Eigen::Vector3i(start_idx.x() + pad, start_idx.y() + pad, z_hi));
  }

  Eigen::Vector3i goal_idx(static_cast<int>(std::floor(search_goal.x() / res)),
                           static_cast<int>(std::floor(search_goal.y() / res)),
                           static_cast<int>(std::floor(search_goal.z() / res)));

  // Slide the target off any obstacle it landed on.
  //
  // The horizon projection is a blind extrapolation along the line to the goal,
  // so it lands inside a wall whenever one happens to sit at that distance. An
  // unreachable target is the worst case for A*: it can never terminate early,
  // so it expands until it hits the time limit and then returns a partial path
  // -- turning what should be the cheap case into the most expensive one.
  // Nudging to the nearest non-obstacle cell costs a handful of lookups.
  //
  // Among the free cells on the SMALLEST shell that has one, take the cell
  // closest to the true goal -- not simply the first one iteration order
  // happens to reach.
  //
  // Taking the first was a directional bias, not a neutral choice: the loops
  // run dx from -r upward, so the winner was whatever sat at the most negative
  // x. On a route heading +x that is the cell furthest BACKWARD, so every nudge
  // pulled the search goal toward the vehicle. It is invisible in sparse maps
  // because the nudge almost never fires, but obstacle-rich ones make it the
  // common path -- at 10% obstacle area roughly a quarter of horizon
  // projections land in inflated space, and at 20% about half -- so the goal
  // was being dragged backwards on every other replan. Symptom: the vehicle
  // creeps and stalls through a forest while the corridors and trajectories
  // themselves all look perfectly healthy.
  //
  // Scoring by distance to end_pos keeps the nudge minimal (the smallest shell
  // still wins outright) while spending that freedom on forward progress.
  if (discrete_grid_ && discrete_grid_->is_occupied_inflated(goal_idx)) {
    const int kMaxNudgeCells = 12;
    bool found = false;
    for (int r = 1; r <= kMaxNudgeCells && !found; ++r) {
      double best_score = std::numeric_limits<double>::infinity();
      Eigen::Vector3i best_cand = goal_idx;
      for (int dx = -r; dx <= r; ++dx) {
        for (int dy = -r; dy <= r; ++dy) {
          for (int dz = -r; dz <= r; ++dz) {
            // Shell at radius r only; the interior was covered by earlier r.
            if (std::abs(dx) != r && std::abs(dy) != r && std::abs(dz) != r)
              continue;
            const Eigen::Vector3i cand = goal_idx + Eigen::Vector3i(dx, dy, dz);
            if (discrete_grid_->is_occupied_inflated(cand))
              continue;
            const Eigen::Vector3d cand_pos((cand.x() + 0.5) * res,
                                           (cand.y() + 0.5) * res,
                                           (cand.z() + 0.5) * res);
            const double score = (cand_pos - end_pos).squaredNorm();
            if (score < best_score) {
              best_score = score;
              best_cand = cand;
              found = true;
            }
          }
        }
      }
      if (found)
        goal_idx = best_cand;
    }
  }

  // 2. Search + Smooth
  auto t_start = std::chrono::high_resolution_clock::now();

  // Clear any previous virtual obstacles
  if (discrete_grid_) {
    discrete_grid_->clear_virtual_obstacles();
  }

  // Convert sensor range from meters to voxel units
  double sensor_range_voxels =
      (config_.sensor_range > 0.0)
          ? config_.sensor_range / config_.voxel_resolution
          : -1.0;

  // Convert min_altitude (positive-up meters) to NED voxel z-index floor.
  // In NED, z is negative-up: min_altitude 0.5m → z_ned_max = -0.5 → z_index =
  // floor(-0.5 / res)
  int max_z_index = static_cast<int>(
      std::floor(-config_.min_altitude / config_.voxel_resolution));

  // max_time_ms is a loose backstop only -- config_.astar_max_nodes is the
  // real, and deterministic, limit. See the search budget comment in
  // astar_sfc.cpp.
  auto run_search = [&](double soft_cost) {
    return astar_planner_->search(
        start_idx, goal_idx, config_.astar_max_time_ms, sensor_range_voxels,
        max_z_index, config_.fov_vertical_min, config_.fov_vertical_max,
        config_.unknown_traversal_cost, config_.astar_max_nodes,
        config_.astar_clearance_cost, config_.heat_weight,
        config_.astar_heuristic_weight, soft_cost,
        config_.astar_min_segment_voxels);
  };

  // Pass 1: obstacles hard-blocked. This is the path the corridor builder wants
  // -- every cell on it is genuinely free.
  auto raw_path = run_search(0.0);

  // Pass 2, only if pass 1 could not reach the goal.
  //
  // A hard search that fails returns whatever it got closest with, which in a
  // cluttered scene is often a stub pointing INTO the obstruction -- the search
  // spent its budget probing a dead end and never learned that the way through
  // is around it. The corridor chain built on that stub is short, the
  // trajectory is short, and the vehicle creeps or stops in air that is visibly
  // open. That is the "stalls when the traversable path is not obvious"
  // symptom.
  //
  // Re-running with obstacles SOFT cannot fail: at a large penalty every free
  // detour still wins outright, so the result is the correct route wherever one
  // exists, and only cuts through an obstacle where nothing else does.
  // Truncating that route at its first blocked cell yields the furthest point
  // along the real route that is actually reachable -- a far better sub-goal
  // than the stub, and free of obstacles by construction, so the corridor
  // builder is still handed an all-free path.
  //
  // Note this is a stand-in for a proper global planner, not a replacement: it
  // is still bounded by the same horizon, so it recovers the route SHAPE within
  // the horizon rather than finding detours beyond it.
  if (config_.astar_soft_obstacle_cost > 0.0 && !raw_path.empty() &&
      raw_path.back() != goal_idx) {
    auto soft_path = run_search(config_.astar_soft_obstacle_cost);
    std::size_t free_prefix = 0;
    while (free_prefix < soft_path.size() &&
           !discrete_grid_->is_occupied_inflated(soft_path[free_prefix])) {
      ++free_prefix;
    }
    // Only adopt it if it gets meaningfully further than the hard search
    // managed.
    if (free_prefix > raw_path.size()) {
      std::cout << "[Front-End] hard search stalled at " << raw_path.size()
                << " cells; soft-obstacle pass reaches " << free_prefix
                << " before the first blocked cell -- using that." << std::endl;
      soft_path.resize(free_prefix);
      raw_path = std::move(soft_path);
    }
  }
  auto t_astar = std::chrono::high_resolution_clock::now();
  const std::size_t raw_cells = raw_path.size();
  auto astar_path = astar_planner_->sfc_smoother();

  auto t_end = std::chrono::high_resolution_clock::now();
  double search_ms =
      std::chrono::duration<double, std::milli>(t_astar - t_start).count();
  double smooth_ms =
      std::chrono::duration<double, std::milli>(t_end - t_astar).count();
  double elapsed_ms = search_ms + smooth_ms;
  // Split, because the two stages have completely different levers: search time
  // is bounded by astar_max_nodes / planning_horizon, while smoothing is driven
  // by how many cells the search returned (each shortcut test walks the ray, so
  // a long meandering path costs quadratically in voxel probes). Lumping them
  // hid which one was responsible for a 95 ms front-end.
  std::cout << "[Front-End] A* search " << search_ms << " ms (" << raw_cells
            << " cells) + smoothing " << smooth_ms << " ms = " << elapsed_ms
            << " ms" << std::endl;

  // What was the search actually asked for, and what state were its endpoints
  // in?
  //
  // "N cells" alone cannot distinguish the cases that matter. A short path
  // means the search gave up, but not WHY: the goal may be unreachable, may sit
  // in unmapped space, may have been nudged somewhere unhelpful, or the START
  // may itself be inside inflated space -- in which case every neighbour is
  // rejected and the vehicle cannot move at all, however open the world looks
  // from outside. Those have completely different fixes and are
  // indistinguishable from out here, which is how a 2-cell path (a 1 m plan,
  // floored by the allocator into a 4 s trajectory, i.e. a hover in clear air)
  // survived several rounds of investigation without being pinned down.
  //
  // Printed only when the path is too short to be a real plan, so healthy
  // replans stay quiet.
  if (raw_cells <= 3 && discrete_grid_) {
    auto describe = [&](const Eigen::Vector3i &c) -> const char * {
      if (discrete_grid_->is_occupied_inflated(c))
        return "INFLATED-BLOCKED";
      if (discrete_grid_->is_free(c))
        return "free";
      return "unknown";
    };
    std::cout << "[Front-End] short path: start idx [" << start_idx.transpose()
              << "] (" << describe(start_idx) << ") -> goal idx ["
              << goal_idx.transpose() << "] (" << describe(goal_idx) << "), "
              << (search_goal - start_pos).norm() << " m apart, horizon "
              << config_.planning_horizon << " m." << std::endl;
  }

  // 3. Convert to continuous path
  std::vector<Eigen::Vector3d> continuous_path;
  continuous_path.reserve(astar_path.size());
  for (const auto &idx : astar_path) {
    Eigen::Vector3d pos =
        idx.cast<double>() * res + Eigen::Vector3d::Constant(res / 2.0);
    continuous_path.push_back(pos);
  }

  // Reaching the projected horizon is not reaching the mission goal -- only
  // claim the latter, or the manager will stop replanning short of the actual
  // waypoint.
  const bool horizon_was_clamped = (search_goal - end_pos).squaredNorm() > 1e-9;
  bool reached_goal = (!astar_path.empty() && astar_path.back() == goal_idx &&
                       !horizon_was_clamped);

  // Anchor exact start/goal
  if (continuous_path.size() >= 2) {
    continuous_path.front() = start_pos;
    if (reached_goal) {
      continuous_path.back() = end_pos;
    }
  }

  // --- Receding-horizon truncation
  // --------------------------------------------- One SFC is generated per path
  // segment, so N waypoints produce N-1 corridors. Capping the corridor count
  // caps the constraint-matrix growth (and therefore the OSQP solve time) on
  // long paths. Truncating means we no longer reach the goal this cycle -- the
  // next replan continues from further along.
  //
  // The cap counts CORRIDORS, not distance. When segments are short --
  // threading a gap makes A* emit one-voxel ones -- three corridors can span
  // under 2 m, every exclusive pool zeroes out (see
  // allocate_dynamic_control_points), and the chain ends where the vehicle
  // already is. The QP then solves start-to-end, succeeds, and commands a
  // hover; the vehicle does not move, so the next cycle rebuilds the same
  // chain. That is an ABSORBING state, measured in flight as T_trav 1662 s at
  // 0.11 m/s with the route unchanged 99.53% of the time.
  //
  // min_chain_length therefore grows the chain past the soft cap until it
  // covers useful ground, bounded by max_sfc_count_hard so the QP cannot grow
  // without limit.
  if (config_.max_sfc_count > 0 &&
      static_cast<int>(continuous_path.size()) - 1 > config_.max_sfc_count) {
    std::size_t keep = static_cast<std::size_t>(config_.max_sfc_count);
    if (config_.min_chain_length > 0.0) {
      const std::size_t hard = static_cast<std::size_t>(
          config_.max_sfc_count_hard > 0 ? config_.max_sfc_count_hard
                                         : 3 * config_.max_sfc_count);
      double covered = 0.0;
      for (std::size_t i = 1; i <= keep && i < continuous_path.size(); ++i) {
        covered += (continuous_path[i] - continuous_path[i - 1]).norm();
      }
      while (covered < config_.min_chain_length &&
             keep < continuous_path.size() - 1 && keep < hard) {
        covered += (continuous_path[keep + 1] - continuous_path[keep]).norm();
        ++keep;
      }
    }
    continuous_path.resize(keep + 1);
    reached_goal = false;
  }

  // 4. Build SFCs.
  // Corridors are kept out of unmeasured space by the map's FREE/UNKNOWN
  // frontier, which SparseVoxelGrid::get_obstacles_in_radius() now returns
  // alongside real obstacles. The previous scheme injected A*'s shadowed voxels
  // here instead: those were whatever cells the search happened to test and
  // fail a line-of-sight check on, so the barrier was incomplete, unstable
  // between replans, and scattered through free space -- it collapsed corridors
  // to zero width. Nothing to inject any more.

  // Tell the map where corridors are about to be built so it only extracts the
  // frontier there. Radius covers the path plus the corridor extensions and
  // lateral drift budget.
  if (discrete_grid_ && continuous_path.size() >= 2) {
    Eigen::Vector3d lo = continuous_path.front(), hi = continuous_path.front();
    for (const auto &p : continuous_path) {
      lo = lo.cwiseMin(p);
      hi = hi.cwiseMax(p);
    }
    const Eigen::Vector3d center = 0.5 * (lo + hi);
    const double margin = std::max(config_.sfc_start_ext, config_.sfc_end_ext) +
                          config_.sfc_width;
    discrete_grid_->set_frontier_focus(center, 0.5 * (hi - lo).norm() + margin);
  }

  sfc::StandaloneWaypointsSFC waypoints_smooth(3);
  waypoints_smooth.add(continuous_path[0], -1, 0.0, false);

  std::vector<sfc::SFCResult> corridors;

  auto build_corridor = [&](size_t i, bool is_goal) {
    double ext_s = config_.sfc_start_ext;
    double ext_e = config_.sfc_end_ext;
    if (config_.spline_type == "natural") {
      if (i == 1)
        ext_s = 30.0;
      if (is_goal)
        ext_e = 30.0;
    }
    if (config_.aircraft_type == "fixed-wing") {
      return static_sfc_manager_.generate_sfc(continuous_path[i - 1],
                                              continuous_path[i], ext_s, ext_e);
    }
    return sfc_manager_.generate_sfc(continuous_path[i - 1], continuous_path[i],
                                     config_.sfc_width, ext_s, ext_e);
  };

  auto t_sfc0 = std::chrono::high_resolution_clock::now();
  for (size_t i = 1; i < continuous_path.size(); ++i) {
    corridors.push_back(build_corridor(i, i == continuous_path.size() - 1));
  }
  // --- REPAIR DEGENERATE CORRIDORS ---------------------------------------------------------
  //
  // A corridor with a lateral half-width of zero is NOT unsafe. The box is already inset by
  // hull + obstacle_margin, so every point inside it -- including on the wall -- keeps the
  // CENTRE OF MASS that far from the obstacle surface, and the trajectory tracks the centre of
  // mass. A zero-width corridor is a line, and a straight trajectory down that line is fine.
  //
  // It is INFEASIBLE. Every control point of the B-spline must lie inside the corridor (the
  // convex-hull property), so a collapsed axis pins them all to a plane and the spline cannot
  // curve on it. That is survivable only for straight travel with the start velocity already
  // aligned; the moment the pinned start state points off-axis, or the next corridor has a
  // different axis and the bridge points must satisfy both, the QP is over-constrained. In
  // flight that reads as "primal infeasible on the corridor block, residual stalled, time
  // stretching cannot fix it" -- measured on hard s0 with narrowest 0.000 m.
  //
  // REPAIR, DO NOT REJECT. Drop the interior waypoint that produced the collapsed box so its
  // segment merges into a neighbour, then rebuild. This is not the reverted
  // min_corridor_half_width, which imposed a width floor after the fact and DROPPED the
  // offending corridors -- fragmenting the chain, clearing reached_goal, and stalling the
  // vehicle. This keeps the chain contiguous and never moves an endpoint.
  //
  // The acceptance test is the same test that found the problem: a merged axis that cuts an
  // obstacle is shrunk to zero by the builder itself, so a bad merge cannot be accepted.
  if (config_.sfc_min_usable_width > 0.0 && corridors.size() > 1) {
    auto narrowest = [](const sfc::SFCResult & c) {
      double m = std::numeric_limits<double>::infinity();
      for (int k = 2; k <= 5; ++k) { m = std::min(m, c.bounds(k)); }
      return m;
    };
    auto n_degenerate = [&](const std::vector<sfc::SFCResult> & cs) {
      int n = 0;
      for (const auto & c : cs) {
        if (narrowest(c) < config_.sfc_min_usable_width) ++n;
      }
      return n;
    };
    auto rebuild_all = [&]() {
      std::vector<sfc::SFCResult> out;
      for (std::size_t i = 1; i < continuous_path.size(); ++i) {
        out.push_back(build_corridor(i, i == continuous_path.size() - 1));
      }
      return out;
    };

    for (int attempt = 0; attempt < config_.sfc_max_repairs && continuous_path.size() > 2;
         ++attempt) {
      int worst = -1;
      for (std::size_t k = 0; k < corridors.size(); ++k) {
        if (narrowest(corridors[k]) < config_.sfc_min_usable_width) { worst = static_cast<int>(k);
          break; }
      }
      if (worst < 0) break;

      const int before = n_degenerate(corridors);
      bool fixed = false;
      // Corridor k spans continuous_path[k] -> [k+1]. Merging it forward drops [k+1]; merging
      // it backward drops [k]. Endpoints are never candidates.
      for (std::size_t drop : {static_cast<std::size_t>(worst) + 1,
                               static_cast<std::size_t>(worst)}) {
        if (drop == 0 || drop + 1 >= continuous_path.size()) continue;
        // DO NOT MERGE DOWN TO A SINGLE CORRIDOR.
        //
        // THE OFFLINE AND CLOSED-LOOP MEASUREMENTS DISAGREE, AND THE CLOSED-LOOP ONE WINS.
        //
        // Offline (25 paired forests, 20% density, v5) says the guard COSTS success, because a
        // bridge charges `degree` points per junction, so three corridors deliver ~8 s over
        // ground one corridor covers in 4 s -- merging removes over-allocation faster than the
        // single-corridor floor (degree*2 = 8 points) imposes it:
        //     guard   repair 0.15 -> 84%      repair 0.30 -> 60%
        //     none    repair 0.15 -> 92%      repair 0.30 -> 84%
        //
        // In the SIMULATOR the same change made hard_s0 v5 go from PASS to FAIL:
        //     guard   T_trav  68.7 s  L_path 157.1 m  v_avg 2.29  front-end max 32.8 ms  PASS
        //     none    T_trav 132.1 s  L_path 175.9 m  v_avg 1.33  front-end max 51.0 ms  FAIL
        //                                                          (2 collisions, 3.3 s contact)
        //
        // The offline harness has NO CONTROLLER AND NO ESTIMATOR, so it cannot see the path
        // that actually decides this: more merging means more chain rebuilds, which lengthens
        // the front end (32.8 -> 51.0 ms worst cycle), which starves the 50 Hz dispatch loop
        // (31.1 -> 29.5 Hz), which degrades tracking (cross-track 0.97 -> 1.22 m, along-track
        // lag 0.11 -> 0.61 m), which is what puts the vehicle into obstacles. A plan-quality
        // win that costs CPU can still lose the run.
        //
        // Both samples are n=1 in sim on a map with documented bimodality, so this is a
        // provisional call, not a proof. Re-check it with several paired runs before removing.
        if (continuous_path.size() <= 3) continue;   // would leave a single corridor
        const auto saved_path = continuous_path;
        const auto saved_cor = corridors;
        continuous_path.erase(continuous_path.begin() + static_cast<long>(drop));
        corridors = rebuild_all();
        if (n_degenerate(corridors) < before) { fixed = true; break; }
        continuous_path = saved_path;
        corridors = saved_cor;
      }
      if (!fixed) break;   // neither merge helped; the geometry is genuinely tight
    }
  }

  double sfc_build_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - t_sfc0)
                            .count();
  std::cout << "[Front-End] SFC construction " << sfc_build_ms << " ms for "
            << corridors.size() << " corridor(s)" << std::endl;

  for (size_t i = 1; i < continuous_path.size(); ++i) {
    waypoints_smooth.add(continuous_path[i], static_cast<int>(i - 1), 0.0,
                         i == continuous_path.size() - 1);
    waypoints_smooth.addSFC(corridors[i - 1]);
  }

  std::cout << "[Front-End] Extracted " << corridors.size()
            << " Safe Flight Corridors via A*." << std::endl;

  // 5. Allocate control points
  FrontEndResult result;
  result.corridors = corridors;
  result.waypoints_smooth = waypoints_smooth;
  result.reached_goal = reached_goal;
  result.actual_end_pos =
      continuous_path.empty() ? end_pos : continuous_path.back();

  if (config_.aircraft_type == "fixed-wing") {
    result.allocation_list =
        allocate_dynamic_control_points_fixed_wing(corridors);
  } else {
    result.allocation_pools = allocate_dynamic_control_points(corridors);
  }

  return result;
}

// ==========================================
// allocate_dynamic_control_points (Multi-Rotor)
// ==========================================
std::vector<ConstraintPool> FrontEndSFC::allocate_dynamic_control_points(
    const std::vector<sfc::SFCResult> &corridors) const {

  int num_corridors = static_cast<int>(corridors.size());
  int degree = config_.degree;
  double v_max = std::max(config_.v_max, 1e-3);
  double a_max = std::max(config_.a_max, 1e-3);
  double pts_per_sec = 1.0;

  std::vector<ConstraintPool> pools;

  // --- Global time budget
  // -------------------------------------------------------------- The
  // acceleration-limited time 2*sqrt(L/a) is the duration of an
  // accelerate-from-rest- to-rest manoeuvre. Evaluating it PER LEG and summing
  // bills the vehicle for stopping and restarting at every intermediate
  // waypoint, which it never does -- it flies straight through. That
  // overcharges by sqrt(N): a 15 m route split into 8 legs was allocated 15.5 s
  // instead of 5.5 s. The vehicle then crawled through a trajectory sized for
  // three times the real flight time, and since duration also sets the
  // control-point count, the QP grew until it stopped converging.
  //
  // So: size the budget ONCE from the whole path, then hand each leg its share
  // by length.
  double total_path_length = 0.0;
  for (int i = 0; i < num_corridors; ++i) {
    total_path_length += corridors[i].getDistancePrimaryToSecondary();
  }
  const double total_time_budget = kinematic_time_budget(
      total_path_length, v_max, a_max, config_.time_budget_factor);

  for (int i = 0; i < num_corridors; ++i) {
    double L_base = corridors[i].getDistancePrimaryToSecondary();

    // Determine overlap from neighbors
    double actual_ext_prev =
        (i > 0) ? corridors[i - 1].bounds(0) -
                      corridors[i - 1].getDistancePrimaryToSecondary()
                : 0.0;
    double actual_ext_next =
        (i < num_corridors - 1) ? corridors[i + 1].bounds(1) : 0.0;

    // This leg's share of the global budget, proportional to how much of the
    // path it is.
    double t_target = (total_path_length > 1e-9)
                          ? total_time_budget * (L_base / total_path_length)
                          : total_time_budget;

    int exclusive_pts;
    if (L_base <= (actual_ext_prev + actual_ext_next)) {
      exclusive_pts = 0; // Subsumed by bridges
    } else {
      exclusive_pts = static_cast<int>(std::ceil(t_target * pts_per_sec));
    }

    if (num_corridors == 1) {
      exclusive_pts = std::max(exclusive_pts, degree * 2);
    }

    if (exclusive_pts > 0) {
      pools.push_back({exclusive_pts, {i}, "exclusive"});
    }

    // Bridge to next SFC
    if (i < num_corridors - 1) {
      pools.push_back({degree, {i, i + 1}, "bridge"});
    }
  }

  if (pools.empty()) {
    return pools;
  }

  // --- Time-budget correction
  // --------------------------------------------------- The knot vector is
  // unit-spaced, so the flight time a spline actually delivers is (total
  // control points - degree) seconds, NOT (total control points). Sizing the
  // pools directly from t_target therefore under-allocates the trajectory
  // duration by `degree` seconds, which makes the MINVO velocity/acceleration
  // constraints infeasible before the solver ever runs (worst for a single
  // corridor, where the shortfall is the full `degree`). Top the pools up until
  // the delivered duration covers what the kinematics need.
  int total_pts = 0;
  for (const auto &pool : pools)
    total_pts += pool.pts;

  // Total flight time must come from the TOTAL path length, not from summing
  // the per-leg times. The acceleration-limited term 2*sqrt(L/a) is the
  // duration of an accelerate-from-rest-to-rest manoeuvre, so summing it over N
  // legs bills the vehicle for stopping and restarting at every intermediate
  // waypoint -- it does not, it flies straight through. Summed, the same path
  // costs sqrt(N) times too much: a 15 m route split into 8 legs was
  // allocated 15.5 s instead of 5.5 s. That inflated duration is why the
  // vehicle crawled, and (because duration sets the control-point count) it is
  // also what blew the QP up to hundreds of milliseconds.
  const double total_time_required = total_time_budget;

  int required_segments =
      static_cast<int>(std::ceil(total_time_required * pts_per_sec));
  int deficit = required_segments - (total_pts - degree);

  // Distribute the shortfall round-robin. Growing a "bridge" pool is safe: its
  // points are constrained by the intersection of both corridors, so every
  // degree+1 window of control points still lies wholly inside at least one
  // corridor (convex hull property).
  for (int k = 0; k < deficit; ++k) {
    pools[k % pools.size()].pts += 1;
  }

  return pools;
}

// ==========================================
// allocate_dynamic_control_points_fixed_wing
// ==========================================
std::vector<int> FrontEndSFC::allocate_dynamic_control_points_fixed_wing(
    const std::vector<sfc::SFCResult> &corridors) const {

  int num_corridors = static_cast<int>(corridors.size());
  int degree = config_.degree;
  double v_max = 3.0, a_max = 2.0, pts_per_sec = 1.5;

  std::vector<int> num_pts_list(num_corridors, 0);

  // Pass 1: Straightaway Baseline
  for (int i = 0; i < num_corridors; ++i) {
    double L = corridors[i].getDistancePrimaryToSecondary();
    double t_cruise = L / v_max;
    double t_accel = 2.0 * std::sqrt(L / a_max);
    double t_target = std::max(t_cruise, t_accel);

    int N_kinematic = static_cast<int>(std::ceil(t_target * pts_per_sec));
    num_pts_list[i] = std::max(N_kinematic, 2 * degree);
  }

  // Pass 2: Apex Injector
  for (int i = 0; i < num_corridors - 1; ++i) {
    Eigen::Vector3d p0 = corridors[i].primaryPosition;
    Eigen::Vector3d p1 = corridors[i].secondaryPosition;
    Eigen::Vector3d p2 = corridors[i + 1].secondaryPosition;

    Eigen::Vector3d v_in = p1 - p0;
    Eigen::Vector3d v_out = p2 - p1;

    double norm_in = v_in.norm();
    double norm_out = v_out.norm();

    if (norm_in > 0.001 && norm_out > 0.001) {
      double cos_theta =
          std::clamp(v_in.dot(v_out) / (norm_in * norm_out), -1.0, 1.0);
      double momentum_shed = 1.0 - cos_theta;

      if (momentum_shed > 0.1) {
        int apex_pool =
            static_cast<int>(std::ceil(momentum_shed * (degree * 3)));
        int half_pool = apex_pool / 2;
        num_pts_list[i] += half_pool;
        num_pts_list[i + 1] += half_pool;
      }
    }
  }

  return num_pts_list;
}

// ==========================================
// compile_system_constraints
// ==========================================
void FrontEndSFC::compile_system_constraints(
    const std::vector<sfc::SFCResult> &corridors,
    const std::vector<ConstraintPool> &pools, Eigen::MatrixXd &A_sfc_out,
    Eigen::VectorXd &b_sfc_out, int &total_points_out) {

  int num_dimensions = 3;
  total_points_out = 0;
  for (const auto &pool : pools) {
    total_points_out += pool.pts;
  }

  std::vector<Eigen::MatrixXd> A_rows;
  std::vector<Eigen::VectorXd> b_rows;
  int global_cp_index = 0;

  for (const auto &pool : pools) {
    // Combine A/b from all SFCs in this pool
    std::vector<Eigen::MatrixXd> A_parts;
    std::vector<Eigen::VectorXd> b_parts;
    int total_ineq = 0;

    for (int sfc_idx : pool.sfc_indices) {
      A_parts.push_back(corridors[sfc_idx].A_mat);
      b_parts.push_back(corridors[sfc_idx].b_vec);
      total_ineq += static_cast<int>(corridors[sfc_idx].A_mat.rows());
    }

    Eigen::MatrixXd A_pool(total_ineq, num_dimensions);
    Eigen::VectorXd b_pool(total_ineq);
    int row = 0;
    for (size_t k = 0; k < A_parts.size(); ++k) {
      int r = static_cast<int>(A_parts[k].rows());
      A_pool.block(row, 0, r, num_dimensions) = A_parts[k];
      b_pool.segment(row, r) = b_parts[k];
      row += r;
    }

    // Lock control points inside the overlapping volume
    for (int j = 0; j < pool.pts; ++j) {
      // Relax constraints for the first 3 and last 3 control points
      // since they are strictly bound by the equality constraints (Boundary
      // conditions)
      if (global_cp_index < 3 || global_cp_index >= total_points_out - 3) {
        global_cp_index++;
        continue;
      }

      Eigen::MatrixXd A_padded =
          Eigen::MatrixXd::Zero(total_ineq, total_points_out * num_dimensions);
      int col_start = global_cp_index * num_dimensions;
      A_padded.block(0, col_start, total_ineq, num_dimensions) = A_pool;

      A_rows.push_back(A_padded);
      b_rows.push_back(b_pool);
      global_cp_index++;
    }
  }

  // Stack into final matrices
  int total_rows = 0;
  for (const auto &a : A_rows)
    total_rows += static_cast<int>(a.rows());

  A_sfc_out.resize(total_rows, total_points_out * num_dimensions);
  b_sfc_out.resize(total_rows);

  int r = 0;
  for (size_t i = 0; i < A_rows.size(); ++i) {
    int nr = static_cast<int>(A_rows[i].rows());
    A_sfc_out.block(r, 0, nr, A_rows[i].cols()) = A_rows[i];
    b_sfc_out.segment(r, nr) = b_rows[i];
    r += nr;
  }
}

} // namespace trajectory_planner
