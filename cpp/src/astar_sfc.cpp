#include "astar_sfc.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace trajectory_planner {
namespace path_finding {

AStarSFCPlanner::AStarSFCPlanner(const VoxelGridData &grid,
                                 const Eigen::Vector3i &bounds_min,
                                 const Eigen::Vector3i &bounds_max)
    : voxel_grid_(grid), min_bounds_(bounds_min), max_bounds_(bounds_max) {
  // Precalculate the 26 directions to avoid math in the inner loop!
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dz = -1; dz <= 1; ++dz) {
        if (dx == 0 && dy == 0 && dz == 0)
          continue;
        double cost = std::sqrt(dx * dx + dy * dy + dz * dz);
        directions_.push_back({dx, dy, dz, cost});
      }
    }
  }
}

std::vector<Eigen::Vector3i> AStarSFCPlanner::reconstruct_path(Node *end_node) {
  path_.clear();
  Node *current = end_node;
  while (current != nullptr) {
    path_.push_back(current->position);
    current = current->parent;
  }
  std::reverse(path_.begin(), path_.end());
  return path_;
}

std::vector<Eigen::Vector3i> AStarSFCPlanner::search(
    const Eigen::Vector3i &start_pos, const Eigen::Vector3i &goal_pos,
    double max_time_ms, double sensor_range_voxels, int max_z_index,
    double fov_vertical_min_deg, double fov_vertical_max_deg,
    double unknown_traversal_cost, int max_nodes, double clearance_penalty,
    double heat_weight, double heuristic_weight, double soft_obstacle_cost,
    double min_segment_voxels) {
  auto search_start = std::chrono::high_resolution_clock::now();
  std::priority_queue<Node *, std::vector<Node *>, CompareNode> open_list;
  std::unordered_set<Eigen::Vector3i, Vector3iHash> closed_set;
  std::unordered_map<Eigen::Vector3i, double, Vector3iHash> best_g;

  shadowed_voxels_.clear();

  // --- MEMORY ARENA (Replaces Python Garbage Collector) ---
  // This safely owns all nodes. When 'search' returns, it instantly deletes all
  // of them.
  std::vector<std::unique_ptr<Node>> node_memory;
  node_memory.reserve(
      50000); // Pre-allocate to prevent mid-search OS bottleneck

  double initial_h =
      (start_pos - goal_pos).cast<double>().norm() * heuristic_weight_;
  node_memory.push_back(
      std::make_unique<Node>(initial_h, 0.0, initial_h, start_pos, nullptr));
  open_list.push(node_memory.back().get());

  Node *best_node = node_memory.back().get();
  double min_h_cost = initial_h;

  int nodes_expanded = 0;

  // Pre-square the sensor range to avoid sqrt in the hot loop
  const double sensor_range_sq = (sensor_range_voxels > 0.0)
                                     ? sensor_range_voxels * sensor_range_voxels
                                     : -1.0;

  // Pre-calculate FOV bounds in radians and store in class for raycasts
  fov_min_rad_ = fov_vertical_min_deg * (M_PI / 180.0);
  fov_max_rad_ = fov_vertical_max_deg * (M_PI / 180.0);
  check_fov_ =
      false; // superseded by the unknown-space penalty; see the neighbour loop
  search_start_pos_ = start_pos;
  unknown_cost_ = unknown_traversal_cost;
  clearance_cost_ = clearance_penalty;
  heat_weight_ = heat_weight;
  min_segment_voxels_ = min_segment_voxels;
  soft_obstacle_cost_ = soft_obstacle_cost;
  heuristic_weight_ = (heuristic_weight > 0.0) ? heuristic_weight : 2.0;

  while (!open_list.empty()) {
    nodes_expanded++;
    Node *current_node = open_list.top();
    open_list.pop();

    // --- LAZY DELETION CHECK ---
    if (closed_set.find(current_node->position) != closed_set.end()) {
      continue;
    }

    // --- BEST NODE TRACKING ---
    if (current_node->h_cost < min_h_cost) {
      min_h_cost = current_node->h_cost;
      best_node = current_node;
    }

    // --- SEARCH BUDGET ---
    //
    // The primary limit is a NODE COUNT, not wall-clock, because the budget
    // decides which path comes out and therefore everything downstream. A
    // wall-clock budget makes the planner non-deterministic: CPU scheduling
    // jitter changes how far the search gets, so the same vehicle in the same
    // place produces a different trajectory run to run. Measured directly --
    // identical binary and config, five runs, two reached the goal and three
    // did not. Unreproducible behaviour is not something to fly.
    //
    // Wall-clock survives only as a backstop against pathological slowness (a
    // huge map, a heavily loaded CPU); it is set loose enough that it should
    // never fire in normal operation, so ordinary runs stay deterministic.
    if (max_nodes > 0 && nodes_expanded >= max_nodes) {
      std::cout << "[ A* ] Node budget reached (" << nodes_expanded
                << " nodes). Returning partial path." << std::endl;
      return reconstruct_path(best_node);
    }
    if (nodes_expanded % 100 == 0) {
      auto now = std::chrono::high_resolution_clock::now();
      double elapsed =
          std::chrono::duration<double, std::milli>(now - search_start).count();
      if (elapsed >= max_time_ms) {
        std::cout
            << "[ A* ] WALL-CLOCK backstop hit (" << elapsed
            << " ms) before the node budget -- this run is not reproducible."
            << std::endl;
        return reconstruct_path(best_node);
      }
    }

    if (current_node->position == goal_pos) {
      std::cout << "[ A* ] Path found! Explored " << nodes_expanded << " nodes."
                << std::endl;
      return reconstruct_path(current_node);
    }

    closed_set.insert(current_node->position);

    // --- CLEARANCE PENALTY ---
    //
    // Plain A* returns the SHORTEST path, and the shortest path hugs obstacles:
    // it runs flush against the inflation boundary, because leaving it costs
    // distance and buys nothing the search can see. Downstream that is
    // expensive. AsymmetricBoxBuilder subtracts the voxel's SAT corner radius r
    // from every obstacle projection, so an obstacle sitting exactly r off the
    // corridor axis gives max(0, r - r) = 0 and the corridor is emitted with
    // ZERO width on that face. Measured: every collapsed face in a sample run
    // was caused by a point at |p| = r_y = 0.354 m, i.e. dead tangent. The
    // trajectory is then pinned to the centreline, and the vehicle visibly
    // crawls through that corridor while flying the rest of the path at speed.
    //
    // Charging a small toll for travelling flush against inflation buys a voxel
    // of margin wherever margin is available, and -- because it is a cost and
    // not a gate -- still lets the search squeeze through a genuinely tight gap
    // when that is the only way.
    //
    // Evaluated once per EXPANSION (6 probes) rather than once per neighbour
    // (26 * 6): the toll is attached to the edges leaving a cramped cell
    // instead of the edges entering one. Every cell on the returned path except
    // the last is expanded, so the path is charged either way, at 1/26th the
    // number of hash probes.
    double clearance_toll = 0.0;
    if (clearance_cost_ > 0.0) {
      const auto &occ = *voxel_grid_.occupied_voxels_inflated;
      for (int axis = 0; axis < 3 && clearance_toll == 0.0; ++axis) {
        for (int s = -1; s <= 1; s += 2) {
          Eigen::Vector3i probe = current_node->position;
          probe[axis] += s;
          if (occ.find(probe) != occ.end()) {
            clearance_toll = clearance_cost_;
            break;
          }
        }
      }
    }

    for (const auto &dir : directions_) {
      Eigen::Vector3i neighbor_pos(current_node->position.x() + dir.dx,
                                   current_node->position.y() + dir.dy,
                                   current_node->position.z() + dir.dz);

      // 1. Geofence Check
      if (neighbor_pos.x() < min_bounds_.x() ||
          neighbor_pos.x() > max_bounds_.x() ||
          neighbor_pos.y() < min_bounds_.y() ||
          neighbor_pos.y() > max_bounds_.y() ||
          neighbor_pos.z() < min_bounds_.z() ||
          neighbor_pos.z() > max_bounds_.z()) {
        continue;
      }

      // 2. Z-Floor Check (prevents underground exploration)
      if (neighbor_pos.z() > max_z_index) {
        continue;
      }

      // 3. Sensor Range Gate (unknown space beyond sensor range = blocked)
      double dx = neighbor_pos.x() - start_pos.x();
      double dy = neighbor_pos.y() - start_pos.y();
      double dz = neighbor_pos.z() - start_pos.z();

      if (sensor_range_sq > 0.0) {
        if (dx * dx + dy * dy + dz * dz > sensor_range_sq) {
          continue;
        }
      }
      // NOTE: the old "FoV blind spot gate" used to sit here, rejecting
      // neighbours whose elevation from the start fell outside the LiDAR's
      // vertical FoV. It is gone, for two reasons. It is subsumed -- a cell
      // outside the FoV is never swept by a beam, so it is UNKNOWN and already
      // penalised below. And it actively fought the accumulating map: a cell
      // observed a moment ago is legitimately FREE even once it drops out of
      // the current FoV, and the gate blocked exactly that. (It also measured
      // elevation in the world frame while the real FoV is body-fixed, so it
      // mis-fired whenever the vehicle pitched.)

      // 4. Collision Check (O(1) hash lookup)
      //
      // With soft_obstacle_cost_ > 0 an inflated cell is TRAVERSABLE at a large
      // penalty rather than blocked. That is SANDO's use_soft_cost_obstacles
      // (obstacle_soft_cost 5.0 against a unit step): their global search can
      // then never fail, never return a partial path, and never wedge -- it
      // always produces some route, and the layers below decide what to do with
      // it.
      //
      // Do NOT enable this for a search whose path is handed straight to the
      // corridor builder. A path through an obstacle yields a zero-width
      // corridor and a primal-infeasible QP, which is the failure this is meant
      // to avoid. It is for finding the SHAPE of a route; see the two-pass
      // subgoal selection in front_end.cpp.
      bool soft_penalty = false;
      const bool neighbour_blocked =
          voxel_grid_.occupied_voxels_inflated->find(neighbor_pos) !=
          voxel_grid_.occupied_voxels_inflated->end();
      bool escaping = false;
      if (neighbour_blocked) {
        // ESCAPING A BLOCKED START.
        //
        // The vehicle is physically wherever it is. Refusing to plan because the MAP says it
        // is inside an obstacle is never useful -- and it happens routinely: the inflation
        // radius is hull + obstacle_margin = 0.690 m while the measured horizontal estimator
        // bias is 0.55-0.66 m, so a vehicle genuinely clear by 0.7 m can have its BELIEVED
        // position land inside a trunk's inflation shell. Every neighbour inside that shell is
        // then refused as well, so the search is trapped and the vehicle hovers with open space
        // all around it. Measured on hard s0 v5: the start cell was INFLATED-BLOCKED in 85 of
        // 137 short-path diagnostics.
        //
        // So: a node that is ITSELF inside inflated space may step to a blocked neighbour, at a
        // steep price. This can only ever walk OUTWARD from the blob containing the start --
        // from a free node a blocked neighbour is still refused, exactly as before -- so it
        // cannot be used to enter an obstacle, only to leave one. Any free route wins outright
        // on cost, so the escape is used only when there is no alternative.
        //
        // Deliberately NOT soft_obstacle_cost_: that is the two-pass sub-goal mechanism, which
        // is off in pass 1, and the escape has to work in pass 1 or the path handed to the
        // corridor builder is the one that never leaves the blob.
        escaping = voxel_grid_.occupied_voxels_inflated->find(current_node->position) !=
                   voxel_grid_.occupied_voxels_inflated->end();
        if (!escaping) {
          if (soft_obstacle_cost_ <= 0.0) {
            continue;
          }
          soft_penalty = true;
        }
      }

      // 5. Skip fully evaluated nodes
      if (closed_set.find(neighbor_pos) != closed_set.end()) {
        continue;
      }

      // 6. Unknown-space penalty.
      //
      // Replaces the old line-of-sight shadow gate, which raycast from the
      // search start to every candidate cell against every obstacle box --
      // O(N_boxes) per neighbour, and a hard block. Hard-blocking unknown space
      // is what made A* unable to route around a corner: everything past the
      // corner is invisible from the start, so the search exhausted its budget
      // and returned a partial path. Charging for it instead lets the search
      // commit to unseen space when it has to, while still preferring
      // measured-free space when that exists. Cost is now a single O(1) hash
      // probe.
      double step_cost = dir.cost;
      if (voxel_grid_.has_free() &&
          voxel_grid_.free_voxels->find(neighbor_pos) ==
              voxel_grid_.free_voxels->end()) {
        if (unknown_cost_ < 0.0)
          continue; // negative == restore hard-block
        step_cost += unknown_cost_ * dir.cost;
      }

      step_cost += clearance_toll * dir.cost;

      // Graded proximity cost. THE INVARIANT THAT MATTERS: the total step
      // multiplier
      //     m = 1 + unknown_cost_ + clearance_cost_ + heat_weight_ * heat
      // must stay strictly below heuristic_weight_. A* only prefers a step
      // toward the goal while m < w -- stepping closer costs m and pays back w,
      // so f moves by (m - w), and at m >= w the search stops preferring the
      // goal and floods outward instead. That is exactly how the old binary
      // clearance toll broke the search: it pushed m to 2.0 against w = 2.0,
      // and A* burned its whole node budget returning 2-cell paths in open air.
      // front_end.cpp checks this at construction and warns.
      if (heat_weight_ > 0.0) {
        step_cost +=
            heat_weight_ * voxel_grid_.heat_at(neighbor_pos) * dir.cost;
      }

      // Soft-obstacle penalty. Large enough that any free detour wins outright,
      // so this only decides the route where NO free route exists.
      if (soft_penalty) {
        step_cost += soft_obstacle_cost_ * dir.cost;
      }

      // Escape toll. Steep enough that the search leaves the blob by the shortest route and
      // never lingers, but finite, so a trapped start still yields a path.
      if (escaping) {
        step_cost += kEscapeCost * dir.cost;
      }

      // Path hysteresis. Applied to the whole step cost, so an on-route cell is
      // cheaper however it was otherwise priced. Only ever reduces the
      // multiplier, so the guidance invariant (see front_end.hpp) is
      // unaffected.
      if (prev_path_cells_ && hysteresis_bonus_ > 0.0 &&
          prev_path_cells_->find(neighbor_pos) != prev_path_cells_->end()) {
        step_cost *= (1.0 - hysteresis_bonus_);
      }

      double new_g = current_node->g_cost + step_cost;

      // Only queue this cell if we have actually found a cheaper way to reach
      // it. Without this the search pushes a fresh node for EVERY edge into a
      // cell and relies purely on lazy deletion at pop time, so the open list
      // fills with dominated duplicates. That was survivable while the shadow
      // gate pruned most neighbours away; now that unknown space is traversable
      // the branching factor is far higher and the duplicates dominate the run
      // time.
      auto g_it = best_g.find(neighbor_pos);
      if (g_it != best_g.end() && g_it->second <= new_g) {
        continue;
      }
      best_g[neighbor_pos] = new_g;

      // WEIGHTED A* (2.0 multiplier matches your Python Magic Number)
      double new_h =
          (neighbor_pos - goal_pos).cast<double>().norm() * heuristic_weight_;
      double new_f = new_g + new_h;

      // Create neighbor and push pointer to Priority Queue
      node_memory.push_back(std::make_unique<Node>(new_f, new_g, new_h,
                                                   neighbor_pos, current_node));
      open_list.push(node_memory.back().get());
    }
  }

  std::cout
      << "[ A* ] Open list emptied, no full path found. Returning partial path."
      << std::endl;
  return reconstruct_path(best_node);
}

std::vector<Eigen::Vector3i> AStarSFCPlanner::sfc_smoother() {
  if (path_.size() <= 2)
    return path_;

  std::vector<Eigen::Vector3i> smoothed_path;
  smoothed_path.push_back(path_[0]);
  int anchor_idx = 0;

  // The Look-Ahead Loop
  for (size_t i = 2; i < path_.size(); ++i) {
    if (!is_line_of_sight_clear(path_[anchor_idx], path_[i])) {
      smoothed_path.push_back(path_[i - 1]);
      anchor_idx = i - 1;
    }
  }
  smoothed_path.push_back(path_.back());

  // --- Drop COLLINEAR waypoints ------------------------------------------------------------
  //
  // A waypoint whose two legs point the same way is redundant BY CONSTRUCTION: it lies on the
  // straight line between its neighbours, so the merged segment is exactly the union of two
  // segments the smoother already accepted. Removing it cannot cut a corner, cannot shorten
  // the path, and cannot move an endpoint -- so unlike the length-based merge below it needs
  // NO line-of-sight test and can never be refused.
  //
  // That distinction is the whole point. The length-based merge asks "may I drop this
  // waypoint?", which re-asks the question that emitted it, and at a tight corner the answer is
  // the same no (see test_segment_merge.cpp). Collinearity is not a question -- it is an
  // identity -- so this pass always fires where it applies.
  //
  // Measured on the hard_s0 v5 stall snapshot: the chain contained P1->P2->P3 with both legs at
  // ux [0.707107 -0.707107 0] to seven digits, i.e. two ONE-VOXEL DIAGONAL segments where one
  // 1.41 m segment belonged. Each got its own corridor, and both came out with a lateral
  // half-width of EXACTLY 0 against a 0.490 m hull -- a QP that is primal-infeasible on
  // geometry, which time-stretching cannot fix, so the vehicle stops. Every other knob was
  // swept against that snapshot (horizon, unknown cost, node budget, sfc extension, obstacle
  // margin) and none of them moved it.
  //
  // Integer arithmetic throughout: cross() == 0 with dot() > 0 is exact collinearity in the
  // same direction. A zero-length leg also satisfies it and is equally redundant. dot() <= 0
  // excludes a reversal, which is NOT redundant -- it is a genuine there-and-back.
  if (collinear_merge_) {
    bool changed = true;
    while (changed && smoothed_path.size() > 2) {
      changed = false;
      for (std::size_t i = 1; i + 1 < smoothed_path.size(); ++i) {
        const Eigen::Vector3i a = smoothed_path[i] - smoothed_path[i - 1];
        const Eigen::Vector3i b = smoothed_path[i + 1] - smoothed_path[i];
        if (a.cross(b) != Eigen::Vector3i::Zero())
          continue;                      // not parallel
        if (a.dot(b) <= 0)
          continue;                      // reversal or degenerate pair; keep the waypoint
        smoothed_path.erase(smoothed_path.begin() + static_cast<long>(i));
        changed = true;
        break;                           // indices shifted; restart the scan
      }
    }
  }

  // --- Merge sub-threshold segments
  // -------------------------------------------------------
  //
  // The look-ahead loop above emits a waypoint every time line-of-sight from
  // the anchor fails, and re-anchors on the cell BEFORE the failure. Around a
  // tight corner it fails on consecutive i, so it emits consecutive cells --
  // segments one voxel long. The final segment is structurally prone to this
  // too: the loop's last check is anchor->back, and when that fails it pushes
  // back-1 and then back, i.e. exactly one cell.
  //
  // A one-voxel segment is a bad corridor spine for two compounding reasons,
  // both measured:
  //   - it is diagonal, so the SFC's SAT corner term is half_res*sqrt(2) =
  //   0.354 m rather than
  //     0.25 m, and that is subtracted from every clearance the corridor
  //     measures;
  //   - the corridor box still extends the full sfc_end_ext (5 m) past it, so
  //   the box is ~8.6x
  //     longer than the segment it represents (against 1.7x for a healthy 15 m
  //     one) and sweeps obstacles that have nothing to do with this piece of
  //     the path.
  // The result was a corridor with one lateral bound at EXACTLY 0 -- observed
  // as "len 0.707107 ... y +/- 0/4.01902 ... r_y 0.353553" and again on the z
  // axis -- which makes the QP primal-infeasible on geometry, so
  // time-stretching cannot help and the vehicle stops.
  //
  // Dropping the waypoint is only done when the merged line is still clear, so
  // this cannot cut a corner into an obstacle. It is therefore BEST EFFORT: at
  // a genuinely tight corner the line-of-sight check will refuse and the short
  // segment survives. That is the safe failure.
  //
  // NOT the reverted min_corridor_half_width: that imposed a width floor after
  // the fact and fragmented the corridor chain. This removes a degenerate
  // segment before a corridor is ever built for it, and leaves the chain
  // contiguous.
  if (min_segment_voxels_ > 0.0) {
    bool changed = true;
    while (changed && smoothed_path.size() > 2) {
      changed = false;
      for (std::size_t i = 1; i + 1 < smoothed_path.size(); ++i) {
        const double before =
            (smoothed_path[i] - smoothed_path[i - 1]).cast<double>().norm();
        const double after =
            (smoothed_path[i + 1] - smoothed_path[i]).cast<double>().norm();
        if (std::min(before, after) >= min_segment_voxels_)
          continue;
        if (!is_line_of_sight_clear(smoothed_path[i - 1], smoothed_path[i + 1]))
          continue;
        smoothed_path.erase(smoothed_path.begin() + static_cast<long>(i));
        changed = true;
        break; // indices shifted; restart the scan
      }
    }
  }

  path_ = smoothed_path;
  return path_;
}

bool AStarSFCPlanner::is_line_of_sight_clear(const Eigen::Vector3i &idx_a,
                                             const Eigen::Vector3i &idx_b,
                                             int margin_cells) {
  double res = voxel_grid_.voxel_resolution;

  // --- 1. DISCRETE VOXEL WALK (preferred) ---
  //
  // Walk the inflated occupancy grid with Amanatides-Woo: O(cells along the
  // ray) with O(1) probes, versus the continuous branch below which tests the
  // segment against EVERY obstacle box. Once occupancy accumulates, that box
  // list runs to thousands of entries and the smoother issues O(path^2) of
  // these calls, so the linear form dominated the whole front-end. The
  // voxelised inflation is a superset of the continuous boxes, so this is the
  // more conservative of the two -- it can only ever refuse a line the
  // continuous test would have allowed, never the reverse.
  if (voxel_grid_.has_occupancy()) {
    return !discrete_ray_hits_obstacle(idx_a, idx_b, margin_cells);
  }

  // --- 2. CONTINUOUS FALLBACK (no voxelised occupancy available) ---
  if ((voxel_grid_.continuous_inflated_bounds &&
       !voxel_grid_.continuous_inflated_bounds->empty())) {
    Eigen::Vector3d p0 =
        idx_a.cast<double>() * res + Eigen::Vector3d::Constant(res / 2.0);
    Eigen::Vector3d p1 =
        idx_b.cast<double>() * res + Eigen::Vector3d::Constant(res / 2.0);
    Eigen::Vector3d d = p1 - p0;

    Eigen::Vector3d inv_d;
    for (int i = 0; i < 3; ++i) {
      // Avoid division by zero!
      inv_d(i) =
          (d(i) == 0.0) ? std::numeric_limits<double>::infinity() : 1.0 / d(i);
    }

    for (const auto &box : *voxel_grid_.continuous_inflated_bounds) {
      Eigen::Vector3d t1, t2;

      // Slab test. The parallel case (d(i) == 0) has to short-circuit, because
      // it cannot be expressed through t1/t2: those get sorted by
      // cwiseMin/cwiseMax below, so there is no pair of values that yields
      // "enter after exit" on one axis.
      //
      // Previously the outside-and-parallel case set t1=+inf, t2=-inf, which
      // sorts to t_min=-inf / t_max=+inf -- i.e. the axis stopped constraining
      // anything and the box was effectively treated as an INFINITE SLAB along
      // it. A level flight path has d.z()==0 exactly, so every horizontal
      // line-of-sight test silently ignored altitude and collided with the
      // accumulated ground sheet directly beneath it. That is why the smoother
      // could not collapse a path across open space: almost every check "hit"
      // the floor, so it emitted a waypoint per A* cell.
      bool parallel_miss = false;
      for (int i = 0; i < 3; ++i) {
        if (d(i) == 0.0) {
          if (p0(i) < box.first(i) || p0(i) > box.second(i)) {
            parallel_miss =
                true; // ray runs alongside this box, never enters it
            break;
          }
          t1(i) = -std::numeric_limits<double>::infinity();
          t2(i) = std::numeric_limits<double>::infinity();
        } else {
          t1(i) = (box.first(i) - p0(i)) * inv_d(i);
          t2(i) = (box.second(i) - p0(i)) * inv_d(i);
        }
      }
      if (parallel_miss)
        continue;

      Eigen::Vector3d t_min = t1.cwiseMin(t2);
      Eigen::Vector3d t_max = t1.cwiseMax(t2);

      double t_enter = t_min.maxCoeff();
      double t_exit = t_max.minCoeff();

      if (t_enter <= t_exit && t_exit >= 0.0 && t_enter <= 1.0) {
        return false; // Hit an obstacle
      }
    }
    return true;
  }
  // If the map is completely empty, there are no obstacles blocking the line of
  // sight.
  return true;
}

bool AStarSFCPlanner::discrete_ray_hits_obstacle(const Eigen::Vector3i &idx_a,
                                                 const Eigen::Vector3i &idx_b,
                                                 int margin_cells) const {
  {
    Eigen::Vector3i current_voxel = idx_a;
    Eigen::Vector3i end_voxel = idx_b;

    if (current_voxel == end_voxel)
      return false; // same cell, nothing to hit

    const auto &occ = *voxel_grid_.occupied_voxels_inflated;
    // With margin_cells > 0 the ray sweeps a thicker tube: each stepped-into
    // cell is tested along with its face neighbours. Probing only the face
    // neighbours (not the full 3x3x3) keeps this at 7 hash lookups per step
    // instead of 27; the diagonal cells it misses get visited by the walk
    // itself on any ray that would actually pass through them.
    auto blocked = [&](const Eigen::Vector3i &c) {
      if (occ.find(c) != occ.end())
        return true;
      for (int m = 1; m <= margin_cells; ++m) {
        for (int axis = 0; axis < 3; ++axis) {
          Eigen::Vector3i n = c;
          n[axis] += m;
          if (occ.find(n) != occ.end())
            return true;
          n[axis] -= 2 * m;
          if (occ.find(n) != occ.end())
            return true;
        }
      }
      return false;
    };

    Eigen::Vector3d dir = (end_voxel - current_voxel).cast<double>();

    // Amanatides-Woo Raycast
    int stepX = (dir.x() > 0) ? 1 : ((dir.x() < 0) ? -1 : 0);
    int stepY = (dir.y() > 0) ? 1 : ((dir.y() < 0) ? -1 : 0);
    int stepZ = (dir.z() > 0) ? 1 : ((dir.z() < 0) ? -1 : 0);

    double tDeltaX = (stepX != 0) ? std::abs(1.0 / dir.x())
                                  : std::numeric_limits<double>::infinity();
    double tDeltaY = (stepY != 0) ? std::abs(1.0 / dir.y())
                                  : std::numeric_limits<double>::infinity();
    double tDeltaZ = (stepZ != 0) ? std::abs(1.0 / dir.z())
                                  : std::numeric_limits<double>::infinity();

    double tMaxX =
        (stepX != 0) ? 0.5 * tDeltaX : std::numeric_limits<double>::infinity();
    double tMaxY =
        (stepY != 0) ? 0.5 * tDeltaY : std::numeric_limits<double>::infinity();
    double tMaxZ =
        (stepZ != 0) ? 0.5 * tDeltaZ : std::numeric_limits<double>::infinity();

    while (current_voxel != end_voxel) {
      if (blocked(current_voxel)) {
        return true; // hit
      }

      if (tMaxX < tMaxY) {
        if (tMaxX < tMaxZ) {
          current_voxel.x() += stepX;
          tMaxX += tDeltaX;
        } else {
          current_voxel.z() += stepZ;
          tMaxZ += tDeltaZ;
        }
      } else {
        if (tMaxY < tMaxZ) {
          current_voxel.y() += stepY;
          tMaxY += tDeltaY;
        } else {
          current_voxel.z() += stepZ;
          tMaxZ += tDeltaZ;
        }
      }
    }

    return voxel_grid_.occupied_voxels_inflated->find(current_voxel) !=
           voxel_grid_.occupied_voxels_inflated->end();
  }
}

double AStarSFCPlanner::get_safe_extension_length(const Eigen::Vector3i &idx_a,
                                                  const Eigen::Vector3i &idx_b,
                                                  double requested_extension) {
  if (requested_extension <= 0.0)
    return 0.0;
  double res = voxel_grid_.voxel_resolution;

  // --- 1. CONTINUOUS FRONT-END ---
  if ((voxel_grid_.continuous_inflated_bounds &&
       !voxel_grid_.continuous_inflated_bounds->empty())) {
    Eigen::Vector3d p0 = idx_a.cast<double>() * res;
    Eigen::Vector3d p1 = idx_b.cast<double>() * res;
    Eigen::Vector3d d = p1 - p0;
    double dist = d.norm();
    if (dist == 0)
      return 0.0;

    Eigen::Vector3d dir_unit = d / dist;
    Eigen::Vector3d inv_d;
    for (int i = 0; i < 3; ++i) {
      inv_d(i) = (dir_unit(i) == 0.0) ? std::numeric_limits<double>::infinity()
                                      : 1.0 / dir_unit(i);
    }

    double min_safe_distance = requested_extension;
    for (const auto &box : *voxel_grid_.continuous_inflated_bounds) {
      Eigen::Vector3d t1 = (box.first - p1).cwiseProduct(inv_d);
      Eigen::Vector3d t2 = (box.second - p1).cwiseProduct(inv_d);

      Eigen::Vector3d t_min = t1.cwiseMin(t2);
      Eigen::Vector3d t_max = t1.cwiseMax(t2);

      double t_enter = t_min.maxCoeff();
      double t_exit = t_max.minCoeff();

      if (t_enter <= t_exit && t_exit >= 0.0) {
        if (t_enter > 0 && t_enter < min_safe_distance) {
          min_safe_distance = std::max(0.0, t_enter - 0.01);
        }
      }
    }
    return min_safe_distance;
  }
  // --- 2. DISCRETE LIDAR ---
  else if (voxel_grid_.has_occupancy()) {
    Eigen::Vector3d p1 = idx_b.cast<double>() * res;
    Eigen::Vector3d p0 = idx_a.cast<double>() * res;
    Eigen::Vector3d d = p1 - p0;
    double dist = d.norm();
    if (dist == 0)
      return 0.0;

    Eigen::Vector3d dir_unit = d / dist;
    Eigen::Vector3d target_pt = p1 + dir_unit * requested_extension;

    Eigen::Vector3i current_voxel = (p1 / res).array().round().cast<int>();
    Eigen::Vector3i end_voxel = (target_pt / res).array().round().cast<int>();

    if (current_voxel == end_voxel)
      return requested_extension;

    Eigen::Vector3d dir_voxel = (target_pt - p1) / res;

    int stepX = (dir_voxel.x() > 0) ? 1 : ((dir_voxel.x() < 0) ? -1 : 0);
    int stepY = (dir_voxel.y() > 0) ? 1 : ((dir_voxel.y() < 0) ? -1 : 0);
    int stepZ = (dir_voxel.z() > 0) ? 1 : ((dir_voxel.z() < 0) ? -1 : 0);

    double tDeltaX = (stepX != 0) ? std::abs(1.0 / dir_voxel.x())
                                  : std::numeric_limits<double>::infinity();
    double tDeltaY = (stepY != 0) ? std::abs(1.0 / dir_voxel.y())
                                  : std::numeric_limits<double>::infinity();
    double tDeltaZ = (stepZ != 0) ? std::abs(1.0 / dir_voxel.z())
                                  : std::numeric_limits<double>::infinity();

    double tMaxX =
        (stepX != 0) ? 0.5 * tDeltaX : std::numeric_limits<double>::infinity();
    double tMaxY =
        (stepY != 0) ? 0.5 * tDeltaY : std::numeric_limits<double>::infinity();
    double tMaxZ =
        (stepZ != 0) ? 0.5 * tDeltaZ : std::numeric_limits<double>::infinity();

    double t_hit = 1.0;

    while (current_voxel != end_voxel) {
      if (voxel_grid_.occupied_voxels_inflated->find(current_voxel) !=
          voxel_grid_.occupied_voxels_inflated->end()) {
        break;
      }

      if (tMaxX < tMaxY) {
        if (tMaxX < tMaxZ) {
          t_hit = tMaxX;
          current_voxel.x() += stepX;
          tMaxX += tDeltaX;
        } else {
          t_hit = tMaxZ;
          current_voxel.z() += stepZ;
          tMaxZ += tDeltaZ;
        }
      } else {
        if (tMaxY < tMaxZ) {
          t_hit = tMaxY;
          current_voxel.y() += stepY;
          tMaxY += tDeltaY;
        } else {
          t_hit = tMaxZ;
          current_voxel.z() += stepZ;
          tMaxZ += tDeltaZ;
        }
      }
    }

    // If we hit a voxel, t_hit is the normalized distance along dir_voxel (0 to
    // 1). Since dir_voxel represents the full requested_extension, physical
    // distance is t_hit * requested_extension.
    if (voxel_grid_.occupied_voxels_inflated->find(current_voxel) !=
        voxel_grid_.occupied_voxels_inflated->end()) {
      return std::max(0.0, t_hit * requested_extension -
                               res); // Back off 1 voxel for safety
    }

    return requested_extension;
  }
  return requested_extension;
}

} // namespace path_finding
} // namespace trajectory_planner