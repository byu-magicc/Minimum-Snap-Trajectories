#pragma once

#include "voxel_grid.hpp"
#include <Eigen/Dense>
#include <cstdint>
#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <unordered_set>
#include <vector>

namespace trajectory_planner {
namespace path_finding {

// ==========================================
// 1. NODE & DATA STRUCTURES (Replaces node.py)
// ==========================================
struct Node {
  double f_cost; // Placed first, mimicking your dataclass order
  double g_cost;
  double h_cost;
  Eigen::Vector3i position;
  Node *parent;

  Node(double f, double g, double h, Eigen::Vector3i pos, Node *p = nullptr)
      : f_cost(f), g_cost(g), h_cost(h), position(pos), parent(p) {}
};

// Custom Hash for Eigen::Vector3i to use O(1) unordered_set lookups
using Vector3iHash = mapping::Vector3iHash;

// Priority Queue Min-Heap Comparator
struct CompareNode {
  bool operator()(const Node *a, const Node *b) const {
    return a->f_cost > b->f_cost;
  }
};

struct BoundingBox {
  Eigen::Vector3d b_min;
  Eigen::Vector3d b_max;
};

// Lightweight Interface to hold your Voxel Grid data
using VoxelSet = std::unordered_set<Eigen::Vector3i, Vector3iHash>;

// Lightweight VIEW over a SparseVoxelGrid.
//
// These are borrowed pointers, not owned copies. The planner is rebuilt on
// every replan, so copying the occupancy and free sets in here cost several
// milliseconds per cycle and grew without bound as the map accumulated. The
// referenced grid must outlive the planner -- FrontEndSFC holds a shared_ptr to
// it, which guarantees that.
struct VoxelGridData {
  double voxel_resolution = 0.5;
  const VoxelSet *occupied_voxels_inflated = nullptr;
  const VoxelSet *free_voxels = nullptr;
  const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>
      *continuous_inflated_bounds = nullptr;
  /// Proximity cost in [0, 1] per cell; null or empty means the field is off.
  const std::unordered_map<Eigen::Vector3i, float, Vector3iHash> *heat =
      nullptr;

  double heat_at(const Eigen::Vector3i &c) const {
    if (!heat) {
      return 0.0;
    }
    auto it = heat->find(c);
    return it == heat->end() ? 0.0 : static_cast<double>(it->second);
  }

  bool has_occupancy() const {
    return occupied_voxels_inflated && !occupied_voxels_inflated->empty();
  }
  bool has_free() const { return free_voxels && !free_voxels->empty(); }
};

// ==========================================
// 2. PLANNER INTERFACE (Replaces astar_sfc.py)
// ==========================================
class AStarSFCPlanner {
public:
  // Pass the pre-computed grid and the discrete bounding limits
  AStarSFCPlanner(const VoxelGridData &grid, const Eigen::Vector3i &bounds_min,
                  const Eigen::Vector3i &bounds_max);

  std::vector<Eigen::Vector3i>
  search(const Eigen::Vector3i &start_pos, const Eigen::Vector3i &goal_pos,
         double max_time_ms = 10.0, double sensor_range_voxels = -1.0,
         int max_z_index = std::numeric_limits<int>::max(),
         double fov_vertical_min_deg = -90.0,
         double fov_vertical_max_deg = 90.0,
         double unknown_traversal_cost = 2.0, int max_nodes = 6000,
         double clearance_penalty = 0.0, double heat_weight = 0.0,
         double heuristic_weight = 2.0, double soft_obstacle_cost = 0.0,
         double min_segment_voxels = 0.0);
  /**
   * Move the search geofence.
   *
   * The bounds were fixed at construction from map_bounds, centred on the WORLD
   * ORIGIN, which silently made them a mission limit rather than a search
   * limit: at map_bounds 100 m and 0.5 m voxels the box is +/-50 m about (0,0),
   * so a vehicle flying a 105 m course hit an invisible wall at exactly the
   * halfway point. Every neighbour beyond it is rejected, so the search can
   * only expand backwards and sideways -- it burns the whole node budget and
   * returns a 2-cell path, in visibly open air, which the allocator then floors
   * into a 4 s trajectory. That reads as an inexplicable hover.
   *
   * A receding-horizon search should be bounded RELATIVE to where it starts.
   * Call this before each search so the box follows the vehicle.
   */
  /**
   * Supply the previously committed route, as a set of voxel cells, to be
   * preferred.
   *
   * Borrowed, not copied -- the caller must keep it alive for the search. Pass
   * nullptr or a zero bonus to disable. See
   * FrontEndConfig::path_hysteresis_bonus for why this exists.
   */
  void set_previous_path(const VoxelSet *cells, double bonus) {
    prev_path_cells_ = cells;
    hysteresis_bonus_ = bonus;
  }

  void set_bounds(const Eigen::Vector3i &bounds_min,
                  const Eigen::Vector3i &bounds_max) {
    min_bounds_ = bounds_min;
    max_bounds_ = bounds_max;
  }

  std::vector<Eigen::Vector3i> sfc_smoother();

  /**
   * @param margin_cells extra clearance demanded either side of the ray, in
   * voxels. 0 is the safety test (does the line itself hit anything); >0
   * additionally refuses lines that merely graze inflation. See sfc_smoother()
   * for why that distinction matters.
   */
  bool is_line_of_sight_clear(const Eigen::Vector3i &idx_a,
                              const Eigen::Vector3i &idx_b,
                              int margin_cells = 0);

  /** Amanatides-Woo walk over the inflated occupancy set. True if the ray hits
   * anything. */
  bool discrete_ray_hits_obstacle(const Eigen::Vector3i &idx_a,
                                  const Eigen::Vector3i &idx_b,
                                  int margin_cells = 0) const;
  double get_safe_extension_length(const Eigen::Vector3i &idx_a,
                                   const Eigen::Vector3i &idx_b,
                                   double requested_extension);

  std::vector<Eigen::Vector3i> get_path() const { return path_; }
  /// Seed the raw path directly. Test seam: it lets sfc_smoother() be exercised on an exact
  /// waypoint sequence (a collinear staircase, a reversal) instead of whatever a search happens
  /// to produce, which is how the collinear-merge cases are pinned down.
  void set_path(const std::vector<Eigen::Vector3i> & path) { path_ = path; }

  /// Toggle the collinear-waypoint merge in sfc_smoother(). Exists so the pass can be A/B'd
  /// offline: it changes the corridor chain, and a change to the chain cannot be judged from a
  /// single closed-loop run when the benchmark's travel time has 2.3x run-to-run spread.
  void set_collinear_merge(bool on) { collinear_merge_ = on; }
  std::vector<Eigen::Vector3i> get_shadowed_voxels() const {
    return shadowed_voxels_;
  }

private:
  std::vector<Eigen::Vector3i> reconstruct_path(Node *end_node);

  const VoxelGridData &voxel_grid_;
  double heat_weight_ = 0.0;
  /// >0 makes inflated cells traversable at this penalty instead of blocked.
  double soft_obstacle_cost_ = 0.0;
  /// Smoothed segments shorter than this (in voxels) are merged away where
  /// line-of-sight allows. 0 disables. See sfc_smoother().
  double min_segment_voxels_ = 0.0;
  /// Previously committed route (borrowed) and the discount applied to cells on
  /// it.
  const VoxelSet *prev_path_cells_ = nullptr;
  double hysteresis_bonus_ = 0.0;
  double heuristic_weight_ = 2.0;
  bool collinear_merge_ = true;
  /// Toll for stepping between inflated cells while escaping a blocked START cell. Steep so any
  /// free route wins outright; finite so a trapped start still produces a path. See the escape
  /// in search().
  static constexpr double kEscapeCost = 50.0;
  Eigen::Vector3i min_bounds_;
  Eigen::Vector3i max_bounds_;

  double fov_min_rad_ = -M_PI;
  double fov_max_rad_ = M_PI;
  bool check_fov_ = false;
  double clearance_cost_ = 1.0;
  double unknown_cost_ = 2.0;
  Eigen::Vector3i search_start_pos_;

  // Store precalculated 26 directions: (dx, dy, dz, move_cost)
  struct Direction {
    int dx, dy, dz;
    double cost;
  };
  std::vector<Direction> directions_;

  std::vector<Eigen::Vector3i> path_;
  std::vector<Eigen::Vector3i> shadowed_voxels_;
};

} // namespace path_finding
} // namespace trajectory_planner