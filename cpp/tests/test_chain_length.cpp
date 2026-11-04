// The corridor chain must cover useful GROUND, not just a corridor count.
//
// max_sfc_count counts corridors. When segments are short -- threading a gap makes A* emit
// one-voxel ones -- three corridors can span under 2 m. Every exclusive control-point pool then
// zeroes out (allocate_dynamic_control_points gives none to a segment shorter than
// sfc_start_ext + sfc_end_ext), the plan becomes nothing but junction bridges, and the chain ENDS
// essentially where the vehicle already is. The QP solves start-to-end, reports success, and
// commands a hover -- permanently, because the vehicle never moves to change the situation.
//
// Measured in flight on hard s0 at v_max 8 before min_chain_length existed: T_trav 1662 s,
// v_avg 0.11 m/s, 14502 replans, route unchanged 99.53%, median route divergence 0.00 deg, with
// the planner reporting success at 10 Hz the entire time.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <vector>

#include "front_end.hpp"
#include "voxel_grid.hpp"

namespace trajectory_planner {
namespace tests {
namespace {

constexpr double kRes = 0.5;
constexpr double kHull = 0.490491;

FrontEndConfig config(double min_chain)
{
  FrontEndConfig c;
  c.voxel_resolution = kRes;
  c.drone_physical_radius = kHull;
  c.v_max = 8.0; c.a_max = 20.0;
  c.spline_type = "clamped"; c.degree = 4;
  c.planning_horizon = 15.0;
  c.min_altitude = 0.5; c.max_altitude = 6.0;
  c.map_bounds = Eigen::Vector3d(240, 120, 24);
  c.astar_max_nodes = 4000; c.astar_max_time_ms = 50.0;
  c.unknown_traversal_cost = 0.5; c.astar_heuristic_weight = 3.0;
  c.heat_radius = 1.0; c.heat_weight = 0.6;
  c.astar_min_segment_voxels = 2.0;
  c.sfc_width = 5.0; c.sfc_start_ext = 1.0; c.sfc_end_ext = 5.0;
  c.max_sfc_count = 3;
  c.min_chain_length = min_chain;
  return c;
}

/// Two trunks leaving a narrow gap, with a third just past it. This is the geometry that
/// reproduced the stall: threading the gap forces A* into one-voxel segments, so three corridors
/// span only a few metres. A staggered slalom does NOT reproduce it -- the segments there stay
/// long enough that the count cap alone already covers useful ground.
std::shared_ptr<mapping::SparseVoxelGrid> gap_scene(double gap = 3.0)
{
  auto grid = std::make_shared<mapping::SparseVoxelGrid>(kRes, kHull + 0.20);
  std::vector<mapping::ObstacleBox> obs;
  const double r = 1.25, h = kRes;
  auto trunk = [&](double cx, double cy) {
    for (double x = cx - r; x <= cx + r + 1e-9; x += h) {
      for (double y = cy - r; y <= cy + r + 1e-9; y += h) {
        if (std::hypot(x - cx, y - cy) <= r) obs.push_back({{x, y, -6.0}, {x + h, y + h, 0.0}});
      }
    }
  };
  trunk(10.0, gap / 2.0 + r);
  trunk(10.0, -(gap / 2.0 + r));
  trunk(14.0, 0.0);                 // clips the run-out just past the gap
  grid->update_from_obstacles(obs);
  return grid;
}

double chain_length(const FrontEndResult & r)
{
  double L = 0.0;
  for (const auto & s : r.corridors) L += (s.secondaryPosition - s.primaryPosition).norm();
  return L;
}

}  // namespace

// THE REGRESSION. With the cap counting corridors only, the chain is free to collapse.
TEST(ChainLength, CoversTheRequestedDistanceEvenWhenSegmentsAreShort)
{
  auto grid = gap_scene();
  const double want = 8.0;

  FrontEndSFC capped(config(0.0), grid);      // corridor count only -- the old behaviour
  FrontEndSFC distance(config(want), grid);   // distance-aware

  bool exercised = false;
  for (double x = 6.0; x <= 9.5; x += 0.5) {
    const Eigen::Vector3d start(x, 0.0, -4.0);
    const Eigen::Vector3d goal(40.0, 0.0, -4.0);
    const auto a = capped.get_corridors_astar(start, goal);
    const auto b = distance.get_corridors_astar(start, goal);
    if (a.corridors.empty() || b.corridors.empty()) continue;

    EXPECT_GE(chain_length(b), std::min(want, chain_length(a)) - 1e-6)
        << "at x = " << x << " the distance-aware chain (" << chain_length(b)
        << " m) is shorter than the count-capped one (" << chain_length(a) << " m)";
    if (chain_length(a) < want - 1e-6) {
      exercised = true;
      EXPECT_GT(chain_length(b), chain_length(a))
          << "at x = " << x << " the count cap gave only " << chain_length(a)
          << " m and min_chain_length did not extend it";
    }
  }
  EXPECT_TRUE(exercised)
      << "no position produced a chain under " << want
      << " m, so this scenario never tested the fix -- the gap geometry has drifted and is no "
         "longer forcing short segments";
}

// The hard ceiling still has to bound QP growth: a chain of one-voxel segments must not be
// allowed to grow without limit chasing the distance.
TEST(ChainLength, HardCeilingStillBoundsCorridorCount)
{
  auto grid = gap_scene();
  FrontEndConfig c = config(1000.0);   // unreachable distance
  c.max_sfc_count_hard = 5;
  FrontEndSFC fe(c, grid);
  const auto r = fe.get_corridors_astar(Eigen::Vector3d(6, 0, -4), Eigen::Vector3d(40, 0, -4));
  EXPECT_LE(r.corridors.size(), 5u)
      << "chain grew to " << r.corridors.size() << " corridors past the hard ceiling";
}

// Disabling must be exact, so the previous behaviour is recoverable.
TEST(ChainLength, ZeroIsANoOp)
{
  auto grid = gap_scene();
  FrontEndSFC off(config(0.0), grid);
  const auto r = off.get_corridors_astar(Eigen::Vector3d(6, 0, -4), Eigen::Vector3d(40, 0, -4));
  EXPECT_LE(r.corridors.size(), 3u) << "with min_chain_length 0 the count cap must still bind";
}

}  // namespace tests
}  // namespace trajectory_planner
