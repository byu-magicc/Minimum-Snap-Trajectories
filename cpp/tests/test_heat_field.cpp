// Proximity ("heat") field: the soft half of obstacle avoidance.
//
// The inflated occupancy set is a BINARY test, so a search using it alone has no reason to
// prefer the middle of a gap over grazing its edge. These cover the properties the A* step cost
// actually relies on -- monotonic falloff, zero inside the untraversable region, zero beyond the
// halo, and max- rather than sum-combining between neighbouring obstacles.

#include <gtest/gtest.h>

#include <cmath>

#include "front_end.hpp"
#include "voxel_grid.hpp"

using trajectory_planner::mapping::SparseVoxelGrid;

namespace
{
constexpr double kRes = 0.5;
constexpr double kInflation = 0.5;
constexpr double kHeatRadius = 1.0;

/// A grid holding one obstacle voxel at the origin, with the field enabled.
SparseVoxelGrid make_grid_with_one_obstacle()
{
  SparseVoxelGrid grid(kRes, kInflation);
  grid.set_heat_radius(kHeatRadius);
  // A single return at the origin. update_from_scan wants a sensor origin and hit points; put
  // the sensor well away along +x so the beam does not sweep the cells being measured.
  grid.update_from_scan(Eigen::Vector3d(10.0, 0.0, 0.0), {Eigen::Vector3d(0.0, 0.0, 0.0)});
  return grid;
}

Eigen::Vector3i idx_at(double x, double y, double z)
{
  return Eigen::Vector3i(static_cast<int>(std::floor(x / kRes)),
                         static_cast<int>(std::floor(y / kRes)),
                         static_cast<int>(std::floor(z / kRes)));
}
}  // namespace

// Cells inside the inflated set are untraversable, so charging them is meaningless -- and would
// put heat where A* never looks while inflating the map's memory.
TEST(HeatField, ZeroInsideTheInflatedSet)
{
  const SparseVoxelGrid grid = make_grid_with_one_obstacle();
  EXPECT_DOUBLE_EQ(grid.heat_at(idx_at(0.0, 0.0, 0.0)), 0.0);
}

// Far from anything there is nothing to avoid; a non-zero floor here would be a flat surcharge
// on the whole search rather than a clearance preference.
TEST(HeatField, ZeroBeyondTheHalo)
{
  const SparseVoxelGrid grid = make_grid_with_one_obstacle();
  EXPECT_DOUBLE_EQ(grid.heat_at(idx_at(6.0, 0.0, 0.0)), 0.0);
}

// The whole point: cost must fall off with distance, so the search has a gradient to follow
// toward the middle of a gap instead of a cliff edge it cannot see past.
TEST(HeatField, DecreasesMonotonicallyWithDistance)
{
  const SparseVoxelGrid grid = make_grid_with_one_obstacle();
  // Start OUTSIDE the inflated set. Inside it heat is legitimately 0 (those cells are
  // untraversable), so a scan starting at the obstacle sees 0 -> positive -> decay, and would
  // report that rise as a monotonicity failure when it is the intended shape.
  double prev = 2.0;   // above any legal value
  bool saw_positive = false;
  bool started = false;
  for (double x = 0.5; x <= 4.0; x += kRes) {
    const Eigen::Vector3i c = idx_at(x, 0.0, 0.0);
    if (!started) {
      if (grid.heat_at(c) <= 0.0) { continue; }   // still inside the inflated region
      started = true;
    }
    const double h = grid.heat_at(c);
    EXPECT_GE(h, 0.0);
    EXPECT_LE(h, 1.0);
    EXPECT_LE(h, prev + 1e-9) << "heat rose with distance at x = " << x;
    if (h > 0.0) { saw_positive = true; }
    prev = h;
  }
  EXPECT_TRUE(saw_positive) << "field never became positive anywhere outside the inflated set";
}

// Heat answers "how close is the nearest obstacle", which is a property of the cell. Summing
// contributions would make a cell midway between two distant trunks score WORSE than one
// grazing a single trunk -- the exact opposite of a clearance cost.
TEST(HeatField, CombinesByMaxNotSum)
{
  SparseVoxelGrid one(kRes, kInflation);
  one.set_heat_radius(kHeatRadius);
  one.update_from_scan(Eigen::Vector3d(0.0, 10.0, 0.0), {Eigen::Vector3d(0.0, 0.0, 0.0)});

  SparseVoxelGrid two(kRes, kInflation);
  two.set_heat_radius(kHeatRadius);
  two.update_from_scan(Eigen::Vector3d(0.0, 10.0, 0.0),
                       {Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(3.0, 0.0, 0.0)});

  // A probe near the first obstacle and far from the second must read identically in both.
  const Eigen::Vector3i probe = idx_at(0.9, 0.0, 0.0);
  EXPECT_DOUBLE_EQ(two.heat_at(probe), one.heat_at(probe));
  EXPECT_LE(two.heat_at(probe), 1.0);
}

// Disabling must be free, not merely cheap -- it is the escape hatch if the field ever costs
// more than it earns.
TEST(HeatField, DisabledWhenRadiusNonPositive)
{
  SparseVoxelGrid grid(kRes, kInflation);
  grid.set_heat_radius(0.0);
  grid.update_from_scan(Eigen::Vector3d(10.0, 0.0, 0.0), {Eigen::Vector3d(0.0, 0.0, 0.0)});
  EXPECT_TRUE(grid.get_heat().empty());
  EXPECT_DOUBLE_EQ(grid.heat_at(idx_at(0.75, 0.0, 0.0)), 0.0);
}

// The A* guidance invariant, asserted as arithmetic so a future config change trips a test
// rather than a flood: A* prefers a step toward the goal only while the summed step-cost
// multiplier stays below the heuristic weight.
TEST(HeatField, DefaultConfigSatisfiesTheGuidanceInvariant)
{
  trajectory_planner::FrontEndConfig cfg;
  const double m_max = 1.0 + cfg.unknown_traversal_cost + cfg.astar_clearance_cost
                     + cfg.heat_weight;
  EXPECT_LT(m_max, cfg.astar_heuristic_weight)
      << "step multiplier " << m_max << " >= heuristic weight " << cfg.astar_heuristic_weight
      << "; A* would stop preferring the goal and flood outward";
}
