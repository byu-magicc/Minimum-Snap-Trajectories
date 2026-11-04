// Planning must not stop because the MAP believes the vehicle is inside an obstacle.
//
// The vehicle is physically wherever it is. A* refusing to expand because the start cell sits
// in inflated space is never a useful answer -- and it is not a rare corner: inflation is
// hull + obstacle_margin = 0.690 m, while the measured horizontal estimator bias on this
// airframe is 0.55-0.66 m. A vehicle genuinely clear by 0.7 m can therefore have its BELIEVED
// position inside a trunk's inflation shell. Every neighbour inside that shell is refused too,
// so the search is trapped and the vehicle hovers with open space all around it.
//
// Measured on hard s0 at v_max 5: the start cell was INFLATED-BLOCKED in 85 of 137 short-path
// diagnostics -- the single largest category.
//
// The escape may only ever walk OUTWARD: from a FREE node a blocked neighbour is still refused,
// so this cannot be used to enter an obstacle, only to leave one. These tests pin both halves.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <vector>

#include "astar_sfc.hpp"

namespace trajectory_planner {
namespace path_finding {
namespace tests {
namespace {

/// Solid block spanning [lo, hi] in x and y, and the FULL z range of the search bounds.
///
/// Spanning z is not incidental. A first version of this filled only z = 0, and every test
/// here passed with the escape stubbed out -- A* simply stepped to z = 1 and flew over the
/// blob. A scenario that leaves a trivial way out cannot detect whether the escape works.
void block(VoxelSet& o, int xlo, int xhi, int ylo, int yhi)
{
  for (int x = xlo; x <= xhi; ++x) {
    for (int y = ylo; y <= yhi; ++y) {
      for (int z = -3; z <= 3; ++z) { o.insert(Eigen::Vector3i(x, y, z)); }
    }
  }
}

std::vector<Eigen::Vector3i> plan(VoxelSet& occ, const Eigen::Vector3i& start,
                                  const Eigen::Vector3i& goal)
{
  VoxelGridData grid;
  grid.voxel_resolution = 1.0;
  grid.occupied_voxels_inflated = &occ;
  AStarSFCPlanner planner(grid, Eigen::Vector3i(-20, -20, -2), Eigen::Vector3i(20, 20, 2));
  return planner.search(start, goal, /*max_time_ms=*/200.0, /*sensor_range=*/-1.0,
                        /*max_z=*/100, -90.0, 90.0, /*unknown=*/0.5, /*max_nodes=*/50000,
                        /*clearance=*/0.0, /*heat_w=*/0.0, /*heuristic_w=*/3.0,
                        /*soft_obstacle=*/0.0, /*min_seg=*/0.0);
}

}  // namespace

// THE FIX. Start buried in the middle of an inflated blob, goal in open space beyond it.
TEST(BlockedStart, PlansOutOfAnInflatedStartCell)
{
  VoxelSet occ;
  block(occ, -2, 2, -2, 2);                       // 5x5 blob centred on the origin
  const Eigen::Vector3i start(0, 0, 0);           // dead centre, fully enclosed
  ASSERT_EQ(occ.count(start), 1u) << "scenario must actually bury the start";

  const auto path = plan(occ, start, Eigen::Vector3i(10, 0, 0));

  ASSERT_GE(path.size(), 2u)
      << "no path from a blocked start -- the search is trapped in the blob and the vehicle "
         "hovers with open space all around it";
  EXPECT_EQ(path.front(), start) << "path must begin at the vehicle";
  EXPECT_EQ(path.back(), Eigen::Vector3i(10, 0, 0)) << "path must reach the goal";
}

// The escape must END. A path that stays inside the blob is not an escape.
TEST(BlockedStart, LeavesTheObstacleAndStaysOut)
{
  VoxelSet occ;
  block(occ, -2, 2, -2, 2);
  const auto path = plan(occ, Eigen::Vector3i(0, 0, 0), Eigen::Vector3i(10, 0, 0));
  ASSERT_GE(path.size(), 2u);

  // Once the path reaches a free cell it must never re-enter inflated space: from a free node
  // the ordinary rejection applies, so a re-entry would mean the escape had leaked.
  bool escaped = false;
  for (const auto& c : path) {
    const bool blocked = occ.count(c) > 0;
    if (!blocked) { escaped = true; }
    else if (escaped) {
      ADD_FAILURE() << "path re-entered inflated space at [" << c.transpose()
                    << "] after reaching free space -- the escape is not one-way";
    }
  }
  EXPECT_TRUE(escaped) << "path never left the blob";
}

// THE HALF THAT MUST NOT CHANGE. From a FREE start, a wall is still a wall: the escape must not
// have turned every obstacle into a toll road.
TEST(BlockedStart, AFreeStartStillCannotEnterAnObstacle)
{
  VoxelSet occ;
  block(occ, 3, 5, -10, 10);                      // wall spanning the direct route

  const auto path = plan(occ, Eigen::Vector3i(0, 0, 0), Eigen::Vector3i(10, 0, 0));
  ASSERT_GE(path.size(), 2u) << "the wall is finite in y, so a route around it exists";
  for (const auto& c : path) {
    EXPECT_EQ(occ.count(c), 0u)
        << "path from a FREE start crosses inflated cell [" << c.transpose()
        << "] -- the blocked-start escape has leaked into ordinary planning";
  }
}

}  // namespace tests
}  // namespace path_finding
}  // namespace trajectory_planner
