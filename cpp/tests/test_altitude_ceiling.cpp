// The altitude ceiling (FrontEndConfig::max_altitude).
//
// min_altitude has always bounded the search downward; nothing bounded it upward except
// map_bounds.z/2, a map-SIZING number acting as a flight limit by accident. In the forest that
// left a window above the inflated trunk tops, and A* used it: a measured path climbed 4.14 ->
// 7.24 m, crossed above the canopy and dived back at 77 degrees from horizontal. That path is
// legal, so A* reported success -- but the corridor chain it implies turns far too sharply for
// consecutive boxes to share volume, and the QP came back primal-infeasible on the corridor
// block with boundary and velocity both satisfied to ~1e-5.
//
// The sign conversion is the part worth pinning down: NED altitude a lives at z = -a, so the
// HIGHEST altitude is the MOST NEGATIVE index and belongs in the geofence's LOWER bound. Getting
// that backwards turns the ceiling into a second floor, which would silently forbid normal
// flight rather than forbid overflight.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <memory>

#include "front_end.hpp"

namespace trajectory_planner {
namespace tests {
namespace {

constexpr double kRes = 0.5;

/// A barrier the search would rather fly over than go around: it spans the direct route but
/// stops at `top_alt`, with clear air beyond it laterally.
std::shared_ptr<mapping::SparseVoxelGrid> barrier_grid(double top_alt)
{
  auto grid = std::make_shared<mapping::SparseVoxelGrid>(kRes, /*inflation=*/0.5);
  std::vector<mapping::ObstacleBox> obs;
  // Wall across the path, from the ground up to top_alt, but only +/-6 m wide.
  obs.push_back({{4.0, -6.0, -top_alt}, {5.0, 6.0, 0.0}});
  grid->update_from_obstacles(obs);
  return grid;
}

FrontEndConfig base_config()
{
  FrontEndConfig c;
  c.voxel_resolution = kRes;
  c.drone_physical_radius = 0.5;
  c.min_altitude = 0.5;
  c.map_bounds = Eigen::Vector3d(60.0, 60.0, 30.0);   // +/-15 m vertically: plenty to fly over
  c.planning_horizon = -1.0;                           // search straight to the goal
  c.astar_max_nodes = 200000;
  c.astar_max_time_ms = 500.0;
  c.unknown_traversal_cost = 0.0;
  c.heat_radius = 0.0;
  c.heat_weight = 0.0;
  return c;
}

/// Highest altitude (metres, positive-up) reached by any corridor waypoint.
double peak_altitude(const FrontEndResult& r)
{
  double peak = -1e18;
  for (const auto& c : r.corridors) {
    peak = std::max({peak, -c.primaryPosition.z(), -c.secondaryPosition.z()});
  }
  return peak;
}

}  // namespace

// Without a ceiling the search climbs over the barrier -- this is the behaviour being removed,
// asserted so the test below is known to be measuring something real rather than a scene the
// search would have routed around anyway.
TEST(AltitudeCeiling, WithoutCeilingTheSearchClimbsOverTheBarrier)
{
  auto grid = barrier_grid(/*top_alt=*/6.0);
  FrontEndConfig cfg = base_config();
  cfg.max_altitude = -1.0;   // disabled

  FrontEndSFC fe(cfg, grid);
  const auto r = fe.get_corridors_astar(Eigen::Vector3d(-6.0, 0.0, -3.0),
                                        Eigen::Vector3d(12.0, 0.0, -3.0));
  ASSERT_FALSE(r.corridors.empty());
  EXPECT_GT(peak_altitude(r), 6.0)
      << "scene does not actually tempt the search upward; the ceiling test below would be "
         "vacuous";
}

// THE FIX. With the ceiling set at the barrier height, overflight is impossible and the search
// must solve the problem horizontally.
TEST(AltitudeCeiling, CeilingForcesHorizontalRouting)
{
  auto grid = barrier_grid(/*top_alt=*/6.0);
  FrontEndConfig cfg = base_config();
  cfg.max_altitude = 6.0;

  FrontEndSFC fe(cfg, grid);
  const auto r = fe.get_corridors_astar(Eigen::Vector3d(-6.0, 0.0, -3.0),
                                        Eigen::Vector3d(12.0, 0.0, -3.0));

  ASSERT_FALSE(r.corridors.empty());
  EXPECT_LE(peak_altitude(r), cfg.max_altitude + kRes)
      << "a corridor waypoint sits above the ceiling; the NED sign conversion is inverted or the "
         "bound is not reaching the geofence";
}

// The ceiling must not become a second floor. If the sign were flipped, the traversable band
// would be empty and even an unobstructed cruise would fail.
TEST(AltitudeCeiling, NormalCruiseIsUnaffected)
{
  auto grid = std::make_shared<mapping::SparseVoxelGrid>(kRes, 0.5);
  grid->update_from_obstacles({});   // empty world

  FrontEndConfig cfg = base_config();
  cfg.max_altitude = 6.0;

  FrontEndSFC fe(cfg, grid);
  const auto r = fe.get_corridors_astar(Eigen::Vector3d(-6.0, 0.0, -3.0),
                                        Eigen::Vector3d(12.0, 0.0, -3.0));

  EXPECT_FALSE(r.corridors.empty())
      << "cruising at 3 m under a 6 m ceiling produced no corridors -- the bound is inverted";
  EXPECT_TRUE(r.reached_goal) << "an empty world must be solvable end to end";
}

// THE CEILING MUST NOT EXCLUDE THE VEHICLE.
//
// max_altitude says where we would LIKE to fly. It is not a reason to be unable to plan from
// where we actually are. If the vehicle is above the ceiling and the geofence excludes it, A* is
// asked to search from a cell outside its own bounds and planning stops dead -- the vehicle
// climbs away at takeoff and then never recovers.
//
// Measured: takeoff overshoots the 3 m cruise altitude by ~4.4 m in every flight (vertical
// peak_true 4.26-4.91 m across ten runs), reaching ~7.4 m against a 6.0 m max_altitude -- z
// index -14/-15 against a geofence floor of -12.
TEST(AltitudeCeiling, PlansFromAboveTheCeilingInsteadOfFailing)
{
  auto grid = std::make_shared<mapping::SparseVoxelGrid>(kRes, 0.5);
  grid->update_from_obstacles({});             // clear world; only the ceiling is in play

  FrontEndConfig cfg = base_config();
  cfg.max_altitude = 6.0;
  FrontEndSFC fe(cfg, grid);

  // Vehicle at 7.5 m -- above the 6 m ceiling, exactly where takeoff overshoot leaves it.
  const auto r = fe.get_corridors_astar(Eigen::Vector3d(-6.0, 0.0, -7.5),
                                        Eigen::Vector3d(12.0, 0.0, -3.0));
  EXPECT_FALSE(r.corridors.empty())
      << "no corridors from 7.5 m under a 6 m ceiling -- the geofence excluded the vehicle and "
         "the search had nowhere to start, which is the takeoff-climb stall";
}

// And it must still route back DOWN rather than treating the excursion as the new normal.
TEST(AltitudeCeiling, RoutesBackDownTowardsLegalAltitude)
{
  auto grid = std::make_shared<mapping::SparseVoxelGrid>(kRes, 0.5);
  grid->update_from_obstacles({});

  FrontEndConfig cfg = base_config();
  cfg.max_altitude = 6.0;
  FrontEndSFC fe(cfg, grid);

  const auto r = fe.get_corridors_astar(Eigen::Vector3d(-6.0, 0.0, -7.5),
                                        Eigen::Vector3d(12.0, 0.0, -3.0));
  ASSERT_FALSE(r.corridors.empty());
  const double end_alt = -r.corridors.back().secondaryPosition.z();
  EXPECT_LT(end_alt, 7.5)
      << "the chain ends at " << end_alt << " m, no lower than the start -- containing the start "
         "in the geofence must not turn the ceiling into a free-for-all";
}

}  // namespace tests
}  // namespace trajectory_planner
