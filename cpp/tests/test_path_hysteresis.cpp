// Path hysteresis: the search should prefer to stay on the route it committed to last cycle.
//
// Replanning at 10 Hz re-decides the route from scratch every cycle. Where two branches cost
// within a few percent of each other -- a junction with more than one way past an obstacle --
// the argmin flips on tiny changes in vehicle position, and the vehicle dithers between branches
// instead of committing. Observed in the forest roughly a third of the way along.
//
// A discount on cells belonging to the previous route makes continuity cheap without making it
// mandatory. It cannot cause a collision: inflated cells remain hard-blocked, so the worst it can
// do is hold a valid-but-suboptimal route slightly longer than necessary.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "front_end.hpp"

namespace trajectory_planner {
namespace tests {
namespace {

/// An obstacle squarely across the direct line, with clear air both sides -- a near-tie.
std::shared_ptr<mapping::SparseVoxelGrid> junction_grid()
{
  auto g = std::make_shared<mapping::SparseVoxelGrid>(0.5, 0.5);
  std::vector<mapping::ObstacleBox> obs;
  obs.push_back({{8.0, -2.0, -6.0}, {10.0, 2.0, 0.0}});
  g->update_from_obstacles(obs);
  return g;
}

FrontEndConfig cfg(double bonus)
{
  FrontEndConfig c;
  c.voxel_resolution = 0.5;
  c.drone_physical_radius = 0.5;
  c.min_altitude = 0.5;
  c.map_bounds = Eigen::Vector3d(80, 80, 30);
  c.planning_horizon = -1.0;
  c.astar_max_nodes = 200000;
  c.astar_max_time_ms = 500.0;
  c.unknown_traversal_cost = 0.0;
  c.heat_radius = 0.0;
  c.heat_weight = 0.0;
  c.path_hysteresis_bonus = bonus;
  c.path_hysteresis_radius_cells = 1;
  return c;
}

std::vector<Eigen::Vector3d> route_of(const FrontEndResult& r)
{
  std::vector<Eigen::Vector3d> p;
  for (const auto& c : r.corridors) { p.push_back(c.primaryPosition); }
  if (!r.corridors.empty()) { p.push_back(r.corridors.back().secondaryPosition); }
  return p;
}

/// Mean lateral offset of the route -- its sign says which side of the obstacle it passed.
double side_of(const FrontEndResult& r)
{
  double sum = 0.0;
  for (const auto& c : r.corridors) { sum += c.primaryPosition.y(); }
  return r.corridors.empty() ? 0.0 : sum / static_cast<double>(r.corridors.size());
}

/// Replan repeatedly while the start drifts laterally, counting how often the chosen side flips.
int count_flips(double bonus, int cycles = 6)
{
  auto grid = junction_grid();
  const Eigen::Vector3d goal(18.0, 0.0, -3.0);

  FrontEndSFC fe(cfg(bonus), grid);
  fe.set_previous_path({});
  const auto first = fe.get_corridors_astar(Eigen::Vector3d(0.0, 0.0, -3.0), goal);
  if (first.corridors.empty()) { return -1; }

  auto prev = route_of(first);
  double last = side_of(first);
  int flips = 0;

  for (int k = 1; k <= cycles; ++k) {
    // A fresh front end each cycle, because the real node rebuilds the planner every replan --
    // which is precisely why the route has to be handed back in rather than kept internally.
    FrontEndSFC fe_k(cfg(bonus), grid);
    fe_k.set_previous_path(prev);
    const double dy = (k % 2 ? 1.0 : -1.0) * 0.35;   // adversarial: drift reverses every cycle
    const auto r = fe_k.get_corridors_astar(Eigen::Vector3d(0.0, dy, -3.0), goal);
    if (r.corridors.empty()) { continue; }
    const double s = side_of(r);
    if ((s > 0.0) != (last > 0.0)) { ++flips; }
    last = s;
    prev = route_of(r);
  }
  return flips;
}

}  // namespace

// The scenario must actually dither without hysteresis, or the test below proves nothing.
TEST(PathHysteresis, ScenarioDithersWhenDisabled)
{
  const int flips = count_flips(0.0);
  ASSERT_GE(flips, 0) << "scenario failed to plan at all";
  EXPECT_GE(flips, 3)
      << "the junction no longer produces a near-tie, so it cannot demonstrate hysteresis";
}

// The fix.
TEST(PathHysteresis, DiscountSuppressesFlipping)
{
  const int without = count_flips(0.0);
  const int with = count_flips(0.20);
  ASSERT_GE(without, 0);
  ASSERT_GE(with, 0);
  EXPECT_LT(with, without)
      << "hysteresis did not reduce route flipping (" << with << " vs " << without << ")";
  EXPECT_LE(with, 1) << "still dithering with hysteresis enabled";
}

// Disabling must be exact -- it is the escape hatch if stickiness ever costs more than it earns.
TEST(PathHysteresis, ZeroBonusIsIdenticalToNoPreviousPath)
{
  auto grid = junction_grid();
  const Eigen::Vector3d start(0.0, 0.35, -3.0), goal(18.0, 0.0, -3.0);

  FrontEndSFC a(cfg(0.0), grid);
  a.set_previous_path({});
  const auto ra = a.get_corridors_astar(start, goal);

  // A previous route that would pull the other way, but with the discount switched off.
  FrontEndSFC b(cfg(0.0), grid);
  b.set_previous_path({Eigen::Vector3d(0, -4, -3), Eigen::Vector3d(18, -4, -3)});
  const auto rb = b.get_corridors_astar(start, goal);

  ASSERT_FALSE(ra.corridors.empty());
  ASSERT_FALSE(rb.corridors.empty());
  EXPECT_DOUBLE_EQ(side_of(ra), side_of(rb))
      << "a zero bonus still influenced the search";
}

}  // namespace tests
}  // namespace trajectory_planner
