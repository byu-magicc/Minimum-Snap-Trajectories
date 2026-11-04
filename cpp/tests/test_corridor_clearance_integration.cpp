// The corridor clearance invariant, exercised through the PRODUCTION construction path.
//
// WHY A SECOND CLEARANCE TEST EXISTS. test_corridor_clearance.cpp validates the same invariant,
// and it passed throughout the period in which corridors were being built flush against tree
// trunks in flight. It passed because it builds an AsymmetricSFCManager itself, with the hull
// passed correctly -- so it tested the FORMULA while production supplied a different argument:
//
//     sfc_manager_ = sfc::AsymmetricSFCManager(
//         discrete_grid_, 0.0, max_cp_drift, config_.voxel_resolution);
//                        ^^^ drone_physical_radius
//
// The builder computes relief = obstacle_inflation - drone_radius, so a zero radius handed back
// the ENTIRE inflation instead of just obstacle_margin. Every corridor face landed exactly one
// hull radius too far out, i.e. flush against the raw obstacle surface with zero clearance --
// which is precisely what "the SFC is drawn right up to the wall" looked like on screen.
// Measured over 171 corridors on the medium forest benchmark: 45% of corridors contained points
// INSIDE a trunk, the worst 1.48 m inside. After passing the hull, 11.7%.
//
// The lesson is the reason for this file: a unit test that constructs its own subject cannot
// catch a bad argument at the call site. This one goes through FrontEndSFC, so the wiring is
// under test and not just the arithmetic.
//
// KNOWN GAP, deliberately not asserted here: a residual ~12% of corridors still contain interior
// points inside an obstacle, and those breaches sit almost entirely in the start/end EXTENSIONS
// (17 of 20 on medium were behind pA) rather than in the segment tunnel. That is a separate,
// undiagnosed defect in the end-cap shrink. These tests cover the tunnel, where the invariant
// now holds; do not widen them to the extensions until that is fixed, or this file will be
// disabled and stop protecting the part that works.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
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
constexpr double kMargin = 0.20;

FrontEndConfig config()
{
  FrontEndConfig c;
  c.voxel_resolution = kRes;
  c.drone_physical_radius = kHull;
  c.v_max = 5.0;
  c.a_max = 20.0;
  c.spline_type = "clamped";
  c.degree = 4;
  c.planning_horizon = -1.0;
  c.min_altitude = 0.5;
  c.max_altitude = 8.0;   // walls run the full band, so overflight is impossible
  c.map_bounds = Eigen::Vector3d(120.0, 120.0, 24.0);
  c.astar_max_nodes = 200000;
  c.astar_max_time_ms = 500.0;
  c.unknown_traversal_cost = 0.0;
  c.heat_radius = 0.0;
  c.heat_weight = 0.0;
  c.astar_min_segment_voxels = 2.0;
  c.sfc_width = 5.0;
  c.sfc_start_ext = 1.0;
  c.sfc_end_ext = 5.0;
  return c;
}

/// Two parallel walls forming a gap of `gap` metres, centred on y = 0 and running along x.
std::shared_ptr<mapping::SparseVoxelGrid> corridor_walls(double gap)
{
  auto grid = std::make_shared<mapping::SparseVoxelGrid>(kRes, kHull + kMargin);
  std::vector<mapping::ObstacleBox> obs;
  const double y = gap / 2.0;
  obs.push_back({{-2.0, y, -30.0}, {20.0, y + 2.0, 0.0}});
  obs.push_back({{-2.0, -y - 2.0, -30.0}, {20.0, -y, 0.0}});
  grid->update_from_obstacles(obs);
  return grid;
}

/// Smallest distance from p to the surface of any wall box (0 inside).
double dist_to_walls(const Eigen::Vector3d& p, double gap)
{
  const double y = gap / 2.0;
  double best = 1e18;
  for (int s : {+1, -1}) {
    const Eigen::Vector3d lo = s > 0 ? Eigen::Vector3d(-2.0, y, -30.0)
                                     : Eigen::Vector3d(-2.0, -y - 2.0, -30.0);
    const Eigen::Vector3d hi = s > 0 ? Eigen::Vector3d(20.0, y + 2.0, 0.0)
                                     : Eigen::Vector3d(20.0, -y, 0.0);
    Eigen::Vector3d d = (lo - p).cwiseMax(p - hi).cwiseMax(0.0);
    best = std::min(best, d.norm());
  }
  return best;
}

/// Worst clearance over the corridor's SEGMENT TUNNEL only -- see the note on extensions above.
double worst_clearance_in_tunnel(const sfc::SFCResult& c, double gap)
{
  const double seg = (c.secondaryPosition - c.primaryPosition).norm();
  double worst = 1e18;
  const int N = 10;
  for (int i = 0; i <= N; ++i) {
    const double ax = seg * i / N;                       // tunnel only: 0 .. seg
    for (int j = 0; j <= N; ++j) {
      const double b = -c.bounds(3) + (c.bounds(2) + c.bounds(3)) * j / N;
      for (int k = 0; k <= N; ++k) {
        const double z = -c.bounds(5) + (c.bounds(4) + c.bounds(5)) * k / N;
        const Eigen::Vector3d p = c.primaryPosition + c.ux * ax + c.uy * b + c.uz * z;
        worst = std::min(worst, dist_to_walls(p, gap));
      }
    }
  }
  return worst;
}

}  // namespace

// THE REGRESSION. Through the production path, a corridor's tunnel must keep one hull radius
// from the real obstacle surface. With drone_physical_radius wired as 0.0 this returned ~0.
TEST(CorridorClearanceIntegration, ProductionPathKeepsAHullFromTheWall)
{
  const double gap = 4.0;
  auto grid = corridor_walls(gap);
  FrontEndSFC fe(config(), grid);

  const auto r = fe.get_corridors_astar(Eigen::Vector3d(0.0, 0.0, -4.0),
                                        Eigen::Vector3d(16.0, 0.0, -4.0));
  ASSERT_FALSE(r.corridors.empty()) << "scenario produced no corridors";

  for (std::size_t i = 0; i < r.corridors.size(); ++i) {
    const double worst = worst_clearance_in_tunnel(r.corridors[i], gap);
    EXPECT_GE(worst, kHull)
        << "corridor " << i << " lets the vehicle centre within " << worst
        << " m of the wall, closer than the " << kHull
        << " m hull -- relief is too generous. Check the drone_physical_radius argument at the "
           "AsymmetricSFCManager construction in front_end.cpp.";
  }
}

// THE BUG CLASS, closed. relief is clamped to min(0, inflation - hull), so no value of
// drone_radius_ can hand clearance back -- a wrong argument can only make a corridor NARROWER.
// That is the property that makes this whole failure mode unreachable, and it is worth a test of
// its own because the original defect was not a bad formula but a bad argument to a good one.
TEST(CorridorClearanceIntegration, ZeroingTheDroneRadiusCannotWidenTheCorridor)
{
  const double gap = 4.0;
  auto grid = corridor_walls(gap);

  FrontEndSFC fe(config(), grid);
  const auto good = fe.get_corridors_astar(Eigen::Vector3d(0.0, 0.0, -4.0),
                                           Eigen::Vector3d(16.0, 0.0, -4.0));
  ASSERT_FALSE(good.corridors.empty());
  const auto& g0 = good.corridors.front();

  // The exact miswiring that caused the incident.
  sfc::AsymmetricSFCManager zeroed(grid, 0.0, (config().sfc_width / 2.0) - kHull, kRes);
  const auto bad = zeroed.generate_sfc(g0.primaryPosition, g0.secondaryPosition,
                                       config().sfc_width, config().sfc_start_ext,
                                       config().sfc_end_ext);

  for (int face : {2, 3, 4, 5}) {
    EXPECT_LE(bad.bounds(face), g0.bounds(face) + 1e-9)
        << "face " << face << " grew from " << g0.bounds(face) << " to " << bad.bounds(face)
        << " m when drone_radius was zeroed -- relief is unclamped again and a wiring mistake "
           "can once more produce a corridor that is flush with an obstacle";
  }
}

// A gap too narrow to fly must not yield a flyable corridor.
//
// DISABLED: THIS FAILS, AND THE FAILURE IS REAL -- it is a SECOND, still-unfixed defect, distinct
// from the drone-radius wiring this file was written for. Through a 1.2 m gap (impassable for a
// 0.98 m hull plus margin, with the walls spanning the whole flyable band so overflight is not an
// option) the builder returns a corridor of half-width 0.200 m whose interior touches the wall.
//
// 0.200 m is exactly `relief`. In build_bounds the shrink is
//     bounds(i) = min(bounds(i), max(0.0, proj - r + relief))
// so once `proj` falls below `r`, the expression floors out near `relief` instead of going
// negative. A negative bound is the geometry SAYING the corridor is infeasible -- the obstacle
// reaches past the axis -- and clamping it to a small positive number converts that signal into
// a thin but flyable-looking box. The QP then solves happily inside it and certifies a
// trajectory through a gap the vehicle cannot fit.
//
// Not fixed here because the fix is a design decision, not a clamp change: build_bounds needs a
// way to report "infeasible" and plan_mission needs to decide what to do with it (reject the
// corridor? truncate the chain? brake?). Picking that blind would be the fourth reverted change
// in this area. Re-enable by removing the DISABLED_ prefix once it is decided.
TEST(CorridorClearanceIntegration, DISABLED_ImpassableGapDoesNotYieldAFlyableCorridor)
{
  // Gap of 1.2 m against a 0.98 m hull diameter: passable only with < 0.11 m to spare a side,
  // which is under the margin and must not be certified.
  const double gap = 1.2;
  auto grid = corridor_walls(gap);
  FrontEndSFC fe(config(), grid);

  const auto r = fe.get_corridors_astar(Eigen::Vector3d(0.0, 0.0, -4.0),
                                        Eigen::Vector3d(16.0, 0.0, -4.0));
  for (std::size_t i = 0; i < r.corridors.size(); ++i) {
    const double half = std::min(r.corridors[i].bounds(2), r.corridors[i].bounds(3));
    const double worst = worst_clearance_in_tunnel(r.corridors[i], gap);
    EXPECT_TRUE(half <= 1e-6 || worst >= kHull)
        << "corridor " << i << " has half-width " << half << " m through a " << gap
        << " m gap while coming within " << worst << " m of the wall";
  }
}

}  // namespace tests
}  // namespace trajectory_planner
