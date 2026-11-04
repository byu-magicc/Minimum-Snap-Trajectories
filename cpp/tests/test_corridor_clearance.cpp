// The corridor's safety property, under two-radii construction.
//
// Obstacle points reach the SFC builder already inflated by (hull + obstacle_margin). Shrinking
// the box to exclude them WITHOUT giving the excess back counts the same clearance twice, and
// the corridor loses that much width everywhere -- measured in flight as a lateral half-width of
// exactly 0 and a primal-infeasible QP. The builder therefore adds
// (obstacle_inflation - drone_radius) back.
//
// That is only sound if the box still keeps the vehicle clear, so these assert the thing the
// whole safety argument rests on: EVERY point inside the corridor is at least one hull radius
// from the raw obstacle surface. If that ever fails, the relief is too generous and the
// trajectory can be certified into a collision.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

#include "dynamic_sfc.hpp"
#include "voxel_grid.hpp"

namespace trajectory_planner {
namespace sfc {
namespace tests {
namespace {

constexpr double kRes = 0.5;
constexpr double kHull = 0.490491;   // as parsed from the real airframe
constexpr double kMargin = 0.20;

/// Distance from a point to the surface of an axis-aligned voxel centred at c.
double dist_to_voxel(const Eigen::Vector3d& p, const Eigen::Vector3d& c)
{
  const double h = kRes / 2.0;
  Eigen::Vector3d d = (p - c).cwiseAbs() - Eigen::Vector3d::Constant(h);
  d = d.cwiseMax(0.0);
  return d.norm();
}

/// Smallest distance from p to any raw obstacle surface.
double clearance(const Eigen::Vector3d& p, const std::vector<Eigen::Vector3d>& raw)
{
  double m = 1e18;
  for (const auto& c : raw) { m = std::min(m, dist_to_voxel(p, c)); }
  return m;
}

/// Worst clearance over a dense sample of the corridor interior.
double worst_clearance_in_box(const SFCResult& c, const std::vector<Eigen::Vector3d>& raw)
{
  double worst = 1e18;
  const int N = 9;
  for (int i = 0; i <= N; ++i) {
    // Along the axis, from the back face to the front face.
    const double x = -c.bounds(1) + (c.bounds(0) + c.bounds(1)) * i / N;
    for (int j = 0; j <= N; ++j) {
      const double y = -c.bounds(3) + (c.bounds(2) + c.bounds(3)) * j / N;
      for (int k = 0; k <= N; ++k) {
        const double z = -c.bounds(5) + (c.bounds(4) + c.bounds(5)) * k / N;
        const Eigen::Vector3d p = c.primaryPosition + c.ux * x + c.uy * y + c.uz * z;
        worst = std::min(worst, clearance(p, raw));
      }
    }
  }
  return worst;
}

/// A grid holding the given raw obstacle cells, inflated by hull + margin.
std::shared_ptr<mapping::SparseVoxelGrid> make_grid(const std::vector<Eigen::Vector3d>& raw)
{
  auto g = std::make_shared<mapping::SparseVoxelGrid>(kRes, kHull + kMargin);
  // Sensor offset along -y so the beams terminate ON the wall rather than sweeping past it, and
  // WITHIN map_window_radius (25 m) -- update_from_scan drops returns beyond the window, so a
  // distant sensor silently yields an empty map.
  g->update_from_scan(Eigen::Vector3d(0.0, -10.0, 0.0), raw);
  return g;
}

}  // namespace

// THE SAFETY INVARIANT. A trajectory is only ever certified by being inside the corridor, so if
// any interior point can come within a hull radius of a real obstacle, the certificate is void.
TEST(CorridorClearance, BoxInteriorStaysOneHullFromRawObstacles)
{
  // A short wall beside the corridor axis -- the geometry that squeezes a corridor in a forest.
  std::vector<Eigen::Vector3d> raw;
  for (int i = -3; i <= 3; ++i) {
    raw.push_back(Eigen::Vector3d(i * kRes, 2.0, 0.0));
  }
  auto grid = make_grid(raw);

  AsymmetricSFCManager mgr(grid, kHull, /*max_cp_drift=*/2.5, kRes);
  const SFCResult c = mgr.generate_sfc(Eigen::Vector3d(-2.0, 0.0, 0.0),
                                       Eigen::Vector3d(2.0, 0.0, 0.0),
                                       /*W=*/5.0, /*ext_start=*/1.0, /*ext_end=*/1.0);

  const double worst = worst_clearance_in_box(c, raw);
  EXPECT_GE(worst, kHull)
      << "corridor interior comes within " << worst << " m of a raw obstacle, closer than the "
      << kHull << " m hull -- the relief is too generous and the QP could certify a collision";
}

// Inflation that the map already applied is NEVER handed back.
//
// This test previously asserted the opposite -- that relief recovers exactly obstacle_margin,
// on the reasoning that shrinking against already-inflated points "double counts" the clearance.
// That policy was measured and rejected. Two reasons:
//
//   obstacle_margin exists to absorb STATE ESTIMATION error, so a corridor that gives it back
//   guarantees only the bare hull, and any estimator bias is then a collision. Measured bias on
//   this airframe is 0.3-0.8 m against a 0.49 m hull.
//
//   And the double-count argument does not hold up: clamping costs nothing measurable. Over 171
//   corridors on the medium forest, solve success stayed 100% at every max_sfc_count with
//   unchanged attempts and solve time, while corridors containing interior points inside an
//   obstacle fell 11.7% -> 8.8% and median clearance rose 0.565 -> 0.776 m.
//
// relief is now min(0, inflation - hull): it only ever INSETS. The safety consequence is that a
// wrong drone_radius can make a corridor narrower but never unsafe -- which matters, because
// exactly that argument was wired as 0.0 at the FrontEndSFC call site and the unclamped formula
// turned it into corridors flush against tree trunks.
TEST(CorridorClearance, InflationAlreadyAppliedIsNeverHandedBack)
{
  std::vector<Eigen::Vector3d> raw;
  for (int i = -3; i <= 3; ++i) {
    raw.push_back(Eigen::Vector3d(i * kRes, 2.0, 0.0));
  }
  auto grid = make_grid(raw);

  const Eigen::Vector3d pA(-2.0, 0.0, 0.0), pB(2.0, 0.0, 0.0);
  const auto obs = grid->get_obstacles_in_radius((pA + pB) / 2.0, 12.0);
  ASSERT_FALSE(obs.empty()) << "scenario must actually present obstacles to the builder";

  // Same obstacle set and geometry; only what the builder is told about their inflation differs.
  AsymmetricBoxBuilder pre_inflated(kHull, 2.5, kRes, /*obstacle_inflation=*/kHull + kMargin);
  AsymmetricBoxBuilder as_if_raw(kHull, 2.5, kRes,   /*obstacle_inflation=*/0.0);

  const auto a = pre_inflated.build_bounds(pA, pB, obs, 1.0, 1.0);
  const auto b = as_if_raw.build_bounds(pA, pB, obs, 1.0, 1.0);
  const auto& ba = std::get<3>(a);
  const auto& bb = std::get<3>(b);

  // The obstacle is on +y, so bounds(2) is the face it squeezes. Treating the same points as raw
  // must inset by exactly one hull and nothing else -- that is the only clearance relief may add.
  EXPECT_GT(ba(2), bb(2)) << "treating pre-inflated points as raw did not inset the box";
  EXPECT_NEAR(ba(2) - bb(2), kHull, 1e-9)
      << "the inset is " << (ba(2) - bb(2)) << " m rather than the hull -- relief is adding "
         "clearance beyond the hull, or handing inflation back";
}

// The margin must survive into the corridor, since absorbing estimation error is its whole job.
// With the old policy this was exactly one hull; it is now hull + margin.
TEST(CorridorClearance, CorridorKeepsTheMarginNotJustTheHull)
{
  std::vector<Eigen::Vector3d> raw;
  for (int i = -3; i <= 3; ++i) {
    raw.push_back(Eigen::Vector3d(i * kRes, 2.0, 0.0));
  }
  auto grid = make_grid(raw);

  AsymmetricSFCManager mgr(grid, kHull, /*max_cp_drift=*/2.5, kRes);
  const SFCResult c = mgr.generate_sfc(Eigen::Vector3d(-2.0, 0.0, 0.0),
                                       Eigen::Vector3d(2.0, 0.0, 0.0),
                                       /*W=*/5.0, /*ext_start=*/1.0, /*ext_end=*/1.0);

  const double worst = worst_clearance_in_box(c, raw);
  EXPECT_GE(worst, kHull + kMargin)
      << "corridor interior comes within " << worst << " m of a raw obstacle, inside the "
      << (kHull + kMargin) << " m the map already cleared -- the margin that covers estimation "
         "error is being handed back to the optimiser";
}

// Raw obstacle points must also work, since that is the configuration SANDO uses and the one a
// future change is most likely to move to.
TEST(CorridorClearance, RawObstaclesAreInsetByTheHull)
{
  std::vector<Eigen::Vector3d> raw;
  for (int i = -3; i <= 3; ++i) {
    raw.push_back(Eigen::Vector3d(i * kRes, 2.0, 0.0));
  }
  // obstacle_inflation = 0 means the points are raw, so relief is negative and insets the box.
  AsymmetricBoxBuilder builder(kHull, 2.5, kRes, /*obstacle_inflation=*/0.0);
  const auto out = builder.build_bounds(Eigen::Vector3d(-2.0, 0.0, 0.0),
                                        Eigen::Vector3d(2.0, 0.0, 0.0), raw, 1.0, 1.0);
  const auto& bounds = std::get<3>(out);

  SFCResult c;
  c.primaryPosition = Eigen::Vector3d(-2.0, 0.0, 0.0);
  c.ux = std::get<0>(out);
  c.uy = std::get<1>(out);
  c.uz = std::get<2>(out);
  c.bounds = bounds;

  const double worst = worst_clearance_in_box(c, raw);
  EXPECT_GE(worst, kHull)
      << "raw-obstacle path let the corridor within " << worst << " m of an obstacle";
}

}  // namespace tests
}  // namespace sfc
}  // namespace trajectory_planner
