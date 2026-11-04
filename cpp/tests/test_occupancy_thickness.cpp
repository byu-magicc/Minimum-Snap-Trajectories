// Occupancy depth behind a LiDAR return.
//
// A beam deposits occupancy in exactly the cell it terminates in. Everything past that is
// sensor shadow -- including the FAR SIDE OF THE OBJECT THE BEAM JUST HIT -- so it is never
// observed and stays UNKNOWN. UNKNOWN is not blocked: A* crosses it for
// unknown_traversal_cost. The consequence is that the interior of a tree the sensor is
// staring straight at is traversable at a small premium, and because the SFC builder shrinks
// only against KNOWN obstacles, a path threading that interior yields a corridor overlapping a
// real trunk with nothing anywhere reporting a fault.
//
// set_occupancy_thickness() continues marking along the beam past the return. The subtlety
// worth protecting is the STOP RULE: the fill must halt at any cell already measured FREE, or
// a genuine swept gap behind a thin obstacle gets walled off and routes are lost.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "voxel_grid.hpp"

namespace trajectory_planner {
namespace mapping {
namespace tests {
namespace {

constexpr double kRes = 0.5;

Eigen::Vector3i idx_of(const Eigen::Vector3d& p)
{
  return Eigen::Vector3i(static_cast<int>(std::floor(p.x() / kRes)),
                         static_cast<int>(std::floor(p.y() / kRes)),
                         static_cast<int>(std::floor(p.z() / kRes)));
}

/// Grid with no inflation, so these tests observe the RAW occupancy the fill writes rather
/// than inflation's separate contribution.
std::shared_ptr<SparseVoxelGrid> grid(double thickness)
{
  auto g = std::make_shared<SparseVoxelGrid>(kRes, /*inflation=*/0.0);
  g->set_accumulate(true);
  g->set_occupancy_thickness(thickness);
  return g;
}

}  // namespace

// Off by default, so enabling it is always a deliberate act.
TEST(OccupancyThickness, DefaultsToOff)
{
  SparseVoxelGrid g(kRes, 0.0);
  EXPECT_EQ(g.occupancy_thickness(), 0.0);
}

// THE BUG. One return at 10 m: with no thickness the cell behind it is UNKNOWN, which A* will
// happily cross. This is the pre-fix behaviour, asserted so the test below is known to be
// measuring a real change rather than a tautology.
TEST(OccupancyThickness, WithoutItTheCellBehindAReturnIsUnknown)
{
  auto g = grid(0.0);
  const Eigen::Vector3d origin(0.0, 0.0, 0.0), hit(10.0, 0.0, 0.0);
  g->update_from_scan(origin, {hit});

  EXPECT_TRUE(g->is_occupied_raw(idx_of(hit)));
  const Eigen::Vector3i behind = idx_of(hit + Eigen::Vector3d(kRes, 0.0, 0.0));
  EXPECT_TRUE(g->is_unknown(behind))
      << "cell behind the return is already known -- scenario no longer reproduces the shadow";
  EXPECT_FALSE(g->is_occupied_raw(behind));
}

// THE FIX.
TEST(OccupancyThickness, MarksCellsBehindTheReturnOccupied)
{
  auto g = grid(1.0);
  const Eigen::Vector3d origin(0.0, 0.0, 0.0), hit(10.0, 0.0, 0.0);
  g->update_from_scan(origin, {hit});

  for (double d : {0.5, 1.0}) {
    const Eigen::Vector3i c = idx_of(hit + Eigen::Vector3d(d, 0.0, 0.0));
    EXPECT_TRUE(g->is_occupied_raw(c))
        << "cell " << d << " m behind the return is not blocked; the shadow is still traversable";
  }
}

// Depth is bounded by the parameter, not open-ended -- otherwise one return would blank out
// everything behind it to the window edge.
TEST(OccupancyThickness, StopsAtTheRequestedDepth)
{
  auto g = grid(1.0);
  const Eigen::Vector3d origin(0.0, 0.0, 0.0), hit(10.0, 0.0, 0.0);
  g->update_from_scan(origin, {hit});

  const Eigen::Vector3i too_far = idx_of(hit + Eigen::Vector3d(3.0, 0.0, 0.0));
  EXPECT_FALSE(g->is_occupied_raw(too_far))
      << "fill ran past occupancy_thickness";
}

// THE STOP RULE, AND THE REASON THIS IS SAFE.
//
// A thin obstacle with a gap behind it that some OTHER beam has swept: the gap is measured
// FREE, and inferred depth must never overrule a measurement. Without this rule a fixed-depth
// fill closes real routes -- which is how a conservative map turns into a stalled vehicle.
TEST(OccupancyThickness, NeverOverwritesMeasuredFreeSpace)
{
  auto g = grid(2.0);
  const Eigen::Vector3d origin(0.0, 0.0, 0.0);

  // Beam 1 sweeps straight past x=10 out to 14 m, so everything along it is measured empty.
  // Beam 2 terminates at 10 m just off-axis. Its depth fill heads into cells beam 1 cleared.
  const Eigen::Vector3d through(14.0, 0.0, 0.0);
  const Eigen::Vector3d hit(10.0, 0.02, 0.0);
  g->update_from_scan(origin, {through, hit});

  const Eigen::Vector3i gap = idx_of(Eigen::Vector3d(11.0, 0.0, 0.0));
  EXPECT_TRUE(g->is_free(gap))
      << "the swept gap was overwritten as occupied -- inferred depth beat a measurement, and "
         "a real route has just been walled off";
  EXPECT_FALSE(g->is_occupied_raw(gap));
}

// WHAT THE FILL IS ACTUALLY FOR, stated at its real size.
//
// TWO EARLIER VERSIONS OF THIS TEST ASSERTED MORE THAN IS TRUE, and the mutation checks caught
// both. Recording that here so the claim is not re-inflated later:
//
//   "A* threads a trunk's hollow interior."  FALSE. Inflation (hull + obstacle_margin = 0.69 m,
//   ~1.4 voxels) reaches inward from a densely-sampled near surface far enough that no chain of
//   A*-acceptable cells crosses a 1.3 m trunk even with the fill OFF. Measured: 18 of 20
//   interior cells are already inflated-blocked at thickness 0.
//
//   "The corridor builder cannot see the shadowed face at all."  ALSO FALSE, for the same
//   reason -- get_obstacles_in_radius returns inflated_centers_, the INFLATED set, so it sees
//   those same 18 of 20 cells.
//
// WHAT IS LEFT IS REAL BUT NARROW: the last ~10%. At thickness 0 two interior cells -- the
// deepest, furthest from any return -- are neither occupied nor inflated, so a corridor built
// in the trunk's shadow may extend roughly half a metre into solid wood before anything clips
// it, against a 0.49 m hull. A small amount of fill closes exactly that.
//
// So this asserts the invariant worth keeping (no interior cell escapes the map) and NOT a
// story about A* routing through trees.
TEST(OccupancyThickness, ClosesTheLastUnblockedCellsInsideAnObstacle)
{
  const double cx = 12.0, cy = 0.0, r = 1.3;
  const double kInflation = 0.490491 + 0.20;

  auto make_hits = [&](const Eigen::Vector3d& o) {
    std::vector<Eigen::Vector3d> hits;
    for (double az = -40.0; az <= 40.0; az += 0.3) {
      const double a = az * M_PI / 180.0;
      const Eigen::Vector3d d(std::cos(a), std::sin(a), 0.0);
      const double ox = o.x() - cx, oy = o.y() - cy;
      const double A = d.x() * d.x() + d.y() * d.y();
      const double B = 2.0 * (ox * d.x() + oy * d.y());
      const double C = ox * ox + oy * oy - r * r;
      const double disc = B * B - 4 * A * C;
      if (disc < 0) continue;
      const double s = (-B - std::sqrt(disc)) / (2 * A);
      if (s > 0.1) hits.push_back(o + s * d);
    }
    return hits;
  };

  // Interior cells NOT blocked in the inflated set -- i.e. invisible to both A* and the
  // corridor builder, since both consult that set.
  auto unblocked_interior = [&](double thickness) {
    auto g = std::make_shared<SparseVoxelGrid>(kRes, kInflation);
    g->set_accumulate(true);
    g->set_occupancy_thickness(thickness);
    for (double x = 0.0; x <= 6.0; x += 2.0) {
      const Eigen::Vector3d o(x, 0.0, 0.0);
      g->update_from_scan(o, make_hits(o));
    }
    int open = 0;
    for (double x = cx - r; x <= cx + r; x += kRes) {
      for (double y = cy - r; y <= cy + r; y += kRes) {
        if ((x - cx) * (x - cx) + (y - cy) * (y - cy) >= r * r) continue;
        if (!g->is_occupied_inflated(idx_of(Eigen::Vector3d(x, y, 0.0)))) ++open;
      }
    }
    return open;
  };

  const int off = unblocked_interior(0.0);
  EXPECT_GT(off, 0)
      << "with the fill off, every cell inside the trunk is already blocked -- there is no gap "
         "to close and this test proves nothing; check the scan geometry";
  EXPECT_EQ(unblocked_interior(0.5), 0)
      << "half a metre of fill left " << unblocked_interior(0.5) << " interior cell(s) open";
  EXPECT_EQ(unblocked_interior(1.0), 0)
      << "the configured default left interior cells open";
}

}  // namespace tests
}  // namespace mapping
}  // namespace trajectory_planner
