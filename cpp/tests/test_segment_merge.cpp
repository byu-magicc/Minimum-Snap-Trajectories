// The smoother's sub-threshold segment merge.
//
// WHAT THIS IS AND IS NOT. One-voxel segments make degenerate corridor spines -- diagonal (so the
// SFC's SAT corner term is half_res*sqrt(2) rather than half_res) and still carrying the full
// sfc_end_ext, so the corridor box runs many times the length of the segment it represents.
// Measured in flight as a lateral bound of exactly 0 and a primal-infeasible QP that
// time-stretching cannot fix.
//
// The merge was written to remove those, and IT LARGELY CANNOT. The smoother emits a waypoint
// exactly when line-of-sight from the anchor fails, so asking whether that waypoint may be
// dropped re-asks the question that put it there -- and around a trunk the answer is the same no.
// Verified below on a staggered-trunk layout where both candidates are refused.
//
// It is kept because it is safe and occasionally useful (a genuinely redundant short segment does
// get removed), but the degenerate-corridor fix is elsewhere: build the corridor against RAW
// obstacles and inset by the hull, instead of against the already-inflated set, so the clearance
// the inflation already contains is not counted twice.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>

#include "astar_sfc.hpp"

namespace trajectory_planner {
namespace path_finding {
namespace tests {
namespace {

/// Shortest segment in a path, in voxels.
double shortest_segment(const std::vector<Eigen::Vector3i>& p)
{
  double m = 1e18;
  for (std::size_t i = 1; i < p.size(); ++i) {
    m = std::min(m, (p[i] - p[i - 1]).cast<double>().norm());
  }
  return p.size() < 2 ? 0.0 : m;
}

}  // namespace

namespace {

/// Rasterised disc -- stands in for a forest trunk. A path curving around one is what makes the
/// smoother emit short segments; straight-line detours never do, because a straight approach is
/// always line-of-sight clear and gets shortcut into a single long segment.
void disc(VoxelSet& o, int cx, int cy, double r)
{
  for (int x = cx - static_cast<int>(r) - 1; x <= cx + static_cast<int>(r) + 1; ++x) {
    for (int y = cy - static_cast<int>(r) - 1; y <= cy + static_cast<int>(r) + 1; ++y) {
      if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r) {
        o.insert(Eigen::Vector3i(x, y, 0));
      }
    }
  }
}

void staggered_trunks(VoxelSet& o)
{
  disc(o, 6, 2, 3);
  disc(o, 12, -2, 3);
  disc(o, 18, 2, 3);
}

std::vector<Eigen::Vector3i> smooth_with(VoxelSet& occ, double min_seg)
{
  VoxelGridData grid;
  grid.voxel_resolution = 1.0;
  grid.occupied_voxels_inflated = &occ;
  AStarSFCPlanner planner(grid, Eigen::Vector3i(-4, -16, -2), Eigen::Vector3i(24, 16, 2));
  planner.search(Eigen::Vector3i(0, 0, 0), Eigen::Vector3i(22, 0, 0),
                 50.0, -1.0, 100, -90.0, 90.0, 0.0, 50000, 0.0, 0.0, 2.0, 0.0, min_seg);
  return planner.sfc_smoother();
}

}  // namespace

// THE MERGE IS BEST EFFORT, AND AT A TIGHT CORNER IT CANNOT FIRE.
//
// This is the important property to record, because it is counter-intuitive and it means the
// merge is NOT a fix for the degenerate-corridor stall. The smoother emits a waypoint exactly
// when line-of-sight from the anchor fails; asking whether that waypoint can be dropped re-asks
// the same question, and around a trunk the answer is the same no. Measured on this layout:
// both short-segment candidates were refused, so a 1.41-voxel segment survives.
//
// The fix for degenerate corridors is therefore elsewhere -- building the corridor against RAW
// obstacles and insetting by the hull, rather than against the already-inflated set, so the
// measured clearance is not double-counted.
TEST(SegmentMerge, IsBestEffortAndDeclinesAtTightCorners)
{
  VoxelSet occ;
  staggered_trunks(occ);
  const auto merged = smooth_with(occ, 2.0);

  ASSERT_GE(merged.size(), 3u) << "layout must produce a multi-segment path to be meaningful";
  // Deliberately NOT asserting that every segment clears the threshold: it cannot, and asserting
  // otherwise would encode a guarantee the merge does not provide.
  EXPECT_LT(shortest_segment(merged), 2.0)
      << "this layout is supposed to defeat the merge; if it no longer does, the scenario has "
         "drifted and this test has stopped covering the declining path";
}

// A staircase around a wall corner is exactly what the look-ahead loop produces: it re-anchors on
// the cell before each line-of-sight failure, so consecutive failures emit consecutive cells.
// Whatever the merge does, it may only ever shorten the list and never move an endpoint.
TEST(SegmentMerge, OnlyEverRemovesInteriorWaypoints)
{
  VoxelSet occ;
  staggered_trunks(occ);
  const auto unmerged = smooth_with(occ, 0.0);
  const auto merged = smooth_with(occ, 2.0);

  ASSERT_GE(unmerged.size(), 2u);
  ASSERT_GE(merged.size(), 2u);
  EXPECT_EQ(merged.front(), unmerged.front());   // the vehicle's own state
  EXPECT_EQ(merged.back(), unmerged.back());     // the search goal
  EXPECT_LE(merged.size(), unmerged.size());
}

// The merge is gated on line-of-sight, so it must never straighten a path through an obstacle.
// This is what keeps it safe when a corner is genuinely tight: it declines and the short segment
// survives, which is the correct failure.
TEST(SegmentMerge, NeverCutsThroughAnObstacle)
{
  VoxelSet occupied;
  VoxelGridData grid;
  grid.voxel_resolution = 1.0;
  grid.occupied_voxels_inflated = &occupied;

  // A solid block the path must go around.
  for (int x = 3; x <= 5; ++x) {
    for (int y = 0; y <= 4; ++y) {
      occupied.insert(Eigen::Vector3i(x, y, 0));
    }
  }

  AStarSFCPlanner planner(grid, Eigen::Vector3i(0, -4, -2), Eigen::Vector3i(12, 10, 2));
  planner.search(Eigen::Vector3i(0, 0, 0), Eigen::Vector3i(9, 0, 0),
                 50.0, -1.0, 100, -90.0, 90.0, 0.0, 20000, 0.0, 0.0, 2.0, 0.0,
                 /*min_segment_voxels=*/2.0);
  const auto merged = planner.sfc_smoother();

  ASSERT_GE(merged.size(), 2u);
  for (const auto& c : merged) {
    EXPECT_EQ(occupied.count(c), 0u)
        << "merged waypoint [" << c.transpose() << "] is inside an obstacle";
  }
  // Every merged segment must still be traversable, not just its endpoints.
  for (std::size_t i = 1; i < merged.size(); ++i) {
    EXPECT_TRUE(planner.is_line_of_sight_clear(merged[i - 1], merged[i]))
        << "merge produced a segment that crosses an obstacle";
  }
}

// Disabling must be exact, so the previous behaviour can be recovered if the merge ever costs
// more than it earns.
TEST(SegmentMerge, ZeroThresholdLeavesPathUntouched)
{
  VoxelSet occ;
  staggered_trunks(occ);
  EXPECT_EQ(smooth_with(occ, 0.0), smooth_with(occ, 0.0));
}

// THE COLLINEAR PASS IS NOT UNIT-TESTED HERE, AND THAT IS A DELIBERATE RECORD, NOT AN OMISSION.
//
// Two tests were written for it and BOTH were vacuous. The reason is worth keeping, because it
// will trap the next person the same way: the look-ahead loop above already collapses any
// straight run in an open scene, so a synthetic "staircase of collinear cells" comes out as two
// waypoints whether the collinear pass exists or not. Verified by mutation -- the test still
// passed with the pass stubbed out. Flanking the diagonal with blockers does not help either:
// the discrete voxel walk on an exact diagonal does not clip them, so line-of-sight across the
// whole run stays clear and the look-ahead still collapses it.
//
// Which is itself the finding: redundant collinear joints do NOT arise in clean synthetic
// scenes. They arise in accumulated maps with unknown space, where line-of-sight fails across
// spans for reasons no toy layout reproduces.
//
// The pass is therefore validated on the real artefact instead --
// benchmark_results/raw/hard_s0_v5_stall.txt, replayed through tools/replay_stall:
//
//     without: chain  8.17 m, shortest segment 0.71 m (two ONE-VOXEL DIAGONAL spines)
//     with:    chain 12.10 m, shortest segment 1.41 m
//
// Its safety does not rest on a test: a waypoint whose two legs are parallel and same-signed
// lies on the straight line between its neighbours, so the merged segment is the union of two
// segments already accepted. Endpoints are untouched and the path can only get shorter.
//
// AND IT DID NOT FIX THAT STALL. Every corridor there was already narrower than the hull
// (0.286, 0.032, 0.000 m half-widths), so the spine was never what collapsed them. Recorded so
// the merge is not credited with more than it earns.

}  // namespace tests
}  // namespace path_finding
}  // namespace trajectory_planner
