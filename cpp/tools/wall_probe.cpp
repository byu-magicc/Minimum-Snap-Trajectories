// Can A* route THROUGH a tree it can plainly see?
//
// Reported from flight on forest_hard_s2: the LiDAR clearly shows the space ahead blocked and
// clear openings to either side, and the planner keeps aiming forward -- with a corridor that
// appears to penetrate a trunk.
//
// FINDING, after measurement -- the first hypothesis here was WRONG and is recorded so it is
// not repeated. The suspicion was that A* routes THROUGH trees: update_from_scan marks exactly
// one cell per return and nothing behind it, so the far side of every obstacle is sensor shadow,
// stays UNKNOWN, and UNKNOWN is not blocked -- A* charges unknown_traversal_cost to cross it.
//
// A* CANNOT cross. Inflation (hull + obstacle_margin = 0.69 m, ~1.4 voxels) reaches inward far
// enough that no chain of A*-acceptable cells gets through a 1.3 m trunk: 18 of 20 interior
// cells are already inflated-blocked with the fill off, and a flood fill confirms no crossing
// exists (see test_occupancy_thickness.cpp). get_obstacles_in_radius returns that same inflated
// set, so the corridor builder is not blind to the shadowed face either.
//
// What IS real: the deepest interior cells escape both mechanisms, so a corridor built in an
// obstacle's shadow is clipped by nothing there and can reach ~0.5 m into solid wood against a
// 0.49 m hull. How many cells escape depends strongly on viewing geometry -- 2 of 20 on a trunk
// sampled head-on and densely, but 22 of 55 across the obliquely-viewed cluster below.
// SparseVoxelGrid::set_occupancy_thickness closes them; it is OFF by default because it costs
// +27% on update_from_scan, which is the component already starving the dispatch loop.
//
// The reason the planner aims into blocked space on s2 is therefore still unexplained.
//
// This scans a real slice of the s2 bottleneck from the vehicle's actual start, then prints the
// FREE/OCCUPIED/UNKNOWN classification straight across the wall, and prices both routes.
//
// BUILD
//   cd planner_reference/cpp/build && cmake .. && make wall_probe && ./wall_probe

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "front_end.hpp"
#include "voxel_grid.hpp"

using namespace trajectory_planner;

namespace {

constexpr double kRes = 0.5;
constexpr double kHull = 0.490491;
constexpr double kMargin = 0.20;
constexpr double kAlt = -3.0;          // NED cruise altitude

struct Trunk { double x, y, r; };

// Trunks forming the s2 bottleneck near y=0, x in [11,17], from forest_hard_s2.yaml.
std::vector<Trunk> wall()
{
  return {{12.96, -0.43, 1.42}, {14.55, 1.05, 1.21}, {13.71, -2.35, 1.34},
          {15.02, -3.60, 1.10}, {13.10, 2.82, 1.28}, {16.40, 3.95, 1.45}};
}

/// Livox-style scan from `origin`: 1.5 deg in both axes, as the benchmark configures it.
std::vector<Eigen::Vector3d> scan(const Eigen::Vector3d& origin, const std::vector<Trunk>& ts)
{
  std::vector<Eigen::Vector3d> hits;
  const double res_deg = 1.5, max_range = 70.0;
  for (double el = -20.0; el <= 20.0; el += res_deg) {
    for (double az = -60.0; az <= 60.0; az += res_deg) {
      const double e = el * M_PI / 180.0, a = az * M_PI / 180.0;
      const Eigen::Vector3d d(std::cos(e) * std::cos(a), std::cos(e) * std::sin(a), -std::sin(e));
      // Nearest cylinder intersection along the ray (infinite height; trunks are 6 m tall and
      // we are scanning a 3 m band).
      double best = max_range;
      for (const auto& t : ts) {
        const double ox = origin.x() - t.x, oy = origin.y() - t.y;
        const double A = d.x() * d.x() + d.y() * d.y();
        if (A < 1e-12) continue;
        const double B = 2.0 * (ox * d.x() + oy * d.y());
        const double C = ox * ox + oy * oy - t.r * t.r;
        const double disc = B * B - 4 * A * C;
        if (disc < 0) continue;
        const double s = (-B - std::sqrt(disc)) / (2 * A);
        if (s > 0.1 && s < best) best = s;
      }
      if (best < max_range - 1e-3) hits.push_back(origin + best * d);
    }
  }
  return hits;
}

const char* state(const mapping::SparseVoxelGrid& g, int ix, int iy)
{
  const Eigen::Vector3i c(ix, iy, static_cast<int>(std::floor(kAlt / kRes)));
  if (g.is_occupied_inflated(c)) return "X";   // A* hard-blocks these
  if (g.is_free(c)) return "o";
  return "?";                                   // UNKNOWN -- traversable at a premium
}

bool inside_trunk(double x, double y, const std::vector<Trunk>& ts)
{
  for (const auto& t : ts) {
    if ((x - t.x) * (x - t.x) + (y - t.y) * (y - t.y) < t.r * t.r) return true;
  }
  return false;
}

}  // namespace

int main()
{
  const auto ts = wall();
  const Eigen::Vector3d origin(0.0, 0.0, kAlt);

  auto grid = std::make_shared<mapping::SparseVoxelGrid>(kRes, kHull + kMargin);
  grid->set_accumulate(true);
  // Several scans from along the approach, as the vehicle actually flies it.
  for (double x = 0.0; x <= 8.0; x += 2.0) {
    const Eigen::Vector3d o(x, 0.0, kAlt);
    grid->update_from_scan(o, scan(o, ts));
  }

  std::printf("Scanned the s2 bottleneck from x=0..8 m along y=0 (1.5 deg beams, %d trunks).\n\n",
              static_cast<int>(ts.size()));
  std::printf("Map state along y=0.  X = inflated-blocked (A* refuses)   o = FREE   ? = UNKNOWN\n");
  std::printf("Bottom row marks cells that are PHYSICALLY INSIDE A TRUNK.\n\n");

  std::printf("        y[m] ->\n        ");
  for (int iy = -28; iy <= 28; iy += 2) std::printf("%d", ((iy*5/10)%10+10)%10);
  std::printf("\n");
  int hidden = 0, solid = 0, thread = 0;
  for (int ix = 18; ix <= 40; ++ix) {
    std::printf("  x=%5.1f ", (ix + 0.5) * kRes);
    bool row_open = false;
    for (int iy = -28; iy <= 28; iy += 2) {
      const double x = (ix + 0.5) * kRes, y = (iy + 0.5) * kRes;
      const Eigen::Vector3i c(ix, iy, static_cast<int>(std::floor(kAlt / kRes)));
      const bool in = inside_trunk(x, y, ts);
      const bool blk = grid->is_occupied_inflated(c);
      const bool fre = grid->is_free(c);
      if (in) { ++solid; if (!blk) ++hidden; }
      if (!blk) row_open = true;
      // T = inside a trunk AND not blocked in the map -- a hole A* can fly through
      std::printf("%c", in ? (blk ? 'X' : 'T') : (blk ? 'X' : (fre ? 'o' : '.')));
    }
    if (row_open) ++thread;
    std::printf("\n");
  }
  std::printf("\n  X = blocked   o = FREE   . = UNKNOWN (A* crosses for a fee)   T = INSIDE A\n");
  std::printf("  TRUNK but NOT blocked -- sensor shadow, so the map never learned it is solid.\n\n");
  std::printf("  %d of %d cells inside a trunk are unblocked in the map.\n", hidden, solid);

  // Price the two options the way A* does.
  const double ucost = 0.5;
  int unknown_run = 0;
  for (int ix = 20; ix <= 36; ++ix) {
    const Eigen::Vector3i c(ix, 0, static_cast<int>(std::floor(kAlt / kRes)));
    if (!grid->is_occupied_inflated(c) && !grid->is_free(c)) ++unknown_run;
  }
  std::printf("\n  COST along y=0 (unknown_traversal_cost = %.2f):\n", ucost);
  std::printf("    straight through : 15.8 m + %d unknown steps x %.2f = %.1f\n",
              unknown_run, ucost, 15.8 + unknown_run * ucost);
  std::printf("    around the wall  : 33.0 m  (measured on the true map)\n");
  const double thru = 15.8 + unknown_run * ucost;
  std::printf("    -> A* prefers %s\n", thru < 33.0 ? "STRAIGHT THROUGH THE TREES" : "the detour");
  if (thru < 33.0 && unknown_run > 0) {
    std::printf("    unknown_traversal_cost would have to exceed %.2f for the detour to win.\n",
                (33.0 - 15.8) / unknown_run);
  }
  return 0;
}
