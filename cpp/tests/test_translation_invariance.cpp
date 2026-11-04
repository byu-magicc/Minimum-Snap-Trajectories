// Planning must not depend on where in the world the problem sits.
//
// The QP was written in WORLD coordinates, so every corridor half-space carried an offset the
// size of the vehicle's distance from the origin, while the quantity that actually matters -- the
// corridor half-width -- is under a metre. Past roughly 45 m that ratio was bad enough that OSQP
// declared primal infeasibility on a comfortably feasible problem.
//
// The consequence was not subtle: the forest benchmark is 105 m long, so the vehicle crossed the
// threshold near its midpoint and every replan from there on failed. It presented as "the planner
// stalls halfway through", and it survived a long investigation because it is invisible in the
// planner's own output -- A* succeeds, the corridors look healthy, and the blame lands on the
// constraint blocks. Widening corridors, lengthening the time budget, lowering the terminal
// velocity and loosening the acceptance tolerance all changed nothing, because none of them
// touched the conditioning.
//
// plan_mission now shifts the problem so the start is the origin and shifts the solution back.
// These tests fail if that is ever removed.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "dynamic_sfc.hpp"
#include "front_end.hpp"
#include "trajectory_planner.hpp"
#include "voxel_grid.hpp"

namespace trajectory_planner {
namespace tests {
namespace {

FrontEndConfig config()
{
  FrontEndConfig c;
  c.voxel_resolution = 0.5;
  c.drone_physical_radius = 0.490491;
  c.v_max = 5.0;
  c.a_max = 20.0;
  c.spline_type = "clamped";
  c.degree = 4;
  c.time_budget_factor = 1.20;
  c.max_solve_time_ms = 60.0;
  c.qp_accept_violation = 0.02;
  return c;
}

/// One corridor with the geometry captured from the flight failure: a 9 m box, 0.84 m lateral
/// half-width, tilted axis. Placed at `origin`.
sfc::SFCResult corridor_at(const Eigen::Vector3d& origin)
{
  const double len = 9.02774;
  const Eigen::Vector3d ux(0.941543, 0.332309, 0.0553849);
  Eigen::Vector3d up(0.0, 0.0, 1.0);
  const Eigen::Vector3d uy = up.cross(ux).normalized();
  const Eigen::Vector3d uz = ux.cross(uy);

  sfc::SFCResult r;
  r.primaryPosition = origin;
  r.secondaryPosition = origin + ux * len;
  r.ux = ux;
  r.uy = uy;
  r.uz = uz;
  r.bounds << 14.0277, 0.967416, 0.843034, 0.843034, 0.913226, 0.913226;
  sfc::OBBAdapters::get_osqp_matrices(origin, ux, uy, uz, r.bounds, r.A_mat, r.b_vec);
  return r;
}

PlanningResult plan_at(const Eigen::Vector3d& origin)
{
  const FrontEndConfig cfg = config();
  auto grid = std::make_shared<mapping::SparseVoxelGrid>(cfg.voxel_resolution,
                                                          cfg.drone_physical_radius);
  TrajectoryPlanner planner(cfg, grid, cfg.v_max, cfg.a_max);

  FrontEndResult fe;
  fe.corridors.push_back(corridor_at(origin));
  fe.allocation_pools = planner.front_end().allocate_for(fe.corridors);
  fe.reached_goal = true;
  fe.actual_end_pos = fe.corridors.back().secondaryPosition;
  planner.set_front_end_override(fe);

  return planner.plan_mission(origin, fe.actual_end_pos,
                              Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
}

}  // namespace

// The regression itself. 48 m was the measured failure point; 500 m is far enough that any
// re-introduced world-frame formulation cannot survive it.
TEST(TranslationInvariance, SolvesAtEveryDistanceFromTheOrigin)
{
  for (double x : {0.0, 25.0, 47.0, 48.0, 60.0, 105.0, 500.0}) {
    const auto r = plan_at(Eigen::Vector3d(x, -5.25, -4.75));
    EXPECT_TRUE(r.success)
        << "identical geometry failed with its origin at x = " << x
        << " m -- the QP is conditioned on absolute coordinates again";
  }
}

// Solving the same shape further out must not silently cost more either: retries are the symptom
// that showed up first (the control-point count crept from 8 to 9 to 10 before failing outright).
TEST(TranslationInvariance, CostsTheSameWhereverItSits)
{
  const auto near = plan_at(Eigen::Vector3d(0.0, -5.25, -4.75));
  const auto far = plan_at(Eigen::Vector3d(500.0, -5.25, -4.75));

  ASSERT_TRUE(near.success);
  ASSERT_TRUE(far.success);
  EXPECT_EQ(near.total_control_points, far.total_control_points);
  EXPECT_EQ(near.solve_attempts, far.solve_attempts);
}

// The solution must come back in WORLD coordinates -- the caller feeds it straight to the
// controller and the collision re-check, both of which are world-frame consumers. A local-frame
// solve that forgets to shift back would still "succeed" while flying the vehicle to the origin.
TEST(TranslationInvariance, ControlPointsAreReturnedInWorldCoordinates)
{
  const Eigen::Vector3d origin(500.0, -5.25, -4.75);
  const auto r = plan_at(origin);
  ASSERT_TRUE(r.success);
  ASSERT_GE(r.control_points.size(), 3);

  const Eigen::Vector3d first = r.control_points.segment<3>(0);
  EXPECT_LT((first - origin).norm(), 2.0)
      << "first control point is at [" << first.transpose() << "] but the trajectory starts at ["
      << origin.transpose() << "] -- the local-frame offset was not added back";

  // And the corridors handed back must be world-frame too, or the collision re-check tests the
  // wrong volume.
  ASSERT_FALSE(r.corridors.empty());
  EXPECT_LT((r.corridors.front().primaryPosition - origin).norm(), 1e-6);
}

}  // namespace tests
}  // namespace trajectory_planner
