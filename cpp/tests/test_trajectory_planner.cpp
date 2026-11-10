#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <chrono>
#include "trajectory_planner.hpp"
#include "minvo_builder.hpp"
#include "min_snap_clamped.hpp"
#include "min_snap_natural.hpp"

namespace trajectory_planner {
namespace tests {

// ==========================================
// Helper: Build a simple obstacle field for testing
// ==========================================

// ---------------------------------------------------------------------------------------
// COORDINATE CONVENTION
//
// These fixtures were originally written for a 0..100 m positive-orthant world with
// POSITIVE-UP z. The planner works in NED: the A* geofence is +/- map_bounds/2 about the
// ORIGIN, and z is NEGATIVE-up (altitude 5 m is z = -5). Under the old coordinates the start
// sat several metres underground -- rejected by the min_altitude floor -- and the goal fell
// outside the geofence entirely, so A* correctly returned a single-point path and every
// assertion on corridor count failed. The scene below is the same layout expressed in NED:
// x/y centred on the origin, altitudes negative and inside the +/- map_bounds.z()/2 band.
// ---------------------------------------------------------------------------------------
static std::vector<mapping::ObstacleBox> build_test_obstacles() {
    std::vector<mapping::ObstacleBox> obstacles;

    // A wall at X=50, spanning Y=[30,70], Z=[0,15]
    obstacles.push_back({{0.0, -20.0, -7.0}, {2.0, 20.0, 0.0}});

    // A pillar at (30, 50), Z=[0,10]
    obstacles.push_back({{-22.0, -2.0, -5.0}, {-18.0, 2.0, 0.0}});

    // A pillar at (70, 50), Z=[0,10]
    obstacles.push_back({{18.0, -2.0, -5.0}, {22.0, 2.0, 0.0}});

    return obstacles;
}

// ==========================================
// TEST 1: TrajectoryPlanner — Full Pipeline Clamped
// ==========================================
TEST(TrajectoryPlannerTest, ClampedFullPipeline) {
    FrontEndConfig config;
    config.voxel_resolution = 0.5;
    config.drone_physical_radius = 0.5;
    config.sfc_height = 5.0;
    config.sfc_width = 5.0;
    config.sfc_start_ext = 5.0;
    config.sfc_end_ext = 5.0;
    config.spline_type = "clamped";
    config.aircraft_type = "multi-rotor";
    config.degree = 4;
    config.map_bounds = Eigen::Vector3d(100.0, 100.0, 15.0);
    // Disable the wall-clock solve budget. It is a real-time safeguard for flight; in a
    // test it makes the outcome depend on runner speed (a 147 ms first attempt on a loaded
    // Windows CI runner bailed before the retry that succeeds everywhere else).
    config.max_solve_time_ms = 0.0;

    auto obstacles = build_test_obstacles();
    auto grid = std::make_shared<mapping::SparseVoxelGrid>(config.voxel_resolution, config.drone_physical_radius);
    grid->update_from_obstacles(obstacles);
    TrajectoryPlanner planner(config, grid, 3.0, 2.0); // v_max=3, a_max=2

    Eigen::Vector3d start(-30.0, -30.0, -4.0);
    Eigen::Vector3d goal(30.0, 30.0, -5.0);

    auto result = planner.plan_mission(start, goal);

    EXPECT_TRUE(result.success) << "OSQP Failed to solve trajectory!";
    EXPECT_GT(result.total_control_points, 0);
    EXPECT_EQ(result.control_points.size(), result.total_control_points * 3);

    std::cout << "\n============================================" << std::endl;
    std::cout << "  C++ TRAJECTORY PLANNER (CLAMPED) RESULTS" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Total Ctrl Points:  " << result.total_control_points << std::endl;
    std::cout << "SFC Time:           " << result.sfc_time_ms << " ms" << std::endl;
    std::cout << "Opt Time:           " << result.opt_time_ms << " ms" << std::endl;
    std::cout << "Overhead Time:      " << result.overhead_ms << " ms" << std::endl;
    std::cout << "Total Time:         " << result.total_time_ms << " ms" << std::endl;
    std::cout << "============================================\n" << std::endl;
}

// ==========================================
// TEST 2: TrajectoryPlanner — Empty Map
// ==========================================
TEST(TrajectoryPlannerTest, EmptyMapNatural) {
    FrontEndConfig config;
    config.voxel_resolution = 1.0;
    config.drone_physical_radius = 0.5;
    config.sfc_height = 5.0;
    config.sfc_width = 5.0;
    config.sfc_start_ext = 5.0;
    config.sfc_end_ext = 5.0;
    config.spline_type = "natural";
    config.aircraft_type = "multi-rotor";
    config.degree = 4;
    config.map_bounds = Eigen::Vector3d(50.0, 50.0, 15.0);
    // Disable the wall-clock solve budget. It is a real-time safeguard for flight; in a
    // test it makes the outcome depend on runner speed (a 147 ms first attempt on a loaded
    // Windows CI runner bailed before the retry that succeeds everywhere else).
    config.max_solve_time_ms = 0.0;

    std::vector<mapping::ObstacleBox> no_obstacles;
    auto grid = std::make_shared<mapping::SparseVoxelGrid>(config.voxel_resolution, config.drone_physical_radius);
    grid->update_from_obstacles(no_obstacles);
    TrajectoryPlanner planner(config, grid, 5.0, 3.0); // v_max=5, a_max=3

    Eigen::Vector3d start(-20.0, -20.0, -4.0);
    Eigen::Vector3d goal(20.0, 20.0, -5.0);

    auto result = planner.plan_mission(start, goal);

    EXPECT_TRUE(result.success) << "OSQP Failed to solve empty map trajectory!";
    EXPECT_GT(result.total_control_points, 0);
    EXPECT_EQ(result.control_points.size(), result.total_control_points * 3);
}

// ==========================================
// TEST 3: TrajectoryPlanner — Floating Blocks Benchmark
// ==========================================
TEST(TrajectoryPlannerTest, FloatingBlocksBenchmark) {
    FrontEndConfig config;
    config.voxel_resolution = 0.5;
    config.drone_physical_radius = 0.5;
    config.sfc_height = 5.0;
    config.sfc_width = 5.0;
    config.sfc_start_ext = 5.0;
    config.sfc_end_ext = 5.0;
    config.spline_type = "clamped";
    config.aircraft_type = "fixed-wing";
    config.degree = 4;
    config.map_bounds = Eigen::Vector3d(100.0, 100.0, 15.0);
    // Disable the wall-clock solve budget. It is a real-time safeguard for flight; in a
    // test it makes the outcome depend on runner speed (a 147 ms first attempt on a loaded
    // Windows CI runner bailed before the retry that succeeds everywhere else).
    config.max_solve_time_ms = 0.0;

    // Build floating blocks
    std::vector<mapping::ObstacleBox> obstacles;
    int num_blocks = 4;
    double block_width = 10.0;
    double x_inc = 100.0 / num_blocks;
    double y_inc = 100.0 / num_blocks;
    double z_inc = 15.0 / num_blocks;
    double x_start = x_inc / 2.0;
    double y_start = y_inc / 2.0;
    double z_start = z_inc / 2.0;

    for (int i = 0; i < num_blocks; ++i) {
        for (int j = 0; j < num_blocks; ++j) {
            for (int k = 0; k < num_blocks; ++k) {
                // Centred on the origin and negative-up, to match the NED geofence.
                double cx = x_start + i * x_inc - 50.0;
                double cy = y_start + j * y_inc - 50.0;
                double cz = -(z_start + k * z_inc) * 0.5;
                obstacles.push_back({
                    {cx - block_width / 2.0, cy - block_width / 2.0, cz - block_width / 2.0},
                    {cx + block_width / 2.0, cy + block_width / 2.0, cz + block_width / 2.0}
                });
            }
        }
    }

    auto grid = std::make_shared<mapping::SparseVoxelGrid>(config.voxel_resolution, config.drone_physical_radius);
    grid->update_from_obstacles(obstacles);
    TrajectoryPlanner planner(config, grid, 3.0, 2.0);

    Eigen::Vector3d start(-30.0, -30.0, -4.0);
    Eigen::Vector3d goal(30.0, 30.0, -6.0);

    auto result = planner.plan_mission(start, goal);

    EXPECT_TRUE(result.success) << "OSQP Failed to solve floating blocks trajectory!";
    EXPECT_GT(result.total_control_points, 0);

    std::cout << "\n============================================" << std::endl;
    std::cout << "  C++ TRAJECTORY PLANNER (FLOATING BLOCKS) RESULTS" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Total Ctrl Points:  " << result.total_control_points << std::endl;
    std::cout << "SFC Time:           " << result.sfc_time_ms << " ms" << std::endl;
    std::cout << "Opt Time:           " << result.opt_time_ms << " ms" << std::endl;
    std::cout << "Overhead Time:      " << result.overhead_ms << " ms" << std::endl;
    std::cout << "Total Time:         " << result.total_time_ms << " ms" << std::endl;
    std::cout << "============================================\n" << std::endl;
}

// ==========================================
// TEST 4: MINVO Builder Shape Check
// ==========================================
TEST(TrajectoryPlannerTest, MinvoBuilderShape) {
    int degree = 4;
    int num_segments = 5;
    MinSnapEvalClamped optimizer(num_segments, degree);

    // Get a dummy D matrix
    Eigen::MatrixXd D_vel = optimizer.get_fast_cascaded_D_matrix(num_segments, degree, 1);
    
    // D_vel is (rows, cols)
    int D_cols = D_vel.cols();

    // Call MINVO builder
    Eigen::SparseMatrix<double> A_minvo = minvo::build_minvo_kinodynamic_matrix(D_vel, degree - 1, "clamped");

    // The mapping matrix M is (num_minvo_points_total, D_rows)
    // D is (D_rows, D_cols)
    // A_minvo is (num_minvo_points_total, D_cols)
    
    EXPECT_EQ(A_minvo.cols(), D_cols) << "A_minvo columns should match D_matrix columns";
    EXPECT_GT(A_minvo.rows(), 0) << "A_minvo rows should be > 0";
    EXPECT_GT(A_minvo.nonZeros(), 0) << "A_minvo should have non-zero elements";
}

} // namespace tests
} // namespace trajectory_planner
