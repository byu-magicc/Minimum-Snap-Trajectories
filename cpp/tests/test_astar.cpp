#include <gtest/gtest.h>
#include <chrono>
#include <Eigen/Dense>
#include "astar_sfc.hpp"

namespace trajectory_planner {
namespace path_finding {
namespace tests {

TEST(AStarParityTest, WallBypassAndSmoothing) {
    // 1. Setup the identical Mock Grid
    // VoxelGridData is a borrowed view now, so the test owns the actual cell set.
    VoxelSet occupied;
    VoxelGridData grid;
    grid.voxel_resolution = 1.0;
    grid.occupied_voxels_inflated = &occupied;
    
    // 2. Build the same wall at X=2
    for(int y = 0; y <= 5; ++y) {
        for(int z = 0; z <= 5; ++z) {
            occupied.insert(Eigen::Vector3i(2, y, z));
        }
    }
    
    Eigen::Vector3i bounds_min(0, 0, 0);
    Eigen::Vector3i bounds_max(10, 10, 10);
    
    AStarSFCPlanner planner(grid, bounds_min, bounds_max);
    
    // 3. Execute Search
    Eigen::Vector3i start(0, 0, 0);
    Eigen::Vector3i goal(5, 5, 5);
    
    // 1. Benchmark the Search Phase
    auto start_search = std::chrono::high_resolution_clock::now();
    std::vector<Eigen::Vector3i> raw_path = planner.search(start, goal);
    auto end_search = std::chrono::high_resolution_clock::now();

    ASSERT_FALSE(raw_path.empty());

    // 2. Benchmark the Smoothing Phase
    auto start_smooth = std::chrono::high_resolution_clock::now();
    std::vector<Eigen::Vector3i> smoothed_path = planner.sfc_smoother();
    auto end_smooth = std::chrono::high_resolution_clock::now();

    // 3. Calculate Durations
    auto search_time = std::chrono::duration_cast<std::chrono::microseconds>(end_search - start_search).count();
    auto smooth_time = std::chrono::duration_cast<std::chrono::microseconds>(end_smooth - start_smooth).count();

    std::cout << "\n--- C++ BENCHMARK ---" << std::endl;
    for (const auto& wp : smoothed_path) {
        std::cout << "[" << wp.x() << ", " << wp.y() << ", " << wp.z() << "]\n";
    }
    std::cout << "------------------------------" << std::endl;
    std::cout << "[ C++ ] Search Time: " << search_time << " µs" << std::endl;
    std::cout << "[ C++ ] Smooth Time: " << smooth_time << " µs" << std::endl;
    std::cout << "------------------------------\n" << std::endl;
}

} // namespace tests
} // namespace path_finding
} // namespace trajectory_planner