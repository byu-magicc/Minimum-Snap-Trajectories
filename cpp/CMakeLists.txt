cmake_minimum_required(VERSION 3.16)
project(MinimumSnapEngine VERSION 1.0 LANGUAGES CXX)

# Force modern C++ standards (Highly recommended for Eigen3)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==========================================
# 1. FIND REQUIRED LIBRARIES
# ==========================================
# Because you installed these natively via apt, CMake will automatically 
# search your system's /usr/include directories for them.
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)
find_package(osqp REQUIRED)

# ==========================================
# 2. BUILD THE MATH ENGINE LIBRARY
# ==========================================
# STATIC on purpose.
#
# As a shared library this ended up existing in two places at once -- cpp/build/ from a
# standalone build, and the copy a consumer installs after pulling this repo in via
# add_subdirectory(). Rebuild one and not the other and the loader silently picks up the
# stale copy, which surfaces as undefined symbols, garbage return values, or
# "free(): invalid pointer" -- none of which look like a linking problem.
#
# Linked statically, the code lands inside each consumer's binary. A stale archive then
# fails at LINK time with a clear undefined-symbol error instead of at runtime with
# nonsense. PIC is kept on so this can still be linked into a shared object (e.g. the
# pybind11 module) if needed.
add_library(min_snap_engine STATIC
    src/min_snap_natural.cpp
    src/min_snap_clamped.cpp
    src/matrix_builder.cpp
    src/optimize.cpp
    src/astar_sfc.cpp
    src/voxel_grid.cpp
    src/static_sfc.cpp
    src/dynamic_sfc.cpp
    src/front_end.cpp
    src/trajectory_planner.cpp
)

# Tell the compiler where to find your .hpp header files
target_include_directories(min_snap_engine PUBLIC include)
set_target_properties(min_snap_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Keep Eigen's bounds checks alive in optimised builds.
#
# Release defines NDEBUG, which deletes every Eigen assertion, so an out-of-range block
# corrupts the heap instead of faulting (see include/eigen_safety.hpp for the bug that
# motivated this). Force-including the header ahead of every translation unit redefines
# eigen_assert to throw, which the planner's existing try/catch turns into a recoverable
# planning failure. PUBLIC so consumers (roscopter) get the same protection.
option(MIN_SNAP_EIGEN_SAFETY "Turn Eigen bounds assertions into exceptions in release builds" ON)
if(MIN_SNAP_EIGEN_SAFETY)
    target_compile_options(min_snap_engine PUBLIC
        -include ${CMAKE_CURRENT_SOURCE_DIR}/include/eigen_safety.hpp)
endif()

# Link the Eigen3 math library to your engine
target_link_libraries(min_snap_engine PUBLIC Eigen3::Eigen osqp::osqp)

# ==========================================
# 3. BUILD THE GOOGLE TEST SUITE
# ==========================================
enable_testing()

add_executable(bspline_tests
    tests/test_min_snap.cpp
    tests/test_optimize.cpp
    tests/test_astar.cpp
    tests/test_blocked_start.cpp
    tests/test_voxel_grid.cpp
    tests/test_occupancy_thickness.cpp
    tests/test_heat_field.cpp
    tests/test_segment_merge.cpp
    tests/test_corridor_clearance.cpp
    tests/test_corridor_clearance_integration.cpp
    tests/test_chain_length.cpp
    tests/test_altitude_ceiling.cpp
    tests/test_translation_invariance.cpp
    tests/test_path_hysteresis.cpp
    tests/test_sfc.cpp
    tests/test_front_end.cpp
    tests/test_trajectory_planner.cpp
)

# Link the test suite to your engine AND to Google Test's main runner
target_link_libraries(bspline_tests
    PRIVATE
    min_snap_engine
    GTest::gtest_main
)

# Make CTest aware of the Google Tests
include(GoogleTest)
gtest_discover_tests(bspline_tests)

# ==========================================
# 3b. EXAMPLES
# ==========================================
# Built by default so that CI catches the example rotting against an API change. It is one
# small translation unit and nothing is executed at build time, so the cost is negligible --
# and a usage example that no longer compiles is worse than no example at all.
option(BUILD_EXAMPLES "Build example programs" ON)

if(BUILD_EXAMPLES)
    add_executable(minimal_planner_loop examples/minimal_planner_loop.cpp)
    target_link_libraries(minimal_planner_loop PRIVATE min_snap_engine)

    # Offline reproduction of a QP failure captured in flight. Replays a logged corridor chain
    # through the real back end, so a hypothesis about why a solve failed costs a millisecond
    # instead of a sim run.
    add_executable(repro_forest_stall tools/repro_forest_stall.cpp)
    target_link_libraries(repro_forest_stall PRIVATE min_snap_engine)

    add_executable(replay_stall tools/replay_stall.cpp)
    target_link_libraries(replay_stall PRIVATE min_snap_engine)

    # The harness uses the SAME directional gate the vehicle flies (header-only, Eigen + STL
    # only). Scoped to this tool: min_snap_engine must not gain a roscopter dependency -- the
    # dependency runs the other way.
    #
    # roscopter is not part of this repo, so the tool is built only when the gate header can
    # be found; otherwise it is skipped and the rest of the build is unaffected. The default
    # hint is the usual ROS workspace layout; point elsewhere with
    #   cmake -DROSCOPTER_INCLUDE_DIR=<path>/roscopter/roscopter/include ..
    find_path(ROSCOPTER_INCLUDE_DIR navigation/route_gate.hpp
        HINTS $ENV{HOME}/rosflight_ws/src/roscopter/roscopter/include
        DOC "roscopter include directory (only needed for tools/monte_carlo.cpp)")
    if(ROSCOPTER_INCLUDE_DIR)
        add_executable(monte_carlo tools/monte_carlo.cpp)
        target_link_libraries(monte_carlo PRIVATE min_snap_engine)
        target_include_directories(monte_carlo PRIVATE ${ROSCOPTER_INCLUDE_DIR})
    else()
        message(STATUS "Skipping monte_carlo: navigation/route_gate.hpp (roscopter) not found. "
                       "Set -DROSCOPTER_INCLUDE_DIR to build it.")
    endif()

    add_executable(wall_probe tools/wall_probe.cpp)
    target_link_libraries(wall_probe PRIVATE min_snap_engine)

    add_executable(vmax_probe tools/vmax_probe.cpp)
    target_link_libraries(vmax_probe PRIVATE min_snap_engine)
endif()

# ==========================================
# 4. PYTHON BINDINGS (PYBIND11)
# ==========================================
# Default OFF: src/python_bindings.cpp still calls the old
# TrajectoryPlanner(config, obstacles, ...) signature, which was replaced by
# TrajectoryPlanner(config, shared_ptr<SparseVoxelGrid>, ...). Leaving this ON made the
# documented `make -j$(nproc)` fail for anyone following the README. Turn it back ON once
# the bindings are updated.
option(BUILD_PYTHON_BINDINGS "Build Python Bindings" OFF)

if(BUILD_PYTHON_BINDINGS)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(min_snap_bindings src/python_bindings.cpp)
    target_include_directories(min_snap_bindings PUBLIC include)
    target_link_libraries(min_snap_bindings PRIVATE min_snap_engine)
endif()