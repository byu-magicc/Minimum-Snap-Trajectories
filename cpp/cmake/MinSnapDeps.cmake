# Resolves the engine's two third-party dependencies, Eigen3 and OSQP, into the targets
# Eigen3::Eigen and osqp::osqp.
#
# MIN_SNAP_BUNDLED_DEPS=OFF (default): use the system copies if find_package() finds them, and
#   download only the missing ones. Consumers that use add_subdirectory() (roscopter) keep
#   linking the same system Eigen they use themselves. Mixing two Eigen versions in one binary
#   is an ODR violation, so the system copy must win whenever it exists.
# MIN_SNAP_BUNDLED_DEPS=ON (what pip sets): always download pinned versions and link OSQP
#   statically, so the Python extension needs nothing installed beyond a C/C++ compiler and has
#   no runtime library dependency.
#
# Downloads need network access at configure time. They are cached in the build tree, so they
# happen once per build directory.

include(FetchContent)

set(MIN_SNAP_EIGEN_VERSION 3.4.0)
set(MIN_SNAP_OSQP_VERSION  v1.0.0)   # the code uses the OSQP 1.x API (OSQPInt, OSQPFloat)

option(MIN_SNAP_BUNDLED_DEPS "Always download and build Eigen3 and OSQP instead of using system copies" OFF)

# ------------------------------------------------------------------------------------------
# Eigen (header-only)
# ------------------------------------------------------------------------------------------
function(min_snap_bundle_eigen)
    message(STATUS "min_snap: using bundled Eigen ${MIN_SNAP_EIGEN_VERSION}")
    FetchContent_Declare(min_snap_eigen
        URL https://gitlab.com/libeigen/eigen/-/archive/${MIN_SNAP_EIGEN_VERSION}/eigen-${MIN_SNAP_EIGEN_VERSION}.tar.gz
        URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
        # Download only. Eigen's own CMakeLists configures its test suite, BLAS and docs;
        # pointing SOURCE_SUBDIR at a directory with no CMakeLists.txt skips all of it.
        SOURCE_SUBDIR do-not-configure)
    FetchContent_MakeAvailable(min_snap_eigen)
    add_library(min_snap_bundled_eigen INTERFACE)
    target_include_directories(min_snap_bundled_eigen SYSTEM INTERFACE ${min_snap_eigen_SOURCE_DIR})
    add_library(Eigen3::Eigen ALIAS min_snap_bundled_eigen)
endfunction()

# ------------------------------------------------------------------------------------------
# OSQP (C library, built static)
# ------------------------------------------------------------------------------------------
# A function so the option overrides and PIC setting stay scoped to OSQP's subdirectory.
function(min_snap_bundle_osqp)
    message(STATUS "min_snap: using bundled OSQP ${MIN_SNAP_OSQP_VERSION} (static)")
    set(OSQP_BUILD_STATIC_LIB ON  CACHE INTERNAL "")
    set(OSQP_BUILD_SHARED_LIB OFF CACHE INTERNAL "")
    set(OSQP_BUILD_DEMO_EXE   OFF CACHE INTERNAL "")
    set(OSQP_BUILD_UNITTESTS  OFF CACHE INTERNAL "")
    # Linked into the Python extension (a shared object), so it must be position independent.
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    FetchContent_Declare(min_snap_osqp
        GIT_REPOSITORY https://github.com/osqp/osqp.git
        GIT_TAG        ${MIN_SNAP_OSQP_VERSION}
        GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(min_snap_osqp)
    # osqpstatic exposes include/public (osqp.h at its root), the same layout as the installed
    # target's include/osqp, so `#include <osqp.h>` works with either.
    add_library(osqp::osqp ALIAS osqpstatic)
endfunction()

if(MIN_SNAP_BUNDLED_DEPS)
    min_snap_bundle_eigen()
    min_snap_bundle_osqp()
else()
    find_package(Eigen3 3.3 QUIET NO_MODULE)
    if(TARGET Eigen3::Eigen)
        message(STATUS "min_snap: using system Eigen ${Eigen3_VERSION}")
    else()
        min_snap_bundle_eigen()
    endif()

    find_package(osqp QUIET)
    if(TARGET osqp::osqp)
        message(STATUS "min_snap: using system OSQP (${osqp_DIR})")
    else()
        min_snap_bundle_osqp()
    endif()
endif()
