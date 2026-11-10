// Python bindings for the minimum-snap engine.
//
// Built as the extension module `min_snap._core`; python/min_snap/__init__.py re-exports it.
// Two paradigms are exposed:
//
//   1. Unconstrained minimum snap -- MinSnapEvalClamped / MinSnapEvalNatural. The optimal
//      control points are a closed-form linear map of the boundary states, C = SE * Q, so a
//      solve is one small matrix product. solve_unconstrained() builds SE for you.
//
//   2. SFC-constrained mission planning -- SparseVoxelGrid + FrontEndConfig +
//      TrajectoryPlanner. A* over the voxel grid, Safe Flight Corridors, then an OSQP solve.
//
// Conversions: Eigen <-> NumPy via pybind11/eigen.h, std::vector/std::string via
// pybind11/stl.h. Every Eigen value returned to Python is a COPY (owning NumPy array), never a
// view into C++ state, so arrays stay valid after the object that produced them changes.
//
// The GIL is deliberately NOT released around planning or map updates. The engine is not
// thread-safe (a map update racing a plan on the same grid is undefined behaviour); holding
// the GIL makes Python threads serialise those calls automatically.

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bspline_eval.hpp"
#include "front_end.hpp"
#include "min_snap_clamped.hpp"
#include "min_snap_natural.hpp"
#include "trajectory_planner.hpp"
#include "voxel_grid.hpp"

namespace py = pybind11;
using namespace trajectory_planner;

namespace {

using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

// ------------------------------------------------------------------------------------------
// Argument helpers
// ------------------------------------------------------------------------------------------

// The C++ solvers take int16_t segments / int8_t degree. Accept a Python int and range-check
// it rather than letting a large value wrap silently.
int16_t checked_segments(int n) {
    if (n < 3 || n > std::numeric_limits<int16_t>::max()) {
        throw py::value_error("num_segments must be in [3, 32767], got " + std::to_string(n));
    }
    return static_cast<int16_t>(n);
}

int8_t checked_degree(int d) {
    if (d < 4 || d > std::numeric_limits<int8_t>::max()) {
        throw py::value_error("degree must be >= 4 for minimum snap, got " + std::to_string(d));
    }
    return static_cast<int8_t>(d);
}

// Accepts an (N, 3) array, a flat array of length 3N, a list of 3-vectors, or an empty
// sequence, and returns the points. One bulk conversion instead of N per-row casts.
std::vector<Vector3d> to_points(const py::object& obj) {
    auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(obj);
    if (!arr) throw py::type_error("expected an (N, 3) array-like of points");
    if (arr.size() == 0) return {};
    if (arr.size() % 3 != 0 || (arr.ndim() == 2 && arr.shape(1) != 3) || arr.ndim() > 2) {
        throw py::value_error("points must have shape (N, 3)");
    }
    const double* d = arr.data();
    const py::ssize_t n = arr.size() / 3;
    std::vector<Vector3d> pts;
    pts.reserve(static_cast<std::size_t>(n));
    for (py::ssize_t i = 0; i < n; ++i) pts.emplace_back(d[3 * i], d[3 * i + 1], d[3 * i + 2]);
    return pts;
}

// Packs a set of grid indices into an (N, 3) int array.
template <class Set>
py::array_t<int> indices_to_array(const Set& s) {
    py::array_t<int> out({static_cast<py::ssize_t>(s.size()), py::ssize_t(3)});
    auto r = out.mutable_unchecked<2>();
    py::ssize_t i = 0;
    for (const auto& v : s) {
        r(i, 0) = v.x(); r(i, 1) = v.y(); r(i, 2) = v.z();
        ++i;
    }
    return out;
}

// Same flooring the grid uses internally (SparseVoxelGrid::to_index is private).
Eigen::Vector3i world_to_index(const mapping::SparseVoxelGrid& g, const Vector3d& p) {
    const double r = g.get_resolution();
    return {static_cast<int>(std::floor(p.x() / r)), static_cast<int>(std::floor(p.y() / r)),
            static_cast<int>(std::floor(p.z() / r))};
}

// ------------------------------------------------------------------------------------------
// Unconstrained solve: C = SE * Q
// ------------------------------------------------------------------------------------------
//
// Both solvers compute Q (6 x N) on a UNIT knot spacing, so the derivative states in SE are
// in knot-time: a physical velocity v over knot spacing dt is v*dt, an acceleration a*dt^2.
// The column ORDER also differs between the two -- it follows how each Q was derived:
//   clamped: [p0, v0, a0, af, vf, pf]   (mirrors the B_d3 boundary block)
//   natural: [p0, v0, a0, pf, vf, af]   (start triple, then end triple)
// These helpers own both details so callers only ever pass physical states.

enum class SEOrder { Clamped, Natural };

MatrixXd build_SE(SEOrder order, const VectorXd& p0, const VectorXd& v0, const VectorXd& a0,
                  const VectorXd& pf, const VectorXd& vf, const VectorXd& af, double dt) {
    if (!(dt > 0.0) || !std::isfinite(dt)) throw py::value_error("dt must be a positive finite number");
    const Eigen::Index dim = p0.size();
    for (const VectorXd* v : {&v0, &a0, &pf, &vf, &af}) {
        if (v->size() != dim) throw py::value_error("all boundary states must have the same dimension");
    }
    if (dim < 1) throw py::value_error("boundary states must be non-empty");

    MatrixXd SE(dim, 6);
    SE.col(0) = p0;
    SE.col(1) = v0 * dt;
    SE.col(2) = a0 * (dt * dt);
    if (order == SEOrder::Clamped) {
        SE.col(3) = af * (dt * dt);
        SE.col(4) = vf * dt;
        SE.col(5) = pf;
    } else {
        SE.col(3) = pf;
        SE.col(4) = vf * dt;
        SE.col(5) = af * (dt * dt);
    }
    return SE;
}

MatrixXd apply_SE(const MatrixXd& SE, const MatrixXd& Q) {
    if (SE.cols() != Q.rows()) {
        std::ostringstream os;
        os << "SE must have " << Q.rows() << " columns (got shape " << SE.rows() << "x" << SE.cols()
           << ")";
        throw py::value_error(os.str());
    }
    return SE * Q;
}

constexpr const char* kSolveDocSE =
    "Return the minimum-snap control points C = SE @ Q.\n\n"
    "SE is a (D, 6) matrix of boundary states in this solver's NATIVE column order and in\n"
    "unit-knot time (see the class docstring). Result has shape (D, num_control_points).";

constexpr const char* kSolveDocStates =
    "Return the minimum-snap control points for the given boundary states.\n\n"
    "States are physical (any dimension D, typically 3). dt is the knot spacing in seconds:\n"
    "velocities are scaled by dt and accelerations by dt**2 to convert into the solver's\n"
    "unit-knot time. Evaluate the result against scaled_knots(dt).\n"
    "Result has shape (D, num_control_points).";

// Degree / segment count recovered from the public accessors (the C++ classes keep these
// private): Q has one column per control point N, knots = N + degree + 1, N = M + degree.
Eigen::Index degree_of(const MatrixXd& Q, const VectorXd& knots) { return knots.size() - Q.cols() - 1; }
Eigen::Index segments_of(const MatrixXd& Q, const VectorXd& knots) { return Q.cols() - degree_of(Q, knots); }

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ minimum-snap B-spline trajectory engine (pybind11 bindings)";

    // ==========================================================================================
    // 1. Unconstrained solvers
    // ==========================================================================================

    py::class_<MinSnapEvalClamped>(m, "MinSnapEvalClamped", R"doc(
Closed-form minimum-snap solver on a CLAMPED uniform B-spline.

The control points minimising integrated snap subject to start/end position, velocity and
acceleration are C = SE @ Q, where Q = get_Q_matrix() is (6, N). The trajectory interpolates
p0 and pf exactly (clamped knots). SE's native column order is [p0, v0, a0, af, vf, pf] in
unit-knot time; solve_unconstrained(p0, v0, a0, pf, vf, af, dt) handles both for you.
)doc")
        .def(py::init([](int num_segments, int degree) {
                 return MinSnapEvalClamped(checked_segments(num_segments), checked_degree(degree));
             }),
             py::arg("num_segments"), py::arg("degree") = 4)
        .def("get_Q_matrix", &MinSnapEvalClamped::get_Q_matrix, "(6, N) boundary-to-control-point map.")
        .def("get_knots", &MinSnapEvalClamped::get_knots, "Clamped knot vector on unit spacing.")
        .def("get_W", &MinSnapEvalClamped::get_W, "Pure minimum-snap penalty matrix (N, N).")
        .def("get_W_matrix", &MinSnapEvalClamped::get_W_matrix,
             py::arg("rho_vel") = 0.0, py::arg("rho_accel") = 0.0, py::arg("rho_snap") = 1.0,
             "Weighted penalty matrix rho_vel*W_vel + rho_accel*W_acc + rho_snap*W_snap.")
        .def("get_B_combined", &MinSnapEvalClamped::get_B_combined)
        .def("update_segments",
             [](MinSnapEvalClamped& s, int n) { s.update_segments(checked_segments(n)); },
             py::arg("num_segments"), "Rebuild Q, W and knots for a new segment count.")
        .def_property_readonly("num_control_points",
                               [](const MinSnapEvalClamped& s) { return s.get_Q_matrix().cols(); })
        .def_property_readonly("degree", [](const MinSnapEvalClamped& s) {
            return degree_of(s.get_Q_matrix(), s.get_knots());
        })
        .def_property_readonly("num_segments", [](const MinSnapEvalClamped& s) {
            return segments_of(s.get_Q_matrix(), s.get_knots());
        })
        .def("scaled_knots",
             [](const MinSnapEvalClamped& s, double dt) { return VectorXd(s.get_knots() * dt); },
             py::arg("dt") = 1.0, "Knot vector in seconds for knot spacing dt.")
        .def("solve_unconstrained",
             [](const MinSnapEvalClamped& s, const MatrixXd& SE) { return apply_SE(SE, s.get_Q_matrix()); },
             py::arg("SE"), kSolveDocSE)
        .def("solve_unconstrained",
             [](const MinSnapEvalClamped& s, const VectorXd& p0, const VectorXd& v0, const VectorXd& a0,
                const VectorXd& pf, const VectorXd& vf, const VectorXd& af, double dt) {
                 return apply_SE(build_SE(SEOrder::Clamped, p0, v0, a0, pf, vf, af, dt), s.get_Q_matrix());
             },
             py::arg("p0"), py::arg("v0"), py::arg("a0"), py::arg("pf"), py::arg("vf"), py::arg("af"),
             py::arg("dt") = 1.0, kSolveDocStates)
        .def_static("build_SE",
             [](const VectorXd& p0, const VectorXd& v0, const VectorXd& a0, const VectorXd& pf,
                const VectorXd& vf, const VectorXd& af, double dt) {
                 return build_SE(SEOrder::Clamped, p0, v0, a0, pf, vf, af, dt);
             },
             py::arg("p0"), py::arg("v0"), py::arg("a0"), py::arg("pf"), py::arg("vf"), py::arg("af"),
             py::arg("dt") = 1.0, "Boundary matrix in the clamped solver's native column order.");

    py::class_<MinSnapEvalNatural>(m, "MinSnapEvalNatural", R"doc(
Closed-form minimum-snap solver on a NATURAL (uniform, unclamped) B-spline.

Same C = SE @ Q formulation as MinSnapEvalClamped, but the knots run uniformly from -degree,
so the curve's valid span is [knots[degree], knots[-degree-1]] = [0, num_segments]. SE's
native column order is [p0, v0, a0, pf, vf, af] in unit-knot time. Both the original
camelCase method names and snake_case aliases are provided.
)doc")
        .def(py::init([](int num_segments, int degree) {
                 return MinSnapEvalNatural(checked_segments(num_segments), checked_degree(degree));
             }),
             py::arg("num_segments"), py::arg("degree") = 4)
        .def("getQMatrix", &MinSnapEvalNatural::getQMatrix)
        .def("getKnots", &MinSnapEvalNatural::getKnots)
        .def("getWMatrix", &MinSnapEvalNatural::getWMatrix,
             py::arg("rho_vel") = 0.0, py::arg("rho_accel") = 0.0, py::arg("rho_snap") = 1.0)
        .def("getW", &MinSnapEvalNatural::getW)
        .def("getBCombined", &MinSnapEvalNatural::getBCombined)
        .def("updateSegments",
             [](MinSnapEvalNatural& s, int n) { s.updateSegments(checked_segments(n)); },
             py::arg("num_segments"))
        // snake_case aliases so both solvers share one Python interface
        .def("get_Q_matrix", &MinSnapEvalNatural::getQMatrix, "(6, N) boundary-to-control-point map.")
        .def("get_knots", &MinSnapEvalNatural::getKnots, "Uniform knot vector starting at -degree.")
        .def("get_W", &MinSnapEvalNatural::getW, "Pure minimum-snap penalty matrix (N, N).")
        .def("get_W_matrix", &MinSnapEvalNatural::getWMatrix,
             py::arg("rho_vel") = 0.0, py::arg("rho_accel") = 0.0, py::arg("rho_snap") = 1.0)
        .def("get_B_combined", &MinSnapEvalNatural::getBCombined)
        .def("update_segments",
             [](MinSnapEvalNatural& s, int n) { s.updateSegments(checked_segments(n)); },
             py::arg("num_segments"))
        .def_property_readonly("num_control_points",
                               [](const MinSnapEvalNatural& s) { return s.getQMatrix().cols(); })
        .def_property_readonly("degree", [](const MinSnapEvalNatural& s) {
            return degree_of(s.getQMatrix(), s.getKnots());
        })
        .def_property_readonly("num_segments", [](const MinSnapEvalNatural& s) {
            return segments_of(s.getQMatrix(), s.getKnots());
        })
        .def("scaled_knots",
             [](const MinSnapEvalNatural& s, double dt) { return VectorXd(s.getKnots() * dt); },
             py::arg("dt") = 1.0, "Knot vector in seconds for knot spacing dt.")
        .def("solve_unconstrained",
             [](const MinSnapEvalNatural& s, const MatrixXd& SE) { return apply_SE(SE, s.getQMatrix()); },
             py::arg("SE"), kSolveDocSE)
        .def("solve_unconstrained",
             [](const MinSnapEvalNatural& s, const VectorXd& p0, const VectorXd& v0, const VectorXd& a0,
                const VectorXd& pf, const VectorXd& vf, const VectorXd& af, double dt) {
                 return apply_SE(build_SE(SEOrder::Natural, p0, v0, a0, pf, vf, af, dt), s.getQMatrix());
             },
             py::arg("p0"), py::arg("v0"), py::arg("a0"), py::arg("pf"), py::arg("vf"), py::arg("af"),
             py::arg("dt") = 1.0, kSolveDocStates)
        .def_static("build_SE",
             [](const VectorXd& p0, const VectorXd& v0, const VectorXd& a0, const VectorXd& pf,
                const VectorXd& vf, const VectorXd& af, double dt) {
                 return build_SE(SEOrder::Natural, p0, v0, a0, pf, vf, af, dt);
             },
             py::arg("p0"), py::arg("v0"), py::arg("a0"), py::arg("pf"), py::arg("vf"), py::arg("af"),
             py::arg("dt") = 1.0, "Boundary matrix in the natural solver's native column order.");

    // ==========================================================================================
    // 2. B-spline evaluation (de Boor) -- works for clamped, natural and planner outputs
    // ==========================================================================================

    py::class_<bspline::TrajectorySampler>(m, "TrajectorySampler", R"doc(
Evaluates position / velocity / acceleration of a 3-D B-spline via de Boor.

control_points may be flat [x0, y0, z0, x1, ...] (as in PlanningResult), (N, 3), or (3, N)
(as returned by solve_unconstrained). Times outside [start_time, end_time] are clamped.
)doc")
        .def(py::init([](const py::object& cps, const VectorXd& knots, int degree) {
                 auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(cps);
                 if (!arr) throw py::type_error("control_points must be array-like");
                 VectorXd flat;
                 if (arr.ndim() == 2 && arr.shape(0) == 3 && arr.shape(1) != 3) {
                     // (3, N): transpose into the flat xyz layout
                     const py::ssize_t n = arr.shape(1);
                     flat.resize(3 * n);
                     auto r = arr.unchecked<2>();
                     for (py::ssize_t i = 0; i < n; ++i)
                         for (int k = 0; k < 3; ++k) flat(3 * i + k) = r(k, i);
                 } else {
                     const auto pts = to_points(cps);
                     flat.resize(3 * static_cast<Eigen::Index>(pts.size()));
                     for (std::size_t i = 0; i < pts.size(); ++i) flat.segment<3>(3 * i) = pts[i];
                 }
                 return bspline::TrajectorySampler(flat, knots, degree);
             }),
             py::arg("control_points"), py::arg("knots"), py::arg("degree"))
        .def_property_readonly("valid", &bspline::TrajectorySampler::valid)
        .def_property_readonly("start_time", &bspline::TrajectorySampler::start_time)
        .def_property_readonly("end_time", &bspline::TrajectorySampler::end_time)
        .def_property_readonly("duration", &bspline::TrajectorySampler::duration)
        .def("position", &bspline::TrajectorySampler::position, py::arg("t"))
        .def("velocity", &bspline::TrajectorySampler::velocity, py::arg("t"))
        .def("acceleration", &bspline::TrajectorySampler::acceleration, py::arg("t"))
        .def("sample",
             [](const bspline::TrajectorySampler& s, const VectorXd& ts) {
                 MatrixXd out(ts.size(), 9);
                 for (Eigen::Index i = 0; i < ts.size(); ++i) {
                     out.row(i).segment<3>(0) = s.position(ts(i));
                     out.row(i).segment<3>(3) = s.velocity(ts(i));
                     out.row(i).segment<3>(6) = s.acceleration(ts(i));
                 }
                 return out;
             },
             py::arg("times"), "Return an (len(times), 9) array of [pos | vel | acc] rows.");

    // ==========================================================================================
    // 3. Map
    // ==========================================================================================

    py::class_<mapping::ObstacleBox>(m, "ObstacleBox", "Axis-aligned obstacle box [min, max] in world metres.")
        .def(py::init([](const Vector3d& mn, const Vector3d& mx) {
                 if ((mx.array() < mn.array()).any()) throw py::value_error("ObstacleBox: max must be >= min");
                 return mapping::ObstacleBox{mn, mx};
             }),
             py::arg("min"), py::arg("max"))
        .def_property("min", [](const mapping::ObstacleBox& b) { return Vector3d(b.min); },
                      [](mapping::ObstacleBox& b, const Vector3d& v) { b.min = v; })
        .def_property("max", [](const mapping::ObstacleBox& b) { return Vector3d(b.max); },
                      [](mapping::ObstacleBox& b, const Vector3d& v) { b.max = v; })
        // Names used by the original bindings.
        .def_property("min_pt", [](const mapping::ObstacleBox& b) { return Vector3d(b.min); },
                      [](mapping::ObstacleBox& b, const Vector3d& v) { b.min = v; })
        .def_property("max_pt", [](const mapping::ObstacleBox& b) { return Vector3d(b.max); },
                      [](mapping::ObstacleBox& b, const Vector3d& v) { b.max = v; })
        .def("__repr__", [](const mapping::ObstacleBox& b) {
            std::ostringstream os;
            os << "ObstacleBox(min=[" << b.min.x() << ", " << b.min.y() << ", " << b.min.z() << "], max=["
               << b.max.x() << ", " << b.max.y() << ", " << b.max.z() << "])";
            return os.str();
        });

    // shared_ptr holder: TrajectoryPlanner takes shared ownership of the grid, so Python and
    // the planner can both hold it, and the grid outlives whichever drops it first.
    py::class_<mapping::SparseVoxelGrid, std::shared_ptr<mapping::SparseVoxelGrid>>(m, "SparseVoxelGrid", R"doc(
Sparse three-state (FREE / OCCUPIED / UNKNOWN) voxel map consumed by TrajectoryPlanner.

update_from_obstacles() marks boxes OCCUPIED and leaves everything else UNKNOWN (a box list
carries no evidence of free space). update_from_scan() raycasts sensor returns, marking cells
a beam passes through FREE and the cell it ends in OCCUPIED; with accumulation on (default)
observations persist across scans. A* treats UNKNOWN as traversable at a cost.
)doc")
        .def(py::init<double, double>(), py::arg("resolution") = 1.0, py::arg("inflation_radius") = 0.0)
        .def("update_from_obstacles", &mapping::SparseVoxelGrid::update_from_obstacles, py::arg("obstacles"))
        .def("update_from_scan",
             [](mapping::SparseVoxelGrid& g, const Vector3d& origin, const py::object& hits) {
                 g.update_from_scan(origin, to_points(hits));
             },
             py::arg("sensor_origin"), py::arg("hits"),
             "Raycast an (N, 3) array (or list) of beam return points from sensor_origin.")
        .def("set_accumulate", &mapping::SparseVoxelGrid::set_accumulate, py::arg("on"),
             py::arg("window_radius_m") = 25.0)
        .def_property_readonly("accumulating", &mapping::SparseVoxelGrid::accumulating)
        .def("set_frontier_blocks_corridors", &mapping::SparseVoxelGrid::set_frontier_blocks_corridors,
             py::arg("on"))
        .def_property_readonly("frontier_blocks_corridors", &mapping::SparseVoxelGrid::frontier_blocks_corridors)
        .def("set_occupancy_thickness", &mapping::SparseVoxelGrid::set_occupancy_thickness, py::arg("metres"))
        .def_property_readonly("occupancy_thickness", &mapping::SparseVoxelGrid::occupancy_thickness)
        .def("set_heat_radius", &mapping::SparseVoxelGrid::set_heat_radius, py::arg("radius_m"))
        .def_property_readonly("resolution", &mapping::SparseVoxelGrid::get_resolution)
        .def_property_readonly("inflation_radius", &mapping::SparseVoxelGrid::get_inflation_radius)
        .def_property_readonly("num_free", &mapping::SparseVoxelGrid::num_free)
        .def_property_readonly("num_occupied", &mapping::SparseVoxelGrid::num_raw)
        .def_property_readonly("num_inflated", &mapping::SparseVoxelGrid::num_inflated)
        .def("world_to_index", &world_to_index, py::arg("point"))
        // State queries take a WORLD point (metres); the grid's indices are an internal detail.
        .def("is_free", [](const mapping::SparseVoxelGrid& g, const Vector3d& p) {
            return g.is_free(world_to_index(g, p));
        }, py::arg("point"))
        .def("is_unknown", [](const mapping::SparseVoxelGrid& g, const Vector3d& p) {
            return g.is_unknown(world_to_index(g, p));
        }, py::arg("point"))
        .def("is_occupied", [](const mapping::SparseVoxelGrid& g, const Vector3d& p) {
            return g.is_occupied_raw(world_to_index(g, p));
        }, py::arg("point"), "True if the cell holding this point is OCCUPIED (uninflated).")
        .def("is_occupied_inflated", [](const mapping::SparseVoxelGrid& g, const Vector3d& p) {
            return g.is_occupied_inflated(world_to_index(g, p));
        }, py::arg("point"))
        .def("is_occupied_radius", &mapping::SparseVoxelGrid::is_occupied_radius, py::arg("point"),
             py::arg("radius"))
        .def("nearest_obstacle_distance", &mapping::SparseVoxelGrid::nearest_obstacle_distance, py::arg("point"))
        .def("occupied_indices", [](const mapping::SparseVoxelGrid& g) {
            return indices_to_array(g.get_occupied_raw());
        }, "(N, 3) int array of OCCUPIED cell indices (multiply by resolution for metres).")
        .def("inflated_indices", [](const mapping::SparseVoxelGrid& g) {
            return indices_to_array(g.get_occupied_inflated());
        })
        .def("free_indices", [](const mapping::SparseVoxelGrid& g) { return indices_to_array(g.get_free()); });

    // ==========================================================================================
    // 4. Front end configuration
    // ==========================================================================================

    py::class_<FrontEndConfig> cfg(m, "FrontEndConfig", R"doc(
Front-end / planner configuration. Construct with defaults, then set attributes, or pass any
fields as keyword arguments: FrontEndConfig(v_max=4.0, planning_horizon=15.0).
)doc");
    cfg.def(py::init([](const py::kwargs& kw) {
        // Route each kwarg through the Python attribute setter so unknown names raise and types
        // are checked exactly as for `cfg.x = ...`.
        py::object self = py::cast(FrontEndConfig{});
        for (auto item : kw) {
            const auto name = py::str(item.first).cast<std::string>();
            if (!py::hasattr(self, name.c_str())) {
                throw py::attribute_error("FrontEndConfig has no field '" + name + "'");
            }
            py::setattr(self, item.first, item.second);
        }
        return self.cast<FrontEndConfig>();
    }));

#define CFG_FIELD(name) cfg.def_readwrite(#name, &FrontEndConfig::name)
    CFG_FIELD(voxel_resolution);
    CFG_FIELD(drone_physical_radius);
    CFG_FIELD(sfc_height);
    CFG_FIELD(sfc_width);
    CFG_FIELD(sfc_start_ext);
    CFG_FIELD(sfc_end_ext);
    CFG_FIELD(spline_type);
    CFG_FIELD(aircraft_type);
    CFG_FIELD(degree);
    CFG_FIELD(sensor_range);
    CFG_FIELD(fov_vertical_min);
    CFG_FIELD(fov_vertical_max);
    CFG_FIELD(min_altitude);
    CFG_FIELD(max_altitude);
    CFG_FIELD(max_sfc_count);
    CFG_FIELD(geofence_contains_vehicle);
    CFG_FIELD(min_chain_length);
    CFG_FIELD(max_sfc_count_hard);
    CFG_FIELD(v_max);
    CFG_FIELD(a_max);
    CFG_FIELD(unknown_traversal_cost);
    CFG_FIELD(astar_clearance_cost);
    CFG_FIELD(heat_radius);
    CFG_FIELD(heat_weight);
    CFG_FIELD(astar_heuristic_weight);
    CFG_FIELD(astar_soft_obstacle_cost);
    CFG_FIELD(astar_min_segment_voxels);
    CFG_FIELD(astar_collinear_merge);
    CFG_FIELD(sfc_clamp_relief);
    CFG_FIELD(sfc_min_usable_width);
    CFG_FIELD(sfc_max_repairs);
    CFG_FIELD(path_hysteresis_bonus);
    CFG_FIELD(path_hysteresis_radius_cells);
    CFG_FIELD(planning_horizon);
    CFG_FIELD(knot_scale);
    CFG_FIELD(astar_max_nodes);
    CFG_FIELD(astar_max_time_ms);
    CFG_FIELD(time_budget_factor);
    CFG_FIELD(terminal_speed_fraction);
    CFG_FIELD(max_solve_time_ms);
    CFG_FIELD(qp_accept_violation);
    CFG_FIELD(enforce_norm_limits);
#undef CFG_FIELD
    // Eigen member: copy in/out (a def_readwrite view would make `cfg.map_bounds[0] = x` act
    // on a temporary).
    cfg.def_property("map_bounds", [](const FrontEndConfig& c) { return Vector3d(c.map_bounds); },
                     [](FrontEndConfig& c, const Vector3d& v) { c.map_bounds = v; },
                     "Map extents (x, y, z) in metres.");
    cfg.def("__repr__", [](const FrontEndConfig& c) {
        std::ostringstream os;
        os << "FrontEndConfig(spline_type='" << c.spline_type << "', aircraft_type='" << c.aircraft_type
           << "', degree=" << c.degree << ", voxel_resolution=" << c.voxel_resolution
           << ", v_max=" << c.v_max << ", a_max=" << c.a_max << ", planning_horizon=" << c.planning_horizon
           << ", ...)";
        return os.str();
    });

    // ==========================================================================================
    // 5. Planner
    // ==========================================================================================

    py::class_<sfc::SFCResult>(m, "Corridor", "One Safe Flight Corridor (oriented box, A x <= b).")
        .def_property_readonly("primary_position", [](const sfc::SFCResult& r) { return Vector3d(r.primaryPosition); })
        .def_property_readonly("secondary_position", [](const sfc::SFCResult& r) { return Vector3d(r.secondaryPosition); })
        .def_property_readonly("A", [](const sfc::SFCResult& r) { return MatrixXd(r.A_mat); })
        .def_property_readonly("b", [](const sfc::SFCResult& r) { return VectorXd(r.b_vec); })
        .def_property_readonly("bounds", [](const sfc::SFCResult& r) { return VectorXd(r.bounds); },
                               "[+x_fwd, -x_back, +y_left, -y_right, +z_up, -z_down] in the local frame.")
        .def_property_readonly("axes", [](const sfc::SFCResult& r) {
            Eigen::Matrix3d R;
            R << r.ux, r.uy, r.uz;
            return R;
        }, "3x3 matrix whose columns are the local frame unit vectors ux, uy, uz.")
        .def_property_readonly("vertices", [](const sfc::SFCResult& r) {
            Eigen::Matrix<double, 8, 3, Eigen::RowMajor> v = r.getAllVertices_3D().transpose();
            return v;
        }, "(8, 3) corners of the oriented box: indices 0-3 are the bottom face (local -z) in "
           "order, 4-7 the top face; vertex i+4 sits above vertex i.")
        .def("contains", [](const sfc::SFCResult& r, const Vector3d& p, double tol) {
            return ((r.A_mat * p - r.b_vec).array() <= tol).all();
        }, py::arg("point"), py::arg("tol") = 1e-9);

    py::class_<PlanningResult>(m, "PlanningResult")
        .def_property_readonly("control_points", [](const PlanningResult& r) { return VectorXd(r.control_points); },
                               "Flat [x0, y0, z0, x1, y1, z1, ...] control points.")
        .def_property_readonly("control_points_xyz", [](const PlanningResult& r) {
            const Eigen::Index n = r.control_points.size() / 3;
            Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> out(n, 3);
            for (Eigen::Index i = 0; i < n; ++i) out.row(i) = r.control_points.segment<3>(3 * i).transpose();
            return out;
        }, "Control points reshaped to (N, 3).")
        .def_property_readonly("knots", [](const PlanningResult& r) { return VectorXd(r.knots); })
        .def_readonly("corridors", &PlanningResult::corridors)
        .def_readonly("degree", &PlanningResult::degree)
        .def_readonly("total_control_points", &PlanningResult::total_control_points)
        .def_readonly("success", &PlanningResult::success)
        .def_readonly("reached_goal", &PlanningResult::reached_goal)
        .def_readonly("sfc_time_ms", &PlanningResult::sfc_time_ms)
        .def_readonly("opt_time_ms", &PlanningResult::opt_time_ms)
        .def_readonly("overhead_ms", &PlanningResult::overhead_ms)
        .def_readonly("total_time_ms", &PlanningResult::total_time_ms)
        .def_readonly("matrix_time_ms", &PlanningResult::matrix_time_ms)
        .def_readonly("solve_attempts", &PlanningResult::solve_attempts)
        .def_readonly("d_free_ahead", &PlanningResult::d_free_ahead)
        .def_readonly("terminal_speed", &PlanningResult::terminal_speed)
        .def("sampler", [](const PlanningResult& r) {
            return bspline::TrajectorySampler(r.control_points, r.knots, r.degree);
        }, "TrajectorySampler over this result's spline.")
        .def("__repr__", [](const PlanningResult& r) {
            std::ostringstream os;
            os << "PlanningResult(success=" << (r.success ? "True" : "False")
               << ", reached_goal=" << (r.reached_goal ? "True" : "False")
               << ", control_points=" << r.total_control_points << ", corridors=" << r.corridors.size()
               << ", total_time_ms=" << r.total_time_ms << ")";
            return os.str();
        });

    py::class_<TrajectoryPlanner>(m, "TrajectoryPlanner", R"doc(
SFC-constrained minimum-snap planner: A* over the voxel grid -> Safe Flight Corridors ->
OSQP minimum-snap solve. The planner shares ownership of the grid, so the grid can keep being
updated from Python between plans. Construction is cheap and the planner keeps no state across
plan_mission calls, so rebuild it per replan as the ROS node does.
)doc")
        .def(py::init<const FrontEndConfig&, std::shared_ptr<mapping::SparseVoxelGrid>, double, double>(),
             py::arg("config"), py::arg("grid"), py::arg("v_max") = 3.0, py::arg("a_max") = 2.0)
        .def("plan_mission", &TrajectoryPlanner::plan_mission,
             py::arg("start_pos"), py::arg("end_pos"),
             py::arg("start_vel") = Vector3d::Zero().eval(), py::arg("start_acc") = Vector3d::Zero().eval(),
             "Plan from start_pos (with start_vel / start_acc) toward end_pos.")
        .def("set_previous_path", &TrajectoryPlanner::set_previous_path, py::arg("world_points"),
             "Previously committed route (list of 3-vectors) for path hysteresis.");
}
