#include "dynamic_sfc.hpp"
#include <cmath>
#include <algorithm>

namespace trajectory_planner {
namespace sfc {

// ==========================================
// SpatialScanner Implementation
// ==========================================
SpatialScanner::SpatialScanner(std::shared_ptr<mapping::SparseVoxelGrid> grid)
    : grid_(std::move(grid)) {}

std::vector<Eigen::Vector3d> SpatialScanner::get_broad_phase_obstacles(
    const Eigen::Vector3d& pA, const Eigen::Vector3d& pB,
    double W, double ext_start, double ext_end) const {

    if (!grid_) return {};

    Eigen::Vector3d v = pB - pA;
    double dist = v.norm();
    Eigen::Vector3d u = (dist > 0) ? v / dist : Eigen::Vector3d(1.0, 0.0, 0.0);

    // Define the mathematical cylinder
    Eigen::Vector3d p_start = pA - u * ext_start;
    double total_len = dist + ext_start + ext_end;

    // Broad-phase: bounding sphere
    Eigen::Vector3d midpoint = p_start + u * (total_len / 2.0);
    double search_radius = std::sqrt((total_len / 2.0) * (total_len / 2.0) + W * W);

    // Use KD-Tree radius search instead of iterating over all points
    std::vector<Eigen::Vector3d> candidate_points = grid_->get_obstacles_in_radius(midpoint, search_radius);

    std::vector<Eigen::Vector3d> result;
    result.reserve(candidate_points.size()); // Reasonable pre-alloc

    for (const auto& pt : candidate_points) {

        // Narrow-phase: project onto cylinder axis
        Eigen::Vector3d vec_to_pt = pt - p_start;
        double t = vec_to_pt.dot(u);

        // Length filter
        if (t < 0.0 || t > total_len) continue;

        // Perpendicular distance filter
        Eigen::Vector3d proj_point = p_start + t * u;
        double perp_dist = (pt - proj_point).norm();
        if (perp_dist <= W) {
            result.push_back(pt);
        }
    }

    return result;
}

// ==========================================
// AsymmetricBoxBuilder Implementation
// ==========================================
AsymmetricBoxBuilder::AsymmetricBoxBuilder(double drone_physical_radius, double max_cp_drift,
                                           double voxel_resolution, double obstacle_inflation,
                                           bool clamp_relief)
    : drone_radius_(drone_physical_radius), voxel_resolution_(voxel_resolution),
      max_drift_(max_cp_drift), obstacle_inflation_(obstacle_inflation),
      clamp_relief_(clamp_relief) {}

std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d, Eigen::Matrix<double, 6, 1>>
AsymmetricBoxBuilder::build_bounds(
    const Eigen::Vector3d& pA, const Eigen::Vector3d& pB,
    const std::vector<Eigen::Vector3d>& obs_points,
    double ext_start, double ext_end) const {

    double dist = (pB - pA).norm();
    Eigen::Vector3d ux = (dist > 0) ? (pB - pA) / dist : Eigen::Vector3d(1.0, 0.0, 0.0);

    Eigen::Vector3d global_up(0.0, 0.0, 1.0);
    if (std::abs(ux.dot(global_up)) > 0.99) global_up = Eigen::Vector3d(1.0, 0.0, 0.0);

    Eigen::Vector3d uy = global_up.cross(ux);
    uy.normalize();
    Eigen::Vector3d uz = ux.cross(uy);

    // How much slack to give back to every measured clearance.
    //
    // obs_points arrive already inflated by obstacle_inflation_ (the grid inflates by
    // drone_physical_radius + obstacle_margin). The corridor only has to stay drone_radius_ from
    // the RAW obstacle surface -- the margin is the vehicle's tracking budget, not extra corridor
    // wall. Subtracting the inflated distance without adding the excess back counts the same
    // clearance twice, and the corridor loses that much width everywhere.
    //
    // It is not a small effect. Measured in a forest with hull 0.490 and margin 0.20: a path
    // legitimately threading a gap produced "y +/- 0/4.019 ... r_y 0.353553" -- one lateral bound
    // at EXACTLY zero, a primal-infeasible QP that time-stretching cannot fix, and a permanent
    // stall in visibly open air. The clearance was there; it had just been spent twice.
    //
    // Written as (inflation - hull) so the same expression is correct whichever obstacle set is
    // supplied: with raw points (obstacle_inflation_ = 0) the relief goes negative and the box is
    // INSET by the hull instead, which is what SANDO does
    // (set_obs(raw) + set_inflate_distance(drone_radius)).
    // How much of the obstacles' inflation may be handed back.
    //
    // Points arrive inflated by obstacle_inflation_ = drone_radius_ + obstacle_margin. Handing
    // back the margin leaves a corridor that clears the raw obstacle by exactly one hull radius
    // -- geometrically "correct", and wrong in practice: obstacle_margin exists to absorb STATE
    // ESTIMATION error, and a corridor that gives it back guarantees nothing once the estimate
    // is off. Measured tracking bias on this airframe is 0.3-0.8 m against a 0.49 m hull, so the
    // margin is precisely what has to still be there at the corridor face.
    //
    // min(0, ...) therefore keeps whatever clearance the map already has, and only ever INSETS:
    //   pre-inflated points (inflation 0.69 > hull 0.49) -> 0.00, keep it all
    //   raw points          (inflation 0.00 < hull 0.49) -> -0.49, inset by the hull
    //
    // The earlier formula (inflation - drone_radius_, no clamp) was justified as undoing a
    // "double count" that made corridors collapse to zero width. That does not hold up: measured
    // over 171 corridors on the medium forest, clamping costs nothing -- solve success stays
    // 100% at every max_sfc_count, attempts and solve time are unchanged -- while corridors whose
    // interior lies inside an obstacle fall 11.7% -> 8.8% and median clearance rises 0.565 ->
    // 0.776 m. It also makes a whole bug class impossible: with no clearance handed back, a wrong
    // drone_radius_ can only make a corridor NARROWER, never unsafe. That matters, because
    // drone_radius_ was wired as 0.0 at the FrontEndSFC call site and the unclamped formula
    // turned that into corridors flush against tree trunks (45% of corridors contained interior
    // points inside a trunk, worst 1.48 m inside).
    // See AsymmetricBoxBuilder's ctor doc. Clamped, this only ever INSETS, so a wrong
    // drone_radius can narrow a corridor but never make one unsafe. Unclamped it hands the
    // margin back: wider boxes, bare-hull guarantee.
    const double raw_relief = obstacle_inflation_ - drone_radius_;
    const double relief = clamp_relief_ ? std::min(0.0, raw_relief) : raw_relief;

    double max_total_drift = 2.0 * max_drift_;
    Eigen::Matrix<double, 6, 1> bounds;
    bounds << dist + ext_end, ext_start,
              max_total_drift, max_total_drift,
              max_total_drift, max_total_drift;

    // Exact OBB-AABB Projection (SAT)
    // The obstacle points are already inflated by drone_radius in the VoxelGrid.
    // We only need the half_res projection to mathematically bound the cubic voxel corners.
    double half_res = voxel_resolution_ / 2.0;
    double r_x = half_res * (std::abs(ux.x()) + std::abs(ux.y()) + std::abs(ux.z()));
    double r_y = half_res * (std::abs(uy.x()) + std::abs(uy.y()) + std::abs(uy.z()));
    double r_z = half_res * (std::abs(uz.x()) + std::abs(uz.y()) + std::abs(uz.z()));

    // Project and sort by true segment distance
    struct ProjectedPoint {
        double seg_dist, proj_x, proj_y, proj_z;
    };
    std::vector<ProjectedPoint> projected;
    projected.reserve(obs_points.size());

    for (const auto& obs : obs_points) {
        Eigen::Vector3d v_obs = obs - pA;
        double px = v_obs.dot(ux);
        double py = v_obs.dot(uy);
        double pz = v_obs.dot(uz);

        if (px >= -(bounds(1) + r_x) && px <= (bounds(0) + r_x)) {
            double seg_dist;
            if (px < 0) {
                seg_dist = std::sqrt(px * px + py * py + pz * pz);
            } else if (px > dist) {
                seg_dist = std::sqrt((px - dist) * (px - dist) + py * py + pz * pz);
            } else {
                seg_dist = std::sqrt(py * py + pz * pz);
            }
            projected.push_back({seg_dist, px, py, pz});
        }
    }

    std::sort(projected.begin(), projected.end(),
              [](const ProjectedPoint& a, const ProjectedPoint& b) { return a.seg_dist < b.seg_dist; });

    // Iteratively shrink the 6 planes
    for (const auto& pt : projected) {
        double eps = 1e-4;
        double proj_x = pt.proj_x, proj_y = pt.proj_y, proj_z = pt.proj_z;

        // 6-Plane active slice check
        bool in_x = -(bounds(1) + r_x) + eps < proj_x && proj_x < (bounds(0) + r_x) - eps;
        bool in_y = -(bounds(3) + r_y) + eps < proj_y && proj_y < (bounds(2) + r_y) - eps;
        bool in_z = -(bounds(5) + r_z) + eps < proj_z && proj_z < (bounds(4) + r_z) - eps;

        if (!(in_x && in_y && in_z)) continue;

        if (proj_x >= 0.0 && proj_x <= dist) {
            // Inside main tunnel
            if (std::abs(proj_y) >= std::abs(proj_z)) {
                if (proj_y >= 0) bounds(2) = std::min(bounds(2), std::max(0.0, proj_y - r_y + relief));
                else             bounds(3) = std::min(bounds(3), std::max(0.0, std::abs(proj_y) - r_y + relief));
            } else {
                if (proj_z >= 0) bounds(4) = std::min(bounds(4), std::max(0.0, proj_z - r_z + relief));
                else             bounds(5) = std::min(bounds(5), std::max(0.0, std::abs(proj_z) - r_z + relief));
            }
        } else {
            // End-caps: 6-Plane equality
            double dist_x = (proj_x < 0) ? std::abs(proj_x) : std::abs(proj_x - dist);

            if (dist_x >= std::abs(proj_y) && dist_x >= std::abs(proj_z)) {
                if (proj_x > dist)  bounds(0) = std::min(bounds(0), std::max(dist + 0.0, proj_x - r_x + relief));
                else if (proj_x < 0) bounds(1) = std::min(bounds(1), std::max(0.0, std::abs(proj_x) - r_x + relief));
            } else if (std::abs(proj_y) >= std::abs(proj_z)) {
                if (proj_y >= 0) bounds(2) = std::min(bounds(2), std::max(0.0, proj_y - r_y + relief));
                else             bounds(3) = std::min(bounds(3), std::max(0.0, std::abs(proj_y) - r_y + relief));
            } else {
                if (proj_z >= 0) bounds(4) = std::min(bounds(4), std::max(0.0, proj_z - r_z + relief));
                else             bounds(5) = std::min(bounds(5), std::max(0.0, std::abs(proj_z) - r_z + relief));
            }
        }
    }
    // --- Enforce Asymmetric Drift Budget ---
    // Y-axis (Horizontal)
    if (bounds(2) + bounds(3) > max_total_drift) {
        if (bounds(2) >= max_drift_ && bounds(3) >= max_drift_) {
            bounds(2) = max_drift_;
            bounds(3) = max_drift_;
        } else if (bounds(2) > bounds(3)) {
            bounds(2) = std::min(bounds(2), max_total_drift - bounds(3));
        } else {
            bounds(3) = std::min(bounds(3), max_total_drift - bounds(2));
        }
    }

    // Z-axis (Vertical)
    if (bounds(4) + bounds(5) > max_total_drift) {
        if (bounds(4) >= max_drift_ && bounds(5) >= max_drift_) {
            bounds(4) = max_drift_;
            bounds(5) = max_drift_;
        } else if (bounds(4) > bounds(5)) {
            bounds(4) = std::min(bounds(4), max_total_drift - bounds(5));
        } else {
            bounds(5) = std::min(bounds(5), max_total_drift - bounds(4));
        }
    }

    return {ux, uy, uz, bounds};
}

// ==========================================
// OBBAdapters Implementation
// ==========================================
void OBBAdapters::get_osqp_matrices(
    const Eigen::Vector3d& pA,
    const Eigen::Vector3d& ux, const Eigen::Vector3d& uy, const Eigen::Vector3d& uz,
    const Eigen::Matrix<double, 6, 1>& bounds,
    Eigen::MatrixXd& A_out, Eigen::VectorXd& b_out) {

    // A = [ux; -ux; uy; -uy; uz; -uz]
    A_out.resize(6, 3);
    A_out.row(0) =  ux.transpose();
    A_out.row(1) = -ux.transpose();
    A_out.row(2) =  uy.transpose();
    A_out.row(3) = -uy.transpose();
    A_out.row(4) =  uz.transpose();
    A_out.row(5) = -uz.transpose();

    // b_base = A @ pA
    Eigen::VectorXd b_base = A_out * pA;

    // b = b_base + bounds
    b_out = b_base + bounds;
}

// ==========================================
// AsymmetricSFCManager Implementation
// ==========================================
AsymmetricSFCManager::AsymmetricSFCManager(std::shared_ptr<mapping::SparseVoxelGrid> grid,
                                             double drone_physical_radius,
                                             double max_cp_drift,
                                             double voxel_resolution,
                                             bool clamp_relief)
    : scanner_(grid),
      // The grid is the authority on how much its obstacle points were inflated by; asking it
      // keeps the two from drifting apart if obstacle_margin is retuned.
      builder_(drone_physical_radius, max_cp_drift, voxel_resolution,
               grid ? grid->get_inflation_radius() : 0.0, clamp_relief) {}

SFCResult AsymmetricSFCManager::generate_sfc(
    const Eigen::Vector3d& pA, const Eigen::Vector3d& pB,
    double W, double ext_start, double ext_end) const {

    // Layer 1: Fetch raw points.
    //
    // The scan cylinder must enclose every point the box could still contain, otherwise
    // an unscanned obstacle can end up INSIDE the finished corridor. The box starts at
    // max_lateral_extent() on each of the y/z faces, so its corners reach
    // sqrt(2) * max_lateral_extent() from the axis -- which exceeds W whenever the box is
    // asymmetric. Scan the larger of the two radii.
    double corner_reach = std::sqrt(2.0) * builder_.max_lateral_extent();
    double scan_radius = std::max(W, corner_reach);
    auto obs_points = scanner_.get_broad_phase_obstacles(pA, pB, scan_radius, ext_start, ext_end);

    // Layer 2: Calculate local frame and shrink bounds
    auto [ux, uy, uz, bounds] = builder_.build_bounds(pA, pB, obs_points, ext_start, ext_end);

    // Layer 3: Translate to OSQP matrices
    Eigen::MatrixXd A_mat;
    Eigen::VectorXd b_vec;
    OBBAdapters::get_osqp_matrices(pA, ux, uy, uz, bounds, A_mat, b_vec);

    SFCResult result;
    result.primaryPosition = pA;
    result.secondaryPosition = pB;
    result.A_mat = A_mat;
    result.b_vec = b_vec;
    result.bounds = bounds;
    result.ux = ux;
    result.uy = uy;
    result.uz = uz;

    return result;
}

} // namespace sfc
} // namespace trajectory_planner
