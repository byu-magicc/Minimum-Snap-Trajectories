#include "voxel_grid.hpp"
#include <cstdlib>
#include <cmath>
#include <limits>
#include <iostream>

namespace trajectory_planner {
namespace mapping {

SparseVoxelGrid::SparseVoxelGrid(double resolution, double inflation_radius)
    : voxel_resolution_(resolution), inflation_radius_(inflation_radius),
      kd_adaptor_(nullptr), kdtree_(nullptr) {}

void SparseVoxelGrid::update_from_scan(const Eigen::Vector3d& sensor_origin,
                                       const std::vector<Eigen::Vector3d>& hits) {
    last_sensor_origin_ = sensor_origin;
    changed_cells_.clear();
    if (!accumulate_) {
        free_voxels_.clear();
        occupied_voxels_raw_.clear();
        occupied_dirty_ = true;
        frontier_valid_ = false;
    }

    // 1. Sweep the beams. Each traversed cell is measured-empty, so it becomes FREE and any
    //    occupancy recorded there previously is cleared -- that is what keeps an
    //    accumulated map from filling up with ghosts of things that have moved.
    //    Sweeping past map_window_radius_ is pure waste: prune_to_window() deletes everything
    //    out there within ten scans, so those cells are walked, hashed, inserted and then
    //    dropped without ever being read. It stays invisible while beams are sparse and becomes
    //    the dominant cost when they are not -- the walk is per BEAM, so it scales with angular
    //    resolution, and it runs in the planner's process on the same single thread as the 50 Hz
    //    dispatch timer. Measured going from 3.0 to 1.5 deg rings (2360 -> 9440 beams/scan) with
    //    a 70 m sensor range against a 25 m window: 1.89 -> 7.55 M cell-ops/s, and dispatch
    //    collapsed from ~45 Hz to 7.4 Hz while the planner itself still finished in 9 ms.
    //    Clamping the sweep to the window cuts that to 3.30 M/s and discards nothing that is kept.
    const double window_sq = map_window_radius_ * map_window_radius_;
    for (const auto& hit : hits) {
        const Eigen::Vector3d ray = hit - sensor_origin;
        const double len_sq = ray.squaredNorm();
        if (len_sq > window_sq && len_sq > 0.0) {
            raycast_mark_free(sensor_origin,
                              sensor_origin + ray * (map_window_radius_ / std::sqrt(len_sq)));
        } else {
            raycast_mark_free(sensor_origin, hit);
        }
    }

    // 2. Returns are OCCUPIED, and win over freedom: a beam grazing the corner of a cell
    //    where another beam terminated must not clear it. Returns beyond the window are skipped
    //    for the same reason the sweep is clamped -- they would be pruned before being read.
    //
    // The depth fill (occupancy_thickness_) runs from HERE, and only for returns that are NEW.
    // That gate is what makes it affordable: the shadow behind a return does not change between
    // scans, so re-walking it for every already-known hit is pure repetition. On a 6480-return
    // scan the unconditional version cost +31% on update_from_scan -- the component that was
    // already starving the 50 Hz dispatch loop -- while gating it on new occupancy makes the
    // cost proportional to how much of the map is actually new, which in an accumulating map
    // decays to near nothing after the first few scans. Pruning removes the hit along with its
    // shadow, so a cell re-entering the window is re-inserted as new and refills correctly.
    const double fill_step = 0.5 * voxel_resolution_;
    const int fill_steps = occupancy_thickness_ > 0.0
        ? static_cast<int>(std::ceil(occupancy_thickness_ / fill_step)) : 0;

    for (const auto& hit : hits) {
        if ((hit - sensor_origin).squaredNorm() > window_sq) continue;
        const Eigen::Vector3i idx = to_index(hit);
        if (free_voxels_.erase(idx)) changed_cells_.push_back(idx);
        const bool new_hit = occupied_voxels_raw_.insert(idx).second;
        if (new_hit) {
            changed_cells_.push_back(idx);
            occupied_dirty_ = true;
        }
        // NOT gated on new_hit. Gating there was tried and reverted: a later vantage sees the
        // same surface at a different angle, so its shadow covers DIFFERENT cells, and skipping
        // already-known returns left interior cells permanently open (the trunk test caught it).

        // GIVE THE RETURN DEPTH.
        //
        // A beam deposits occupancy in exactly the cell it ends in and nothing past it, so the
        // far side of the object it just hit is sensor shadow: never observed, hence UNKNOWN,
        // and UNKNOWN is not blocked. Inflation covers most of a trunk from a densely-sampled
        // near surface -- measured 18 of 20 interior cells on a 1.3 m trunk -- but not the
        // deepest cells, and a corridor built in that shadow is clipped by nothing and may
        // reach ~0.5 m into solid wood against a 0.49 m hull.
        //
        // STOP AT MEASURED-FREE. If a beam has been through a cell it is empty, and inference
        // from a neighbouring return must never overrule a measurement -- that rule is what
        // keeps a swept gap behind a thin obstacle open instead of walling off a real route.
        // free_voxels_ already holds this scan's sweep, because step 1 ran first.
        if (fill_steps == 0) continue;
        const Eigen::Vector3d ray = hit - sensor_origin;
        const double len = ray.norm();
        if (len < 1e-9) continue;
        const Eigen::Vector3d dir = ray / len;
        Eigen::Vector3i last_idx = idx;
        for (int k = 1; k <= fill_steps; ++k) {
            const Eigen::Vector3d p = hit + dir * (k * fill_step);
            if ((p - sensor_origin).squaredNorm() > window_sq) break;
            const Eigen::Vector3i c = to_index(p);
            if (c == last_idx) continue;      // half-res sampling revisits cells; skip cheaply
            last_idx = c;
            if (free_voxels_.find(c) != free_voxels_.end()) break;
            if (occupied_voxels_raw_.insert(c).second) {
                changed_cells_.push_back(c);
                occupied_dirty_ = true;
            }
        }
    }

    // 3. Bound memory and query cost. Pruning walks the whole map, so amortise it rather
    //    than paying it inside every 10 Hz callback.
    if (accumulate_ && ++scans_since_prune_ >= kScansPerPrune) {
        prune_to_window(sensor_origin);
        scans_since_prune_ = 0;
        frontier_valid_ = false;   // wholesale removal, incremental update cannot cover it
    }

    // 4. Derived structures (inflated set / continuous bounds / KD-tree) follow occupancy.
    // 4. Derived structures (inflated set / continuous bounds / KD-tree) are a pure function of
    //    occupied_voxels_raw_, so they only need rebuilding when THAT changed. Rebuilding
    //    unconditionally was the dominant per-scan cost once the map got dense: the rebuild
    //    stamps a 5x5x5 inflation box per occupied voxel and then builds a KD-tree over the
    //    result, so at ~3400 occupied cells in the window it was ~425k hash inserts plus a
    //    425k-point tree, every scan, in the planner's process on the 50 Hz dispatch thread.
    //    It never showed up in the shakeout world (a handful of pillars) and became the wall in
    //    a 163-trunk forest.
    //
    //    Free-space changes are frequent, but OCCUPANCY changes are not: in a static scene most
    //    scans re-hit cells already known solid, and insert() reports no change. Skipping the
    //    rebuild then is exact, not an approximation -- there is nothing new to derive.
    if (occupied_dirty_) {
        rebuild_derived_from_occupied();
        occupied_dirty_ = false;
    }

    // 5. Frontier is computed lazily on demand -- see get_unknown_frontier().
    frontier_dirty_ = true;
}

void SparseVoxelGrid::prune_to_window(const Eigen::Vector3d& sensor_origin) {
    const double r2 = map_window_radius_ * map_window_radius_;
    auto out_of_window = [&](const Eigen::Vector3i& idx) {
        return (to_center(idx) - sensor_origin).squaredNorm() > r2;
    };
    for (auto it = free_voxels_.begin(); it != free_voxels_.end();) {
        it = out_of_window(*it) ? free_voxels_.erase(it) : std::next(it);
    }
    for (auto it = occupied_voxels_raw_.begin(); it != occupied_voxels_raw_.end();) {
        if (out_of_window(*it)) { it = occupied_voxels_raw_.erase(it); occupied_dirty_ = true; }
        else { ++it; }
    }
}

void SparseVoxelGrid::rebuild_derived_from_occupied() {
    occupied_voxels_inflated_.clear();
    continuous_inflated_bounds_.clear();
    continuous_inflated_bounds_.reserve(occupied_voxels_raw_.size());

    const double half = voxel_resolution_ / 2.0;
    for (const auto& idx : occupied_voxels_raw_) {
        const Eigen::Vector3d c = to_center(idx);
        const Eigen::Vector3d lo = c - Eigen::Vector3d::Constant(half + inflation_radius_);
        const Eigen::Vector3d hi = c + Eigen::Vector3d::Constant(half + inflation_radius_);
        continuous_inflated_bounds_.emplace_back(lo, hi);

        const Eigen::Vector3i a = to_index(lo);
        const Eigen::Vector3i b = to_index(hi);
        for (int x = a.x(); x <= b.x(); ++x)
            for (int y = a.y(); y <= b.y(); ++y)
                for (int z = a.z(); z <= b.z(); ++z)
                    occupied_voxels_inflated_.insert(Eigen::Vector3i(x, y, z));
    }

    // Proximity field, stamped in the SAME pass so it costs one extra shell rather than a
    // second sweep of the map. Only the cells between the inflated boundary and heat_radius_
    // beyond it get an entry: anything closer is untraversable, anything further is free.
    //
    // Combined with max() rather than summed. Heat is "how close is the nearest obstacle",
    // which is a property of the cell, not a tally of its neighbours -- adding contributions
    // would make a cell between two distant trunks look worse than one grazing a single trunk,
    // which is the opposite of what a clearance cost should say.
    heat_.clear();
    if (heat_radius_ > 0.0) {
        const double halo = inflation_radius_ + heat_radius_;
        for (const auto& idx : occupied_voxels_raw_) {
            const Eigen::Vector3d c = to_center(idx);
            const Eigen::Vector3i a = to_index(c - Eigen::Vector3d::Constant(half + halo));
            const Eigen::Vector3i b = to_index(c + Eigen::Vector3d::Constant(half + halo));
            for (int x = a.x(); x <= b.x(); ++x) {
                for (int y = a.y(); y <= b.y(); ++y) {
                    for (int z = a.z(); z <= b.z(); ++z) {
                        const Eigen::Vector3i cell(x, y, z);
                        // Inside the inflated set the cell is untraversable; no heat needed.
                        if (occupied_voxels_inflated_.count(cell)) { continue; }
                        // Measure from the INFLATED BOUNDARY, not the obstacle centre. The
                        // inflated set is stamped as a box reaching (half + inflation_radius_)
                        // from the centre, so subtracting inflation_radius_ alone leaves every
                        // non-inflated cell already at d >= heat_radius_ and the field comes out
                        // silently EMPTY -- the two measures have to share an origin.
                        const double d = (to_center(cell) - c).norm()
                                       - (half + inflation_radius_);
                        if (d >= heat_radius_) { continue; }
                        // Linear falloff, clamped to [0, 1]. Squared falloff was considered but
                        // linear keeps the gradient non-zero right out to the rim, which is where
                        // the search most needs a reason to pick the wider gap.
                        const float h = static_cast<float>(
                            std::max(0.0, std::min(1.0, 1.0 - d / heat_radius_)));
                        auto it = heat_.find(cell);
                        if (it == heat_.end()) { heat_.emplace(cell, h); }
                        else if (h > it->second) { it->second = h; }
                    }
                }
            }
        }
    }

    rebuild_kdtree();
}

void SparseVoxelGrid::raycast_mark_free(const Eigen::Vector3d& origin,
                                        const Eigen::Vector3d& endpoint) {
    Eigen::Vector3d d = endpoint - origin;
    const double len = d.norm();
    if (len < 1e-9) return;
    d /= len;

    Eigen::Vector3i cur = to_index(origin);
    const Eigen::Vector3i end = to_index(endpoint);
    if (cur == end) return;

    // Amanatides & Woo: tMax[i] is the ray parameter (in metres, since d is normalised) at
    // which the ray crosses the next voxel boundary on axis i; tDelta[i] is the spacing
    // between successive crossings on that axis.
    int step[3];
    double tMax[3], tDelta[3];
    const double kInf = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 3; ++i) {
        if (d[i] > 0.0) {
            step[i] = 1;
            tMax[i] = ((cur[i] + 1) * voxel_resolution_ - origin[i]) / d[i];
            tDelta[i] = voxel_resolution_ / d[i];
        } else if (d[i] < 0.0) {
            step[i] = -1;
            tMax[i] = (cur[i] * voxel_resolution_ - origin[i]) / d[i];
            tDelta[i] = -voxel_resolution_ / d[i];
        } else {
            step[i] = 0;
            tMax[i] = kInf;
            tDelta[i] = kInf;
        }
    }

    // Bound the walk: a malformed ray must never spin here inside a 10 Hz callback.
    const int max_steps = static_cast<int>(len / voxel_resolution_) + 4;
    for (int n = 0; n < max_steps && cur != end; ++n) {
        // Only a cell that was NOT already free represents new information. With an
        // accumulated map the overwhelming majority of swept cells are re-confirmations, so
        // tracking just the genuine transitions is what makes the incremental frontier
        // update cheap -- a few hundred cells per scan instead of the whole free set.
        if (free_voxels_.insert(cur).second) {
            changed_cells_.push_back(cur);
        }
        if (occupied_voxels_raw_.erase(cur)) {   // measured through it -> it is not solid
            occupied_dirty_ = true;
            changed_cells_.push_back(cur);
        }

        int axis = (tMax[0] < tMax[1]) ? ((tMax[0] < tMax[2]) ? 0 : 2)
                                       : ((tMax[1] < tMax[2]) ? 1 : 2);
        if (tMax[axis] > len) break;   // stepped past the return point
        cur[axis] += step[axis];
        tMax[axis] += tDelta[axis];
    }
}

void SparseVoxelGrid::compute_unknown_frontier() const {
    static const Eigen::Vector3i kNeighbors[6] = {
        Eigen::Vector3i( 1, 0, 0), Eigen::Vector3i(-1, 0, 0),
        Eigen::Vector3i( 0, 1, 0), Eigen::Vector3i( 0,-1, 0),
        Eigen::Vector3i( 0, 0, 1), Eigen::Vector3i( 0, 0,-1)
    };

    // A cell is on the frontier if it is UNKNOWN, touches measured-free space, and is not a
    // "pinhole". Pinholes exist because the beam pattern is angular: past ~9.5 m (3 deg
    // beams, 0.5 m voxels) neighbouring beams are more than a voxel apart, leaving unsampled
    // cells BETWEEN them. Those are not a boundary of knowledge -- the space really was swept
    // -- so a candidate ringed by free cells is rejected.
    constexpr int kMaxFreeNeighborsForFrontier = 2;
    const double fr2 = frontier_radius_ * frontier_radius_;
    const Eigen::Vector3d focus = has_frontier_focus_ ? frontier_focus_ : last_sensor_origin_;

    auto in_region = [&](const Eigen::Vector3i& idx) {
        return frontier_radius_ <= 0.0 || (to_center(idx) - focus).squaredNorm() <= fr2;
    };
    auto classify = [&](const Eigen::Vector3i& idx) {
        // Returns true if idx belongs on the frontier.
        if (free_voxels_.count(idx) || occupied_voxels_raw_.count(idx)) return false;
        if (!in_region(idx)) return false;
        int free_neighbors = 0;
        for (const auto& n : kNeighbors) if (free_voxels_.count(idx + n)) ++free_neighbors;
        return free_neighbors > 0 && free_neighbors <= kMaxFreeNeighborsForFrontier;
    };

    // --- Incremental path -------------------------------------------------------------
    // Only cells that actually changed state this scan, and their immediate neighbours, can
    // have changed frontier membership. With an accumulated map that is a few hundred cells
    // rather than the entire free set, which is the difference between ~0.5 ms and ~10 ms.
    if (frontier_valid_ && !changed_cells_.empty()) {
        for (const auto& c : changed_cells_) {
            frontier_set_.erase(c);
            if (classify(c)) frontier_set_.insert(c);
            for (const auto& n : kNeighbors) {
                const Eigen::Vector3i nb = c + n;
                frontier_set_.erase(nb);
                if (classify(nb)) frontier_set_.insert(nb);
            }
        }
    } else {
        // --- Full rebuild (first call, or after a prune removed cells wholesale) -------
        // Counts free neighbours DURING discovery -- each adjacent free cell contributes one
        // increment -- so a candidate is classified without ever re-probing its neighbourhood.
        // (Using classify() here instead costs ~8x more probes per free cell.)
        std::unordered_map<Eigen::Vector3i, int, Vector3iHash> counts;
        counts.reserve(free_voxels_.size() / 4);
        for (const auto& f : free_voxels_) {
            if (!in_region(f)) continue;
            for (const auto& n : kNeighbors) {
                const Eigen::Vector3i nb = f + n;
                if (free_voxels_.count(nb)) continue;
                if (occupied_voxels_raw_.count(nb)) continue;
                ++counts[nb];
            }
        }
        frontier_set_.clear();
        frontier_set_.reserve(counts.size());
        for (const auto& e : counts) {
            if (e.second > kMaxFreeNeighborsForFrontier) continue;   // pinhole
            frontier_set_.insert(e.first);
        }
        frontier_valid_ = true;
    }

    unknown_frontier_.clear();
    unknown_frontier_.reserve(frontier_set_.size());
    for (const auto& idx : frontier_set_) {
        if (!in_region(idx)) continue;   // focus may have moved since insertion
        unknown_frontier_.push_back(to_center(idx));
    }
}

double SparseVoxelGrid::heat_at(const Eigen::Vector3i& idx) const {
    if (heat_.empty()) { return 0.0; }
    auto it = heat_.find(idx);
    return it == heat_.end() ? 0.0 : static_cast<double>(it->second);
}

bool SparseVoxelGrid::is_free(const Eigen::Vector3i& idx) const {
    return free_voxels_.count(idx) > 0;
}

bool SparseVoxelGrid::is_unknown(const Eigen::Vector3i& idx) const {
    return free_voxels_.count(idx) == 0 && occupied_voxels_raw_.count(idx) == 0;
}

void SparseVoxelGrid::update_from_obstacles(const std::vector<ObstacleBox>& obstacles) {
    // Clear old data
    occupied_voxels_inflated_.clear();
    occupied_voxels_raw_.clear();
    continuous_inflated_bounds_.clear();
    inflated_centers_.clear();

    for (const auto& obstacle : obstacles) {
        // Inflated bounds
        Eigen::Vector3d inflated_min = obstacle.min.array() - inflation_radius_;
        Eigen::Vector3d inflated_max = obstacle.max.array() + inflation_radius_;

        // Save the continuous inflated bounds for the LoS smoother
        continuous_inflated_bounds_.emplace_back(inflated_min, inflated_max);

        // Convert to voxel indices (floor division)
        Eigen::Vector3i inf_min_idx(
            static_cast<int>(std::floor(inflated_min.x() / voxel_resolution_)),
            static_cast<int>(std::floor(inflated_min.y() / voxel_resolution_)),
            static_cast<int>(std::floor(inflated_min.z() / voxel_resolution_))
        );
        Eigen::Vector3i inf_max_idx(
            static_cast<int>(std::floor(inflated_max.x() / voxel_resolution_)),
            static_cast<int>(std::floor(inflated_max.y() / voxel_resolution_)),
            static_cast<int>(std::floor(inflated_max.z() / voxel_resolution_))
        );

        // Raw (non-inflated) bounds
        Eigen::Vector3i raw_min_idx(
            static_cast<int>(std::floor(obstacle.min.x() / voxel_resolution_)),
            static_cast<int>(std::floor(obstacle.min.y() / voxel_resolution_)),
            static_cast<int>(std::floor(obstacle.min.z() / voxel_resolution_))
        );
        Eigen::Vector3i raw_max_idx(
            static_cast<int>(std::floor(obstacle.max.x() / voxel_resolution_)),
            static_cast<int>(std::floor(obstacle.max.y() / voxel_resolution_)),
            static_cast<int>(std::floor(obstacle.max.z() / voxel_resolution_))
        );

        // Fill inflated voxels
        for (int x = inf_min_idx.x(); x <= inf_max_idx.x(); ++x) {
            for (int y = inf_min_idx.y(); y <= inf_max_idx.y(); ++y) {
                for (int z = inf_min_idx.z(); z <= inf_max_idx.z(); ++z) {
                    occupied_voxels_inflated_.insert(Eigen::Vector3i(x, y, z));
                }
            }
        }

        // Fill raw voxels
        for (int x = raw_min_idx.x(); x <= raw_max_idx.x(); ++x) {
            for (int y = raw_min_idx.y(); y <= raw_max_idx.y(); ++y) {
                for (int z = raw_min_idx.z(); z <= raw_max_idx.z(); ++z) {
                    occupied_voxels_raw_.insert(Eigen::Vector3i(x, y, z));
                }
            }
        }
    }

    // This path builds the derived structures itself, so nothing is left pending.
    occupied_dirty_ = false;
    rebuild_kdtree();
}

void SparseVoxelGrid::rebuild_kdtree() {
    inflated_centers_.clear();
    inflated_centers_.reserve(occupied_voxels_inflated_.size());
    for (const auto& idx : occupied_voxels_inflated_) {
        Eigen::Vector3d pt = idx.cast<double>() * voxel_resolution_
                             + Eigen::Vector3d::Constant(voxel_resolution_ / 2.0);
        inflated_centers_.push_back(pt);
    }

    if (!inflated_centers_.empty()) {
        kd_adaptor_ = std::make_unique<Vector3dAdaptor>(inflated_centers_);
        kdtree_ = std::make_unique<KDTreeType>(3, *kd_adaptor_, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
        kdtree_->buildIndex();
    } else {
        kdtree_.reset();
        kd_adaptor_.reset();
    }
}

bool SparseVoxelGrid::is_occupied_inflated(const Eigen::Vector3i& idx) const {
    return occupied_voxels_inflated_.count(idx) > 0;
}

bool SparseVoxelGrid::is_occupied_raw(const Eigen::Vector3i& idx) const {
    return occupied_voxels_raw_.count(idx) > 0;
}

double SparseVoxelGrid::nearest_obstacle_distance(const Eigen::Vector3d& pt) const {
    if (!kdtree_ || inflated_centers_.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    
    double query_pt[3] = {pt.x(), pt.y(), pt.z()};
    unsigned int out_index = 0;
    double out_dist_sq = 0.0;
    
    kdtree_->knnSearch(&query_pt[0], 1, &out_index, &out_dist_sq);
    return std::sqrt(out_dist_sq);
}

bool SparseVoxelGrid::is_occupied_radius(const Eigen::Vector3d& pt, double radius) const {
    if (!kdtree_ || inflated_centers_.empty()) {
        return false;
    }
    
    double query_pt[3] = {pt.x(), pt.y(), pt.z()};
    double search_radius = radius * radius; // nanoflann uses squared radius
    std::vector<nanoflann::ResultItem<uint32_t, double>> ret_matches;
    
    nanoflann::SearchParameters params;
    size_t nMatches = kdtree_->radiusSearch(&query_pt[0], search_radius, ret_matches, params);
    
    return nMatches > 0;
}

std::vector<Eigen::Vector3d> SparseVoxelGrid::get_obstacles_in_radius(const Eigen::Vector3d& pt, double radius) const {
    std::vector<Eigen::Vector3d> result;
    double search_radius_sq = radius * radius;
    
    // 1. Physical Obstacles
    if (kdtree_ && !inflated_centers_.empty()) {
        double query_pt[3] = {pt.x(), pt.y(), pt.z()};
        std::vector<nanoflann::ResultItem<uint32_t, double>> ret_matches;
        nanoflann::SearchParameters params;
        
        kdtree_->radiusSearch(&query_pt[0], search_radius_sq, ret_matches, params);
        
        result.reserve(ret_matches.size() + virtual_obstacles_.size());
        for (const auto& match : ret_matches) {
            result.push_back(inflated_centers_[match.first]);
        }
    } else {
        result.reserve(virtual_obstacles_.size());
    }

    // 2. Unknown-space frontier -- OPT-IN, see set_frontier_blocks_corridors().
    //
    // The idea was that a corridor may only claim space actually measured empty, so the
    // FREE/UNKNOWN boundary acts as a barrier exactly like a physical obstacle. In practice
    // that is far too aggressive and it is OFF by default. Two reasons.
    //
    // It is inconsistent with the rest of the pipeline. A* is free to route THROUGH unknown
    // space -- it only pays unknown_traversal_cost -- so the search commits to a path and the
    // corridor builder then walls that same path in. The symptom is specific: the corridor
    // around a waypoint the sensor has not swept yet collapses to ZERO width while every
    // corridor behind it comes out full size. Measured on a straight 6.4 m leg, bounds
    // (+y -y +z -z) were 2.010 2.010 0.000 0.000 with the frontier included and
    // 3.394 0.625 2.010 2.010 with it excluded, on a bit-identical path.
    //
    // And the frontier is not the clean surface it sounds like. Beam divergence means that at
    // range the swept-free region is a fan of thin lines with unsampled cells between them.
    // compute_unknown_frontier() rejects those "pinholes", but only by counting FREE FACE
    // neighbours -- and beam lines run diagonally, so a gap cell scores 1-2 by that metric and
    // survives. Measured 679 surviving frontier cells within 0.75 m of one corridor axis, of
    // which the modal cell had NINE free neighbours in its 26-neighbourhood: embedded in
    // measured-free space, not on any boundary of knowledge.
    //
    // Safety against unknown space is handled where it belongs, and by mechanisms that do not
    // depend on this one: A* pays unknown_traversal_cost to enter it, replan_loop re-checks the
    // committed setpoints against the freshest map at 10 Hz, and the terminal-velocity logic in
    // trajectory_planner.cpp sizes end-of-plan speed so the vehicle can brake inside the space
    // it can actually see. For reference, SANDO builds its decomposition obstacle set from
    // OCCUPIED voxels (getVecOccupied) and treats unknown-boundary inflation as a separately
    // gated option -- it does not wall corridors off with unknown space either.
    if (frontier_blocks_corridors_) {
        for (const auto& f : get_unknown_frontier()) {
            if ((f - pt).squaredNorm() <= search_radius_sq) {
                result.push_back(f);
            }
        }
    }

    // 3. Virtual Obstacles (manual injection point; unused by the default pipeline)
    for (const auto& v_obs : virtual_obstacles_) {
        if ((v_obs - pt).squaredNorm() <= search_radius_sq) {
            result.push_back(v_obs);
        }
    }

    return result;
}

} // namespace mapping
} // namespace trajectory_planner
