#include <cstdlib>
#include <cstdio>
#include "optimize.hpp"
#include <osqp.h>  // not <osqp/osqp.h>: the bundled (build-tree) OSQP has no osqp/ prefix
#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>

namespace trajectory_planner {

Eigen::VectorXd run_qp_solver(
    const Eigen::SparseMatrix<double>& P, 
    const Eigen::VectorXd& q, 
    const Eigen::SparseMatrix<double>& A, 
    const Eigen::VectorXd& l, 
    const Eigen::VectorXd& u,
    double* prim_res_out,
    QpFailureInfo* failure_out,
    double time_limit_s,
    double accept_violation)
{
    OSQPInt n = P.cols();
    OSQPInt m = A.rows();

    // ==========================================
    // 1. SAFE DATA CONVERSION (Eigen -> OSQP v1.0)
    // ==========================================
    // OSQP v1.0 uses OSQPInt (usually 64-bit). Eigen uses 32-bit ints by default. 
    // We copy the sparse index arrays to guarantee memory safety.
    
    std::vector<OSQPInt> P_i(P.innerIndexPtr(), P.innerIndexPtr() + P.nonZeros());
    std::vector<OSQPInt> P_p(P.outerIndexPtr(), P.outerIndexPtr() + P.cols() + 1);

    std::vector<OSQPInt> A_i(A.innerIndexPtr(), A.innerIndexPtr() + A.nonZeros());
    std::vector<OSQPInt> A_p(A.outerIndexPtr(), A.outerIndexPtr() + A.cols() + 1);

    OSQPCscMatrix* P_csc = OSQPCscMatrix_new(
        n, n, 
        (OSQPInt)P.nonZeros(), 
        (OSQPFloat*)P.valuePtr(), 
        P_i.data(), 
        P_p.data()
    );

    OSQPCscMatrix* A_csc = OSQPCscMatrix_new(
        m, n, 
        (OSQPInt)A.nonZeros(), 
        (OSQPFloat*)A.valuePtr(), 
        A_i.data(), 
        A_p.data()
    );

    // ==========================================
    // 2. SETTINGS SETUP
    // ==========================================
    OSQPSettings* settings = OSQPSettings_new();
    if (!settings) {
        OSQPCscMatrix_free(P_csc);
        OSQPCscMatrix_free(A_csc);
        throw std::runtime_error("Failed to allocate OSQP settings.");
    }
    
    settings->alpha = 1.6;
    // NOTE: relaxing these to 5e-3 was tried, on the theory that millimetre-accurate
    // constraint satisfaction is pointless against metre-wide corridors. It did not help
    // consistently (one geometry improved, another got worse), so the tighter value stays --
    // there is no reason to loosen a safety-relevant tolerance for no measured gain.
    settings->eps_abs = 1e-3;
    settings->eps_rel = 1e-3;
    // Bound a failing solve by TIME, not iterations.
    //
    // OSQP can only certify primal infeasibility after exhausting its budget, so a doomed solve
    // is ~50x the cost of a successful one and is what produces replan latency spikes. The first
    // attempt at this capped max_iter at 2500 (measured: successes need median 75 iterations,
    // p99 1050, max 2225; failures ran the full default 4000). That did cut the max -- but it
    // also converted solves that merely needed MANY CHEAP iterations into failures, and each
    // such failure triggers a retry with a bigger QP. Net effect in sim: max fell 289 -> 159 ms
    // while the MEAN rose 21.3 -> 26.1 ms. Iteration count is a poor proxy for time.
    //
    // time_limit bounds the thing we actually care about and is indifferent to how many
    // iterations fit inside it, so a fast-converging-but-iteration-hungry problem is no longer
    // penalised. Requires OSQP_ENABLE_PROFILING, which this build has. Status comes back as
    // OSQP_TIME_LIMIT_REACHED, which the status check below already treats as failure.
    if (time_limit_s > 0.0) settings->time_limit = time_limit_s;
    settings->polishing = 1;  
    settings->verbose = 0;        
    settings->warm_starting = 1;  // Note: renamed from 'warm_start' in v1.0

    // ==========================================
    // 3. WORKSPACE SETUP
    // ==========================================
    OSQPSolver* solver = nullptr; // Note: renamed from 'osqp_workspace'
    
    // v1.0 setup signature requires passing m and n directly
    OSQPInt setup_status = osqp_setup(&solver, P_csc, (OSQPFloat*)q.data(), A_csc, (OSQPFloat*)l.data(), (OSQPFloat*)u.data(), m, n, settings);
    
    if (setup_status != 0) {
        OSQPSettings_free(settings);
        OSQPCscMatrix_free(P_csc);
        OSQPCscMatrix_free(A_csc);
        throw std::runtime_error("OSQP Setup Failed! Error Code: " + std::to_string(setup_status));
    }

    // ==========================================
    // 4. EXECUTE OPTIMIZATION
    // ==========================================
    OSQPInt solve_status = osqp_solve(solver);
    
    // osqp_solve() returns 0 whenever the routine ran without an INTERNAL error --
    // including when it concludes the problem is primal infeasible. The solution
    // status therefore has to be read from info->status_val, and the two checks must
    // be OR'd: an && here short-circuits to "accept" for every status, silently
    // returning an infeasible / unconverged iterate as if it were an optimal solution.
    // Status 1 = OSQP_SOLVED, Status 2 = OSQP_SOLVED_INACCURATE
    // --- NON-CONVERGENCE THAT IS STILL FLYABLE -------------------------------------------
    //
    // OSQP needs BOTH the primal and dual residuals inside tolerance to declare success. A run
    // can therefore stop at max_iter (or the time limit) holding an iterate whose CONSTRAINTS
    // are satisfied to well within anything that matters, purely because the dual residual --
    // i.e. optimality -- has not settled. Observed in flight: status 7, primal residual
    // 6.79e-4, with exactly ONE of 357 rows violated, by 6.79e-4 m/s, against v_max = 3.0.
    // Every boundary, corridor and acceleration row was satisfied exactly.
    //
    // Rejecting that is the wrong trade. In this QP the objective encodes SMOOTHNESS while the
    // constraints encode SAFETY, so a suboptimal-but-feasible trajectory is a slightly less
    // pretty trajectory -- whereas rejecting it costs a whole replan, and if it keeps happening
    // the vehicle hovers and never departs the waypoint. Accept the iterate when its worst
    // constraint violation is negligible next to the margins the constraints are enforcing
    // (corridors already carry the drone-radius inflation; 1e-2 m against 0.49 m, or 1e-2 m/s
    // against 3 m/s, is noise).
    //
    // This applies ONLY to non-convergence. A PRIMAL_INFEASIBLE verdict means OSQP has produced
    // a certificate that no solution exists, and its iterate is a diverging ray, not a nearly
    // feasible point -- accepting that would fly garbage. Same for DUAL_INFEASIBLE.
    const OSQPInt status_val = solver->info->status_val;
    const bool non_convergence = (status_val == OSQP_MAX_ITER_REACHED ||
                                  status_val == OSQP_TIME_LIMIT_REACHED);
    if (solve_status == 0 && non_convergence && accept_violation > 0.0 &&
        solver->info->prim_res <= accept_violation) {
        std::cout << "[QP] " << solver->info->status << " but worst constraint violation is "
                  << solver->info->prim_res << " (<= " << accept_violation
                  << "); accepting the iterate as feasible-but-suboptimal." << std::endl;
        if (prim_res_out) *prim_res_out = solver->info->prim_res;
        Eigen::VectorXd result = Eigen::Map<Eigen::VectorXd>(solver->solution->x, n);
        osqp_cleanup(solver);
        OSQPSettings_free(settings);
        OSQPCscMatrix_free(P_csc);
        OSQPCscMatrix_free(A_csc);
        return result;
    }

    if (solve_status != 0 || (solver->info->status_val != 1 && solver->info->status_val != 2)) {
        std::string status_str = solver->info->status;
        OSQPInt status_code = solver->info->status_val;
        double prim_res = solver->info->prim_res;
        if (prim_res_out) *prim_res_out = prim_res;
        if (failure_out) {
            failure_out->status_val = static_cast<long long>(status_code);
            failure_out->prim_res = prim_res;
            failure_out->x = Eigen::Map<Eigen::VectorXd>(solver->solution->x, n);
            // The certificate only exists for a primal-infeasible verdict.
            if (status_code == 3 || status_code == 4) {
                failure_out->prim_inf_cert =
                    Eigen::Map<Eigen::VectorXd>(solver->solution->prim_inf_cert, m);
            }
        }
        osqp_cleanup(solver);
        OSQPSettings_free(settings);
        OSQPCscMatrix_free(P_csc);
        OSQPCscMatrix_free(A_csc);
        throw std::runtime_error("OSQP Solve Failed! Status: " + status_str
                                 + " (code " + std::to_string(status_code)
                                 + ", primal residual " + std::to_string(prim_res) + ")");
    }

    // ==========================================
    // 5. EXTRACT RESULTS AND CLEANUP
    // ==========================================
    if (prim_res_out) *prim_res_out = solver->info->prim_res;
    Eigen::VectorXd optimized_control_points = Eigen::Map<Eigen::VectorXd>(solver->solution->x, n);

    // Free the C-allocated structs to prevent memory leaks
    osqp_cleanup(solver);
    OSQPSettings_free(settings);
    OSQPCscMatrix_free(P_csc);
    OSQPCscMatrix_free(A_csc);

    return optimized_control_points;
}

} // namespace trajectory_planner