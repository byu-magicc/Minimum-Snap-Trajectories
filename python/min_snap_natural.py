"""
Visual and simple computational representation of snap minimization
using natural uniform B-Splines.

SNAP OPTIMIZATION:
- C_p = [S E] @ Q_d4_M
- S = [ p(0) dp(0)/dt d^2p(0)/dt^2 ]
- E = [ p(M) dp(M)/dt d^2p(M)/dt^2 ]

C++ for Deployment: When you eventually wrap this into a ROS C++ node
for the physical quadcopter, you will swap back to Cholesky, 
use fixed-size Eigen matrices, and achieve sub-microsecond 
trajectory generation.
"""

# ==========================================
# IMPORTS
# ==========================================
import time
import math
import numpy as np
from core.optimize import run_qp_solver
from core.b_spline_constants import M_STENCILS, S_STENCILS, D_STENCILS, T_STENCILS
from core.minvo_bounds import MINVO_STENCILS

# ==========================================
# CORE SOLVER CLASS
# ==========================================
class MinSnapEvalNatural:
    """
    Evaluator for generating Minimum Snap Trajectories using 
    Natural Uniform B-Splines via Singular Value Decomposition (SVD).
    """

    def __init__(self, num_segments, degree=4):
        # 1. Intuitive Safety Checks
        if degree < 4:
            raise ValueError(f"Minimum Snap requires a polynomial of at least degree 4. You provided degree {degree}.")
        if num_segments < 3:
            raise ValueError(f"To satisfy 6 physical constraints, you need at least 3 flight segments. You provided {num_segments}.")
        
        self.degree = degree
        # Only initialize state here, do NOT do the math yet.
        self.update_segments(num_segments)

    def update_segments(self, new_num_segments):
        """
        Updates the segment count and recalculates the structural matrices.
        Call this if the optimizer needs to add segments to satisfy a constraint.
        """
        self.M = new_num_segments
        self.num_control_points = self.M + self.degree
        self.knots = np.arange(-self.degree, self.M + self.degree + 1)
        
        # Now trigger the heavy math
        self._calculate_Q()

    def _calculate_Q(self):
        """
        Internal method to compute the Q mapping matrix via SVD.
        """
        self.B_combined, U1, U2, Sigma, V = self._create_SVD(self.num_control_points)
        self.W = self.get_W_matrix()
        
        A_bar = (U2.T @ self.W @ U2).T
        B_bar = (self.W @ U2).T
        
        X_bar = np.linalg.solve(A_bar, B_bar)
        X = X_bar.T
        
        self.Q = V @ np.linalg.inv(Sigma) @ U1.T @ (np.eye(self.num_control_points) - X @ U2.T)


    def get_Q_matrix(self):
        """Returns the pre-computed Q mapping matrix."""
        return self.Q
    

    def _get_M_matrix(self, degree):
        """
        Dynamically generates the Basis Mapping Matrix (M) for ANY degree uniform B-spline.
        Maps control points to polynomial coefficients for tau in [0, 1].
        """
        M = np.zeros((degree + 1, degree + 1), dtype=int)
        
        for r in range(degree + 1):
            for j in range(degree + 1):
                val = 0
                for k in range(degree - r + 1):
                    term1 = (-1)**k
                    term2 = math.comb(degree + 1, k)
                    term3 = math.comb(degree, j)
                    
                    # Handle 0^0 safely
                    base = degree - r - k
                    term4 = (base**j) if not (base == 0 and j == 0) else 1
                    
                    val += term1 * term2 * term3 * term4
                    
                M[r, j] = val
        
        scalar = math.factorial(degree)
        return M, scalar


    def _get_T_vector(self, degree, derivative_order, tau):
        """
        Dynamically generates the Time vector T(tau) and its derivatives.
        """
        T = np.zeros((degree + 1, 1))
        for i in range(degree + 1):
            power = degree - i
            if power >= derivative_order:
                # Calculate the cascaded derivative scalar using the power rule
                scalar = math.prod(range(power - derivative_order + 1, power + 1)) if derivative_order > 0 else 1
                T[i, 0] = scalar * (tau ** (power - derivative_order))
        return T
    
    def _get_boundary_states(self, M_matrix, degree):
        """
        Calculates the active constraint blocks for the start (tau=0) 
        and end (tau=1) of the trajectory using a hybrid lookup/dynamic approach.
        """
        # --- FAST PATH: Hardcoded Lookup ---
        if degree in T_STENCILS:
            T_start_combined = T_STENCILS[degree]['start']
            T_end_combined = T_STENCILS[degree]['end']
            
            # Execute a single matrix multiplication for all 3 constraints simultaneously
            B_d_M0 = M_matrix @ T_start_combined
            B_d_MM = M_matrix @ T_end_combined
            
            return B_d_M0, B_d_MM

        # --- FALLBACK PATH: Dynamic Generation ---
        # TAU = 0 (Start Constraints)
        start_pos = self._get_T_vector(degree, 0, 0)
        start_vel = self._get_T_vector(degree, 1, 0)
        start_acc = self._get_T_vector(degree, 2, 0)
        T_start_combined = np.hstack((start_pos, start_vel, start_acc))
        B_d_M0 = M_matrix @ T_start_combined

        # TAU = 1 (End Constraints)
        end_pos = self._get_T_vector(degree, 0, 1)
        end_vel = self._get_T_vector(degree, 1, 1)
        end_acc = self._get_T_vector(degree, 2, 1)
        T_end_combined = np.hstack((end_pos, end_vel, end_acc))
        B_d_MM = M_matrix @ T_end_combined

        return B_d_M0, B_d_MM


    def _create_SVD(self, num_control_points):
        """
        Constructs the boundary constraint matrices and performs SVD to isolate
        the null space (free control points) for snap optimization.
        """
        
        # --- 1. RESOLVE THE M MATRIX ---
        if self.degree in M_STENCILS:
            # Fast Path: Memory Lookup
            M_matrix = M_STENCILS[self.degree]
        else:
            # Fallback Path: Dynamic Generation
            print(f"Warning: Generating M^{self.degree} on the fly.")
            M_int, scalar = self._get_M_matrix(self.degree)
            M_matrix = M_int / scalar 

        # --- 2. GENERATE BOUNDARY BLOCKS ---
        B_d_M0, B_d_MM = self._get_boundary_states(M_matrix, self.degree) # block representing the active control points at t=0 and t=M
        

        # Initialize the full-size boundary matrices with zeros
        B_0_full = np.zeros((num_control_points, 3))
        B_M_full = np.zeros((num_control_points, 3))

        # Paste the active blocks into their respective ends# The injection window size is exactly degree + 1
        window_size = self.degree + 1
        B_0_full[0:window_size, :] = B_d_M0 
        B_M_full[-window_size:, :] = B_d_MM

        # Glue them together horizontally for the SVD: [B(0) B(M)]
        B_combined = np.hstack((B_0_full, B_M_full))

        # Run the Singular Value Decomposition
        U, s, Vh = np.linalg.svd(B_combined, full_matrices=True)

        # We have 6 total constraints (3 start + 3 end)
        num_constraints = B_combined.shape[1]

        # Extract the components required for the Q matrix
        # U1 represents the constrained space; U2 represents the free null space
        U1 = U[:, :num_constraints]
        U2 = U[:, num_constraints:]
        Sigma = np.diag(s)
        V = Vh.T

        return B_combined, U1, U2, Sigma, V
    
    
    def _get_fast_cascaded_D_matrix(self, M, degree, derivative_order):
        """
        Direct O(N) LUT implementation of the book's cascaded D matrix.
        Yields an (M+d) x (M+d-l) matrix matching the exact output of D^d * D^{d-1}...
        """
        if derivative_order not in D_STENCILS:
            raise NotImplementedError(f"Stencil for derivative {derivative_order} not hardcoded.")

        stencil = D_STENCILS[derivative_order]

        # The book's dimensions: mapping from (M + d - l) up to (M + d)
        rows = M + degree
        cols = M + degree - derivative_order

        D_cascaded = np.zeros((rows, cols))

        # The book's D matrix cascades column-wise
        for i in range(cols):
            D_cascaded[i : i + len(stencil), i] = stencil

        return D_cascaded
    
    def _get_S_matrix(self, M, k):
        """
        Dynamically populates the S matrix using the pre-calculated memory LUT.
        Applies exact boundary truncation patches for splines bleeding out of [0, M].
        """
        if k not in S_STENCILS:
            raise NotImplementedError(f"Integral stencil for k={k} is not yet hardcoded.")

        size = M + k
        S = np.zeros((size, size))
        stencil = S_STENCILS[k]
        
        # 1. Populate the main diagonal (s_0)
        S += np.diag(np.full(size, stencil[0]))
        
        # 2. Populate the sub and super diagonals (s_1, s_2, etc.)
        for i in range(1, len(stencil)):
            S += np.diag(np.full(size - i, stencil[i]), k=i)   # Super-diagonal
            S += np.diag(np.full(size - i, stencil[i]), k=-i)  # Sub-diagonal
            
        # 3. APPLY BOUNDARY TRUNCATION PATCHES
        if k == 1: # [[1/3, 2/3], [1/6]]
            # The first and last basis functions lose exactly half their area 
            # because they bleed outside the [0, M] integral bounds.
            S[0, 0] = 1/3
            S[-1, -1] = 1/3

        elif k == 2: # [[1/20, 1/2, 11/20], [13/120, 26/120], [1/120]]
            # Main Diagonal (b_0*b_0 and b_1*b_1)
            S[0, 0] = 1/20
            S[1, 1] = 1/2
            S[-1, -1] = 1/20
            S[-2, -2] = 1/2
            
            # First Off-Diagonal (b_0*b_1)
            S[0, 1] = S[1, 0] = 13/120
            S[-1, -2] = S[-2, -1] = 13/120
            
            # Note: The second off-diagonal (b_0*b_2) doesn't bleed into 
            # negative time, so it remains the infinite value of 1/120!

        elif k == 3: # [[1/252, 151/630, 599/1260, 151/315], [43/1680, 59/280, 397/1680], [1/84, 1/42], [1/5040]]
            # 1. Main Diagonal Patches (S_0,0 / S_1,1 / S_2,2)
            S[0, 0] = 1/252
            S[1, 1] = 151/630
            S[2, 2] = 599/1260
            
            S[-1, -1] = 1/252
            S[-2, -2] = 151/630
            S[-3, -3] = 599/1260

            # --- First Off-Diagonal Patches ---
            S[0, 1] = S[1, 0] = 43/1680
            S[1, 2] = S[2, 1] = 59/280   # (Which is exactly 354/1680)
            
            S[-1, -2] = S[-2, -1] = 43/1680
            S[-2, -3] = S[-3, -2] = 59/280

            # --- Second Off-Diagonal Patches ---
            S[0, 2] = S[2, 0] = 1/84

            S[-1, -3] = S[-3, -1] = 1/84


        elif k > 3:
            # Placeholder: The boundary patches for k=2 and higher are matrices 
            # (e.g., a 2x2 corner block for k=2) because the overlap bleeds further.
            raise NotImplementedError(f"Boundary patches for k={k} not yet hardcoded.")
        
        # print(f"S:\n{S}")
        return S

    def get_W_matrix(self, rho_vel=0.0, rho_accel=0.0, rho_snap=1.0):
        """
        Generates the Minimum Snap Penalty matrix (W) for a Natural Uniform Spline.
        Cascades 4 derivative matrices to represent the 4th derivative (Snap).
        """
        M = self.M
        W_total = np.zeros((self.num_control_points, self.num_control_points))

        if rho_snap > 0:
            # 1. Calculate Snap W (4th derivative)
            snap_minimization = 4        
            D_snap = self._get_fast_cascaded_D_matrix(M, self.degree, snap_minimization)
            S_snap = self._get_S_matrix(M, self.degree - snap_minimization)
            W_total += rho_snap*(D_snap @ S_snap @ D_snap.T)

        if rho_accel > 0 and self.degree <= 4:
            # 2. Calculate Accel W (2nd derivative)
            accel_minimization = 2
            D_accel = self._get_fast_cascaded_D_matrix(M, self.degree, accel_minimization)
            S_accel = self._get_S_matrix(M, self.degree - accel_minimization)
            W_total += rho_accel*(D_accel @ S_accel @ D_accel.T)

        if rho_vel > 0 and self.degree <= 4:
            # 2. Calculate Vel W (1st derivative)
            vel_minimization = 1
            D_vel = self._get_fast_cascaded_D_matrix(M, self.degree, vel_minimization)
            S_vel = self._get_S_matrix(M, self.degree - vel_minimization)
            W_total += rho_vel*(D_vel @ S_vel @ D_vel.T)

        return W_total
        

    def get_sfc_matrices(self, sfc_constraints, num_pts_list):
        """
        Translates Dean's SFC constraints into full-scale QP matrices.
        """
        A_ineq_list = []
        b_ineq_list = []
        
        start_idx = 0 
        
        for i, sfc in enumerate(sfc_constraints):
            A_mat = sfc['A'] 
            b_vec = np.array(sfc['b']).flatten() 
            
            num_pts_in_box = num_pts_list[i]
            num_inequalities = A_mat.shape[0]
            
            for j in range(num_pts_in_box):
                global_cp_index = start_idx + j
                
                # Create the padded matrix for this specific control point
                A_padded = np.zeros((num_inequalities, self.num_control_points * 3))
                
                col_start = global_cp_index * 3
                col_end = col_start + 3
                A_padded[:, col_start:col_end] = A_mat
                
                A_ineq_list.append(A_padded)
                b_ineq_list.append(b_vec)
                
            # Step forward, accounting for the intersection overlap!
            start_idx += (num_pts_in_box - self.degree)
            
        # Stack them vertically to create one massive matrix for the QP solver
        A_sfc_total = np.vstack(A_ineq_list)
        b_sfc_total = np.concatenate(b_ineq_list)
        
        return A_sfc_total, b_sfc_total

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # ----------------------------------------------------
    # DEMO: SINGLE FLIGHT PATH GENERATION
    # ----------------------------------------------------
    degree = 4
    BASE_SEGMENTS = 15

    print("Pre-computing Q Matrix...")
    
    start_time = time.perf_counter()
    
    min_snap_evaluator = MinSnapEvalNatural(BASE_SEGMENTS, degree)
    knots = min_snap_evaluator.knots

    W = min_snap_evaluator.get_W_matrix()
    Q = min_snap_evaluator.Q

    # D_vel = min_snap_evaluator._get_fast_cascaded_D_matrix(BASE_SEGMENTS, degree, 1).T
    # D_accel = min_snap_evaluator._get_fast_cascaded_D_matrix(BASE_SEGMENTS, degree, 2).T

    i_tot = 1
    for i in range(i_tot):
        snap_num_segments = BASE_SEGMENTS

        # Define the map dimensions (X, Y, Z)
        # map_size = np.array([[40.0], [40.0], [5.0]])

        # Generate random start and end conditions
        # p0 = np.random.rand(3, 1) * 10
        # v0 = np.random.rand(3, 1) * 5 - 2.5
        # a0 = np.random.rand(3, 1) * 2 - 1
        
        # pf = np.random.rand(3, 1) * 10
        # vf = np.random.rand(3, 1) * 5 - 2.5
        # af = np.random.rand(3, 1) * 2 - 1

        p0 = np.array([[0],[0],[0]])
        v0 = np.array([[-10],[-10],[10]])
        a0 = np.array([[0],[0],[0]])
        pf = np.array([[50],[50],[30]])
        vf = np.array([[-10],[-10],[-10]])
        af = np.array([[0],[0],[0]])
        
        S = np.hstack((p0, v0, a0))
        E = np.hstack((pf, vf, af))
        SE = np.hstack((S, E))

        # 1. Initialize the math structures

        # 2. Get the "Starter Motor" path (The Unconstrained Answer)
        # Core optimization calculation
        C_p_min_snap = SE @ Q
        use_constraints = False

        if not use_constraints:
            # We are done! This is the fastest O(1) path.
            C_p_min_snap_constrained = C_p_min_snap 

        else:
            # 3. Setup the "Glass Box" (The Constrained Answer)

            V_max = 3
            A_max = 2

            max_segments = 30 
            success = False

            # 4. Fire up the QP Solver with Optimized Temporal Scaling
            while snap_num_segments <= max_segments:
                try:
                    print(f"Attempting trajectory with {snap_num_segments} segments...")
                    
                    # A. Try the solver immediately using the CURRENT matrices
                    C_p_min_snap_constrained = run_qp_solver(
                        objective_matrix=W,
                        equality_constraints=SE,
                        inequality_constraints=(D_vel, D_accel, V_max, A_max), 
                        initial_guess=C_p_min_snap,
                        A_eq=min_snap_evaluator.B_combined.T,
                        degree=degree
                    ) 
                    
                    # B. If it passes, break out! No need to rebuild anything.
                    print("Success! Trajectory is physically feasible.")
                    success = True
                    break 
                    
                except ValueError as e:
                    print("Physics impossible! Adding time and rebuilding matrices...")
                    snap_num_segments += 2 
                    
                    # C. ONLY rebuild the matrices if we are going to loop again
                    if snap_num_segments <= max_segments:
                        min_snap_evaluator.update_segments(snap_num_segments)
                        
                        # Pull the newly sized matrices
                        W = min_snap_evaluator.get_W_matrix()
                        Q = min_snap_evaluator.Q
                        D_vel = min_snap_evaluator._get_fast_cascaded_D_matrix(snap_num_segments, degree, 1).T
                        D_accel = min_snap_evaluator._get_fast_cascaded_D_matrix(snap_num_segments, degree, 2).T
                        A_eq = min_snap_evaluator.B_combined.T
                        
                        # Calculate the new starting guess
                        C_p_min_snap = SE @ Q

            if not success:
                print("CRITICAL FAILURE: Could not find a feasible path within the segment limit.")

        
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time = total_time / i_tot
    
    print(f"Total time for {i_tot} trajectories: {total_time:.6f} seconds")
    print(f"Average time per trajectory: {avg_time:.6f} seconds ({avg_time * 1000:.3f} ms)")

    print(C_p_min_snap_constrained.T)

    # Plot the last trajectory from the loop
    from utils.visualization import plot_trajectory, plot_kinematics
    plot_trajectory(C_p_min_snap_constrained, min_snap_evaluator.knots, degree,spline_type='natural', minvo_stencils=False)
    # plot_trajectory(C_p_min_snap_constrained, min_snap_evaluator.knots, degree, minvo_stencils=MINVO_STENCILS)
    # plot_kinematics(C_p_min_snap_constrained, min_snap_evaluator.knots, degree, V_max, A_max) # Verify the physics!



    # ----------------------------------------------------
    # OPTIONAL BENCHMARKS
    # (Uncomment the lines below to run them)
    # ----------------------------------------------------
    #
    from utils.benchmarks import run_batch_performance_test, run_performance_benchmark
    # run_batch_performance_test()
    # run_performance_benchmark(max_control_points=100)