'''
Visual and simple computational 
representation of snap minimization
using clamped B-Splines

TO DO:
- Optimize 4th derivative of position (snap)
- Optimize 2nd derivative of heading (course spline)

SNAP OPTIMIZATION:
- C_p = A_p @ Q_d4_M

HEADING OPTIMIZATION:
- C_psi = A_psi @ Q_d2_M

Q_d4_M = B_d3 @ U_1 @ (I - W_d4_M @ U_2 @ inv(U_2.T@W_d4_M@U_2) @ U_2.T)
Q_d2_M = U_1_psi.T @ (I - W_d2_M @ U_2_psi @ inv(U_2_psi.T@W_d2_M@U_2_psi) @ U_2_psi.T)

Will need to define matrices and how to calculate them
'''

import numpy as np
from numpy.linalg import inv
from core.clamped_constants import CASCADED_S_STENCILS, INTEGRAL_STENCILS
from core.minvo_bounds_clamped import MINVO_CLAMPED_STENCILS
from core.minvo_bounds import MINVO_STENCILS
from fractions import Fraction
import time


# ==========================================
# MINIMUM SNAP EVALUATOR (3D POSITION)
# ==========================================

class MinSnapEvalClamped:
    '''
    Visual and simple computational 
    representation of snap minimization
    using clamped uniform B-Splines
    '''

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
        start_time = 0
        self.knots = self._create_clamped_knot_points(self.num_control_points, self.degree, start_time)
        
        # Now trigger the heavy math
        self._calculate_Q()

    def _calculate_Q(self):
        B_d_3 = self._get_B_d3_matrix(self.degree)
        U1, U2 = self._get_U_matrices(self.num_control_points)

        # Expose the API properties expected by the trajectory planner
        self.B_combined = U1 # For OSQP A_eq extraction

        # 1. Grab the Blended W Matrix
        self.W = self.get_W_matrix()

        # S_d4_M, snap_knots = self._get_S_matrix(self.degree, self.degree, self.knots, self.num_control_points)

        # # print(f'S_snap=\n{S_snap}\n\nS_d4_M=\n{S_d4_M}\n')
        # self.W = self._get_W_matrix(S_d4_M, snap_knots)

        # 2. Optimize the Inverse via LU Decomposition Linear Solve
        A_bar = U2.T @ self.W @ U2
        B_bar = U2.T @ self.W
        
        X_T = np.linalg.solve(A_bar, B_bar)
        X = X_T.T

        # 3. Final Analytical Q Calculation
        self.Q = B_d_3 @ U1.T @ (np.eye(self.num_control_points) - X @ U2.T)

    def get_Q_matrix(self):
        return self.Q

    # --- Internal Class Methods ---

    def _create_clamped_knot_points(self, num_ctrl_pts, degree, start_time):
        number_of_knot_points = num_ctrl_pts + degree + 1
        number_of_unique_knot_points = number_of_knot_points - 2*degree
        unique_knot_points = np.arange(0,number_of_unique_knot_points) + start_time
        knot_points = np.zeros(number_of_knot_points) + start_time
        knot_points[degree : degree + number_of_unique_knot_points] = unique_knot_points
        knot_points[degree + number_of_unique_knot_points: 2*degree + number_of_unique_knot_points] = unique_knot_points[-1]
        return knot_points

    def _get_B_d3_matrix(self, d):
        d_inv = 1.0 / d
        accel_scalar = 2.0 / (d * (d - 1))
        B_d3 = np.array([
            [1,  1,       1,             0,             0,        0],
            [0,  d_inv,   3 * d_inv,     0,             0,        0],
            [0,  0,       accel_scalar,  0,             0,        0],
            [0,  0,       0,             accel_scalar,  0,        0],
            [0,  0,       0,            -3 * d_inv,    -d_inv,    0],
            [0,  0,       0,             1,             1,        1]
        ])
        return B_d3
    
    def _get_U_matrices(self, num_control_points):
        I = np.eye(num_control_points)
        U1 = np.hstack((I[:, 0:3], I[:, -3:]))
        U2 = I[:, 3:-3]
        return U1, U2
    
    def get_W_matrix(self, rho_vel=0.0, rho_accel=1.0, rho_snap=0.0):
        """
        Generates the Penalty matrix (W) for a Clamped Uniform Spline.
        Blends the integrated matrices in O(1) time.
        """
        M = self.M
        W_total = np.zeros((self.num_control_points, self.num_control_points))

        if rho_snap > 0:
            D_snap = self._get_fast_cascaded_D_matrix(M, self.degree, 4)
            if self.degree - 4 == 0: W_snap = D_snap.T @ D_snap
            else: W_snap = D_snap.T @ self._get_basis_integral_matrix(M, self.degree, 4) @ D_snap
            W_total += rho_snap * W_snap

        if rho_accel > 0 and self.degree > 2:
            D_accel = self._get_fast_cascaded_D_matrix(M, self.degree, 2)
            W_accel = D_accel.T @ self._get_basis_integral_matrix(M, self.degree, 2) @ D_accel
            W_total += rho_accel * W_accel

        if rho_vel > 0 and self.degree > 1:
            D_vel = self._get_fast_cascaded_D_matrix(M, self.degree, 1)
            W_vel = D_vel.T @ self._get_basis_integral_matrix(M, self.degree, 1) @ D_vel
            W_total += rho_vel * W_vel

        return W_total
    
    def _get_basis_integral_matrix(self, M, degree, derivative_order):
        """
        O(1) generation of the integral of b(t)b(t)^T for clamped B-splines.
        """
        d_minus_j = degree - derivative_order
        
        if d_minus_j not in INTEGRAL_STENCILS:
            raise NotImplementedError(f"Integral stencil for (d-j)={d_minus_j} not found.")
            
        # 1. Calculate active matrix size based on our physical active control formula
        N = M + degree - derivative_order 
        W_int = np.zeros((N, N))
        stencils = INTEGRAL_STENCILS[d_minus_j]
        
        # 2. Populate the shift-invariant interior bands using np.fill_diagonal
        interior_bands = stencils['interior']
        for offset, val in enumerate(interior_bands):
            np.fill_diagonal(W_int[offset:], val)        # Upper band
            if offset > 0:
                np.fill_diagonal(W_int[:, offset:], val) # Lower band
                
        # 3. Overwrite the boundary corners with the clamped squish blocks
        block = stencils['boundary_block']
        block_size = block.shape[0]
        
        # Top-Left Overwrite
        W_int[:block_size, :block_size] = block
        
        # Bottom-Right Overwrite (np.flip rotates it 180 degrees perfectly!)
        W_int[-block_size:, -block_size:] = np.flip(block)
        
        return W_int
    
    def _get_fast_cascaded_D_matrix(self, M, degree, derivative_order):
        """
        O(1) dynamic generation of the cascaded derivative mapping matrix.
        Checks the dictionary first; falls back to symbolic generation if missing.
        """
        # 1. Check Dictionary or Trigger Fallback
        if degree in CASCADED_S_STENCILS and derivative_order in CASCADED_S_STENCILS[degree]:
            stencils = CASCADED_S_STENCILS[degree][derivative_order]
            interior_band = stencils['interior']
            boundary_block = stencils['boundary_block']
        else:
            print(f"Warning: Stencil for d={degree}, j={derivative_order} not found. Triggering symbolic fallback...")
            interior_band, boundary_block = self._generate_fallback_S_stencil(degree, derivative_order)
            
            # Optionally cache it so we don't calculate it again this run!
            if degree not in CASCADED_S_STENCILS: CASCADED_S_STENCILS[degree] = {}
            CASCADED_S_STENCILS[degree][derivative_order] = {'interior': interior_band, 'boundary_block': boundary_block}

        # 2. Build the Matrix
        rows = M + degree - derivative_order
        cols = M + degree
        S_cascaded = np.zeros((rows, cols))
        
        # 3. Tile the shift-invariant interior (Pascal's Triangle)
        for i in range(rows):
            S_cascaded[i, i : i + len(interior_band)] = interior_band
            
        # 4. Overwrite the Top-Left with the squished boundary block
        block_rows, block_cols = boundary_block.shape
        S_cascaded[:block_rows, :block_cols] = boundary_block
        
        # 5. Overwrite the Bottom-Right (Rotated and signed)
        sign = 1 if derivative_order % 2 == 0 else -1
        S_cascaded[-block_rows:, -block_cols:] = np.flip(boundary_block) * sign
        
        return S_cascaded
    
    # ==========================================
    # SYMBOLIC FALLBACK GENERATORS
    # ==========================================

    def _get_single_D_step(self, k, M_dummy=15):
        """Generates a single derivative step matrix D^k using Fractions."""
        size = M_dummy + k - 1
        diag = []
        for i in range(1, k): diag.append(Fraction(k, i))
        
        num_ones = size - 2 * (k - 1)
        for i in range(num_ones): diag.append(Fraction(1, 1))
        
        for i in range(k - 1, 0, -1): diag.append(Fraction(k, i))
            
        D = np.zeros((size, size + 1), dtype=object)
        for r in range(size):
            D[r, r] = -diag[r]
            D[r, r+1] = diag[r]
        return D

    def _generate_fallback_S_stencil(self, d, j):
        """Cascades the matrices and extracts the exact arrays for the solver."""
        M_dummy = 15 # Large enough to prevent boundary collision
        D_cascaded = None
        
        # Cascade from degree d down to (d - j + 1)
        for k in range(d - j + 1, d + 1):
            D_current = self._get_single_D_step(k, M_dummy)
            if D_cascaded is None:
                D_cascaded = D_current
            else:
                D_cascaded = np.dot(D_cascaded, D_current)
                
        # Extract shapes
        boundary_rows = d
        boundary_cols = d + j
        
        # Slice the object arrays and cast them down to fast numpy floats!
        top_left_block = D_cascaded[:boundary_rows, :boundary_cols].astype(float)
        interior = D_cascaded[boundary_rows + 1, boundary_rows + 1 : boundary_rows + 1 + j + 1].astype(float)
        
        return interior.tolist(), top_left_block
    
    # ==========================================
    # LEGACY S, D, W GENERATORS
    # ==========================================

    def _get_D_matrix(self, degree, knots, num_control_points):
        num_derivative_cps = num_control_points - 1
        diag_values = np.zeros(num_derivative_cps)
        for i in range(num_derivative_cps):
            denominator = knots[i + degree + 1] - knots[i + 1]
            diag_values[i] = degree / denominator
        D_bar = np.diag(diag_values)
        zero_row = np.zeros((1, num_derivative_cps))
        block_1 = np.vstack((D_bar, zero_row))
        block_2 = np.vstack((zero_row, D_bar))
        D_matrix = -block_1 + block_2
        return D_matrix

    def _get_S_matrix(self, degree, derivative_level, knots, num_control_points):
        S_matrix = None
        current_degree = degree
        current_num_cp = num_control_points
        current_knots = np.copy(knots)
        
        for i in range(derivative_level):
            D_current = self._get_D_matrix(current_degree, current_knots, current_num_cp)
            if S_matrix is None:
                S_matrix = D_current
            else:
                S_matrix = np.dot(S_matrix, D_current)
                
            current_degree -= 1
            current_num_cp -= 1
            current_knots = current_knots[1:-1]
            
        return S_matrix, current_knots

    def _get_W_matrix(self, S_matrix, snap_knots):
        num_intervals = len(snap_knots) - 1
        dt_values = np.zeros(num_intervals)
        for i in range(num_intervals):
            dt_values[i] = snap_knots[i+1] - snap_knots[i]
        integral_matrix = np.diag(dt_values)
        W_matrix = np.dot(S_matrix, np.dot(integral_matrix, S_matrix.T))
        return W_matrix


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    
    # -------------------------
    # 1. RUN SNAP OPTIMIZATION
    # -------------------------
    snap_degree = 4
    BASE_SEGMENTS = 15

    # Pre-compute the Q matrix (This simulates the drone "booting up" on the ground)
    print("Pre-computing Q Matrix...")
    start_time = time.perf_counter()

    min_snap_evaluator = MinSnapEvalClamped(BASE_SEGMENTS, snap_degree)
    Q = min_snap_evaluator.get_Q_matrix()
    
    # Start the high-precision timer

    i_tot = 1
    for i in range(i_tot):
        # Generate random 3x1 column vectors for the states
        # The scalars give them reasonable physical ranges (e.g., 0 to 10 meters for position)
        # p0 = np.random.rand(3, 1) * 10 
        # v0 = np.random.rand(3, 1) * 5 - 2.5
        # a0 = np.random.rand(3, 1) * 2 - 1
        
        # pf = np.random.rand(3, 1) * 10 
        # vf = np.random.rand(3, 1) * 5 - 2.5
        # af = np.random.rand(3, 1) * 2 - 1

        p0 = np.array([[0],[0]])
        v0 = np.array([[-10],[-10]])
        a0 = np.array([[0],[0]])
        pf = np.array([[50],[50]])
        vf = np.array([[-10],[-10]])
        af = np.array([[0],[0]])
        
        # Build the boundary constraint matrix
        A_p = np.hstack((p0, v0, a0, af, vf, pf))
        
        # Calculate the exact optimal 3D flight path in a single dot product
        C_p_snap = A_p @ Q
        
    # Stop the timer
    end_time = time.perf_counter()
    
    # Calculate and print the results
    total_time = end_time - start_time
    avg_time = total_time / i_tot
    
    print(f"Total time for {i_tot} trajectories: {total_time:.6f} seconds")
    print(f"Average time per trajectory: {avg_time:.6f} seconds ({avg_time * 1000:.3f} ms)")


    # print("\n--- 3D Minimum Snap Control Points ---")
    print(C_p_snap.T)
    from utils.visualization import plot_trajectory
    plot_trajectory(C_p_snap, min_snap_evaluator.knots, snap_degree, minvo_stencils=False)


    # from utils.benchmarks import run_batch_performance_test_clamped, run_clamped_performance_benchmark
    # run_batch_performance_test_clamped()
    # run_clamped_performance_benchmark_clamped()