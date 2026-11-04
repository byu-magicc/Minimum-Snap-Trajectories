
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from core.minvo_bounds_clamped import MINVO_CLAMPED_STENCILS
from core.minvo_bounds import MINVO_STENCILS
import numpy as np

# ==========================================
# UTILITY & VISUALIZATION
# ==========================================
def plot_trajectory(ctrl_pts, knots, degree, spline_type="clamped", minvo_stencils=False):
    """
    Evaluates and plots the 3D Minimum Snap Trajectory and its control polygon.
    """
    pts = ctrl_pts.T 
    is_3d = pts.shape[1] == 3
    
    fig = plt.figure(figsize=(10, 8))
    if is_3d:
        ax = fig.add_subplot(111, projection='3d')
        # 1. Plot the Control Polygon (The mathematical "gravitational anchors")
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'ro--', alpha=0.4, linewidth=1.5, label='Control Points')
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='red', s=30)
    else:
        ax = fig.add_subplot(111)
        # 1. Plot the Control Polygon
        ax.plot(pts[:, 0], pts[:, 1], 'ro--', alpha=0.4, linewidth=1.5, label='Control Points')
        ax.scatter(pts[:, 0], pts[:, 1], c='red', s=30)

    # 2. Plot the MINVO Polygons (The tight "shrink-wrap")
    if minvo_stencils:
        
        # Calculate how many segments make up this flight path
        num_segments = len(pts) - degree
        # Define a list of colors to cycle through
        colors = ['g', 'c', 'm', 'y', 'orange'] 
        
        # Slide the window across each segment!
        for s in range(num_segments):
            C_local = pts[s : s + degree + 1]

            if spline_type == "clamped":
                stencils = MINVO_CLAMPED_STENCILS[degree]
                if s < degree:
                    F = stencils[f'start_{s}']
                elif s >= num_segments - degree:
                    end_idx = num_segments - 1 - s
                    F = stencils[f'end_{end_idx}']
                else:
                    F = stencils['interior']
            else:
                F = MINVO_STENCILS[degree]

            V_local = F @ C_local
            V_closed = np.vstack((V_local, V_local[0]))
            
            # Pick a color based on the segment index
            c = colors[s % len(colors)]
            
            label = 'MINVO Bounds' if s == 0 else ""
            
            # Plot using the cycling color!
            if is_3d:
                ax.plot(V_closed[:, 0], V_closed[:, 1], V_closed[:, 2], 
                        color=c, linestyle='--', marker='.', alpha=0.6, linewidth=1.5, label=label)
            else:
                ax.plot(V_closed[:, 0], V_closed[:, 1], 
                        color=c, linestyle='--', marker='.', alpha=0.6, linewidth=1.5, label=label)
    
    # 2. Evaluate and Plot the actual Minimum Snap B-Spline Curve
    spline = BSpline(knots, pts, degree)
    t_smooth = np.linspace(knots[degree], knots[-degree-1], 100)
    curve = spline(t_smooth)
    
    if is_3d:
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], 'b-', linewidth=3, label='B-Spline Trajectory')
        # Plot Start and End constraint points
        ax.scatter(*curve[0], c='green', s=100, marker='*', label='Start')
        ax.scatter(*curve[-1], c='purple', s=100, marker='*', label='End')
    else:
        ax.plot(curve[:, 0], curve[:, 1], 'b-', linewidth=3, label='B-Spline Trajectory')
        # Plot Start and End constraint points
        ax.scatter(curve[0, 0], curve[0, 1], c='green', s=100, marker='*', label='Start')
        ax.scatter(curve[-1, 0], curve[-1, 1], c='purple', s=100, marker='*', label='End')
    
    # Formatting
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    # ax.set_zlabel('Z Position')
    ax.set_title('B-Spline Minimum Snap Trajectory')
    ax.legend()
    if is_3d:
        ax.set_box_aspect([1, 1, 1]) 
    else:
        ax.set_aspect('equal', 'box')
    plt.show()


def plot_kinematics(C_p, knots, degree, V_max, A_max):
    """
    Evaluates the continuous velocity and acceleration profiles and plots 
    them against the physical solver limits.
    """
    pts = C_p.T  # Transpose to N x 3 for SciPy
    
    # 1. Construct the continuous position spline
    pos_spline = BSpline(knots, pts, degree)
    
    # 2. Extract the analytical derivative splines
    vel_spline = pos_spline.derivative(nu=1)
    acc_spline = pos_spline.derivative(nu=2)
    
    # 3. Evaluate smoothly over the flight time
    t_smooth = np.linspace(knots[degree], knots[-degree-1], 500)
    velocities = vel_spline(t_smooth)
    accelerations = acc_spline(t_smooth)
    
    # 4. Visualization Setup
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # --- VELOCITY PLOT ---
    ax1.plot(t_smooth, velocities[:, 0], 'r-', linewidth=2, label='Vx')
    ax1.plot(t_smooth, velocities[:, 1], 'g-', linewidth=2, label='Vy')
    ax1.plot(t_smooth, velocities[:, 2], 'b-', linewidth=2, label='Vz')
    
    # Draw the physical constraints
    ax1.axhline(V_max, color='k', linestyle='--', linewidth=2, label=f'Limit (±{V_max} m/s)')
    ax1.axhline(-V_max, color='k', linestyle='--', linewidth=2)
    
    ax1.set_title('Velocity Profiles vs. Physical Limits', fontweight='bold')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.4)
    
    # --- ACCELERATION PLOT ---
    ax2.plot(t_smooth, accelerations[:, 0], 'r-', linewidth=2, label='Ax')
    ax2.plot(t_smooth, accelerations[:, 1], 'g-', linewidth=2, label='Ay')
    ax2.plot(t_smooth, accelerations[:, 2], 'b-', linewidth=2, label='Az')
    
    # Draw the physical constraints
    ax2.axhline(A_max, color='k', linestyle='--', linewidth=2, label=f'Limit (±{A_max} m/s²)')
    ax2.axhline(-A_max, color='k', linestyle='--', linewidth=2)
    
    ax2.set_title('Acceleration Profiles vs. Physical Limits', fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Acceleration (m/s²)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.4)
    
    plt.tight_layout()
    plt.show()


def plot_course_trajectory(ctrl_pts, knots, degree):
    pts = ctrl_pts.T  # Shape becomes (N, 1)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Generate the physical time array for the X-axis of the control points
    cp_time = np.linspace(knots[0], knots[-1], len(pts))
    ax.plot(cp_time, pts, 'ro--', alpha=0.5, label='Course Control Polygon')
    
    # Evaluate smooth spline
    spline = BSpline(knots, pts, degree)
    t_smooth = np.linspace(knots[degree], knots[-degree-1], 100)
    curve = spline(t_smooth)
    
    ax.plot(t_smooth, curve, 'b-', linewidth=3, label='Optimized Course Trajectory')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Course Angle (Degrees or Rads)')
    ax.set_title('Minimum 2nd-Derivative Course Spline')
    ax.legend()
    ax.grid(True)
    plt.show()