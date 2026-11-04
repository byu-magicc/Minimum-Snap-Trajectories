import sys
import os

# Add parent directory to path to import core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import BSpline

try:
    from python.core.minvo_bounds_clamped import MINVO_CLAMPED_STENCILS
except ImportError:
    from core.minvo_bounds_clamped import MINVO_CLAMPED_STENCILS

# 1. Define winding control points
C_p = np.array([
    [0.0, 0.0],
    [1.0, 5.0],
    [3.0, 8.0],
    [6.0, 7.0],
    [8.0, 2.0],
    [10.0, -3.0],
    [13.0, -5.0]
])

degree = 4
num_pts = len(C_p)
num_segments = num_pts - degree

# Create clamped knots
knots = np.zeros(num_pts + degree + 1)
knots[degree : num_pts + 1] = np.arange(num_segments + 1)
knots[num_pts + 1 :] = num_segments
knots[:degree] = 0

spline = BSpline(knots, C_p, degree)
t_eval = np.linspace(0, num_segments, 500)
curve = spline(t_eval)

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_aspect('equal', 'box')

# Plot overall curve and control polygon
ax.plot(curve[:, 0], curve[:, 1], 'b-', linewidth=2, label='B-Spline Curve')
ax.plot(C_p[:, 0], C_p[:, 1], 'o:', color='darkorange', alpha=0.9, linewidth=5, markersize=8, markeredgecolor='black', markeredgewidth=1.5, label='Global Control Polygon')

# Animation elements
minvo_poly, = ax.plot([], [], 'ro-', linewidth=2, markersize=8, label='Active MINVO Control Points')
active_cp, = ax.plot([], [], 'o-', color='darkorange', markersize=10, linewidth=3, label='Active B-Spline Points')
active_curve, = ax.plot([], [], 'c-', linewidth=4, label='Active Curve Segment')

# ax.legend()
ax.axis('off')

def update(s):
    # 1. Active control points
    C_local = C_p[s : s + degree + 1]
    active_cp.set_data(C_local[:, 0], C_local[:, 1])
    
    # 2. MINVO Stencil
    stencils = MINVO_CLAMPED_STENCILS[degree]
    if s < degree:
        F = stencils[f'start_{s}']
    elif s >= num_segments - degree:
        end_idx = num_segments - 1 - s
        F = stencils[f'end_{end_idx}']
    else:
        F = stencils['interior']
        
    V_local = F @ C_local
    V_closed = np.vstack((V_local, V_local[0]))
    minvo_poly.set_data(V_closed[:, 0], V_closed[:, 1])
    
    # 3. Active curve segment
    t_seg = np.linspace(s, s+1, 50)
    seg_curve = spline(t_seg)
    active_curve.set_data(seg_curve[:, 0], seg_curve[:, 1])
    
    return minvo_poly, active_cp, active_curve

anim = FuncAnimation(fig, update, frames=num_segments, interval=1000, blit=True, repeat_delay=2000)

if __name__ == "__main__":
    plt.show()
