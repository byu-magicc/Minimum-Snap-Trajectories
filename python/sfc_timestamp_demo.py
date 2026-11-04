import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import BSpline

# 1. Define Safe Flight Corridors (SFCs) as [x, y, z, width, height, depth]
sfcs = [
    (0, 0, 0, 3, 2, 10),    # SFC 1
    (2, 1, 0, 3, 4, 10),    # SFC 2
    (4, 4, 0, 4, 2, 10),    # SFC 3
    (7, 1, 0, 3, 4, 10)     # SFC 4
]

# 2. Define Control Points that lie inside the SFC intersections (now 3D)
C_p = np.array([
    [0.5, 0.5, 5.0],   # SFC 1
    [1.5, 1.0, 5.0],   # SFC 1
    [2.5, 1.5, 5.0],   # Bridge 1-2
    [3.0, 2.5, 5.0],   # SFC 2
    [4.5, 4.5, 5.0],   # Bridge 2-3
    [5.5, 5.0, 5.0],   # SFC 3
    [6.5, 5.5, 5.0],   # SFC 3
    [7.5, 4.5, 5.0],   # Bridge 3-4
    [8.5, 3.0, 5.0],   # SFC 4
    [9.5, 2.0, 5.0]    # SFC 4
])

degree = 4
num_pts = len(C_p)
num_segments = num_pts - degree

# 3. Create clamped knots
knots = np.zeros(num_pts + degree + 1)
knots[degree : num_pts + 1] = np.arange(num_segments + 1)
knots[num_pts + 1 :] = num_segments
knots[:degree] = 0

spline = BSpline(knots, C_p, degree)

# 4. Setup 3D Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.axis('off')
ax.set_box_aspect([1, 1, 1])

# Set camera to view directly from above
ax.view_init(elev=90, azim=-90)

# Plot SFCs as 3D Wireframes
def plot_wireframe_box(ax, x, y, z, w, h, d, color='red'):
    # Corners
    x2, y2, z2 = x + w, y + h, z + d
    
    # Bottom and top loops
    bottom = np.array([[x, y, z], [x2, y, z], [x2, y2, z], [x, y2, z], [x, y, z]])
    top = np.array([[x, y, z2], [x2, y, z2], [x2, y2, z2], [x, y2, z2], [x, y, z2]])
    
    ax.plot(bottom[:,0], bottom[:,1], bottom[:,2], color=color, linewidth=2)
    ax.plot(top[:,0], top[:,1], top[:,2], color=color, linewidth=2)
    
    # Pillars
    for px, py in [(x, y), (x2, y), (x2, y2), (x, y2)]:
        ax.plot([px, px], [py, py], [z, z2], color=color, linewidth=2)

for i, (x, y, z, w, h, d) in enumerate(sfcs):
    plot_wireframe_box(ax, x, y, z, w, h, d)
    
    # Add label in the corner of the box
    ax.text(x + 0.2, y + h - 0.4, z + d, f'SFC {i+1}', fontsize=16, fontweight='bold', 
            color='black', alpha=1.0)

# Plot Control Polygon
ax.plot(C_p[:, 0], C_p[:, 1], C_p[:, 2], 'o:', color='darkgray', alpha=0.8, linewidth=2.5, 
        markersize=8, markeredgecolor='black', markeredgewidth=1.5)

# Helper function to draw a stylized top-down drone mockup matching the image
def draw_drone(ax, x, y, z, size=0.4):
    # 1. Thick X-frame body
    ax.plot([x - size, x + size], [y - size, y + size], [z, z], color='black', linewidth=7, zorder=10, solid_capstyle='round')
    ax.plot([x - size, x + size], [y + size, y - size], [z, z], color='black', linewidth=7, zorder=10, solid_capstyle='round')
    ax.plot([x], [y], [z], 'o', color='black', markersize=14, zorder=11) # Center hub
    
    # 2. Rotors (Guards and Propellers)
    theta = np.linspace(0, 2*np.pi, 30)
    r_guard = size * 0.55
    
    rotors = [
        (x - size, y + size, np.pi/4),   # Top-left (/)
        (x + size, y + size, -np.pi/4),  # Top-right (\)
        (x - size, y - size, -np.pi/4),  # Bottom-left (\)
        (x + size, y - size, np.pi/4)    # Bottom-right (/)
    ]
    
    for px, py, angle in rotors:
        # Hollow guard
        cx = px + r_guard * np.cos(theta)
        cy = py + r_guard * np.sin(theta)
        ax.plot(cx, cy, np.full_like(cx, z), color='black', linewidth=3.5, zorder=12)
        
        # Center motor hub
        ax.plot([px], [py], [z], 'o', color='black', markersize=6, zorder=13)
        
        # Dashed propeller blades
        dx = np.cos(angle) * r_guard * 0.75
        dy = np.sin(angle) * r_guard * 0.75
        ax.plot([px - dx, px + dx], [py - dy, py + dy], [z, z], 
                color='black', linewidth=3, linestyle='--', dashes=(3, 2), zorder=14)

# 5. Plot segments and mark timestamps
segment_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for t in range(num_segments + 1):
    pt = spline(t)
    
    if t < num_segments:
        # Plot the segment from t to t+1 in a unique color
        t_seg = np.linspace(t, t+1, 100)
        seg_curve = spline(t_seg)
        ax.plot(seg_curve[:, 0], seg_curve[:, 1], seg_curve[:, 2], '-', 
                color=segment_colors[t], linewidth=5, alpha=0.9, zorder=4,
                label=f't = {t}s to {t+1}s')
    
    if t == 0:
        draw_drone(ax, pt[0], pt[1], pt[2])
    else:
        # Small dot to indicate the exact knot location
        ax.plot([pt[0]], [pt[1]], [pt[2]], 'o', color='black', markersize=5, zorder=5)

# Add a highly legible legend for the timestamps
ax.legend(loc='lower right', fontsize=18, framealpha=0.95, edgecolor='black', 
          title='Trajectory Timeline', title_fontsize=22)

plt.tight_layout()

if __name__ == "__main__":
    plt.show()
