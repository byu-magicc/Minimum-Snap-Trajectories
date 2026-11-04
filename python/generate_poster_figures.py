#!/usr/bin/env python3
"""
Generate poster figures for COAST Trajectory Planner matching the user's reference style:
- Figure 1: Safe Flight Corridors (SFCs) adapting to surrounding obstacles (Image 1 style:
  red/salmon 3D obstacle boxes, purple wireframe SFC corridor prisms, flight path).
- Figure 2: 4th-degree B-Spline curve showing how control points dictate trajectory shape
  (Image 2 style: blue curve, dark orange control polygon C_0..C_6, cyan active segment,
  red active MINVO bounding polytope).
- Combined side-by-side figure ready for the bottom-left poster box.
- Consistent color palette across both figures.
- High-resolution outputs in PNG (standard and transparent), PDF (vector), and SVG (vector).
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import BSpline

# Add repository root to path for core imports
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from python.core.minvo_bounds_clamped import MINVO_CLAMPED_STENCILS
except ImportError:
    try:
        from core.minvo_bounds_clamped import MINVO_CLAMPED_STENCILS
    except ImportError:
        MINVO_CLAMPED_STENCILS = None

# Typography matching academic conference poster standards
plt.rcParams.update({
    "font.family": ["Noto Sans", "Liberation Sans", "DejaVu Sans", "sans-serif"],
    "font.size": 11.5,
    "axes.titlesize": 13.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 11.5,
    "figure.titlesize": 14.5,
    "figure.titleweight": "bold",
    "pdf.fonttype": 42,
})

# ==============================================================================
# CONSISTENT COLOR PALETTE (Matched across both figures & user references)
# ==============================================================================
OBS_FACE = "#FFCDD2"       # Semi-transparent red/salmon (Image 1: facecolors='red', alpha=0.15-0.25)
OBS_EDGE = "#B71C1C"       # Dark red wireframe edge (Image 1: edgecolors='darkred')
SFC_COLOR = "#6A1B9A"      # Purple wireframe corridor (Image 1: color="purple")
SFC_FACE = "#F3E5F5"       # Light purple translucent volume fill
SPLINE_BLUE = "#0044CC"    # B-Spline curve blue (Image 2: 'b-')
CTRL_ORANGE = "darkorange"  # Control points & polygon (Image 2: 'darkorange' 'o:')
ACTIVE_CYAN = "#00BCD4"    # Active curve segment (Image 2: 'c-')
ACTIVE_RED = "#D32F2F"     # Active MINVO bounding polygon (Image 2: 'ro-')
NAVY = "#002E5D"           # BYU Navy for section headings
CHARCOAL = "#1D2226"       # Dark neutral for high-contrast labels
GRID_COLOR = "#E2E8F0"     # Clean subtle grid lines


def get_box_faces(x, y, z, dx, dy, dz):
    """Generates 3D polygonal faces and vertex arrays for an axis-aligned box."""
    x1, y1, z1 = x + dx, y + dy, z + dz
    v = np.array([
        [x, y, z], [x1, y, z], [x1, y1, z], [x, y1, z],
        [x, y, z1], [x1, y, z1], [x1, y1, z1], [x, y1, z1]
    ])
    faces = [
        [v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]],
        [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]],
        [v[0], v[3], v[7], v[4]], [v[1], v[2], v[6], v[5]]
    ]
    return faces, v


# ==============================================================================
# FIGURE 1: SAFTE FLIGHT CORRIDORS (SFCS) ADAPTING TO ENVIRONMENT
# ==============================================================================
def create_sfc_adaptation_figure(out_prefix, transparent=False, elev=68, azim=-78):
    """
    Renders Safe Flight Corridors adapting in size and direction to navigate through
    surrounding red obstacle prisms:
    - Open start chamber
    - Narrow constricted corridor between obstacles 1 & 2
    - Corner transition corridor around obstacle 3
    - Expansion corridor past the bottleneck
    - Wide goal chamber
    """
    fig = plt.figure(figsize=(7.5, 6.0), facecolor="none" if transparent else "white", dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("none" if transparent else "white")

    # 1. Obstacle blocks (red/salmon faces with dark red edges, matching Image 1)
    obstacles = [
        # (x, y, z, dx, dy, dz)
        (8.0, 2.0, 0.0, 14.0, 13.0, 11.0),   # Left obstacle at bottleneck
        (8.0, 25.0, 0.0, 14.0, 14.0, 11.0),  # Right obstacle at bottleneck
        (30.0, 16.0, 0.0, 14.0, 23.0, 13.0), # Corner obstacle forcing turn
        (32.0, 2.0, 0.0, 12.0, 8.0, 9.0),    # Side obstacle
        (52.0, 2.0, 0.0, 14.0, 15.0, 11.0),  # Obstacle defining goal approach
    ]

    for obs in obstacles:
        x, y, z, dx, dy, dz = obs
        faces, _ = get_box_faces(x, y, z, dx, dy, dz)
        poly = Poly3DCollection(faces, facecolors=OBS_FACE, edgecolors=OBS_EDGE, linewidths=1.2, alpha=0.25)
        ax.add_collection3d(poly)
        for f in faces:
            cf = np.vstack([f, f[0]])
            ax.plot(cf[:, 0], cf[:, 1], cf[:, 2], color=OBS_EDGE, lw=1.2, alpha=0.85)

    # 2. Safe Flight Corridors (Purple wireframe boxes visibly adapting to free space)
    sfcs = [
        (0.0, 5.0, 0.0, 12.0, 20.0, 7.5),   # SFC 1: Wide start chamber
        (10.0, 15.0, 0.0, 14.0, 10.0, 7.5), # SFC 2: Narrowed to fit constriction
        (22.0, 10.0, 0.0, 14.0, 10.0, 7.5), # SFC 3: Corner transition
        (34.0, 10.0, 0.0, 16.0, 13.0, 7.5), # SFC 4: Expanding out of constriction
        (48.0, 17.0, 0.0, 18.0, 18.0, 7.5), # SFC 5: Wide goal chamber
    ]

    for sfc in sfcs:
        x, y, z, dx, dy, dz = sfc
        faces, _ = get_box_faces(x, y, z, dx, dy, dz)
        poly_sfc = Poly3DCollection(faces, facecolors=SFC_FACE, edgecolors=SFC_COLOR, linewidths=2.3, alpha=0.15)
        ax.add_collection3d(poly_sfc)
        for f in faces:
            cf = np.vstack([f, f[0]])
            ax.plot(cf[:, 0], cf[:, 1], cf[:, 2], color=SFC_COLOR, lw=2.3, alpha=0.95, zorder=5)

    # Start and Goal markers
    start_pt = (2.0, 15.0, 3.8)
    goal_pt = (62.0, 26.0, 3.8)

    ax.scatter(*start_pt, color="#2E7D32", s=100, edgecolors="black", linewidths=1.6, zorder=10)
    ax.text(start_pt[0], start_pt[1] - 4.5, start_pt[2] + 4.5, "Start",
            fontsize=11.5, weight="bold", color="#1B5E20", ha="center", zorder=12)

    ax.scatter(*goal_pt, color="#C62828", s=100, edgecolors="black", linewidths=1.6, zorder=10)
    ax.text(goal_pt[0], goal_pt[1] + 4.5, goal_pt[2] + 4.5, "Goal",
            fontsize=11.5, weight="bold", color="#B71C1C", ha="center", zorder=12)

    # Flight path centerline smoothly threading through the adapting SFCs
    path_pts = np.array([
        start_pt,
        (12.0, 17.0, 3.8),
        (21.0, 18.0, 3.8),
        (32.0, 15.0, 3.8),
        (44.0, 16.5, 3.8),
        goal_pt
    ])
    ax.plot(path_pts[:, 0], path_pts[:, 1], path_pts[:, 2], "--", color=SPLINE_BLUE,
            lw=2.2, dashes=(4, 3), alpha=0.85, label="Trajectory Centerline", zorder=6)

    # Ground reference grid
    for gx in np.arange(0, 71, 10):
        ax.plot([gx, gx], [0, 42], [-0.05, -0.05], color=GRID_COLOR, lw=0.7)
    for gy in np.arange(0, 45, 10):
        ax.plot([0, 68], [gy, gy], [-0.05, -0.05], color=GRID_COLOR, lw=0.7)

    # Formatting and Camera
    ax.set_title("Safe Flight Corridors (SFCs)\nAdapting to Surrounding Obstacles", pad=12, color=NAVY, fontsize=13.5)
    ax.set_xlim(-2, 70)
    ax.set_ylim(-2, 42)
    ax.set_zlim(0, 14)

    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect([1.8, 1.1, 0.45])

    ax.set_xlabel("X Position (m)", labelpad=4, color=CHARCOAL)
    ax.set_ylabel("Y Position (m)", labelpad=4, color=CHARCOAL)
    ax.set_zticks([])
    ax.set_zlabel("")

    # Clean transparent panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("none")
    ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.set_edgecolor("none")

    legend_elements = [
        plt.Line2D([0], [0], color=OBS_EDGE, lw=2, label="Obstacles"),
        plt.Line2D([0], [0], color=SFC_COLOR, lw=2.3, label="SFC Prisms (Adapting)"),
        plt.Line2D([0], [0], color=SPLINE_BLUE, lw=2, linestyle="--", label="Trajectory Centerline")
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.92, edgecolor=GRID_COLOR, fontsize=10.0)

    fig.tight_layout()
    fig.savefig(f"{out_prefix}.png", dpi=300, bbox_inches="tight", transparent=transparent)
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved SFC Adaptation figure to: {out_prefix}.png (.pdf, .svg)")


# ==============================================================================
# FIGURE 2: 4TH-DEGREE B-SPLINE CURVE (CONTROL POINTS DICTATING SHAPE)
# ==============================================================================
def create_bspline_shape_figure(out_prefix, transparent=False):
    """
    Renders 4th-degree B-Spline curve in the exact visual style of minvo_sliding_window_demo:
    - Global control points and polygon in dark orange ('o:', color='darkorange').
    - Blue smooth 4th-degree B-Spline curve ('b-').
    - Active B-Spline points and segment ('o-', color='darkorange', 'c-' active curve).
    - Active MINVO bounding polygon in red ('ro-').
    - Uncluttered lower-left legend ensuring zero overlap with geometry and labels.
    """
    fig, ax = plt.subplots(figsize=(7.5, 6.0), facecolor="none" if transparent else "white", dpi=300)
    ax.set_facecolor("white" if not transparent else "none")

    # Control points (matching minvo_sliding_window_demo.py setup)
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
    num_segments = num_pts - degree  # 7 - 4 = 3 segments

    # Clamped uniform knot vector
    knots = np.zeros(num_pts + degree + 1)
    knots[degree : num_pts + 1] = np.arange(num_segments + 1)
    knots[num_pts + 1 :] = num_segments
    knots[:degree] = 0

    spline = BSpline(knots, C_p, degree)

    # 1. Full B-Spline Curve (Blue line, matching Image 2: 'b-')
    t_eval = np.linspace(0, num_segments, 500)
    curve = spline(t_eval)
    ax.plot(curve[:, 0], curve[:, 1], '-', color=SPLINE_BLUE, linewidth=2.8,
            label="4th-Degree B-Spline Curve", zorder=3)

    # 2. Global Control Polygon & Control Points (Dark orange dotted line with circular markers, Image 2)
    ax.plot(C_p[:, 0], C_p[:, 1], "o:", color=CTRL_ORANGE, alpha=0.9, linewidth=4.2,
            markersize=8, markeredgecolor="black", markeredgewidth=1.2,
            label="Control Polygon (Dictates Shape)", zorder=2)

    # 3. Active Segment (Segment s = 0: t in [0, 1], cyan line, Image 2)
    s = 0
    t_seg = np.linspace(s, s + 1, 100)
    seg_curve = spline(t_seg)
    ax.plot(seg_curve[:, 0], seg_curve[:, 1], '-', color=ACTIVE_CYAN, linewidth=4.5,
            label="Active Curve Segment", zorder=4)

    # 4. Active MINVO Bounding Polytope (red line with markers, Image 2)
    C_local = C_p[s : s + degree + 1]
    if MINVO_CLAMPED_STENCILS is not None:
        stencils = MINVO_CLAMPED_STENCILS[degree]
        F = stencils[f"start_{s}"]
        V_local = F @ C_local
        V_closed = np.vstack((V_local, V_local[0]))
        ax.plot(V_closed[:, 0], V_closed[:, 1], "ro-", linewidth=2.0, markersize=7,
                markeredgecolor="darkred", label="Active Bounding Polytope (MINVO)", zorder=5)

    # Non-overlapping control point labels C_0 .. C_6
    label_pos = [
        (C_p[0, 0] - 0.75, C_p[0, 1] - 0.4),  # C_0
        (C_p[1, 0] - 0.85, C_p[1, 1] + 0.1),  # C_1
        (C_p[2, 0] - 0.1, C_p[2, 1] + 0.45),  # C_2: cleanly above peak
        (C_p[3, 0] + 0.35, C_p[3, 1] + 0.3),  # C_3: above-right
        (C_p[4, 0] + 0.45, C_p[4, 1] + 0.15), # C_4: right
        (C_p[5, 0] + 0.55, C_p[5, 1] - 0.1),  # C_5: right
        (C_p[6, 0] + 0.45, C_p[6, 1] - 0.2)   # C_6: right
    ]
    for i, (lx, ly) in enumerate(label_pos):
        ax.text(lx, ly, rf"$\mathbf{{C_{{{i}}}}}$",
                fontsize=12, weight="bold", color=CHARCOAL, zorder=10)

    # Subtle grid and clean frame
    ax.grid(True, color=GRID_COLOR, linestyle="--", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)

    ax.set_title("4th-Degree B-Spline Curve\nControl Points Dictate Trajectory Shape",
                 pad=12, color=NAVY, fontsize=13.5)
    ax.set_xlabel("X Position (m)", color=CHARCOAL)
    ax.set_ylabel("Y Position (m)", color=CHARCOAL)
    ax.set_aspect("equal", "box")
    ax.set_xlim(-2.0, 15.0)
    ax.set_ylim(-7.0, 9.8)

    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
        spine.set_linewidth(1.2)

    # Legend located in the lower-left corner where it never touches any curve or point
    ax.legend(loc="lower left", framealpha=0.95, edgecolor=GRID_COLOR, fontsize=10.0)

    fig.tight_layout()
    fig.savefig(f"{out_prefix}.png", dpi=300, bbox_inches="tight", transparent=transparent)
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved B-Spline Shape figure to: {out_prefix}.png (.pdf, .svg)")


# ==============================================================================
# COMBINED FIGURE: BOTH FIGURES SIDE-BY-SIDE (FOR POSTER BOTTOM-LEFT BOX)
# ==============================================================================
def create_combined_poster_figure(out_prefix, transparent=False, elev=68, azim=-78):
    """
    Renders both figures side-by-side with matched styles, ready for direct
    insertion into the bottom-left poster box ("COAST PLANNER: SPEED AND SAFETY").
    """
    fig = plt.figure(figsize=(14.2, 5.6), facecolor="none" if transparent else "white", dpi=300)

    # ------------------ (a) SFC Adaptation (3D) ------------------
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.set_facecolor("none" if transparent else "white")

    obstacles = [
        (8.0, 2.0, 0.0, 14.0, 13.0, 11.0),
        (8.0, 25.0, 0.0, 14.0, 14.0, 11.0),
        (30.0, 16.0, 0.0, 14.0, 23.0, 13.0),
        (32.0, 2.0, 0.0, 12.0, 8.0, 9.0),
        (52.0, 2.0, 0.0, 14.0, 15.0, 11.0),
    ]

    for obs in obstacles:
        x, y, z, dx, dy, dz = obs
        faces, _ = get_box_faces(x, y, z, dx, dy, dz)
        poly = Poly3DCollection(faces, facecolors=OBS_FACE, edgecolors=OBS_EDGE, linewidths=1.2, alpha=0.25)
        ax1.add_collection3d(poly)
        for f in faces:
            cf = np.vstack([f, f[0]])
            ax1.plot(cf[:, 0], cf[:, 1], cf[:, 2], color=OBS_EDGE, lw=1.2, alpha=0.85)

    sfcs = [
        (0.0, 5.0, 0.0, 12.0, 20.0, 7.5),
        (10.0, 15.0, 0.0, 14.0, 10.0, 7.5),
        (22.0, 10.0, 0.0, 14.0, 10.0, 7.5),
        (34.0, 10.0, 0.0, 16.0, 13.0, 7.5),
        (48.0, 17.0, 0.0, 18.0, 18.0, 7.5),
    ]

    for sfc in sfcs:
        x, y, z, dx, dy, dz = sfc
        faces, _ = get_box_faces(x, y, z, dx, dy, dz)
        poly_sfc = Poly3DCollection(faces, facecolors=SFC_FACE, edgecolors=SFC_COLOR, linewidths=2.2, alpha=0.15)
        ax1.add_collection3d(poly_sfc)
        for f in faces:
            cf = np.vstack([f, f[0]])
            ax1.plot(cf[:, 0], cf[:, 1], cf[:, 2], color=SFC_COLOR, lw=2.2, alpha=0.95)

    start_pt = (2.0, 15.0, 3.8)
    goal_pt = (62.0, 26.0, 3.8)
    ax1.scatter(*start_pt, color="#2E7D32", s=90, edgecolors="black", linewidths=1.6, zorder=10)
    ax1.text(start_pt[0], start_pt[1] - 4.5, start_pt[2] + 4.5, "Start",
             fontsize=11, weight="bold", color="#1B5E20", ha="center")
    ax1.scatter(*goal_pt, color="#C62828", s=90, edgecolors="black", linewidths=1.6, zorder=10)
    ax1.text(goal_pt[0], goal_pt[1] + 4.5, goal_pt[2] + 4.5, "Goal",
             fontsize=11, weight="bold", color="#B71C1C", ha="center")

    path_pts = np.array([
        start_pt,
        (12.0, 17.0, 3.8),
        (21.0, 18.0, 3.8),
        (32.0, 15.0, 3.8),
        (44.0, 16.5, 3.8),
        goal_pt
    ])
    ax1.plot(path_pts[:, 0], path_pts[:, 1], path_pts[:, 2], "--", color=SPLINE_BLUE,
             lw=2.0, dashes=(4, 3), alpha=0.85)

    for gx in np.arange(0, 71, 10):
        ax1.plot([gx, gx], [0, 42], [-0.05, -0.05], color=GRID_COLOR, lw=0.6)
    for gy in np.arange(0, 45, 10):
        ax1.plot([0, 68], [gy, gy], [-0.05, -0.05], color=GRID_COLOR, lw=0.6)

    ax1.set_title("(a) Safe Flight Corridors (SFCs)\nAdapting to Surrounding Obstacles", pad=12, color=NAVY, fontsize=13)
    ax1.set_xlim(-2, 70)
    ax1.set_ylim(-2, 42)
    ax1.set_zlim(0, 14)
    ax1.view_init(elev=elev, azim=azim)
    ax1.set_box_aspect([1.8, 1.1, 0.45])

    ax1.set_xlabel("X Position (m)", labelpad=4, color=CHARCOAL, fontsize=10.5)
    ax1.set_ylabel("Y Position (m)", labelpad=4, color=CHARCOAL, fontsize=10.5)
    ax1.set_zticks([])
    ax1.set_zlabel("")
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.xaxis.pane.set_edgecolor("none")
    ax1.yaxis.pane.set_edgecolor("none")
    ax1.zaxis.pane.set_edgecolor("none")

    legend_elements1 = [
        plt.Line2D([0], [0], color=OBS_EDGE, lw=2, label="Obstacles"),
        plt.Line2D([0], [0], color=SFC_COLOR, lw=2.2, label="SFC Prisms (Adapting)"),
        plt.Line2D([0], [0], color=SPLINE_BLUE, lw=1.8, linestyle="--", label="Trajectory Centerline")
    ]
    ax1.legend(handles=legend_elements1, loc="upper right", framealpha=0.92, edgecolor=GRID_COLOR, fontsize=9.2)

    # ------------------ (b) B-Spline Control Points ------------------
    ax2 = fig.add_subplot(122)
    ax2.set_facecolor("white" if not transparent else "none")

    C_p = np.array([
        [0.0, 0.0], [1.0, 5.0], [3.0, 8.0], [6.0, 7.0],
        [8.0, 2.0], [10.0, -3.0], [13.0, -5.0]
    ])
    degree = 4
    num_pts = len(C_p)
    num_segments = num_pts - degree

    knots = np.zeros(num_pts + degree + 1)
    knots[degree : num_pts + 1] = np.arange(num_segments + 1)
    knots[num_pts + 1 :] = num_segments
    knots[:degree] = 0

    spline = BSpline(knots, C_p, degree)

    t_eval = np.linspace(0, num_segments, 500)
    curve = spline(t_eval)
    ax2.plot(curve[:, 0], curve[:, 1], '-', color=SPLINE_BLUE, linewidth=2.6,
             label="4th-Degree B-Spline Curve", zorder=3)
    ax2.plot(C_p[:, 0], C_p[:, 1], "o:", color=CTRL_ORANGE, alpha=0.9, linewidth=4.0,
             markersize=7.5, markeredgecolor="black", markeredgewidth=1.2,
             label="Control Polygon (Dictates Shape)", zorder=2)

    s = 0
    t_seg = np.linspace(s, s + 1, 100)
    seg_curve = spline(t_seg)
    ax2.plot(seg_curve[:, 0], seg_curve[:, 1], '-', color=ACTIVE_CYAN, linewidth=4.0,
             label="Active Curve Segment", zorder=4)

    C_local = C_p[s : s + degree + 1]
    if MINVO_CLAMPED_STENCILS is not None:
        stencils = MINVO_CLAMPED_STENCILS[degree]
        F = stencils[f"start_{s}"]
        V_local = F @ C_local
        V_closed = np.vstack((V_local, V_local[0]))
        ax2.plot(V_closed[:, 0], V_closed[:, 1], "ro-", linewidth=1.8, markersize=6.5,
                 markeredgecolor="darkred", label="Active Bounding Polytope (MINVO)", zorder=5)

    label_pos = [
        (C_p[0, 0] - 0.75, C_p[0, 1] - 0.4),
        (C_p[1, 0] - 0.85, C_p[1, 1] + 0.1),
        (C_p[2, 0] - 0.1, C_p[2, 1] + 0.45),
        (C_p[3, 0] + 0.35, C_p[3, 1] + 0.3),
        (C_p[4, 0] + 0.45, C_p[4, 1] + 0.15),
        (C_p[5, 0] + 0.55, C_p[5, 1] - 0.1),
        (C_p[6, 0] + 0.45, C_p[6, 1] - 0.2)
    ]
    for i, (lx, ly) in enumerate(label_pos):
        ax2.text(lx, ly, rf"$\mathbf{{C_{{{i}}}}}$", fontsize=11, weight="bold", color=CHARCOAL, zorder=10)

    ax2.grid(True, color=GRID_COLOR, linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.set_axisbelow(True)
    ax2.set_title("(b) 4th-Degree B-Spline Curve\nControl Points Dictate Trajectory Shape",
                  pad=12, color=NAVY, fontsize=13)
    ax2.set_xlabel("X Position (m)", color=CHARCOAL, fontsize=10.5)
    ax2.set_ylabel("Y Position (m)", color=CHARCOAL, fontsize=10.5)
    ax2.set_aspect("equal", "box")
    ax2.set_xlim(-2.0, 15.0)
    ax2.set_ylim(-7.0, 9.8)

    for spine in ax2.spines.values():
        spine.set_color(GRID_COLOR)
        spine.set_linewidth(1.2)

    ax2.legend(loc="lower left", framealpha=0.95, edgecolor=GRID_COLOR, fontsize=9.2)

    fig.tight_layout()
    fig.savefig(f"{out_prefix}.png", dpi=300, bbox_inches="tight", transparent=transparent)
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Combined figure to: {out_prefix}.png (.pdf, .svg)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate COAST poster figures.")
    parser.add_argument("--outdir", default=None, help="Output directory")
    args = parser.parse_args()

    out_dir = args.outdir or os.path.join(REPO_ROOT, "python", "poster_figures")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating updated figures in user reference style into: {out_dir}")
    
    # Standard white background
    create_sfc_adaptation_figure(os.path.join(out_dir, "figure_sfc_adapting"))
    create_bspline_shape_figure(os.path.join(out_dir, "figure_bspline_control_points"))
    create_combined_poster_figure(os.path.join(out_dir, "figure_coast_bottom_left_combined"))

    # Transparent background variants
    create_sfc_adaptation_figure(os.path.join(out_dir, "figure_sfc_adapting_transparent"), transparent=True)
    create_bspline_shape_figure(os.path.join(out_dir, "figure_bspline_control_points_transparent"), transparent=True)
    create_combined_poster_figure(os.path.join(out_dir, "figure_coast_bottom_left_combined_transparent"), transparent=True)

    print("All poster figures generated successfully!")
