---
# JOSS paper skeleton (#9). Everything marked TODO needs the authors; see #10, #11 and #12.
# Before filling in, check the section headings and length guidance against JOSS's current
# example paper: https://joss.readthedocs.io/en/latest/example_paper.html
title: 'TODO: title, e.g. "min_snap: real-time minimum-snap B-spline trajectories through Safe Flight Corridors"'
tags:
  - C++
  - Python
  - robotics
  - trajectory optimization
  - motion planning
  - quadrotors
  - B-splines
authors:
  - name: Alejandro Pelaez
    orcid: 0009-0009-0848-4365   
    affiliation: 1
    corresponding: true
  - name: Brady Moon        
    orcid: 0000-0003-4297-3938   
    affiliation: 2
  - name: Randal Beard
    orcid: 0000-0002-2548-5329  
    affiliation: 1
affiliations:
  - name: MAGICC Lab, Brigham Young University, United States
    index: 1
  - name: FROST Lab, Brigham Young University, United States
    index: 2
date: TODO 1 January 2027
bibliography: paper.bib
---

# Summary

<!-- TODO (#10). What the software does, for a non-specialist reader. Suggested points:
     a planner that turns a start state, a goal and LiDAR returns into a smooth, collision-free
     B-spline trajectory fast enough to re-plan at 10 Hz on board; C++17 core with Python
     bindings; no ROS dependency. -->

# Statement of need

<!-- TODO (#10). Who needs this and what gap it fills: e.g. researchers building multirotor
     autonomy in cluttered, partially-unknown environments who want minimum-snap quality
     trajectories with hard corridor constraints, without mixed-integer programming, as a
     library they can embed (C++) or script (Python). -->

# State of the field

<!-- TODO (#10). Compare with existing tools and say what is different. Candidates already in
     paper.bib: minimum-snap polynomial generation [@mellinger2011minimum], SFC-based planning
     [@liu2017planning], FASTER [@tordesillas2022faster], EGO-Planner [@zhou2021ego];
     add SANDO and any others. -->

# Software design

<!-- TODO (#10). Suggested structure, matching the README's "How it works":
     1. Three-state (FREE / OCCUPIED / UNKNOWN) sparse voxel map, raycast with
        Amanatides–Woo [@amanatides1987fast], accumulated across scans.
     2. A* front end (unknown space traversable at a cost) -> smoothed waypoints -> Safe
        Flight Corridors.
     3. Minimum-snap QP solved with OSQP [@stellato2020osqp], with MINVO-based
        [@tordesillas2022minvo] kinodynamic bounds and corridor containment via exclusive and
        bridge control-point pools.
     4. Closed-form unconstrained solvers (C = SE * Q) for clamped and natural B-splines.
     5. Python bindings via pybind11 [@pybind11], Eigen [@eigenweb], nanoflann [@nanoflann];
        self-contained pip package; CI on Linux, macOS and Windows. -->

# Research impact

<!-- TODO (#12). Evidence of real research use (JOSS rejects "aspirational" claims): the ACC
     paper / arXiv preprint, the roscopter integration and flights or simulations run with it,
     theses, other groups using it. -->

# AI usage disclosure

<!-- TODO (#11). MANDATORY for JOSS. Must state (1) which AI tools and versions were used and
     exactly where, (2) the scope of assistance, (3) that the authors reviewed, modified and
     validated all AI-generated content and made the primary design decisions.

     Starting point (factual list of work done with Claude, an AI assistant by Anthropic;
     the authors must complete and verify it):
       - Python bindings and pip packaging (PR #2)
       - bundled Eigen/OSQP/GoogleTest, MSVC fixes, cross-platform CI (PR #7)
       - README overhaul (PR #8), repository tidy (PR #22)
       - JOSS paper and citation skeletons (this file's structure, CITATION.cff)
       - planner hardening commit c04994b (A* front end, SFC chain, voxel map)
       - TODO: any other AI-assisted code, documentation or paper text -->

# Acknowledgements

<!-- TODO (#10). Funding sources (and whether the sponsor had any role), people who helped,
     and a conflict-of-interest statement if needed. -->

# References
