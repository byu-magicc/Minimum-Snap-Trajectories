"""Visualise an unconstrained solve and an obstacle-avoiding plan.

This demo is part of the package: after `pip install "coast-planner[viz]"`, run it anywhere with

    coast-planner-demo              # opens windows
    coast-planner-demo --save out   # writes PNGs to ./out instead

This script runs the same demo from a clone of the repository. The code is in
python/coast_planner/_demo.py.
"""

from coast_planner._demo import main

if __name__ == "__main__":
    main()
