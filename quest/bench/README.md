# Replay bench (offline, research)

Scripts behind the numbers in the README. They read the LeRobot datasets copied off the NUC (paths at the top of
bench_ik.py and resp.py) and need numpy, pandas, pyarrow>=21, mujoco and mink 1.1.0. `final.py` prints the
before/after table. `ratio.py` measures tool turn per wrist travel (human leader arm vs Quest IK). `guard.py` is the
prototype of continuity.WristFlipGuard.
