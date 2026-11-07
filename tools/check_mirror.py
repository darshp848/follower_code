"""Why leader_follower.py offers joint copy only: search all 2^6 joint-sign maps for one that makes a YAM reproduce
the mirror image (about the base xz- or yz-plane) of another YAM's pose. Prints the residual of the best map per
plane; an exact mirror would give ~0 m.

    python tools/check_mirror.py
"""
import itertools

import mujoco as mj
import numpy as np
from i2rt.robots.get_robot import ArmType, GripperType, combine_arm_and_gripper_xml

N_ARM, N_POSES = 6, 6

m = mj.MjModel.from_xml_path(str(combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310)))
a, b = mj.MjData(m), mj.MjData(m)
links = [m.body(f'link{i}').id for i in range(1, 7)] + [m.body('gripper').id]
rng = np.random.default_rng(0)
poses = [rng.uniform(-0.6, 0.6, N_ARM) for _ in range(N_POSES)]

for plane, flip in (('xz (left/right)', [1, -1, 1]), ('yz (front/back)', [-1, 1, 1])):
    best = None
    for signs in itertools.product((1., -1.), repeat=N_ARM):
        err = 0.
        for q in poses:
            a.qpos[:], b.qpos[:] = 0, 0
            a.qpos[:N_ARM], b.qpos[:N_ARM] = q, np.array(signs) * q
            mj.mj_kinematics(m, a)
            mj.mj_kinematics(m, b)
            err += np.linalg.norm(a.xpos[links] * flip - b.xpos[links], axis=1).mean()
        if best is None or err < best[0]:
            best = (err, signs)
    print(f'{plane}: best signs {best[1]}, mean link error {best[0] / N_POSES * 100:.1f} cm')
