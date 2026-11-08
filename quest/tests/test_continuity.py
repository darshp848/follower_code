"""continuity.py on its own: numpy only, no kit, no Quest.

    python -m pytest -q quest/tests/test_continuity.py
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from continuity import TrackingJumpGuard, WristFlipGuard, per_joint_caps, rotation_angle_deg  # noqa: E402


def _rz(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    pose = np.eye(4)
    pose[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return pose


def _drive(guard, wrist_step_deg, tool_step_deg, ticks, dt=1 / 60):
    q, tool, flagged = np.zeros(6), np.eye(4), []
    for _ in range(ticks):
        q2 = q.copy()
        q2[3] += math.radians(wrist_step_deg)
        q2[5] -= math.radians(wrist_step_deg)          # joint4 and joint6 counter-rotate, as in the recordings
        tool2 = _rz(tool_step_deg) @ tool
        flagged.append(guard.observe(q, q2, tool, tool2, dt))
        q, tool = q2, tool2
    return flagged


def test_flags_a_counter_rotating_wrist_that_barely_turns_the_tool():
    # unplug-split-v1 episode 0 at 3.3 s: joint4 -166 deg and joint6 -135 deg in 1.9 s, the tool turned 15 deg.
    assert any(_drive(WristFlipGuard(), wrist_step_deg=1.4, tool_step_deg=0.13, ticks=30))


def test_leaves_a_real_wrist_turn_alone():
    # A human on the leader arm: at least ~0.5 deg of tool turn per deg of wrist travel.
    assert not any(_drive(WristFlipGuard(), wrist_step_deg=1.4, tool_step_deg=1.6, ticks=120))


def test_small_motions_never_count():
    assert not any(_drive(WristFlipGuard(), wrist_step_deg=0.1, tool_step_deg=0.0, ticks=120))


def test_rebase_holds_the_tools_orientation_and_reset_clears_it():
    guard = WristFlipGuard()
    tool, hand = _rz(30), _rz(80)
    held = guard.rebase(tool, hand)
    assert rotation_angle_deg(held[:3, :3] @ tool[:3, :3].T) < 1e-9
    later = guard.apply(_rz(90))                      # the hand turns 10 deg more: the tool target turns 10 deg
    assert abs(rotation_angle_deg(later[:3, :3] @ tool[:3, :3].T) - 10.0) < 1e-6
    guard.reset()
    assert np.allclose(guard.apply(hand), hand)


def test_jump_guard_flags_glitches_not_moves():
    guard = TrackingJumpGuard()
    pose = np.eye(4)
    assert not guard.is_glitch(pose)
    moved = pose.copy()
    moved[0, 3] = 0.03                                # 3 cm in one reading: a fast hand at 30-60 Hz
    assert not guard.is_glitch(moved)
    jumped = moved.copy()
    jumped[0, 3] = 0.30                               # 27 cm in one reading: tracking glitch
    assert guard.is_glitch(jumped)
    assert guard.is_glitch(_rz(60) @ jumped)          # 60 deg in one reading
    assert not guard.is_glitch(None) and not guard.is_glitch(pose)   # lost tracking restarts the comparison


def test_per_joint_caps():
    np.testing.assert_allclose(per_joint_caps(2.5, 4.0), [2.5, 2.5, 2.5, 4.0, 4.0, 4.0])
    np.testing.assert_allclose(per_joint_caps(2.5, None), [2.5] * 6)
