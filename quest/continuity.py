"""Guards that keep Quest teleop continuous: no wrist flips, no lunges on a tracking glitch. Kit-independent.

Found in the station's Quest recordings (unplug-v1, unplug-split-v1, kit teleop at 30 Hz, 1.5 rad/s):

- **Wrist flips.** With joint5 resting on its limit (-82 deg, the IK's margin), a small hand turn the wrist cannot
  make sends the differential IK the long way round: joint4 and joint6 counter-rotate 130-190 deg at the speed cap
  for 1.5-2.5 s while the tool turns only 8-40 deg. Over half a second of large wrist motion, a human driving the
  leader arm gets at least 0.48 deg of tool turn per deg of wrist travel (5th percentile, 58 windows); the Quest
  episodes median 0.11-0.23, and 59-88 % of their windows are under 0.3. WristFlipGuard spots that self-motion and
  holds the tool's orientation from there (position keeps following), so the wrist stops instead of whirling.
- **Tracking glitches.** A controller pose that jumps further in one reading than a hand can move is a tracking
  glitch, not a motion. TrackingJumpGuard flags it, and the clutch re-anchors at the new pose instead of chasing it.
"""

import collections
import math

import numpy as np
import numpy.typing as npt


def rotation_angle_deg(rotation: npt.NDArray[np.float64]) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(rotation)) - 1.0) / 2.0))))


class WristFlipGuard:
    """Detects wrist self-motion (wrist joints turning a lot, tool barely turning) and re-anchors the orientation.

    The guard owns an orientation offset applied to every target: `target_used = offset @ target` (base frame).
    On a flip, the offset is set so the target's orientation equals the tool's orientation now; from then on the
    hand's further turns act from there. reset() (at every engage) clears it.

    @param window_s The look-back window.
    @param min_travel_deg Wrist travel (sum of |step| over the wrist joints) in the window that can count as a flip.
    @param ratio A flip is tool turn < ratio * wrist travel over the window.
    @param wrist Indices of the wrist joints in the six.
    """

    def __init__(self, window_s: float = 0.3, min_travel_deg: float = 15.0, ratio: float = 0.3,
                 wrist: tuple[int, ...] = (3, 4, 5)) -> None:
        self.window_s, self.min_travel_deg, self.ratio, self.wrist = window_s, min_travel_deg, ratio, list(wrist)
        self.reset()

    def reset(self) -> None:
        self.offset = np.eye(3)
        self._window: collections.deque[tuple[float, float]] = collections.deque()
        self.rebases = 0

    def apply(self, target: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        out = np.array(target, dtype=np.float64, copy=True)
        out[:3, :3] = self.offset @ out[:3, :3]
        return out

    def observe(self, joints_before: npt.ArrayLike, joints_after: npt.ArrayLike,
                tool_before: npt.NDArray[np.float64], tool_after: npt.NDArray[np.float64], dt_s: float) -> bool:
        """Adds one tick's step; True when the window shows a flip (the caller then calls rebase())."""
        step = np.abs(np.asarray(joints_after, float) - np.asarray(joints_before, float))[self.wrist]
        self._window.append((math.degrees(float(step.sum())),
                             rotation_angle_deg(tool_after[:3, :3] @ tool_before[:3, :3].T)))
        while len(self._window) > max(1, round(self.window_s / dt_s)):
            self._window.popleft()
        travel = sum(w for w, _ in self._window)
        turned = sum(r for _, r in self._window)
        return travel > self.min_travel_deg and turned < self.ratio * travel

    def rebase(self, tool_now: npt.NDArray[np.float64], target: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Holds the tool's current orientation: returns the target to solve instead, with the new offset applied."""
        self.offset = tool_now[:3, :3] @ np.asarray(target, float)[:3, :3].T
        self._window.clear()
        self.rebases += 1
        return self.apply(target)


class TrackingJumpGuard:
    """Flags a controller reading that moved further since the last one than a hand can (a tracking glitch).

    @param max_jump_m Position change in one reading that counts as a glitch.
    @param max_jump_deg Orientation change in one reading that counts as a glitch.
    """

    def __init__(self, max_jump_m: float = 0.08, max_jump_deg: float = 45.0) -> None:
        self.max_jump_m, self.max_jump_deg = max_jump_m, max_jump_deg
        self._last: npt.NDArray[np.float64] | None = None
        self.glitches = 0

    def reset(self) -> None:
        self._last = None

    def is_glitch(self, pose: npt.NDArray[np.float64] | None) -> bool:
        if pose is None:
            self._last = None
            return False
        last, self._last = self._last, np.array(pose, dtype=np.float64, copy=True)
        if last is None:
            return False
        jumped = (np.linalg.norm(pose[:3, 3] - last[:3, 3]) > self.max_jump_m
                  or rotation_angle_deg(pose[:3, :3] @ last[:3, :3].T) > self.max_jump_deg)
        self.glitches += int(jumped)
        return bool(jumped)


def per_joint_caps(base_rad_s: float, wrist_rad_s: float | None, n: int = 6,
                   wrist: tuple[int, ...] = (3, 4, 5)) -> npt.NDArray[np.float64]:
    """Joint speed caps: `base_rad_s` everywhere, `wrist_rad_s` (if given) on the wrist joints.

    The wrist sat on the kit's 1.5 rad/s cap in 30-69 % of the frames of the unplug-split episodes, so hand turns
    lagged; the Metal's wrist motors are rated 20 rad/s in its URDF, and MakerMods' follower cap is 280 deg/s.
    """
    caps = np.full(n, float(base_rad_s))
    if wrist_rad_s is not None:
        caps[list(wrist)] = float(wrist_rad_s)
    return caps
