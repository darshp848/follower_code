"""The hackathon kit's `dhk-collect` with better Quest teleop: heading calibration, filtering, a wrist lock, faster IK.

Runs the kit's own session (github.com/dreamlabs-co/physical-ai-hack-2026, Apache-2.0): its Quest reader, cameras,
dashboard, device checks, station lock, homing, lag guard and LeRobot v2.1 writer. Only three pieces are swapped,
for this process only; none of the kit's files change:

- **The clutch** (wrong direction, wrist goes wild). The kit maps the Quest's frame to each arm's base with a fixed
  `quest_to_base`, but the Quest's frame turns with wherever the headset faced when tracking started, so the arm
  moves rotated or mirrored, and hand turns land on the wrong axes. Press **Y**, move the right controller about
  20 cm in the arm's forward direction (the way the gripper points when the arm reaches out), press **Y** again: the
  heading is measured and saved (~/.cache/follower_code/quest_heading.json) for later sessions. Hand turns are
  scaled (`--rotation-scale`) after a small dead band (`--rotation-deadband-deg`) that drops wobble. **X** locks the
  wrist: the tool keeps its orientation and only follows the hand's motion; X again unlocks.
- **A One Euro filter** on each controller's pose (jittery): smooth when the hand is slow or still, little lag when
  it moves fast. It restarts at every engage, so engaging never jumps.
- **The IK** (laggy, jittery): the loop runs at `--control-hz` (60, against the kit's 30), the joint speed cap is
  `--max-joint-speed` (2.5 rad/s, against 1.5), and the solver's damping is raised (`--ik-damping`), which stops the
  joints chattering near singular poses.
- **Gravity feedforward** (Metal only, on by default, `--no-gravity-ff` turns it off): fills each joint's gravity
  torque into the follower's MIT frames, so the arm stops sitting under its target (shoulder_lift 4 to 6 deg low
  without it). It is metal/leader_collect.py's FollowerGravity, MakerMods' own Metal model, which took the
  leader/follower episodes' shoulder_lift bias from +4.5 to +0.2 deg on the station's arms.

    cd ~/physical-ai-hacks
    .venv/bin/python ~/follower_code/quest/quest_collect.py --config configs/station.yaml --dry-run
    .venv/bin/python ~/follower_code/quest/quest_collect.py --config configs/station.yaml --dataset-dir ~/hackathon_data/x

Buttons: the kit's (grip drives an arm, trigger closes its gripper, A record/save, B discard, right stick click home)
plus X (wrist lock) and Y (heading calibration, twice). Check the dashboard's loop rate reads ~60 Hz; if it cannot,
pass `--control-hz 30`: a loop slower than its rate stretches the recording's time.
"""

import argparse
import dataclasses
import json
import math
import pathlib
import sys
from typing import Any

import mink
import numpy as np
import numpy.typing as npt

from dream_hackathon import arms, collect, config, ik, quest, teleop

HEADING_FILE = pathlib.Path("~/.cache/follower_code/quest_heading.json").expanduser()
## The calibration move must be at least this long, horizontally, to count.
MIN_CALIBRATION_M = 0.10
## The Quest's up axis (its frame is gravity-aligned, y up): a heading error is a turn about it.
QUEST_UP = np.array([0.0, 1.0, 0.0])


# ------------------------------------------------------------------------------------------------ rotations
def yaw(theta: float) -> npt.NDArray[np.float64]:
    """Rotation by `theta` about the Quest's up axis (y), right-handed."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def axis_angle(rotation: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], float]:
    """The unit axis and angle (rad, 0..pi) of a rotation matrix."""
    angle = math.acos(max(-1.0, min(1.0, (np.trace(rotation) - 1.0) / 2.0)))
    if angle < 1e-9:
        return np.array([1.0, 0.0, 0.0]), 0.0
    if math.pi - angle < 1e-6:                        # 180 deg: the axis from the symmetric part
        axis = np.sqrt(np.maximum(np.diag(rotation) + 1.0, 0.0) / 2.0)
        return axis / np.linalg.norm(axis), angle
    axis = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]])
    return axis / (2.0 * math.sin(angle)), angle


def rotation_about(axis: npt.NDArray[np.float64], angle: float) -> npt.NDArray[np.float64]:
    """Rodrigues: rotation by `angle` about the unit `axis`."""
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * k @ k


def heading_from_move(move_quest: npt.NDArray[np.float64], quest_to_base: npt.NDArray[np.float64]) -> float:
    """The turn about the Quest's up axis that makes a hand move `move_quest` (Quest frame) map to the arm's +x.

    @throws ValueError The move is too short, horizontally, to measure a heading from.
    """
    flat = np.asarray(move_quest, float) - QUEST_UP * float(np.dot(move_quest, QUEST_UP))
    if np.linalg.norm(flat) < MIN_CALIBRATION_M:
        raise ValueError(f"the move was {np.linalg.norm(flat) * 100:.0f} cm across; make it at least "
                         f"{MIN_CALIBRATION_M * 100:.0f} cm")
    want = quest_to_base.T @ np.array([1.0, 0.0, 0.0])   # the Quest direction the config maps to the arm's +x
    want = want - QUEST_UP * float(np.dot(want, QUEST_UP))
    a, b = flat / np.linalg.norm(flat), want / np.linalg.norm(want)
    return math.atan2(float(np.dot(np.cross(a, b), QUEST_UP)), float(np.dot(a, b)))


def load_heading(path: pathlib.Path = HEADING_FILE) -> float:
    try:
        return float(json.loads(path.read_text())["heading_rad"])
    except (OSError, ValueError, KeyError):
        return 0.0


def save_heading(theta: float, path: pathlib.Path = HEADING_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"heading_rad": theta, "heading_deg": math.degrees(theta)}))


# ------------------------------------------------------------------------------------------------ filtering
class OneEuro:
    """Casiez et al.'s One Euro filter on a vector: cutoff = min_cutoff + beta * |speed|."""

    def __init__(self, min_cutoff_hz: float, beta: float, d_cutoff_hz: float = 1.0) -> None:
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff_hz, beta, d_cutoff_hz
        self.value: npt.NDArray[np.float64] | None = None
        self._speed: npt.NDArray[np.float64] | None = None

    @staticmethod
    def _alpha(cutoff_hz: float, dt_s: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff_hz)
        return 1.0 / (1.0 + tau / dt_s)

    def reset(self) -> None:
        self.value = self._speed = None

    def __call__(self, x: npt.NDArray[np.float64], dt_s: float) -> npt.NDArray[np.float64]:
        x = np.asarray(x, float)
        if self.value is None:
            self.value, self._speed = x.copy(), np.zeros_like(x)
            return x.copy()
        speed = (x - self.value) / dt_s
        self._speed = self._speed + self._alpha(self.d_cutoff, dt_s) * (speed - self._speed)
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._speed))
        self.value = self.value + self._alpha(cutoff, dt_s) * (x - self.value)
        return self.value.copy()


class PoseFilter:
    """One Euro on a 4x4 pose: on the position, and on the rotation along the turn from the filtered one to the raw.

    The rotation's cutoff rises with the filtered turning speed (rad/s) as the position's does with its speed (m/s).
    """

    def __init__(self, min_cutoff_hz: float, beta_m: float, beta_rad: float, d_cutoff_hz: float = 1.0) -> None:
        self._position = OneEuro(min_cutoff_hz, beta_m, d_cutoff_hz)
        self._min_cutoff, self._beta_rad, self._d_cutoff = min_cutoff_hz, beta_rad, d_cutoff_hz
        self._rotation: npt.NDArray[np.float64] | None = None
        self._turn_speed = 0.0

    def reset(self) -> None:
        self._position.reset()
        self._rotation, self._turn_speed = None, 0.0

    def __call__(self, pose: npt.NDArray[np.float64], dt_s: float) -> npt.NDArray[np.float64]:
        out = np.eye(4)
        out[:3, 3] = self._position(pose[:3, 3], dt_s)
        if self._rotation is None:
            self._rotation = pose[:3, :3].copy()
        else:
            axis, angle = axis_angle(pose[:3, :3] @ self._rotation.T)
            self._turn_speed += OneEuro._alpha(self._d_cutoff, dt_s) * (angle / dt_s - self._turn_speed)
            alpha = OneEuro._alpha(self._min_cutoff + self._beta_rad * self._turn_speed, dt_s)
            self._rotation = rotation_about(axis, alpha * angle) @ self._rotation
        out[:3, :3] = self._rotation
        return out


# ------------------------------------------------------------------------------------------------ clutch
@dataclasses.dataclass
class SmoothClutch(teleop.Clutch):
    """The kit's clutch with a measured heading, a pose filter, scaled turns with a dead band, and a wrist lock."""

    heading_rad: float = 0.0
    rotation_scale: float = 1.0
    rotation_deadband_rad: float = math.radians(2.0)
    wrist_locked: bool = False
    dt_s: float = 1.0 / 60.0
    filter: PoseFilter | None = None

    def update(self, grip: float, controller: Any, tool: Any) -> Any:
        if controller is not None and self.filter is not None:
            if not self.engaged and grip >= teleop.GRIP_ENGAGE:
                self.filter.reset()                    # engage on the hand's pose as it is, no filter lag
            controller = self.filter(controller, self.dt_s)
        return super().update(grip, controller, tool)

    def _target(self, controller: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        assert self._controller_start is not None and self._tool_start is not None
        rotation = self.quest_to_base @ yaw(self.heading_rad)
        moved = rotation @ (controller[:3, 3] - self._controller_start[:3, 3]) * self.translation_scale
        target = np.eye(4)
        if self.wrist_locked:
            target[:3, :3] = self._tool_start[:3, :3]
        else:
            axis, angle = axis_angle(rotation @ controller[:3, :3] @ self._controller_start[:3, :3].T @ rotation.T)
            angle = max(0.0, angle - self.rotation_deadband_rad) * self.rotation_scale
            target[:3, :3] = rotation_about(axis, angle) @ self._tool_start[:3, :3]
        start = self._tool_start[:3, 3]
        target[:3, 3] = np.clip(start + moved, np.minimum(self.workspace_min_m, start),
                                np.maximum(self.workspace_max_m, start))
        return target


# ------------------------------------------------------------------------------------------------ IK
class SmoothIK(ik.ArmIK):
    """The kit's ArmIK with its solver damping as a setting (the kit's 1e-3 chatters near singular poses)."""

    damping: float = 1e-2

    @classmethod
    def adopt(cls, solver: ik.ArmIK, damping: float, orientation_cost: float) -> "SmoothIK":
        solver.__class__ = cls
        solver.damping = damping
        solver._task = mink.FrameTask(solver._site, "site", position_cost=1.0, orientation_cost=orientation_cost,
                                      lm_damping=damping)
        return solver  # type: ignore[return-value]

    def step(self, target, joints_rad, dt_s):  # noqa: ANN001 - ArmIK.step with `damping` in place of 1e-3
        seed = np.asarray(joints_rad, dtype=np.float64) * self._signs
        self._set_model_joints(np.clip(seed, self._low, self._high))
        self._task.set_target(mink.SE3.from_matrix(target))
        dt_iteration = dt_s / ik._ITERATIONS_PER_TICK
        try:
            for _ in range(ik._ITERATIONS_PER_TICK):
                velocity = mink.solve_ik(self._configuration, [self._task], dt_iteration, "daqp",
                                         damping=self.damping, limits=self._limits)
                self._configuration.integrate_inplace(velocity, dt_iteration)
        except mink.NoSolutionFound:
            return np.asarray(joints_rad, dtype=np.float64).copy()
        solved = np.clip(self._configuration.q[self._qpos_index], self._low, self._high)
        max_step = self._max_speed * dt_s
        return np.clip(solved, seed - max_step, seed + max_step) * self._signs


# ------------------------------------------------------------------------------------------------ collector
@dataclasses.dataclass(frozen=True)
class Tuning:
    rotation_scale: float = 1.0
    rotation_deadband_deg: float = 2.0
    filter_min_cutoff_hz: float = 0.7
    filter_beta_m: float = 10.0
    filter_beta_rad: float = 0.5
    heading_file: pathlib.Path = HEADING_FILE


class _RecordingHeadset:
    """The Quest, keeping its last reading for the calibration gesture."""

    def __init__(self, headset: Any) -> None:
        self._headset = headset
        self.last: quest.QuestState | None = None

    def read(self) -> quest.QuestState:
        self.last = self._headset.read()
        return self.last

    def __getattr__(self, name: str) -> Any:
        return getattr(self._headset, name)


class QuestCollector(collect.Collector):
    """The kit's Collector with SmoothClutch, and X (wrist lock) and Y (heading calibration) on the left controller."""

    tuning: Tuning = Tuning()

    def __init__(self, station: config.StationConfig, task: str, robot: Any, headset: Any, *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(station, task, robot, _RecordingHeadset(headset), *args, **kwargs)
        t = self.tuning
        self._heading = load_heading(t.heading_file)
        self._calibration_start: npt.NDArray[np.float64] | None = None
        self._clutches = {
            side: SmoothClutch(quest_to_base=arm.quest_to_base, workspace_min_m=arm.workspace_min_m,
                               workspace_max_m=arm.workspace_max_m,
                               translation_scale=station.teleop.translation_scale, heading_rad=self._heading,
                               rotation_scale=t.rotation_scale,
                               rotation_deadband_rad=math.radians(t.rotation_deadband_deg), dt_s=self._dt_s,
                               filter=PoseFilter(t.filter_min_cutoff_hz, t.filter_beta_m, t.filter_beta_rad))
            for side, arm in station.arms.items()
        }
        self._log(f"quest heading {math.degrees(self._heading):+.0f} deg "
                  f"({'saved' if t.heading_file.exists() else 'not calibrated: press Y, move, Y'}); "
                  "X locks the wrist")

    def _handle_buttons(self, pressed: frozenset[str]) -> None:
        super()._handle_buttons(pressed)
        if "X" in pressed:
            locked = not next(iter(self._clutches.values())).wrist_locked
            for clutch in self._clutches.values():
                clutch.wrist_locked = locked
                clutch.engaged = False                 # re-engage where the hand and tool are now: no jump
            self._log("wrist locked: the tool keeps its orientation" if locked else "wrist unlocked")
        if "Y" in pressed:
            self._calibrate()

    def _calibrate(self) -> None:
        last = self._headset.last
        pose = None if last is None else last.poses.get("right")
        if pose is None:
            self._warn("heading calibration needs the right controller tracked")
            return
        if self._calibration_start is None:
            self._calibration_start = pose[:3, 3].copy()
            for clutch in self._clutches.values():
                clutch.engaged = False
            self._log("heading calibration: move the right controller ~20 cm in the arm's forward direction, "
                      "then press Y")
            return
        move, self._calibration_start = pose[:3, 3] - self._calibration_start, None
        any_arm = next(iter(self._station.arms.values()))
        try:
            theta = heading_from_move(move, any_arm.quest_to_base)
        except ValueError as error:
            self._warn(f"heading calibration: {error}; press Y to try again")
            return
        self._heading = theta
        save_heading(theta, self.tuning.heading_file)
        for clutch in self._clutches.values():
            clutch.heading_rad, clutch.engaged = theta, False
        self._log(f"heading calibrated: {math.degrees(theta):+.0f} deg, saved to {self.tuning.heading_file}")

    def _follow(self, state: quest.QuestState, measured: arms.Joints) -> None:
        if self._calibration_start is not None:        # calibrating: the arms hold still
            return
        super()._follow(state, measured)


# ------------------------------------------------------------------------------------------------ gravity
def _follower_gravity() -> Any:
    """metal/leader_collect.py's FollowerGravity: MakerMods' Metal gravity model, proven on the station's arms."""
    metal = str(pathlib.Path(__file__).resolve().parents[1] / "metal")
    if metal not in sys.path:
        sys.path.append(metal)
    from leader_collect import FollowerGravity

    return FollowerGravity


def _with_gravity(open_arms: Any) -> Any:
    def opened(station: config.StationConfig, *, energize: bool, sim: bool = False) -> Any:
        if not (energize and not sim and station.arm == "metal"):
            return open_arms(station, energize=energize, sim=sim)
        # The model loads before the arms energize, so a missing URDF or Pinocchio stops here with torque off.
        from lerobot.teleoperators.metal_leader.gravity import MetalGravityModel
        from lerobot.teleoperators.metal_leader.urdf import metal_urdf_path

        follower_gravity, model = _follower_gravity(), MetalGravityModel(str(metal_urdf_path()))
        robot = open_arms(station, energize=energize, sim=sim)
        for device in robot._devices.values():
            follower_gravity(device.bus, model)
        return robot
    return opened


# ------------------------------------------------------------------------------------------------ launcher
def install(*, tuning: Tuning, ik_damping: float, orientation_cost: float, gravity_ff: bool) -> None:
    """Swaps the clutch, the IK and (optionally) the arm opener into the kit's collect module, for this process."""
    QuestCollector.tuning = tuning
    collect.Collector = QuestCollector  # type: ignore[misc]
    quest.BUTTONS = tuple(dict.fromkeys((*quest.BUTTONS, "X", "Y")))  # the kit's reader looks the tuple up per read
    open_ik = arms.open_ik

    def smooth_ik(station: config.StationConfig) -> dict[str, ik.ArmIK]:
        return {side: SmoothIK.adopt(solver, ik_damping, orientation_cost) for side, solver in open_ik(station).items()}

    arms.open_ik = smooth_ik  # type: ignore[assignment]
    if gravity_ff:
        arms.open_arms = _with_gravity(arms.open_arms)  # type: ignore[assignment]


def tuned_station(station: config.StationConfig, control_hz: int, max_joint_speed: float) -> config.StationConfig:
    if control_hz % station.fps:
        raise config.ConfigError(f"--control-hz {control_hz} must be a whole multiple of fps {station.fps}")
    return dataclasses.replace(station, control_hz=control_hz,
                               teleop=dataclasses.replace(station.teleop, max_joint_speed_rad_s=max_joint_speed))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("--task")
    p.add_argument("--dataset-dir")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quest-ip")
    p.add_argument("--dashboard-port", type=int, default=collect.dashboard.DEFAULT_PORT)
    p.add_argument("--control-hz", type=int, default=60)
    p.add_argument("--max-joint-speed", type=float, default=2.5, help="rad/s")
    p.add_argument("--rotation-scale", type=float, default=1.0, help="tool turn per hand turn")
    p.add_argument("--rotation-deadband-deg", type=float, default=2.0)
    p.add_argument("--filter-min-cutoff", type=float, default=0.7, help="Hz: lower is smoother when still")
    p.add_argument("--filter-beta", type=float, default=10.0, help="higher is less lag when moving fast")
    p.add_argument("--ik-damping", type=float, default=1e-2)
    p.add_argument("--orientation-cost", type=float, default=0.5)
    p.add_argument("--gravity-ff", action=argparse.BooleanOptionalAction, default=True,
                   help="Metal: gravity torque in the follower's frames (on by default; --no-gravity-ff turns it off)")
    p.add_argument("--reset-heading", action="store_true", help="forget the saved heading calibration")
    a = p.parse_args(argv)
    if a.reset_heading:
        HEADING_FILE.unlink(missing_ok=True)
    try:
        station = tuned_station(config.load(a.config), a.control_hz, a.max_joint_speed)
    except config.ConfigError as error:
        sys.exit(f"config: {error}")
    if a.dataset_dir:
        station = dataclasses.replace(station, dataset_dir=pathlib.Path(a.dataset_dir).expanduser())
    install(tuning=Tuning(a.rotation_scale, a.rotation_deadband_deg, a.filter_min_cutoff, a.filter_beta),
            ik_damping=a.ik_damping, orientation_cost=a.orientation_cost, gravity_ff=a.gravity_ff)
    try:
        collect.run(station, collect.choose_task(a.task, station.task), dry_run=a.dry_run, sim=False,
                    quest_ip=a.quest_ip, dashboard_port=a.dashboard_port)
    except (collect.station_lock.StationBusyError, quest.QuestError, arms.ArmsError) as error:
        sys.exit(str(error))
    except config.ConfigError as error:
        sys.exit(f"config: {error}")
    except collect.dataset.DatasetError as error:
        sys.exit(f"dataset: {error}")


if __name__ == "__main__":
    main()
