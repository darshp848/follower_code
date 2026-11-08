"""Record demonstrations on a MakerMods Metal station with one arm as a gravity-compensated leader.

The leader arm runs MakerMods' `metal_leader` (their LeRobot fork): kp = 0 with Pinocchio gravity and friction
feedforward, so you move it by hand and squeeze its gripper. The follower arm copies its joints and gripper through
the same MetalFollower, gripper hold and speed cap the hackathon kit uses. Everything else is the kit's `dhk-collect`
(github.com/dreamlabs-co/physical-ai-hack-2026): its dashboard, cameras, device checks, station lock, homing, lag
check and LeRobot v2.1 writer, so the dataset is the one Dream fine-tuning and `tools/add_eef.py` expect. The kit's
files are not changed.

    cd ~/physical-ai-hacks
    .venv/bin/python ~/follower_code/metal/leader_collect.py --config configs/station.yaml --dry-run
    .venv/bin/python ~/follower_code/metal/leader_collect.py --config configs/station.yaml --task "unplug the cable"

Keys (type, then Enter): Enter = the Quest's A (opens the arms; then starts / saves an episode), `d` = B (discard
the episode), `h` = right joystick click (follower goes home; not while recording). Ctrl+C ends the session: the
follower and the leader are walked back to where they started, then released.

What is recorded for the leader's side: by default its PARKED pose (where it was when the arms opened) in both
`observation.state` and `action`, because at deploy time that arm stays parked while the follower works; the
leader's real motion would teach a model to move it. Each episode's task text is unchanged; the session prints the
choice. `--record-leader` records the leader's measured joints instead.

The dry run opens both arms with torque off (the kit's dry run): nothing is energized, the leader is limp, so
support it; the follower's command (shown on the dashboard) follows the leader and is never sent.
"""

import argparse
import concurrent.futures
import dataclasses
import logging
import math
import pathlib
import queue
import signal
import sys
import threading
import time
from typing import Any

import numpy as np

from dream_hackathon import arms, collect, config, dashboard, dataset, devices, quest, station_lock, teleop
from dream_hackathon.arms import Grips, Joints
from dream_hackathon.arms.makermods import GRIPPER, MOTORS, _grip, _gripper_deg

_log = logging.getLogger(__name__)

## While the follower is still far from the leader (just after the arms open), it closes the gap this slowly.
ENGAGE_SPEED_RAD_S = 0.4
## The follower counts as caught up, and moves at the config's teleop speed, once every joint is this close.
ENGAGED_WITHIN_RAD = 0.05


class LeaderFollowerArms:
    """The kit's Arms for a leader/follower pair: read() both sides, command() only the follower.

    Attributes:
    - start_joints: Per side, the joints when opened (the leader's is also its parked pose).
    - energized: False for a dry run: both arms torque off, read only.
    - leader_joints, leader_grip: The leader's measured joints and gripper, refreshed by read().
    """

    def __init__(self, station: config.StationConfig, *, energize: bool, leader: str, record_leader: bool) -> None:
        from lerobot.robots.metal_follower import MetalFollower, MetalFollowerConfig
        from makermodslab.arms import METAL

        if station.arm != "metal":
            raise arms.ArmsError(f"this recorder drives MakerMods Metal arms; the config's arm is {station.arm!r}")
        mm = station.makermods
        assert mm is not None
        self.energized = energize
        self.leader, self.follower = leader, ("right" if leader == "left" else "left")
        self._record_leader = record_leader
        self._arms = station.arms
        self._family: Any = METAL
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="leader-follower")
        self._devices: dict[str, Any] = {}
        self._rest_poses: list[tuple[Any, dict[str, float]]] = []
        try:
            follower = MetalFollower(MetalFollowerConfig(port=station.arms[self.follower].port,
                                                         id=f"dream_hackathon_{self.follower}",
                                                         can_interface=mm.can_interface))
            self._open_follower(follower)
            self._devices[self.follower] = follower
            if energize:
                build = METAL._metal_leader_config_class()   # MakerMods Lab's builder: hold_kp_on_disconnect = 0
                lead = _metal_leader(build(port=station.arms[self.leader].port, id=f"follower_code_{self.leader}",
                                           can_interface=mm.can_interface))
                lead.connect()                              # builds the gravity model, then enables torque
            else:
                lead = MetalFollower(MetalFollowerConfig(port=station.arms[self.leader].port,
                                                         id=f"dream_hackathon_{self.leader}",
                                                         can_interface=mm.can_interface))
                lead.bus.connect(handshake=False)
                lead.bus.disable_torque()
            self._devices[self.leader] = lead
        except BaseException:
            self.close()
            raise
        if energize:
            self._rest_poses.extend(self._family.capture_rest_poses(self._devices[self.follower]))
            self._rest_poses.extend(self._family.capture_leader_rest_poses(self._devices[self.leader]))
        self._parked: tuple[np.ndarray, float] | None = None
        self.start_joints, grips = self.read()
        self._parked = (self.start_joints[self.leader].copy(), grips[self.leader])
        self.recorded_leader_grip = grips[self.leader]

    def read(self) -> tuple[Joints, Grips]:
        """The follower as measured; the leader as measured, or parked (see the module). The measured leader is
        always in leader_joints / leader_grip, which the collector follows."""
        sides = list(self._devices)
        degrees = dict(zip(sides, self._pool.map(self._read_degrees, sides), strict=True))
        joints = {side: np.radians([deg[m] for m in MOTORS]) for side, deg in degrees.items()}
        grips = {side: _grip(self._arms[side], deg[GRIPPER]) for side, deg in degrees.items()}
        self.leader_joints, self.leader_grip = joints[self.leader].copy(), grips[self.leader]
        if self._parked is not None and not self._record_leader:
            joints[self.leader], grips[self.leader] = self._parked[0].copy(), self._parked[1]
        self.recorded_leader_grip = grips[self.leader]
        return joints, grips

    def command(self, joints: Joints, grips: Grips) -> None:
        if not self.energized:
            raise RuntimeError("a dry run never commands the arms")
        action = {f"{m}.pos": math.degrees(v) for m, v in zip(MOTORS, joints[self.follower], strict=True)}
        action[f"{GRIPPER}.pos"] = _gripper_deg(self._arms[self.follower], grips[self.follower])
        self._devices[self.follower].send_action(action)

    def close(self) -> None:
        """Walks both arms back to where they started (MakerMods' return), releases torque, then disconnects."""
        if self._rest_poses:
            for ok, reason in self._family.return_to_rest(self._rest_poses) or []:
                if not ok:
                    _log.error("an arm did not return to its start pose: %s", reason)
            self._rest_poses = []
        for side, device in self._devices.items():
            for problem in self._family.release_torque(device, f"{side} arm"):
                _log.error(problem)
            try:
                if self.energized:
                    device.disconnect()
                else:
                    device.bus.disconnect(False)
            except Exception as error:  # noqa: BLE001 - the other arm must still be closed
                _log.error("%s arm: disconnect failed: %s", side, error)
        self._devices = {}
        self._pool.shutdown(wait=True)

    def _open_follower(self, device: Any) -> None:
        """The kit's follower start-up (arms/makermods.py), or torque off for a dry run."""
        if not self.energized:
            device.bus.connect(handshake=False)
            device.bus.disable_torque()
            return
        from makermodslab import metal_gripper, torque

        metal_gripper.install_metal_gripper(device, "")
        self._family.prepare_teleoperation(device)
        self._family.prepare_leader_following(device)
        try:
            device.connect(calibrate=False)
        except BaseException:
            torque.de_energize_can_device(device, "follower arm")
            raise

    def _read_degrees(self, side: str) -> dict[str, float]:
        device = self._devices[side]
        if not self.energized:
            return {m: float(device.bus.read("Present_Position", m)) for m in (*MOTORS, GRIPPER)}
        reading = device.get_action() if side == self.leader else device.get_observation()
        return {m: float(reading[f"{m}.pos"]) for m in (*MOTORS, GRIPPER)}


def _metal_leader(cfg: Any) -> Any:
    from lerobot.teleoperators.metal_leader import MetalLeader

    return MetalLeader(cfg)


class KeyboardHeadset:
    """Stands in for the Quest: the kit only needs its buttons here. Enter = A, `d` = B, `h` = right joystick."""

    KEYS = {"": "A", "d": "B", "h": "RJ"}

    def __init__(self, stream: Any = None) -> None:
        self._pressed: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._read, args=(stream or sys.stdin,), daemon=True, name="keys").start()

    def _read(self, stream: Any) -> None:
        for line in stream:
            button = self.KEYS.get(line.strip().lower())
            if button is not None:
                self._pressed.put(button)

    def press(self, button: str) -> None:
        self._pressed.put(button)

    def read(self) -> quest.QuestState:
        pressed = set()
        while True:
            try:
                pressed.add(self._pressed.get_nowait())
            except queue.Empty:
                break
        sides = ("left", "right")
        return quest.QuestState({s: None for s in sides}, {s: 0.0 for s in sides}, {s: 0.0 for s in sides},
                                frozenset(pressed))

    def data_age_s(self) -> float:
        return 0.0

    def reconnect(self) -> None:
        pass

    def close(self) -> None:
        pass


class LeaderCollector(collect.Collector):
    """The kit's Collector with the leader arm in place of the Quest's clutch."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._engaged = False

    def _follow(self, state: quest.QuestState, measured: Joints) -> None:
        robot: LeaderFollowerArms = self._robot  # type: ignore[assignment]
        side = robot.follower
        joints = self._command.joints[side]
        gap = robot.leader_joints - joints
        if not self._engaged and float(np.max(np.abs(gap))) <= ENGAGED_WITHIN_RAD:
            self._engaged = True
            self._log(f"{side} arm is following the {robot.leader} arm")
        speed = self._station.teleop.max_joint_speed_rad_s if self._engaged else ENGAGE_SPEED_RAD_S
        self._command.joints[side] = teleop.step_toward(joints, robot.leader_joints, speed * self._dt_s)
        # The leader's gripper as a fraction of its own stroke, onto the follower's stroke (open = 1).
        self._command.grips[side] = robot.leader_grip
        # The leader side's command is what is recorded as its action: the same value as its recorded state.
        self._command.joints[robot.leader] = measured[robot.leader].copy()
        self._command.grips[robot.leader] = robot.recorded_leader_grip
        for clutch_side, clutch in self._clutches.items():
            clutch.engaged = clutch_side == side and self._engaged

    def _step_home(self) -> None:
        super()._step_home()
        self._engaged = False        # after homing, catch up with the leader slowly again

    def _resync_lagging(self, measured: Joints) -> None:
        super()._resync_lagging(measured)
        if not self._clutches[self._robot.follower].engaged:  # type: ignore[attr-defined]
            self._engaged = False


def run(station: config.StationConfig, task: str, *, dry_run: bool, leader: str, record_leader: bool,
        dashboard_port: int = dashboard.DEFAULT_PORT, keys: Any = None) -> None:
    """collect.run() and collect._run_locked() with the leader arm and the keyboard in place of the Quest."""
    iks = arms.open_ik(station)

    def make_writer() -> dataset.DatasetWriter:
        return dataset.DatasetWriter(station.dataset_dir, station.fps, station.arm, collect._state_names(station),
                                     list(collect.layouts.video_keys(station.family).values()),
                                     station.record_width, station.record_height)

    controls = collect.SessionControls(make_writer, task)
    lock = station_lock.acquire("follower_code leader_collect", station.dataset_dir.resolve())
    try:
        signal.signal(signal.SIGTERM, collect._stop)
        signal.signal(signal.SIGHUP, collect._hang_up)
        headset = KeyboardHeadset(keys)
        found = devices.DeviceSet(station, quest_ip="keyboard", open_quest=lambda _ip: headset)
        recorded = "its measured motion" if record_leader else "its parked pose (where it was when the arms opened)"
        collect._say(f"{leader} arm leads, the other follows; the leader's side records {recorded}")

        def frame(role: str) -> Any:
            camera = found.cameras.get(role)
            if camera is None:
                raise collect.cameras.CameraError(f"camera '{role}' is not open yet")
            return camera.latest()

        def frames() -> dict[str, Any]:
            return {role: frame(role) for role in collect.layouts.CAMERA_ROLES}

        def log(line: str) -> None:
            collect._say(line)
            if board is not None:
                board.event(line)

        def wrap(action: Any) -> Any:
            def call(arg: Any) -> tuple[bool, str]:
                ok, message = action(arg)
                log(f"dashboard: {message}")
                return ok, message
            return call

        board = collect._open_dashboard(station, frame, dashboard_port, on_fix=wrap(found.apply_fix),
                                        on_task=wrap(controls.set_task), on_delete=wrap(controls.delete),
                                        task_of=lambda: controls.task)
        monitor = None
        try:
            reported: dict[str, str | None] = {}
            collect._say("press Enter (= A) once everything is ready to open the arms")
            try:
                robot = collect.wait_and_open_arms(
                    station, found, board, log, dry_run=dry_run, sim=False, reported=reported,
                    open_arms=lambda st, energize, sim: LeaderFollowerArms(st, energize=energize, leader=leader,
                                                                           record_leader=record_leader))
            except KeyboardInterrupt:
                log("session ended before the arms were opened")
                return
            monitor = collect._Monitor(found, board, log, reported)
            collector = None
            try:
                collector = LeaderCollector(station, controls.task, robot, headset, frames, iks, controls.writer,
                                            log=log, controls=controls)
                collect._say("keys: Enter = record / save, d = discard, h = follower home, Ctrl+C = quit")
                collect._loop(collector, 1.0 / station.control_hz, board)
            finally:
                for stop_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                    signal.signal(stop_signal, signal.SIG_IGN)
                try:
                    if collector is not None:
                        collector.discard()
                        if robot.energized:
                            log("returning the arms to where they started (the emergency stop halts them)...")
                            collector.go_home()
                finally:
                    robot.close()
                    log("session ended")
        finally:
            if monitor is not None:
                monitor.stop()
            found.close()
            if board is not None:
                board.close()
    finally:
        lock.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="the station config, e.g. configs/station.yaml")
    parser.add_argument("--task", help="the task text recorded with each episode")
    parser.add_argument("--dataset-dir", help="the dataset folder episodes are added to (default: the config's)")
    parser.add_argument("--dry-run", action="store_true", help="both arms torque off, read only, command nothing")
    parser.add_argument("--leader", choices=("left", "right"), default="left", help="the arm you move by hand")
    parser.add_argument("--record-leader", action="store_true",
                        help="record the leader's measured joints instead of its parked pose")
    parser.add_argument("--dashboard-port", type=int, default=dashboard.DEFAULT_PORT)
    args = parser.parse_args(argv)
    try:
        station = config.load(args.config)
    except config.ConfigError as error:
        sys.exit(f"config: {error}")
    if args.dataset_dir:
        station = dataclasses.replace(station, dataset_dir=pathlib.Path(args.dataset_dir).expanduser())
    try:
        run(station, collect.choose_task(args.task, station.task), dry_run=args.dry_run, leader=args.leader,
            record_leader=args.record_leader, dashboard_port=args.dashboard_port)
    except (station_lock.StationBusyError, arms.ArmsError) as error:
        sys.exit(str(error))
    except config.ConfigError as error:
        sys.exit(f"config: {error}")
    except dataset.DatasetError as error:
        sys.exit(f"dataset: {error}")


if __name__ == "__main__":
    main()
