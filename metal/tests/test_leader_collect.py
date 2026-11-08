"""Offline tests: the kit's real Collector and LeRobot v2.1 writer over fake Metal devices (no CAN, no motors).

Run on a station from ~/physical-ai-hacks (the kit's venv, its configs/station.yaml and the Metal URDF):

    .venv/bin/python -m pytest -q ~/follower_code/metal/tests
"""
import dataclasses
import io
import json
import math
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import leader_collect as lc  # noqa: E402
from dream_hackathon import arms, collect, config  # noqa: E402

KIT = pathlib.Path.home() / "physical-ai-hacks"
MOTORS = (*lc.MOTORS, lc.GRIPPER)
LEADER_START = {"shoulder_pan": 0.0, "shoulder_lift": -20.0, "elbow_flex": 40.0, "wrist_flex": 0.0,
                "wrist_yaw": 0.0, "wrist_roll": 0.0, "gripper": 5.0}
FOLLOWER_START = {"shoulder_pan": 2.0, "shoulder_lift": -18.0, "elbow_flex": 35.0, "wrist_flex": 1.0,
                  "wrist_yaw": 0.0, "wrist_roll": 0.0, "gripper": 0.0}


class FakeBus:
    def __init__(self, device):
        self.device = device

    def connect(self, handshake=True):
        pass

    def disable_torque(self):
        self.device.torque = False

    def disconnect(self, disable_torque=True):
        pass

    def read(self, register, motor):
        return self.device.deg[motor]

    def sync_write_metal(self, commands):
        """The MIT write MetalFollower.send_action uses (velocity_feedforward on): (kp, kd, deg, deg/s, N.m)."""
        self.device.frames.append(commands)
        for motor, (_kp, _kd, position, _v, _t) in commands.items():
            self.device.deg[motor] += 0.6 * (position - self.device.deg[motor])


class FakeFollower:
    """MetalFollower: each frame moves the arm 60% of the way (a lagging servo), through bus.sync_write_metal."""

    def __init__(self, cfg):
        self.cfg, self.deg, self.torque, self.sent, self.frames = cfg, dict(FOLLOWER_START), True, [], []
        self.bus = FakeBus(self)

    def connect(self, calibrate=False):
        pass

    def get_observation(self):
        return {f"{m}.pos": v for m, v in self.deg.items()}

    def send_action(self, action):
        self.sent.append(action)
        self.bus.sync_write_metal({key[:-4]: (100.0, 5.0, value, 0.0, 0.0) for key, value in action.items()})

    def disconnect(self):
        pass


class FakeLeader:
    """MetalLeader: the 'operator' moves it along a script, one step per get_action."""

    def __init__(self, cfg):
        self.cfg, self.deg, self.k, self.torque, self.frames = cfg, dict(LEADER_START), 0, True, []
        self.bus = FakeBus(self)
        self.speed = 1.0

    def connect(self, calibrate=True):
        pass

    def get_action(self):
        self.k += 1
        t = self.k / 30.0
        self.deg["shoulder_pan"] = LEADER_START["shoulder_pan"] + 20 * math.sin(0.8 * self.speed * t)
        self.deg["elbow_flex"] = LEADER_START["elbow_flex"] + 15 * math.sin(0.5 * t)
        self.deg["gripper"] = 5.0 + 50.0 * (0.5 + 0.5 * math.sin(0.6 * t))
        return {f"{m}.pos": v for m, v in self.deg.items()}

    def disconnect(self):
        pass


class FakeFamily:
    def __init__(self):
        self.calls = []

    def _metal_leader_config_class(self):
        def build(port, id=None, **kwargs):  # noqa: A002
            kwargs.setdefault("hold_kp_on_disconnect", 0.0)
            return dict(port=port, id=id, **kwargs)
        return build

    def capture_rest_poses(self, device):
        self.calls.append(("rest", "follower"))
        return [(device, dict(device.deg))]

    def capture_leader_rest_poses(self, device):
        self.calls.append(("rest", "leader"))
        return [(device, dict(device.deg))]

    def return_to_rest(self, poses):
        self.calls.append(("return", len(poses)))
        for device, pose in poses:
            device.deg.update(pose)
        return [(True, "")] * len(poses)

    def release_torque(self, device, label):
        self.calls.append(("release", label))
        device.torque = False
        return []

    def prepare_teleoperation(self, device):
        pass

    def prepare_leader_following(self, device):
        pass


@pytest.fixture
def station(tmp_path, monkeypatch):
    import lerobot.robots.metal_follower as mf
    import makermodslab.metal_gripper
    import makermodslab.arms as mm_arms

    family = FakeFamily()
    monkeypatch.setattr(mf, "MetalFollower", FakeFollower)
    monkeypatch.setattr(mf, "MetalFollowerConfig", lambda **kw: kw)
    monkeypatch.setattr(mm_arms, "METAL", family)
    monkeypatch.setattr(lc, "_metal_leader", FakeLeader)
    monkeypatch.setattr(makermodslab.metal_gripper, "install_metal_gripper", lambda device, name: None, raising=False)
    st = config.load(KIT / "configs" / "station.yaml")
    st = dataclasses.replace(st, dataset_dir=tmp_path / "leader-test")
    return st, family


def _collector(st, robot, headset, logs):
    from dream_hackathon import dataset

    writer = dataset.DatasetWriter(st.dataset_dir, st.fps, st.arm, collect._state_names(st),
                                   list(collect.layouts.video_keys(st.family).values()),
                                   st.record_width, st.record_height)
    blank = np.zeros((st.record_height, st.record_width, 3), np.uint8)
    frames = lambda: {role: blank for role in collect.layouts.CAMERA_ROLES}  # noqa: E731
    return lc.LeaderCollector(st, "move the arm", robot, headset, frames, arms.open_ik(st), writer, log=logs.append)


def _episode(st):
    data = sorted((st.dataset_dir / "data").rglob("*.parquet"))
    assert len(data) == 1
    table = pq.read_table(data[0]).to_pydict()
    return np.array(table["observation.state"]), np.array(table["action"])


def test_records_follower_tracking_leader_with_leader_parked(station):
    st, family = station
    logs = []
    robot = lc.LeaderFollowerArms(st, energize=True, leader="left", record_leader=False)
    headset = lc.KeyboardHeadset(io.StringIO(""))
    c = _collector(st, robot, headset, logs)
    for _ in range(90):                       # 3 s to catch up and follow
        c.tick()
    assert any("is following" in line for line in logs), logs
    headset.press("A")
    for _ in range(120):                      # a 4 s episode at 30 fps
        c.tick()
    headset.press("A")
    c.tick()
    assert any("saved episode 0" in line for line in logs), logs
    state, action = _episode(st)
    assert state.shape[1] == action.shape[1] == 14 and len(state) >= 49
    # MakerMods layout: [left 6, left grip mm, right 6, right grip mm]; left is the leader, parked.
    parked = robot.start_joints["left"]          # the leader where it was when the arms opened
    assert np.allclose(parked, np.radians([LEADER_START[m] for m in lc.MOTORS]), atol=np.radians(1))
    assert np.allclose(state[:, :6], parked, atol=1e-5) and np.allclose(action[:, :6], parked, atol=1e-5)
    assert np.ptp(state[:, 6]) == 0 and np.ptp(action[:, 6]) == 0
    # The follower's action tracks the leader, within one tick of the configured joint speed.
    assert np.ptp(action[:, 7]) > np.radians(15)
    step = np.abs(np.diff(action[:, 7:13], axis=0)).max()
    assert step <= lc.LeaderCollector.follow_speed_rad_s / st.control_hz + 1e-6
    # The follower's gripper action follows the leader's stroke fraction (mm on a 100 mm scale).
    assert np.ptp(action[:, 13]) > 20
    robot.close()
    assert [c for c in family.calls if c[0] == "return"] == [("return", 2)]
    assert ("release", "left arm") in family.calls and ("release", "right arm") in family.calls
    assert family.calls.index(("return", 2)) < family.calls.index(("release", "left arm"))


def test_record_leader_records_its_motion(station):
    st, _ = station
    logs = []
    robot = lc.LeaderFollowerArms(st, energize=True, leader="left", record_leader=True)
    headset = lc.KeyboardHeadset(io.StringIO(""))
    c = _collector(st, robot, headset, logs)
    headset.press("A")
    for _ in range(70):
        c.tick()
    headset.press("A")
    c.tick()
    state, action = _episode(st)
    assert np.ptp(state[:, 0]) > np.radians(10)
    assert np.allclose(state[:, :6], action[:, :6], atol=1e-5)
    robot.close()


def test_dry_run_reads_both_commands_nothing(station):
    st, _ = station
    logs = []
    robot = lc.LeaderFollowerArms(st, energize=False, leader="left", record_leader=False)
    headset = lc.KeyboardHeadset(io.StringIO(""))
    c = _collector(st, robot, headset, logs)
    headset.press("A")
    for _ in range(10):
        c.tick()
    assert any("a dry run does not record" in line for line in logs)
    follower = robot._devices["right"]
    assert follower.sent == [] and not follower.torque and not robot._devices["left"].torque
    robot.close()


def test_keyboard_maps_keys_to_quest_buttons():
    h = lc.KeyboardHeadset(io.StringIO("\nd\nh\nx\n"))
    import time
    time.sleep(0.1)
    assert h.read().pressed == frozenset({"A", "B", "RJ"})
    assert h.read().pressed == frozenset()


def test_non_metal_station_is_refused(station):
    st, _ = station
    with pytest.raises(arms.ArmsError):
        lc.LeaderFollowerArms(dataclasses.replace(st, arm="maker"), energize=False, leader="left",
                              record_leader=False)


def test_dataset_passes_dreams_checks(station, tmp_path):
    """The recorded folder goes through the kit's tools/add_eef.py, which runs Dream's dataset checks."""
    import subprocess

    st, _ = station
    logs = []
    robot = lc.LeaderFollowerArms(st, energize=True, leader="left", record_leader=False)
    headset = lc.KeyboardHeadset(io.StringIO(""))
    c = _collector(st, robot, headset, logs)
    headset.press("A")
    for _ in range(70):
        c.tick()
    headset.press("A")
    c.tick()
    robot.close()
    out = subprocess.run([str(pathlib.Path.home() / "dream-env/bin/python"), str(KIT / "tools/add_eef.py"),
                          str(st.dataset_dir), "--arm", "metal"], capture_output=True, text=True, cwd=KIT)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    info = json.loads((st.dataset_dir.parent / f"{st.dataset_dir.name}-eef" / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v2.1" and info["fps"] == 30


def test_follower_frames_carry_metal_gravity_torque(station):
    """The arm joints' torque field is MakerMods' Metal gravity model at the frame's targets; the gripper's is not."""
    from lerobot.teleoperators.metal_leader.gravity import MetalGravityModel
    from lerobot.teleoperators.metal_leader.urdf import metal_urdf_path

    st, _ = station
    robot = lc.LeaderFollowerArms(st, energize=True, leader="left", record_leader=False)
    c = _collector(st, robot, lc.KeyboardHeadset(io.StringIO("")), [])
    for _ in range(30):
        c.tick()
    frame = robot._devices["right"].frames[-1]
    model = MetalGravityModel(str(metal_urdf_path()))
    want = model.feedforward_torque([math.radians(frame[m][2]) for m in lc.MOTORS], [0.0] * 6)
    got = [frame[m][4] for m in lc.MOTORS]
    np.testing.assert_allclose(got, want, atol=1e-9)
    assert abs(got[1]) > 0.5 or abs(got[2]) > 0.5          # the shoulder or elbow carries real weight
    assert frame["gripper"][4] == 0.0
    robot.close()


def test_no_gravity_ff_sends_zero_torque(station):
    st, _ = station
    robot = lc.LeaderFollowerArms(st, energize=True, leader="left", record_leader=False, gravity_ff=False)
    c = _collector(st, robot, lc.KeyboardHeadset(io.StringIO("")), [])
    for _ in range(5):
        c.tick()
    assert all(v[4] == 0.0 for v in robot._devices["right"].frames[-1].values())
    robot.close()


def test_a_leader_that_keeps_moving_fast_is_still_followed(station):
    """The old catch-up waited to be within 0.05 rad at 0.4 rad/s, which a leader moving faster never allows."""
    st, _ = station
    logs = []
    robot = lc.LeaderFollowerArms(st, energize=True, leader="left", record_leader=False)
    robot._devices["left"].speed = 3.0                       # shoulder_pan swings at up to 0.84 rad/s
    c = _collector(st, robot, lc.KeyboardHeadset(io.StringIO("")), logs)
    for _ in range(int(30 * (lc.ENGAGE_MAX_S + 0.5))):
        c.tick()
    assert c._engaged, logs
    gaps = []
    for _ in range(90):
        c.tick()
        gaps.append(float(np.max(np.abs(robot.leader_joints - c._command.joints["right"]))))
    assert max(gaps) < np.radians(1.0), np.degrees(max(gaps))   # the command keeps up with the leader
    robot.close()


def test_tracking_report_reads_an_episode(station):
    import subprocess

    st, _ = station
    robot = lc.LeaderFollowerArms(st, energize=True, leader="left", record_leader=False)
    headset = lc.KeyboardHeadset(io.StringIO(""))
    c = _collector(st, robot, headset, [])
    headset.press("A")
    for _ in range(70):
        c.tick()
    headset.press("A")
    c.tick()
    robot.close()
    tool = pathlib.Path(__file__).resolve().parents[1] / "tools" / "tracking_report.py"
    out = subprocess.run([sys.executable, str(tool), str(st.dataset_dir)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "shoulder_lift" in out.stdout and "episode_000000" in out.stdout
