"""quest_collect over the kit's own test fakes (FakeArms, FakeQuest, its six-joint test arm): no Quest, no motors.

    cd ~/physical-ai-hacks && .venv/bin/python -m pytest -q ~/follower_code/quest/tests
"""
import dataclasses
import math
import pathlib
import sys

import numpy as np
import pytest

KIT_TESTS = pathlib.Path.home() / "physical-ai-hacks" / "tests"
sys.path[:0] = [str(KIT_TESTS), str(pathlib.Path(__file__).resolve().parents[1])]
import quest_collect as qc  # noqa: E402
from conftest import FLANGE, JOINT_NAMES, arm_urdf, filled_template  # noqa: E402
from test_collect import START, FakeArms, FakeQuest  # noqa: E402

from dream_hackathon import collect, config, dataset, ik, layouts  # noqa: E402

KIT_COLLECTOR = collect.Collector


def _collector(tmp_path, cls=None, tuning=None, logs=None, hz=60, damping=1e-2):
    tmp_path.mkdir(parents=True, exist_ok=True)
    station = config.load(filled_template("yam", tmp_path))
    roomy = {side: dataclasses.replace(arm, workspace_min_m=np.full(3, -2.0), workspace_max_m=np.full(3, 2.0))
             for side, arm in station.arms.items()}
    station = dataclasses.replace(station, dataset_dir=tmp_path / "task", record_width=320, record_height=180,
                                  arms=roomy)
    station = qc.tuned_station(station, hz, 2.5)
    model = ik.urdf_model(arm_urdf(), FLANGE, (0.0, 0.0, 0.08))
    iks = {side: ik.ArmIK(model, ik.TCP_SITE, JOINT_NAMES, 2.5, 0.02) for side in ("left", "right")}
    if cls is qc.QuestCollector:
        iks = {side: qc.SmoothIK.adopt(s, damping, 0.5) for side, s in iks.items()}
        qc.QuestCollector.tuning = tuning or qc.Tuning(heading_file=tmp_path / "heading.json")
    keys = list(layouts.video_keys("yam").values())
    writer = dataset.DatasetWriter(station.dataset_dir, 30, "yam", [f"v{i}" for i in range(14)], keys, 320, 180)
    frame = np.zeros((180, 320, 3), np.uint8)
    arms, headset = FakeArms(), FakeQuest()
    c = (cls or KIT_COLLECTOR)(station, "t", arms, headset, lambda: {r: frame for r in layouts.CAMERA_ROLES},
                                   iks, writer, log=(logs if logs is not None else []).append)
    return c, arms, headset


def _tool(c, arms):
    return c._iks["right"].fk(arms.joints["right"])


def test_install_swaps_in_the_pieces_and_reads_x_and_y(monkeypatch):
    for module, name in ((collect, "Collector"), (qc.quest, "BUTTONS"), (qc.arms, "open_ik"), (qc.arms, "open_arms")):
        monkeypatch.setattr(module, name, getattr(module, name))   # restored after the test
    qc.install(tuning=qc.Tuning(), ik_damping=1e-2, orientation_cost=0.5, gravity_ff=True)
    assert {"A", "B", "RJ", "X", "Y"} <= set(qc.quest.BUTTONS)
    assert collect.Collector is qc.QuestCollector


def test_still_reaches_a_5_cm_move(tmp_path):
    c, arms, headset = _collector(tmp_path, qc.QuestCollector)
    before = _tool(c, arms)[:3, 3]
    headset.grip["right"] = 1.0
    c.tick()
    headset.poses["right"][2, 3] = -0.05           # 5 cm forward (Quest -z), as the kit's own test
    for _ in range(240):
        c.tick()
    np.testing.assert_allclose(_tool(c, arms)[:3, 3] - before, [0.05, 0, 0], atol=2e-3)


@pytest.mark.parametrize("parked_deg", [90.0, 180.0, -35.0])
def test_heading_calibration_fixes_a_turned_quest_frame(tmp_path, parked_deg):
    """The Quest's frame is turned by `parked_deg` about up; after Y-move-Y, forward is forward again."""
    logs = []
    c, arms, headset = _collector(tmp_path, qc.QuestCollector, logs=logs)
    turned = qc.yaw(math.radians(parked_deg))    # physical -> this Quest's frame
    forward_physical = np.array([0.0, 0.0, -1.0])  # the config's forward (Quest -z with no turn)

    def hand(at):
        pose = np.eye(4)
        pose[:3, 3] = turned @ at
        return pose

    headset.poses["right"] = hand(np.zeros(3))
    c.tick()
    headset.press.add("Y")
    c.tick()
    headset.poses["right"] = hand(0.2 * forward_physical + np.array([0.0, 0.03, 0.0]))  # a bit of lift too
    c.tick()
    headset.press.add("Y")
    c.tick()
    assert any("heading calibrated" in line for line in logs), logs
    assert (tmp_path / "heading.json").exists()
    before = _tool(c, arms)[:3, 3]
    headset.grip["right"] = 1.0
    c.tick()
    headset.poses["right"] = hand(0.2 * forward_physical + 0.05 * forward_physical + np.array([0.0, 0.03, 0.0]))
    for _ in range(240):
        c.tick()
    np.testing.assert_allclose(_tool(c, arms)[:3, 3] - before, [0.05, 0, 0], atol=3e-3)


def test_the_kit_moves_a_turned_quest_the_wrong_way():
    """The problem being fixed: with the Quest's frame turned 90 deg the kit's clutch sends forward sideways."""
    q2b = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)
    clutch = qc.teleop.Clutch(quest_to_base=q2b, workspace_min_m=np.full(3, -1.0), workspace_max_m=np.full(3, 1.0),
                              translation_scale=1.0)
    clutch.update(1.0, np.eye(4), np.eye(4))
    hand = np.eye(4)
    hand[:3, 3] = qc.yaw(math.radians(90)) @ np.array([0.0, 0.0, -0.05])   # 5 cm physically forward
    moved = clutch.update(1.0, hand, np.eye(4))[:3, 3]
    assert abs(moved[0]) < 1e-9 and abs(abs(moved[1]) - 0.05) < 1e-9


def _jitter_and_rise(tmp_path, cls, hz):
    """Tool jitter (mm, std) under 2 mm of tracking noise, and the time (s) to 90 % of a 5 cm hand move in 0.25 s."""
    c, arms, headset = _collector(tmp_path, cls, hz=hz)
    rng = np.random.default_rng(0)
    headset.grip["right"] = 1.0
    c.tick()
    tool = []
    for _ in range(3 * hz):
        headset.poses["right"][:3, 3] = rng.normal(0, 0.002, 3)
        c.tick()
        tool.append(_tool(c, arms)[:3, 3])
    jitter_mm = float(np.std(np.array(tool[hz:]), axis=0).max()) * 1e3
    start = _tool(c, arms)[0, 3]
    for k in range(3 * hz):
        headset.poses["right"][:3, 3] = [0.0, 0.0, -0.05 * min(1.0, k / (0.25 * hz))]
        c.tick()
        if _tool(c, arms)[0, 3] - start >= 0.045:
            return jitter_mm, k / hz
    return jitter_mm, math.inf


def test_less_jitter_at_about_the_same_lag(tmp_path):
    kit_jitter, kit_rise = _jitter_and_rise(tmp_path / "kit", None, 30)
    jitter, rise = _jitter_and_rise(tmp_path / "ours", qc.QuestCollector, 60)
    assert jitter < 0.35 * kit_jitter, (jitter, kit_jitter)
    assert rise <= kit_rise + 0.04, (rise, kit_rise)


def test_wrist_lock_keeps_the_tool_orientation(tmp_path):
    logs = []
    c, arms, headset = _collector(tmp_path, qc.QuestCollector, logs=logs)
    start = _tool(c, arms)
    headset.press.add("X")
    c.tick()
    assert any("wrist locked" in line for line in logs)
    headset.grip["right"] = 1.0
    c.tick()
    pose = np.eye(4)
    pose[:3, :3] = qc.rotation_about(np.array([1.0, 0.0, 0.0]), math.radians(40))
    pose[2, 3] = -0.03
    headset.poses["right"] = pose
    for _ in range(240):
        c.tick()
    end = _tool(c, arms)
    _, turned = qc.axis_angle(end[:3, :3] @ start[:3, :3].T)
    assert math.degrees(turned) < 1.0
    np.testing.assert_allclose(end[:3, 3] - start[:3, 3], [0.03, 0, 0], atol=3e-3)


def test_rotation_scale_and_deadband_on_the_clutch():
    clutch = qc.SmoothClutch(quest_to_base=np.eye(3), workspace_min_m=np.full(3, -1.0),
                             workspace_max_m=np.full(3, 1.0), translation_scale=1.0, rotation_scale=0.5,
                             rotation_deadband_rad=math.radians(2), filter=None)
    tool = np.eye(4)
    assert clutch.update(1.0, np.eye(4), tool) is not None
    hand = np.eye(4)
    hand[:3, :3] = qc.rotation_about(np.array([0.0, 0.0, 1.0]), math.radians(42))
    _, angle = qc.axis_angle(clutch.update(1.0, hand, tool)[:3, :3])
    assert math.degrees(angle) == pytest.approx(20.0, abs=1e-6)
    hand[:3, :3] = qc.rotation_about(np.array([0.0, 0.0, 1.0]), math.radians(1.5))
    _, angle = qc.axis_angle(clutch.update(1.0, hand, tool)[:3, :3])
    assert angle == pytest.approx(0.0, abs=1e-9)


def test_engaging_never_jumps(tmp_path):
    c, arms, headset = _collector(tmp_path, qc.QuestCollector)
    headset.poses["right"][:3, 3] = [0.3, -0.2, 0.1]   # the hand far from where the filter last saw it
    for _ in range(5):
        c.tick()
    before = arms.joints["right"].copy()
    headset.grip["right"] = 1.0
    for _ in range(5):
        c.tick()
    np.testing.assert_allclose(arms.joints["right"], before, atol=1e-3)


def test_heading_math_round_trips():
    q2b = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)
    for deg in (0, 45, 90, 180, -120):
        move = qc.yaw(math.radians(deg)) @ np.array([0.0, 0.0, -0.2])
        theta = qc.heading_from_move(move, q2b)
        mapped = q2b @ qc.yaw(theta) @ move
        np.testing.assert_allclose(mapped / np.linalg.norm(mapped), [1, 0, 0], atol=1e-9)
    with pytest.raises(ValueError):
        qc.heading_from_move(np.array([0.0, 0.3, 0.02]), q2b)    # straight up: no heading in it


class _Bus:
    def __init__(self):
        self.sent = []

    def sync_write_metal(self, commands):
        self.sent.append(commands)


def _opened_with_gravity(arm, energize):
    import types

    from dream_hackathon.arms.makermods import MOTORS

    robot = types.SimpleNamespace(_devices={"left": types.SimpleNamespace(bus=_Bus())})
    opened = qc._with_gravity(lambda station, *, energize, sim=False: robot)(
        types.SimpleNamespace(arm=arm), energize=energize)
    bus = opened._devices["left"].bus
    pose_deg = [10.0, -40.0, 60.0, -20.0, 5.0, 0.0]
    bus.sync_write_metal({m: (40.0, 1.0, d, 0.0, 0.0) for m, d in zip(MOTORS, pose_deg)})
    return MOTORS, pose_deg, bus.sent[-1]


def test_gravity_is_on_by_default_and_uses_the_makermods_model(monkeypatch):
    pytest.importorskip("pinocchio")
    from lerobot.teleoperators.metal_leader.gravity import MetalGravityModel
    from lerobot.teleoperators.metal_leader.urdf import metal_urdf_path

    seen = {}
    monkeypatch.setattr(qc, "install", lambda **kw: seen.update(kw))
    monkeypatch.setattr(qc.collect, "run", lambda *a, **kw: None)
    monkeypatch.setattr(qc.collect, "choose_task", lambda *a: "t")
    monkeypatch.setattr(qc.config, "load", lambda path: __import__("types").SimpleNamespace(task="t"))
    monkeypatch.setattr(qc, "tuned_station", lambda station, hz, speed: station)
    qc.main(["--config", str(pathlib.Path(__file__))])
    assert seen["gravity_ff"] is True
    qc.main(["--config", str(pathlib.Path(__file__)), "--no-gravity-ff"])
    assert seen["gravity_ff"] is False

    motors, pose_deg, sent = _opened_with_gravity("metal", energize=True)
    want = MetalGravityModel(str(metal_urdf_path())).feedforward_torque([math.radians(d) for d in pose_deg], [0.0] * 6)
    np.testing.assert_allclose([sent[m][4] for m in motors], want, atol=1e-9)
    assert any(abs(t) > 0.1 for t in want)


@pytest.mark.parametrize(("arm", "energize"), [("metal", False), ("yam", True)])
def test_no_gravity_on_a_dry_run_or_other_arms(arm, energize):
    motors, _, sent = _opened_with_gravity(arm, energize)
    assert all(sent[m][4] == 0.0 for m in motors)


def _moved_away(tmp_path, logs):
    """A collector whose right arm was driven ~0.6 rad from its start; the Quest released."""
    c, arms, headset = _collector(tmp_path, qc.QuestCollector, logs=logs)
    arms.joints["right"] = arms.start_joints["right"] + np.array([0.6, -0.4, 0.3, 0.0, 0.0, 0.0])
    c._command.joints["right"] = arms.joints["right"].copy()
    return c, arms, headset


def test_home_while_recording_needs_a_second_press_then_discards_and_homes(tmp_path):
    logs = []
    c, arms, headset = _moved_away(tmp_path, logs)
    headset.press.add("A")
    c.tick()
    assert c.episode is not None
    headset.press.add("RJ")
    c.tick()
    assert c.episode is not None and not c.homing
    assert any("press the right stick again" in line for line in logs), logs
    headset.press.add("RJ")
    c.tick()
    assert c.episode is None and c.homing
    for _ in range(60 * 10):
        c.tick()
    assert not c.homing
    np.testing.assert_allclose(arms.joints["right"], arms.start_joints["right"], atol=2e-3)


def test_a_blocked_arm_stops_homing_instead_of_freezing_teleop(tmp_path):
    logs = []
    c, arms, headset = _moved_away(tmp_path, logs)
    arms.stuck = True                                  # the arm cannot move (against the table, say)
    headset.press.add("RJ")
    c.tick()
    assert c.homing
    for _ in range(60 * 3):
        c.tick()
    assert not c.homing
    assert any("cannot reach home" in line for line in logs), logs


def test_a_tracking_glitch_does_not_move_the_arm(tmp_path):
    c, arms, headset = _collector(tmp_path, qc.QuestCollector)
    headset.grip["right"] = 1.0
    for _ in range(5):
        c.tick()
    before = arms.joints["right"].copy()
    headset.poses["right"][:3, 3] = [0.4, 0.1, -0.3]   # 50 cm in one reading: tracking jumped
    for _ in range(60):
        c.tick()
    np.testing.assert_allclose(arms.joints["right"], before, atol=1e-3)


def test_flip_guard_is_on_and_resets_at_every_engage(tmp_path):
    c, arms, headset = _collector(tmp_path, qc.QuestCollector)
    guard = c._iks["right"].flip_guard
    assert guard is not None
    guard.offset = qc.rotation_about(np.array([0.0, 0.0, 1.0]), 0.5)
    headset.grip["right"] = 1.0
    c.tick()
    np.testing.assert_allclose(guard.offset, np.eye(3))


def test_homing_lifts_a_low_tool_before_moving_the_joints(tmp_path):
    c, arms, headset = _collector(tmp_path, qc.QuestCollector)
    home_z = _tool(c, arms)[2, 3]
    headset.grip["right"] = 1.0
    c.tick()
    headset.poses["right"][1, 3] = -0.06              # hand 6 cm down (Quest -y): the tool goes below home
    for _ in range(240):
        c.tick()
    headset.grip["right"] = 0.0
    c.tick()
    low = _tool(c, arms)[:3, 3].copy()
    assert low[2] < home_z - 0.04
    headset.press.add("RJ")
    heights, sideways = [], []
    for _ in range(60 * 15):
        c.tick()
        tool = _tool(c, arms)[:3, 3]
        heights.append(tool[2])
        if tool[2] < home_z - 0.01:                   # still lifting: the tool moves straight up
            sideways.append(np.linalg.norm(tool[:2] - low[:2]))
    assert not c.homing
    assert np.all(np.diff(heights[:30]) > -1e-4)      # it starts by going up, never down
    assert max(sideways) < 0.01
    np.testing.assert_allclose(arms.joints["right"], arms.start_joints["right"], atol=2e-3)
