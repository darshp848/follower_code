# follower_code

> **Two recorders here, one per arm type.**
> - `metal/leader_collect.py`: MakerMods **Metal** stations, such as the MakerMods Physical AI Hackathon kit. Start
>   here on those.
> - `leader_follower.py`: i2rt **YAM** arms. It refuses to run unless `~/.dreamlabs/station.yaml` says `body: yam`.

## Better Quest teleop: `quest/quest_collect.py`

This runs the kit's own `dhk-collect` with four pieces swapped in. They apply to this process only, and none of
the kit's files change.

| Problem | Fix |
|---|---|
| Wrong direction | **Heading calibration.** The Quest's frame turns with wherever the headset faced when tracking started, so the config's fixed `quest_to_base` can be off by any angle. Press **Y**, move the right controller about 20 cm in the arm's forward direction, then press **Y** again. The heading is saved to `~/.cache/follower_code/quest_heading.json`. |
| Jittery | A **One Euro filter** on each controller pose, which restarts at every engage so engaging never jumps. The IK damping is also raised from 1e-3 to 1e-2. |
| Laggy | The loop runs at **60 Hz** instead of 30, and the joint cap is **2.5 rad/s** instead of 1.5. `--gravity-ff` (Metal only) adds gravity torque so the tool stops sitting 2.5 to 6.5 cm low. It's adapted from Daniel's `teleop-tuning` branch and hasn't run on hardware. |
| Wrist goes wild | A 2° rotation dead band, `--rotation-scale` (tool turn per hand turn), and **X** to lock or unlock the wrist (the tool keeps its orientation). |

```bash
cd ~/physical-ai-hacks
.venv/bin/python ~/follower_code/quest/quest_collect.py --config configs/station.yaml --dry-run
.venv/bin/python ~/follower_code/quest/quest_collect.py --config configs/station.yaml --dataset-dir ~/hackathon_data/my-task
```

- **Tuning:** `--filter-min-cutoff` lower is smoother when still. `--filter-beta` higher is less lag when moving
  fast. `--rotation-scale 0.5` halves wrist turns, and `--control-hz 30` restores the kit's rate.
- **Loop rate:** check the dashboard's loop rate. If it can't hold 60 Hz, use `--control-hz 30`, or recordings
  will be stretched in time.
- **Measured offline:** on the kit's test arm with 2 mm of tracking noise, the tool jitters 4.7× less than the
  kit's (0.47 mm against 2.2 mm), at about 17 ms more lag on a 5 cm move.
- **Tests:** `.venv/bin/python -m pytest -q ~/follower_code/quest/tests` runs them over the kit's own fakes (11
  pass). The kit's own suite still passes.
- **Not tested:** a real Quest or real arms.

## Metal stations: `metal/leader_collect.py`

One Metal arm is the leader, running MakerMods' gravity-compensated `metal_leader` (kp 0, Pinocchio gravity and
friction feedforward). You move it by hand and squeeze its gripper, and the other arm follows. Everything else is
the hackathon kit's `dhk-collect`: its dashboard, cameras, device checks, station lock, homing, lag check and
**LeRobot v2.1** writer. So the dataset is the one `tools/add_eef.py` and Dream fine-tuning take. None of the kit's
files are changed.

```bash
# on the station, inside tmux; the kit's venv needs Pinocchio: uv pip install "pin>=3.1,<4" "cmeel-urdfdom>=4,<5" "cmeel-tinyxml2<11" numpy==2.2.6
cd ~/physical-ai-hacks
.venv/bin/python ~/follower_code/metal/leader_collect.py --config configs/station.yaml --dry-run
.venv/bin/python ~/follower_code/metal/leader_collect.py --config configs/station.yaml --task "unplug the cable"
~/dream-env/bin/python tools/add_eef.py ~/hackathon_data/my-task --arm metal --replace
```

- **Keys** (type, then Enter): Enter opens the arms, then starts or saves an episode. `d` discards the episode, `h`
  sends the follower home, and Ctrl+C quits. On quit both arms are walked back to their start pose and released,
  MakerMods' way.
- **Leader:** `--leader left` is the default.
- **What's recorded for the leader's side:** its *parked* pose by default, because at deploy time that arm stays
  parked. `--record-leader` records its real motion instead. The left wrist camera still shows the leader and your
  hand.
- **Follower motion:** the follower closes the gap to the leader at 0.4 rad/s, then follows at the config's
  `teleop.max_joint_speed_rad_s`. If it falls more than 0.35 rad behind, the kit re-syncs it.
- **Dry run:** both arms open with torque off. Nothing is energized and the leader is limp, so support it.
- **Tests:** `cd ~/physical-ai-hacks && .venv/bin/python -m pytest -q ~/follower_code/metal/tests` runs the kit's
  real Collector and writer over fake Metal devices, then runs `tools/add_eef.py`, which applies Dream's dataset
  checks. All 6 pass on the hackathon NUC.
- **Not covered by the tests:** the real arms. Gravity compensation, how the leader feels and the follower's lag
  are untested until a dry run and a real run on hardware.

## YAM arms: `leader_follower.py`

Leader/follower teleoperation and demo recording for a pair of YAM arms driven by
[i2rt](https://github.com/i2rt-robotics/i2rt).

- One arm runs in i2rt zero-gravity mode and is the **leader**: you move it by hand, gripper included.
- The other arm is the **follower**. Every tick it is commanded to the leader's joint angles (joint copy),
  rate-limited to 2 rad/s.
- Each episode records the leader joints, the follower command, the follower joint position/velocity/effort and
  the gripper, plus an optional mp4 per camera.

Mirroring isn't supported. The YAM chain has lateral offsets, so the same arm can't reach the mirror image of a
pose: the MJCF puts it about 3 cm off per link. Joint copy also keeps the two arms moving in parallel, so they
can't swing into each other.

## Run

Needs Python with `i2rt`, `numpy` and `opencv-python`.

```bash
python leader_follower.py --sim     # no hardware: i2rt SimRobot, scripted leader, one episode
python leader_follower.py --leader can_follower_l --follower can_follower_r \
    --cam scene=/dev/video0 --cam wrist_r=/dev/video2
```

`--sim` only checks the plumbing. i2rt's SimRobot snaps to each command, so the sim doesn't cover dynamics,
gravity compensation or tracking lag.

Keys (type, then Enter): Enter starts or stops an episode, `s` or `f` labels the last episode success or fail,
`q` quits. Episodes are saved to `runs_teleop/<time>/episode_NNN/` as `data.npz`, `meta.json`, `<cam>.mp4` and
`<cam>_t.npy` (frame timestamps).

## Safety

- Start with both arms in their rest pose. On exit both arms go back to their start-up pose before the CAN buses
  close, because closing sets motor torque to zero. A hard kill skips that step.
- At start-up the follower moves slowly to the leader pose before it starts copying.
- If the follower lags its command by more than 0.35 rad for 0.5 s, tracking stops.
- Only one process can own the CAN links. Stop any other controller (station agent, policy runner) first. If
  stopping it switches the motors off, support the arms while you do it.

## Files

- `leader_follower.py` runs teleop and recording.
- `tests/test_sim.py` is the SimRobot plumbing test (`python -m pytest -q tests`).
- `tools/check_mirror.py` is the evidence for joint copy: the best YAM mirror map is still 3.0 cm off per link.
- `HANDOVER.md` covers taking the arms from a Dream station agent and giving them back.
- `requirements.txt` lists the packages. On a Dream station, use its existing env, which already has them.
