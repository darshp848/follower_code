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
| Laggy | The loop runs at **60 Hz** instead of 30, and the joint cap is **2.5 rad/s** instead of 1.5. |
| Arm sits low | **Gravity feedforward, on by default** (Metal only; `--no-gravity-ff` turns it off). It's the same MakerMods gravity model the leader/follower uses, which took shoulder_lift from 4–6° low to +0.2° on the station's arms. |
| Wrist goes wild | `--rotation-scale` (tool turn per hand turn), and **X** to lock or unlock the wrist (the tool keeps its orientation). |
| Sudden big joint jumps | A **wrist-flip guard** (`quest/continuity.py`). In the station's Quest episodes, when joint5 sits on its limit (-82°) a small hand turn makes the IK go the long way round: joint4 and joint6 swing 130-190° for 1.5-2.5 s while the tool turns only 8-40°. The guard sees the wrist turning a lot for little tool turn, and holds the tool's orientation from there while its position keeps following. `--no-flip-guard` turns it off. |
| Wrist lags hand turns | **Wrist joints capped at 4 rad/s** (`--wrist-max-joint-speed`) instead of the arm's 2.5, a **faster rotation filter** (0.5 Hz when still, rising 3 Hz per rad/s), and **no rotation dead band** (the old 2° band left a 2.8° median turn error). |
| Tracking glitch lunges | A controller pose that jumps 8 cm or 45° in one reading is treated as a glitch. The arm holds, and the clutch re-engages on the new pose. `--no-jump-guard` turns it off. |
| Home button does nothing | The kit ignores the right stick while recording. Here, a press warns you, and a **second press within 2 s discards the episode and homes**. Homing used to run forever on an arm that can't reach home (when it's blocked, say), which froze teleop. Now it stops with a warning, and it also stops after 30 s. Homing from below the home height **lifts the tool straight up first**, and only then moves the joints home, so the kit's joint-space path can't drag the gripper along the table. |

```bash
cd ~/physical-ai-hacks
.venv/bin/python ~/follower_code/quest/quest_collect.py --config configs/station.yaml --dry-run
.venv/bin/python ~/follower_code/quest/quest_collect.py --config configs/station.yaml --dataset-dir ~/hackathon_data/my-task
```

- **Tuning:** `--filter-min-cutoff` lower is smoother when still. `--filter-beta` higher is less lag when moving
  fast. `--rot-filter-beta` is the same trade-off for hand turns: 2 is smoother, 8 follows sooner. `--rotation-scale 0.5`
  halves wrist turns, and `--control-hz 30` restores the kit's rate.
- **Replay bench** (offline, on the Metal URDF): 40 fast-wrist events from `unplug-v1` and `unplug-split-v1` replayed
  through the IK. With the kit's IK, 7 of the 40 swing joint4 or joint6 more than 90° (the worst swings 194°). With the
  guard, 2 of the 40 do (the worst swings 116°), and the tool's position stays within 8 mm. Human motion recorded on
  the leader arm (14 episodes), with 1.5 mm and 0.4° of Quest-like noise added: the wrist follows hand turns 27 ms
  behind, where the previous defaults lagged 105 ms. The median turn error is 1.0° (it was 2.8°), and rotation jitter
  is 40 % below the unfiltered kit's.
- **Loop rate:** check the dashboard's loop rate. If it can't hold 60 Hz, use `--control-hz 30`, or recordings
  will be stretched in time.
- **Measured offline:** on the kit's test arm with 2 mm of tracking noise, the tool jitters 4.7× less than the
  kit's (0.47 mm against 2.2 mm), at about 17 ms more lag on a 5 cm move.
- **Tests:** `.venv/bin/python -m pytest -q ~/follower_code/quest/tests` runs them over the kit's own fakes. The 5 new
  ones, for homing, glitches and the flip guard, have not been run on the NUC yet. `test_continuity.py` needs only numpy
  (6 pass).
- **Not tested:** a real Quest. The gravity model is proven on the arms through the leader/follower, not yet through this launcher.

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
- **Follower motion:** the follower closes the gap to the leader at 0.4 rad/s, for at most 3 s, then follows at
  `--follow-speed` (4 rad/s; MakerMods' 280°/s follower cap still applies). If it falls more than 0.35 rad behind,
  the kit re-syncs it.
- **Gravity feedforward (on by default):** without it, every joint sits under its command by its gravity torque
  over kp. The first real episodes showed shoulder_lift +4 to +6° and elbow_flex +2.5 to +4.3° on average, and the
  old 1.5 rad/s cap was hit on every joint. The follower's MIT frames now carry MakerMods' own Metal gravity torque
  (the leader's model). `--no-gravity-ff` turns it off.
- **Checking tracking:** `.venv/bin/python ~/follower_code/metal/tools/tracking_report.py <dataset>` prints the
  per-joint error of any recorded dataset.
- **Dry run:** both arms open with torque off. Nothing is energized and the leader is limp, so support it.
- **Tests:** `cd ~/physical-ai-hacks && .venv/bin/python -m pytest -q ~/follower_code/metal/tests` runs the kit's
  real Collector and writer over fake Metal devices, then runs `tools/add_eef.py`, which applies Dream's dataset
  checks. All 10 pass on the hackathon NUC.
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
