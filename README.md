# follower_code

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
