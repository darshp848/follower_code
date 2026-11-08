# Quest teleop, branch quest-continuity

The full usage section is "Metal stations" in the [top-level README](../README.md). This note says what this branch
changed on 7-8 Nov and what still has to run on the station.

## What changed

`quest_collect.py` runs the kit's own `dhk-collect` with fixes on top. It changes none of the kit's files.

- The wrist-flip guard (`continuity.py`) stops the big joint swings near the joint5 limit. In a replay of 40 fast
  wrist events, 7 swung a wrist joint more than 90 degrees with the kit's IK and 2 did with the guard.
- Wrist joints can go 4 rad/s instead of 2.5, and the rotation filter is faster. Wrist lag went from 105 ms to 27 ms
  in the offline replay.
- A controller jump of 8 cm or 45 degrees in one reading counts as a glitch. The arm holds until the clutch
  re-engages.
- Gravity feedforward is on for Metal arms by default (`--no-gravity-ff` turns it off).
- A low tool now lifts straight up before the joint move home, at 0.1 m/s.
- Pressing home twice within 2 s discards the episode and homes. One press only warns.

## Still to do on the NUC

These numbers are offline. None of this has run with a real Quest and real arms.

```bash
cd ~/physical-ai-hacks
.venv/bin/python ~/follower_code/quest/quest_collect.py --config configs/station.yaml --dry-run
.venv/bin/python -m pytest -q ~/follower_code/quest/tests
```

The dry run was held on 8 Nov at 02:49 UTC because the team was recording. The 5 new tests for homing, glitches and
the flip guard have not run on the NUC either. Run both when the arms are free, then try one recorded episode and
check the dashboard's loop rate. If it can't hold 60 Hz, use `--control-hz 30`.
