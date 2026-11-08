"""How well a follower arm tracked its command, per joint, in recorded MakerMods-layout LeRobot v2.1 episodes.

    .venv/bin/python ~/follower_code/metal/tools/tracking_report.py ~/hackathon_data/my-task [--arm right]

For each episode: mean (bias: a steady sag shows here), RMS and max of command - measured in degrees, the peak
command speed (a value at the cap means the command was trailing the leader), and the gripper's error in mm.
The layout is the kit's MakerMods one: [left joints x6, left grip mm, right joints x6, right grip mm].
"""
import argparse
import pathlib

import numpy as np
import pyarrow.parquet as pq

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll")


def report(path: pathlib.Path, arm: str, fps: float) -> str:
    table = pq.read_table(path).to_pydict()
    state, action = np.array(table["observation.state"]), np.array(table["action"])
    lo = 0 if arm == "left" else 7
    err = np.degrees(action[:, lo:lo + 6] - state[:, lo:lo + 6])
    if len(action) > 1:
        speed = np.degrees(np.abs(np.diff(action[:, lo:lo + 6], axis=0))).max(axis=0) * fps
    else:
        speed = np.zeros(6)
    grip = action[:, lo + 6] - state[:, lo + 6]
    rows = [f"{path.stem}: {len(state)} frames ({len(state) / fps:.1f} s), {arm} arm, command - measured (deg)",
            f"  {'joint':14s} {'mean':>7s} {'rms':>7s} {'max':>7s}  {'peak cmd deg/s':>14s}"]
    for i, name in enumerate(JOINTS):
        rows.append(f"  {name:14s} {err[:, i].mean():+7.2f} {np.sqrt((err[:, i] ** 2).mean()):7.2f} "
                    f"{np.abs(err[:, i]).max():7.1f}  {speed[i]:14.0f}")
    rows.append(f"  {'gripper (mm)':14s} {grip.mean():+7.1f} {np.sqrt((grip ** 2).mean()):7.1f} "
                f"{np.abs(grip).max():7.1f}")
    return "\n".join(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("dataset", type=pathlib.Path)
    p.add_argument("--arm", choices=("left", "right"), default="right", help="the follower (default right)")
    p.add_argument("--fps", type=float, default=30.0)
    a = p.parse_args()
    files = sorted(a.dataset.expanduser().glob("data/*/*.parquet"))
    if not files:
        raise SystemExit(f"no episodes under {a.dataset}/data")
    for f in files:
        print(report(f, a.arm, a.fps))


if __name__ == "__main__":
    main()
