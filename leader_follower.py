"""Use one YAM arm as a gravity-compensated leader and the other as a position follower, and record demos.

The leader arm runs i2rt's zero-gravity mode (gravity + Coulomb compensation, small kd, no position gain), so the
operator moves it by hand, including its gripper. Each tick the follower is commanded to the leader's joint angles
(joint copy, gripper included), rate-limited so it can never jump. Copy keeps the two arms moving in parallel, so
they never converge. A mirror map is not offered: the YAM chain has lateral offsets, so the same arm cannot reach
the mirror image of a pose (about 3 cm per link off in the MJCF).

    python leader_follower.py --sim                       # no hardware: i2rt SimRobot, leader is wiggled by a test signal
    python leader_follower.py --leader can_follower_l --follower can_follower_r --cam scene=0 --cam wrist_r=2

Keys (type then Enter): [Enter] start/stop an episode, s = mark last episode success, f = fail, q = quit.
Episodes go to runs_teleop/<timestamp>/episode_NNN/{data.npz, <cam>.mp4, <cam>_t.npy, meta.json}.

Start with both arms in their rest pose. On exit both arms are driven back to the pose they had at start-up before
the CAN buses are closed, because closing sets the motor torques to zero.
The real Dream station holds these CAN links and arms.lock: stop it first, with the organisers' permission.
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parent / 'runs_teleop'


def move(robot, target, duration: float, hz: float = 100.):
    """Smooth (minimum-jerk) joint move from the current pose, for any i2rt Robot (SimRobot has no move_joints)."""
    q0, target = robot.get_joint_pos().copy(), np.asarray(target, float)
    n = max(1, int(duration * hz))
    for i in range(1, n + 1):
        u = i / n
        robot.command_joint_pos(q0 + (target - q0) * (10 * u**3 - 15 * u**4 + 6 * u**5))
        time.sleep(1. / hz)


# ---------------------------------------------------------------------------------------------------------- cameras
class Camera(threading.Thread):
    """OpenCV capture thread. While `recording`, frames go to an mp4 and their wall-clock stamps to a list."""

    def __init__(self, name: str, device: str, fps: float = 30., size=(640, 480)):
        super().__init__(daemon=True, name=f'cam-{name}')
        import cv2
        self.cv2, self.name, self.fps = cv2, name, fps
        self.cap = cv2.VideoCapture(int(device) if device.isdigit() else device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise RuntimeError(f'camera {name} ({device}) did not open')
        self.lock, self.writer, self.stamps, self.stop = threading.Lock(), None, [], threading.Event()

    def begin(self, path: Path):
        with self.lock:
            self.path, self.writer, self.stamps = path, None, []

    def end(self) -> list[float]:
        with self.lock:
            if self.writer is not None:
                self.writer.release()
            self.writer, stamps, self.stamps = None, self.stamps, []
            self.path = None
            return stamps

    def run(self):
        self.path = None
        while not self.stop.is_set():
            ok, frame = self.cap.read()
            t = time.time()
            if not ok:
                time.sleep(0.005)
                continue
            with self.lock:
                if self.path is not None:
                    if self.writer is None:
                        h, w = frame.shape[:2]
                        self.writer = self.cv2.VideoWriter(str(self.path), self.cv2.VideoWriter_fourcc(*'mp4v'),
                                                           self.fps, (w, h))
                    self.writer.write(frame)
                    self.stamps.append(t)
        self.cap.release()


# ---------------------------------------------------------------------------------------------------------- teleop
def station_body(path: Path = Path.home() / '.dreamlabs' / 'station.yaml') -> str | None:
    """The `body:` a Dream station config names, or None without one."""
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.startswith('body:'):
            return line.split(':', 1)[1].split('#')[0].strip()
    return None


class Teleop:
    def __init__(self, args):
        from i2rt.robots.get_robot import get_yam_robot
        if not args.sim:
            body = station_body()
            if body != 'yam' and not args.arms_are_yam:
                raise SystemExit(
                    f'refusing: this script drives i2rt YAM arms, and the station config says body: {body}. '
                    'Running YAM motor ids, directions and gravity model on another arm (e.g. a MakerMods Metal) '
                    'can drop or drive it. Pass --arms-are-yam only if the arms really are YAMs.')
        self.args = args
        self.leader = get_yam_robot(channel=args.leader, zero_gravity_mode=True, sim=args.sim)
        self.follower = get_yam_robot(channel=args.follower, zero_gravity_mode=False, sim=args.sim)
        self.q_leader0 = self.leader.get_joint_pos().copy()
        self.q_follower0 = self.follower.get_joint_pos().copy()
        n = len(self.q_follower0)
        self.n_arm = n - 1                                          # last index is the gripper
        self.vmax = np.r_[np.full(self.n_arm, args.max_joint_speed), args.max_gripper_speed]
        self.cams = [Camera(*c.split('=', 1)) for c in args.cam]
        for c in self.cams:
            c.start()
        self.session = RUNS / time.strftime('%Y%m%dT%H%M%S')
        self.session.mkdir(parents=True, exist_ok=True)
        self.keys: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._read_keys, daemon=True).start()
        self.episodes: list[Path] = []
        self.rec: dict[str, list] | None = None
        print(f'leader {args.leader}  follower {args.follower}  '
              f'session {self.session}')

    def _read_keys(self):
        for line in sys.stdin:
            self.keys.put(line.strip().lower())

    # ---- recording
    def _start_episode(self):
        ep = self.session / f'episode_{len(self.episodes):03d}'
        ep.mkdir()
        self.episodes.append(ep)
        self.rec = {k: [] for k in ('t', 'leader_q', 'action', 'q', 'v', 'eff', 'gripper', 'gripper_eff')}
        for c in self.cams:
            c.begin(ep / f'{c.name}.mp4')
        print(f'REC {ep.name}  (Enter to stop)')

    def _stop_episode(self):
        ep, rec, self.rec = self.episodes[-1], self.rec, None
        for c in self.cams:
            np.save(ep / f'{c.name}_t.npy', np.array(c.end()))
        np.savez_compressed(ep / 'data.npz', **{k: np.array(v) for k, v in rec.items()})
        meta = dict(leader=self.args.leader, follower=self.args.follower, hz=self.args.hz, sim=self.args.sim, cams=[c.name for c in self.cams],
                    steps=len(rec['t']), duration_s=float(rec['t'][-1] - rec['t'][0]) if rec['t'] else 0.,
                    success=None, columns=dict(action='rate-limited follower joint command, gripper last',
                                               leader_q='leader joint pos, gripper last',
                                               q='follower joint pos (arm)', gripper='follower gripper pos'))
        (ep / 'meta.json').write_text(json.dumps(meta, indent=1))
        print(f'saved {ep.name}: {meta["steps"]} steps, {meta["duration_s"]:.1f} s  (s/f to label)')

    def _label(self, ok: bool):
        if not self.episodes:
            return
        f = self.episodes[-1] / 'meta.json'
        if f.exists():
            meta = json.loads(f.read_text())
            meta['success'] = ok
            f.write_text(json.dumps(meta, indent=1))
            print(f'{self.episodes[-1].name} labelled {"success" if ok else "fail"}')

    # ---- motion
    def engage(self):
        """Bring the follower to the mapped leader pose slowly before tracking starts."""
        target = (self.leader.get_joint_pos())
        err = np.abs(target[:self.n_arm] - self.follower.get_joint_pos()[:self.n_arm]).max()
        duration = max(2., err / (0.5 * self.args.max_joint_speed))
        print(f'engaging: follower moves {err:.2f} rad max over {duration:.1f} s, hold the leader still')
        move(self.follower, target, duration)

    def run(self):
        dt = 1. / self.args.hz
        self.engage()
        cmd = self.follower.get_joint_pos().copy()
        over, t_next, k = 0., time.perf_counter(), 0
        print('tracking. Enter = start/stop episode, s/f = label, q = quit')
        while True:
            try:
                key = self.keys.get_nowait()
            except queue.Empty:
                key = None
            if key == 'q':
                break
            if key == '':
                self._start_episode() if self.rec is None else self._stop_episode()
            elif key in ('s', 'f'):
                self._label(key == 's')

            if self.args.sim:                                   # test signal: the 'operator' wiggles the leader
                w = self.q_leader0.copy()
                w[:self.n_arm] += 0.25 * np.sin(2 * np.pi * 0.2 * k * dt + np.arange(self.n_arm))
                w[self.n_arm] = 0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * k * dt)
                self.leader.command_joint_pos(w)

            q_lead = self.leader.get_joint_pos().copy()
            target = (q_lead)
            cmd = cmd + np.clip(target - cmd, -self.vmax * dt, self.vmax * dt)
            self.follower.command_joint_pos(cmd.copy())
            obs = self.follower.get_observations()

            lag = np.abs(obs['joint_pos'] - cmd[:self.n_arm]).max()
            over = over + dt if lag > self.args.max_tracking_error else 0.
            if over > 0.5:
                print(f'STOP: follower lag {lag:.2f} rad for >0.5 s (blocked or overloaded)')
                break

            if self.rec is not None:
                r = self.rec
                r['t'].append(time.time()); r['leader_q'].append(q_lead); r['action'].append(cmd.copy())
                r['q'].append(obs['joint_pos']); r['v'].append(obs['joint_vel']); r['eff'].append(obs['joint_eff'])
                r['gripper'].append(obs['gripper_pos']); r['gripper_eff'].append(obs['gripper_eff'])
                if self.args.sim and len(r['t']) >= self.args.sim_episode_steps:
                    self._stop_episode()

            if self.args.sim and k == 50:
                self.keys.put('')                                # auto-start one episode in the self-test
            if self.args.sim and k > 50 + self.args.sim_episode_steps + 20:
                break
            k += 1
            t_next += dt
            sleep = t_next - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            elif sleep < -5 * dt:
                t_next = time.perf_counter()

    def shutdown(self):
        if self.rec is not None:
            self._stop_episode()
        print('returning both arms to their start-up pose')
        hold = self.leader.get_joint_pos().copy()
        self.leader.command_joint_pos(hold)                    # leader stops being free before it is moved
        move(self.follower, self.q_follower0, 3.)
        move(self.leader, self.q_leader0, 3.)
        for c in self.cams:
            c.stop.set()
        for r in (self.follower, self.leader):
            r.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--leader', default='can_follower_l', help='CAN link of the arm moved by hand')
    p.add_argument('--follower', default='can_follower_r', help='CAN link of the arm that records the task')
    p.add_argument('--hz', type=float, default=100.)
    p.add_argument('--max-joint-speed', type=float, default=2.0, help='rad/s cap on the follower command')
    p.add_argument('--max-gripper-speed', type=float, default=4.0, help='gripper units/s')
    p.add_argument('--max-tracking-error', type=float, default=0.35, help='rad; stop if exceeded for 0.5 s')
    p.add_argument('--cam', action='append', default=[], help='name=device, e.g. scene=0 or wrist_r=/dev/video2')
    p.add_argument('--sim', action='store_true', help='i2rt SimRobot for both arms, scripted leader, one episode')
    p.add_argument('--sim-episode-steps', type=int, default=300)
    p.add_argument('--arms-are-yam', action='store_true',
                   help='override the station-config check when the arms are YAMs but no config says so')
    t = Teleop(p.parse_args())
    try:
        t.run()
    except KeyboardInterrupt:
        print('interrupted')
    finally:
        t.shutdown()


if __name__ == '__main__':
    main()
