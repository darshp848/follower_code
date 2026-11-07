"""Plumbing test on i2rt SimRobot (no hardware): one scripted episode is recorded with consistent arrays.
SimRobot snaps to each command, so this does not test dynamics, gravity compensation or tracking lag.

    python -m pytest -q tests
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_sim_episode():
    before = set((ROOT / 'runs_teleop').glob('*')) if (ROOT / 'runs_teleop').exists() else set()
    out = subprocess.run([sys.executable, str(ROOT / 'leader_follower.py'), '--sim', '--sim-episode-steps', '200'],
                         stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    session = (set((ROOT / 'runs_teleop').glob('*')) - before).pop()
    ep = session / 'episode_000'
    meta = json.loads((ep / 'meta.json').read_text())
    d = np.load(ep / 'data.npz')
    n = meta['steps']
    assert n == 200 and meta['sim']
    assert d['leader_q'].shape == (n, 7) and d['action'].shape == (n, 7) and d['q'].shape == (n, 6)
    assert d['gripper'].shape == (n, 1) and d['t'].shape == (n,)
    dt = np.diff(d['t'])
    assert 80 < 1 / dt.mean() < 120                                   # ~100 Hz loop
    assert np.abs(np.diff(d['action'][:, :6], axis=0)).max() <= 2.0 / 100 + 1e-9   # rate limit holds
    assert np.abs(d['action'][:, :6] - d['q']).max() < 1e-6            # SimRobot tracks exactly
