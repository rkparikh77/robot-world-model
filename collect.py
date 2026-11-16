"""Collect random-exploration episodes: images, actions, and (for evaluation and the
fingertip readout) the true fingertip position.

    python collect.py --episodes 1200 --out data/train.npz
    python collect.py --episodes 150 --seed 1 --out data/test.npz
"""
import argparse
import os

import numpy as np

from env import SIZE, ArmEnv, random_actions

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1200)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/train.npz")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    env = ArmEnv()
    frames = np.zeros((a.episodes, a.steps + 1, SIZE, SIZE, 3), np.uint8)
    actions = np.zeros((a.episodes, a.steps, 3), np.float32)
    tips = np.zeros((a.episodes, a.steps + 1, 2), np.float32)
    states = np.zeros((a.episodes, a.steps + 1, 6), np.float32)
    for e in range(a.episodes):
        env.reset(rng)
        actions[e] = random_actions(rng, a.steps)
        for t in range(a.steps + 1):
            frames[e, t], tips[e, t], states[e, t] = env.render(), env.tip(), env.get_state()
            if t < a.steps:
                env.step(actions[e, t])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, frames=frames, actions=actions, tips=tips, states=states)
    print(f"{a.episodes} episodes x {a.steps} steps -> {a.out} ({frames.nbytes/1e6:.0f} MB of images)")
