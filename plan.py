"""Sampling-based MPC (MPPI) that can plan with either the learned world model or the true simulator.

Every control step: sample N action sequences around the current plan, roll each one
forward H steps with the model, score where the fingertip ends up, and move the plan
toward the good ones. Execute the first action, shift the plan, repeat.

The two "models" expose the same call, rollout(actions) -> fingertip positions:
  LearnedModel   rolls out in latent space. It has never seen joint angles, only images.
  OracleModel    rolls out in MuJoCo itself from the true state. The upper bound.
"""
from __future__ import annotations

import mujoco
import numpy as np
import torch
from mujoco import rollout as mj_rollout

from env import FRAME_SKIP, MAX_VEL, N_JOINTS, ArmEnv, tip_from_qpos
from train import TIP_SCALE, to_img


class LearnedModel:
    def __init__(self, wm, device):
        self.wm, self.dev = wm.eval(), device

    @torch.no_grad()
    def observe(self, env: ArmEnv, prev_frame: np.ndarray, frame: np.ndarray):
        f = to_img(torch.from_numpy(np.stack([prev_frame, frame])).to(self.dev))
        self.z = self.wm.encode(f[:1], f[1:])

    @torch.no_grad()
    def rollout(self, actions: np.ndarray) -> np.ndarray:
        a = torch.from_numpy(actions).to(self.dev)                     # (N, H, 3)
        z = self.z.expand(a.shape[0], -1)
        tips = []
        for k in range(a.shape[1]):
            z = self.wm.step(z, a[:, k])
            tips.append(self.wm.tip(z))
        return (torch.stack(tips, 1) / TIP_SCALE).cpu().numpy()        # (N, H, 2) metres


class OracleModel:
    def __init__(self, env: ArmEnv, threads: int = 8):
        self.model = env.model
        self.datas = [mujoco.MjData(env.model) for _ in range(threads)]

    def observe(self, env: ArmEnv, prev_frame=None, frame=None):
        self.state = np.concatenate([[env.data.time], env.data.qpos, env.data.qvel])

    def rollout(self, actions: np.ndarray) -> np.ndarray:
        n, h, _ = actions.shape
        ctrl = np.repeat(np.clip(actions, -1, 1) * MAX_VEL, FRAME_SKIP, axis=1)       # hold each action for 10 physics steps
        state, _ = mj_rollout.rollout(self.model, self.datas, np.tile(self.state, (n, 1)), ctrl)
        q = state[:, FRAME_SKIP - 1::FRAME_SKIP, 1:1 + N_JOINTS]                       # joint angles after each env step
        return tip_from_qpos(q).astype(np.float32)


class MPPI:
    def __init__(self, model, horizon: int = 12, samples: int = 512, sigma: float = 0.7,
                 temperature: float = 0.02, action_cost: float = 1e-4, seed: int = 0):
        self.model, self.H, self.N = model, horizon, samples
        self.sigma, self.temp, self.action_cost = sigma, temperature, action_cost
        self.rng = np.random.default_rng(seed)
        self.mean = np.zeros((horizon, N_JOINTS), np.float32)

    def reset(self):
        self.mean[:] = 0

    def act(self, target: np.ndarray) -> np.ndarray:
        noise = self.rng.normal(0, self.sigma, (self.N, self.H, N_JOINTS)).astype(np.float32)
        noise[0] = 0                                                   # always evaluate the current plan itself
        acts = np.clip(self.mean[None] + noise, -1, 1)
        tips = self.model.rollout(acts)
        d2 = ((tips - target[None, None]) ** 2).sum(-1) * 1e4          # squared distance in cm^2, (N, H)
        w = np.linspace(0.5, 1.5, self.H, dtype=np.float32)            # later steps count more
        cost = (d2 * w).mean(1) + 2.0 * d2[:, -1] + self.action_cost * 1e4 * (acts ** 2).mean((1, 2))
        cost = cost - cost.min()
        p = np.exp(-cost / (self.temp * (cost.std() + 1e-6) * 10))
        p /= p.sum()
        self.mean = (p[:, None, None] * acts).sum(0)
        a = self.mean[0].copy()
        self.mean = np.concatenate([self.mean[1:], self.mean[-1:]])    # shift the plan one step
        return a
