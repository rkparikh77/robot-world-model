"""A 3-link planar robot arm in MuJoCo, seen by a top-down camera.

The arm is driven by velocity servos with real inertia and lag, so the next
image depends on how fast the arm is already moving, not just on the command.
One environment step is 0.05 s (20 Hz control, 10 physics steps of 5 ms).

Observation: a 64x64 RGB image. Action: 3 joint-velocity commands in [-1, 1].
"""
from __future__ import annotations

import mujoco
import numpy as np

SIZE = 64
FRAME_SKIP = 10
N_JOINTS = 3
LINKS = (0.13, 0.11, 0.08)
REACH = sum(LINKS)
JOINT_RANGE = np.deg2rad([170.0, 150.0, 150.0])

XML = f"""
<mujoco model="planar_arm">
  <option timestep="0.005" gravity="0 0 0"/>
  <visual>
    <global offwidth="{SIZE}" offheight="{SIZE}"/>
    <headlight ambient="1 1 1" diffuse="0 0 0" specular="0 0 0"/>
    <quality shadowsize="0"/>
  </visual>
  <default>
    <geom contype="0" conaffinity="0" type="capsule" size="0.016"/>
    <joint type="hinge" axis="0 0 1" damping="0.02" armature="0.002" limited="true"/>
  </default>
  <worldbody>
    <camera name="top" pos="0 0 1" xyaxes="1 0 0 0 1 0" fovy="{2 * np.rad2deg(np.arctan(1.12 * REACH)):.3f}"/>
    <geom type="cylinder" size="0.022 0.005" rgba="0.55 0.55 0.6 1"/>
    <body name="link1">
      <joint name="j1" range="-170 170"/>
      <geom fromto="0 0 0 {LINKS[0]} 0 0" rgba="0.25 0.75 1 1" mass="0.3"/>
      <body name="link2" pos="{LINKS[0]} 0 0">
        <joint name="j2" range="-150 150"/>
        <geom fromto="0 0 0 {LINKS[1]} 0 0" rgba="1 0.55 0.2 1" mass="0.2"/>
        <body name="link3" pos="{LINKS[1]} 0 0">
          <joint name="j3" range="-150 150"/>
          <geom fromto="0 0 0 {LINKS[2]} 0 0" rgba="0.5 1 0.5 1" mass="0.1"/>
          <site name="tip" pos="{LINKS[2]} 0 0" size="0.001"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <velocity joint="j1" kv="0.06" gear="1" ctrlrange="-3 3"/>
    <velocity joint="j2" kv="0.03" gear="1" ctrlrange="-3 3"/>
    <velocity joint="j3" kv="0.012" gear="1" ctrlrange="-3 3"/>
  </actuator>
</mujoco>
"""
MAX_VEL = 3.0          # rad/s commanded at action = 1


class ArmEnv:
    def __init__(self, render: bool = True):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tip")
        self.renderer = mujoco.Renderer(self.model, SIZE, SIZE) if render else None

    # ---- state ---------------------------------------------------------------
    def reset(self, rng: np.random.Generator, qpos=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = rng.uniform(-0.9, 0.9, N_JOINTS) * JOINT_RANGE if qpos is None else qpos
        mujoco.mj_forward(self.model, self.data)

    def get_state(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).copy()

    def set_state(self, s):
        self.data.qpos[:], self.data.qvel[:] = s[:N_JOINTS], s[N_JOINTS:]
        mujoco.mj_forward(self.model, self.data)

    def tip(self):
        return self.data.site_xpos[self.tip_id, :2].copy()

    # ---- dynamics --------------------------------------------------------------
    def step(self, action):
        self.data.ctrl[:] = np.clip(action, -1, 1) * MAX_VEL
        mujoco.mj_step(self.model, self.data, FRAME_SKIP)

    def render(self):
        self.renderer.update_scene(self.data, camera="top")
        return self.renderer.render().copy()                 # (64, 64, 3) uint8


def random_actions(rng: np.random.Generator, n: int) -> np.ndarray:
    """Exploration: commands held for a random number of steps, with pauses and reversals."""
    out = np.zeros((n, N_JOINTS), dtype=np.float32)
    t = 0
    while t < n:
        hold = int(rng.integers(1, 12))
        a = rng.uniform(-1, 1, N_JOINTS)
        a[rng.random(N_JOINTS) < 0.25] = 0.0
        out[t:t + hold] = a
        t += hold
    return out


def tip_from_qpos(q):
    """Forward kinematics of the fingertip (for sampling reachable targets)."""
    a = np.cumsum(q, axis=-1)
    return np.stack([(np.cos(a) * LINKS).sum(-1), (np.sin(a) * LINKS).sum(-1)], -1)
