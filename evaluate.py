"""Two evaluations, both against the real simulator.

1. Open-loop prediction: encode one real observation, then imagine up to 50 steps (2.5 s)
   from the actions alone. How far is the imagined fingertip from the real one?

2. Closed-loop control: reach 100 random targets by planning inside the model.
   Compared with the same planner using the true simulator, and with random actions.

    python evaluate.py            # writes results/metrics.json and the figures
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from env import JOINT_RANGE, N_JOINTS, REACH, ArmEnv, tip_from_qpos
from model import WorldModel, n_params
from plan import MPPI, LearnedModel, OracleModel
from train import TIP_SCALE, dev, to_img

HORIZON_OPEN = 50
EP_STEPS = 80                  # 4 s to reach and hold
SUCCESS_CM = 1.0


def load(path):
    m = WorldModel().to(dev).eval()
    m.load_state_dict(torch.load(path, map_location=dev))
    return m


@torch.no_grad()
def open_loop(m, d):
    """Fingertip error (cm) and image error at every imagined step, over all test episodes."""
    frames = torch.from_numpy(d["frames"]).to(dev)
    actions = torch.from_numpy(d["actions"]).to(dev)
    tips = d["tips"]
    z = m.encode(to_img(frames[:, 0]), to_img(frames[:, 1]))
    tip_err, img_err, imagined = [], [], []
    for k in range(HORIZON_OPEN):
        z = m.step(z, actions[:, 1 + k])
        pred = (m.tip(z) / TIP_SCALE).cpu().numpy()
        tip_err.append(100 * np.linalg.norm(pred - tips[:, 2 + k], axis=1))
        img = torch.sigmoid(m.decode(z))
        img_err.append(((img - to_img(frames[:, 2 + k])) ** 2).mean((1, 2, 3)).cpu().numpy())
        imagined.append((img[:8].movedim(1, -1) * 255).byte().cpu().numpy())
    return np.array(tip_err), np.array(img_err), np.stack(imagined, 1)       # (K, E), (K, E), (8, K, 64, 64, 3)


def reach_tasks(n, seed=123):
    rng = np.random.default_rng(seed)
    starts = rng.uniform(-0.9, 0.9, (n, N_JOINTS)) * JOINT_RANGE
    targets = tip_from_qpos(rng.uniform(-0.9, 0.9, (n, N_JOINTS)) * JOINT_RANGE).astype(np.float32)
    return starts, targets


def run_reaching(make_model, starts, targets, needs_images, record=0, seed=0):
    env = ArmEnv(render=True)
    rng = np.random.default_rng(seed)
    finals, times, traces = [], [], []
    for i, (q0, target) in enumerate(zip(starts, targets)):
        env.reset(rng, qpos=q0)
        model = make_model(env)
        ctrl = MPPI(model, seed=i) if model is not None else None
        prev = frame = env.render()
        dist, vid = [], []
        for t in range(EP_STEPS):
            if ctrl is None:
                a = rng.uniform(-1, 1, N_JOINTS)
            else:
                t0 = time.perf_counter()
                model.observe(env, prev, frame)
                a = ctrl.act(target)
                times.append(time.perf_counter() - t0)
            env.step(a)
            prev, frame = frame, (env.render() if (needs_images or i < record) else frame)
            dist.append(100 * np.linalg.norm(env.tip() - target))
            if i < record:
                vid.append(frame)
        finals.append(np.mean(dist[-5:]))                # average over the last 0.25 s: it has to stay there
        if i < record:
            traces.append((np.stack(vid), target, np.array(dist)))
    finals = np.array(finals)
    return {"success_rate": float((finals < SUCCESS_CM).mean()), "median_final_cm": float(np.median(finals)),
            "mean_final_cm": float(finals.mean()), "p90_final_cm": float(np.quantile(finals, 0.9)),
            "ms_per_plan": float(1000 * np.mean(times)) if times else 0.0, "episodes": len(finals)}, finals, traces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    d = np.load("data/test.npz")
    out = {"success_threshold_cm": SUCCESS_CM, "arm_reach_cm": 100 * REACH, "open_loop": {}, "reaching": {}}
    models = {name: load(f"ckpt/{name}.pt") for name in ("multistep", "onestep") if os.path.exists(f"ckpt/{name}.pt")}
    out["params_M"] = round(n_params(next(iter(models.values()))) / 1e6, 2)

    curves, dreams = {}, {}
    for name, m in models.items():
        tip_err, img_err, imagined = open_loop(m, d)
        curves[name], dreams[name] = tip_err, imagined
        out["open_loop"][name] = {f"tip_error_cm_at_{k}": round(float(np.median(tip_err[k - 1])), 3) for k in (1, 5, 10, 20, 50)}
        out["open_loop"][name]["image_mse_at_50"] = float(img_err[-1].mean())
        print("open loop", name, out["open_loop"][name], flush=True)

    starts, targets = reach_tasks(a.episodes)
    planners = [("random actions", lambda env: None, False)]
    for name, m in models.items():
        planners.append((f"learned world model ({name})", lambda env, m=m: LearnedModel(m, dev), True))
    planners.append(("true simulator (oracle)", lambda env: OracleModel(env), False))
    finals, traces = {}, {}
    for label, make, needs in planners:
        r, f, tr = run_reaching(make, starts, targets, needs, record=3)
        out["reaching"][label], finals[label], traces[label] = r, f, tr
        print("reaching", label, {k: round(v, 3) for k, v in r.items()}, flush=True)
        json.dump(out, open("results/metrics.json", "w"), indent=2)
    np.savez("results/raw.npz", **{f"curve_{k}": v for k, v in curves.items()},
             **{f"final_{k}": v for k, v in finals.items()}, real=d["frames"][:8, 2:2 + HORIZON_OPEN],
             **{f"dream_{k}": v for k, v in dreams.items()},
             **{f"reach_{k}_{i}_{part}": tr[i][j] for k, tr in traces.items() for i in range(len(tr))
                for j, part in enumerate(("video", "target", "dist"))})


if __name__ == "__main__":
    main()
