"""Train the world model.

    python train.py --unroll 10 --out ckpt/multistep.pt     # the model used for planning
    python train.py --unroll 1  --out ckpt/onestep.pt       # ablation: classic one-step training

With --unroll K the model encodes one real observation and is then rolled forward K
steps in latent space with no further images. Every one of those imagined states has
to decode to the right picture and read out the right fingertip position, so errors
that would compound during planning are penalized during training.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import WorldModel, n_params

dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
TIP_SCALE = 10.0            # metres -> units of 10 cm, so the readout loss is not tiny


def to_img(u8):             # (..., 64, 64, 3) uint8 -> (..., 3, 64, 64) float
    return u8.movedim(-1, -3).float() / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unroll", type=int, default=10)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data", default="data/train.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    d = np.load(a.data)
    frames = torch.from_numpy(d["frames"]).to(dev)               # (E, T+1, 64, 64, 3) uint8
    actions = torch.from_numpy(d["actions"]).to(dev)
    tips = torch.from_numpy(d["tips"]).to(dev) * TIP_SCALE
    E, T1 = frames.shape[:2]
    K = a.unroll

    m = WorldModel().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    print(f"unroll={K}: {n_params(m)/1e6:.2f}M params, {E*(T1-1)} transitions, device={dev}", flush=True)

    t0 = time.time()
    for step in range(a.steps):
        e = torch.randint(0, E, (a.bs,), device=dev)
        t = torch.randint(1, T1 - K, (a.bs,), device=dev)        # need frame t-1 for the encoder
        z = m.encode(to_img(frames[e, t - 1]), to_img(frames[e, t]))
        rec = F.binary_cross_entropy_with_logits(m.decode(z), to_img(frames[e, t]))
        tip = F.mse_loss(m.tip(z), tips[e, t])
        for k in range(1, K + 1):
            z = m.step(z, actions[e, t + k - 1])
            rec = rec + F.binary_cross_entropy_with_logits(m.decode(z), to_img(frames[e, t + k]))
            tip = tip + F.mse_loss(m.tip(z), tips[e, t + k])
        loss = (rec * 30.0 + tip) / (K + 1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0)
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == a.steps - 1:
            print(f"step {step:5d}  recon {rec.item()/(K+1):.5f}  tip rmse {100*(tip.item()/(K+1))**0.5/TIP_SCALE:.2f} cm  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(m.state_dict(), a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
