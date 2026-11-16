"""Figures and GIFs for the README, from results/raw.npz and results/metrics.json."""
import json

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from env import REACH, SIZE

S = 4                                   # upscale factor for GIFs
EXTENT = 1.12 * REACH                   # half-width of the camera view in metres


def up(img):
    return np.kron(img, np.ones((S, S, 1), np.uint8))


def to_px(xy):
    return (xy[0] / EXTENT * 0.5 + 0.5) * SIZE * S, (0.5 - xy[1] / EXTENT * 0.5) * SIZE * S


def mark(img, xy, color, r=5):
    img = img.copy()
    cx, cy = to_px(xy)
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    ring = np.abs(np.hypot(xx - cx, yy - cy) - r) < 1.2
    img[ring] = color
    return img


def pad(img, c=30):
    return np.pad(img, ((6, 6), (6, 6), (0, 0)), constant_values=c)


def main():
    raw = np.load("results/raw.npz")
    met = json.load(open("results/metrics.json"))

    # 1. dream vs reality: same actions, model never sees another image after frame 0
    real, dream = raw["real"], raw["dream_multistep"]
    frames = []
    for k in range(real.shape[1]):
        rows = [np.concatenate([pad(up(real[i, k])), pad(up(dream[i, k]))], 1) for i in range(3)]
        frames.append(np.concatenate(rows, 0))
    imageio.mimsave("results/dream.gif", frames + [frames[-1]] * 10, duration=1 / 15, loop=0)

    # 2. reaching with the learned model
    key = "learned world model (multistep)"
    vids = []
    for i in range(3):
        v, target = raw[f"reach_{key}_{i}_video"], raw[f"reach_{key}_{i}_target"]
        vids.append([pad(mark(up(f), target, (255, 60, 90))) for f in v])
    n = min(map(len, vids))
    imageio.mimsave("results/reach.gif", [np.concatenate([v[k] for v in vids], 1) for k in range(n)] +
                    [np.concatenate([v[n - 1] for v in vids], 1)] * 12, duration=1 / 20, loop=0)

    # 3. open-loop error vs horizon
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    k = np.arange(1, raw["curve_multistep"].shape[0] + 1) * 0.05
    for name, color, label in (("onestep", "#9498a0", "trained one step at a time"),
                               ("multistep", "#4269d0", "trained on 10-step imagined rollouts")):
        if f"curve_{name}" in raw:
            c = raw[f"curve_{name}"]
            ax.plot(k, np.median(c, 1), color=color, lw=2.5, label=label)
            ax.fill_between(k, np.quantile(c, 0.25, 1), np.quantile(c, 0.75, 1), color=color, alpha=0.18, lw=0)
    ax.axvline(0.5, color="black", ls=":", lw=1)
    ax.text(0.47, ax.get_ylim()[1] * 0.62, "training rollout length", fontsize=8, rotation=90, ha="right", va="center")
    ax.set_xlabel("seconds imagined with no new image"); ax.set_ylabel("fingertip error (cm), median and IQR")
    ax.set_title("How fast does the imagined arm drift from the real one?", fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    order = ["random actions", "learned world model (onestep)", "learned world model (multistep)", "true simulator (oracle)"]
    names = [n for n in order if n in met["reaching"]]
    vals = [100 * met["reaching"][n]["success_rate"] for n in names]
    colors = ["#c8ccd4", "#9498a0", "#4269d0", "#1a1f2e"]
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=[colors[order.index(n)] for n in names], height=0.6)
    for yy, v, n in zip(y, vals, names):
        ax.text(v + 1.5, yy, f"{v:.0f}%   (median {met['reaching'][n]['median_final_cm']:.1f} cm)", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels([n.replace(" (", "\n(") for n in names], fontsize=9)
    ax.set_xlim(0, 135); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(f"targets reached within {met['success_threshold_cm']:.0f} cm (%)")
    ax.set_title("Reaching 100 random targets by planning in the model", fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("results/results.png", dpi=160)
    print("wrote results/dream.gif, results/reach.gif, results/results.png")


if __name__ == "__main__":
    main()
