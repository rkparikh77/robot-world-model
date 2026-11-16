"""Latent world model: images in, a compact state, dynamics that run without images.

    encoder   two consecutive frames (so velocity is visible) -> z, 64 numbers
    dynamics  (z, action) -> next z                 the only thing the planner calls in its inner loop
    decoder   z -> image                            lets us look at what the model imagines
    readout   z -> fingertip (x, y)                 the planner's cost is computed from this
"""
import torch
import torch.nn as nn

Z = 64
A = 3


def conv(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.GroupNorm(8, o), nn.SiLU())


def deconv(i, o):
    return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.GroupNorm(8, o), nn.SiLU())


class WorldModel(nn.Module):
    def __init__(self, z: int = Z, hidden: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(conv(6, 32), conv(32, 64), conv(64, 128), conv(128, 256),      # 64 -> 4
                                     nn.Flatten(), nn.Linear(256 * 16, 512), nn.SiLU(), nn.Linear(512, z))
        self.dynamics = nn.Sequential(nn.Linear(z + A, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                                      nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                                      nn.Linear(hidden, z))
        self.dec_in = nn.Sequential(nn.Linear(z, 512), nn.SiLU(), nn.Linear(512, 256 * 16), nn.SiLU())
        self.decoder = nn.Sequential(deconv(256, 128), deconv(128, 64), deconv(64, 32),
                                     nn.ConvTranspose2d(32, 3, 4, 2, 1))
        self.readout = nn.Sequential(nn.Linear(z, 256), nn.SiLU(), nn.Linear(256, 2))

    def encode(self, prev_frame, frame):
        """Frames are (B, 3, 64, 64) in [0, 1]."""
        return self.encoder(torch.cat([prev_frame, frame], 1))

    def step(self, z, action):
        return z + self.dynamics(torch.cat([z, action], -1))

    def decode(self, z):
        return self.decoder(self.dec_in(z).view(-1, 256, 4, 4))            # logits

    def tip(self, z):
        return self.readout(z)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
