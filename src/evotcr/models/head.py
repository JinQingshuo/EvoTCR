"""Audited residual MLP equations from the 2026-11-09 training-only run."""

import numpy as np
import torch
from torch import nn
from scipy.special import expit
from scipy.stats import rankdata


def encode(raw, config, mean=None, sd=None):
    if mean is None:
        mean = raw.mean(0)
        sd = np.maximum(raw.std(0), 1e-8)
    z = expit(np.clip((raw - mean) / sd / config["input_temperature"], -12, 12))
    if config["representation"] == "dual":
        z = np.c_[
            z, (rankdata(raw.reshape(-1, 10, 17), axis=1, method="average") / 10).reshape(-1, 17)
        ]
    return z.astype(np.float32), mean, sd


class Model(nn.Module):
    def __init__(self, x, cfg, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.cfg = cfg
        p = x.shape[1]
        z = torch.logit(torch.as_tensor(x).clamp(1e-5, 1 - 1e-5))
        self.register_buffer("mean", z.mean(0))
        self.register_buffer("sd", z.std(0, unbiased=False).clamp_min(1e-6))
        self.linear = nn.Linear(p, 1, bias=False)
        nn.init.zeros_(self.linear.weight)
        layers = []
        for i in range(cfg["depth"]):
            layers.extend(
                [
                    nn.Linear(p if i == 0 else cfg["width"], cfg["width"]),
                    {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[cfg["activation"]](),
                    nn.Dropout(cfg["dropout"]),
                ]
            )
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(cfg["width"], 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, parts=False):
        z = (torch.logit(x.clamp(1e-5, 1 - 1e-5)) - self.mean) / self.sd
        r = self.out(self.body(z)).reshape(-1) * self.cfg["residual_scale"]
        s = self.linear(x).reshape(-1) + r
        return (s, r) if parts else s


def normalize(score, mode, train_sd):
    if mode == "raw":
        return score.copy()
    x = score.reshape(-1, 10)
    if mode == "center":
        return (x - x.mean(1, keepdims=True)).ravel()
    temp = float(mode.rsplit("_", 1)[1]) * max(float(train_sd), 1e-8)
    q = x / temp
    q -= q.max(1, keepdims=True)
    q = np.exp(q)
    return (q / q.sum(1, keepdims=True)).ravel()
