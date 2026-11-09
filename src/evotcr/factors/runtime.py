"""Independent frozen factor evaluation with shared, lazy projection caches."""

import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from evotcr.parameters import factor_parameters


class FrozenFactors:
    def __init__(self, root: Path, data: dict):
        self.root = Path(root)
        self.fit = self.root / "fitted_factors"
        self.data = data
        self.source = dict(np.load(self.root / "train.npz", allow_pickle=False))
        selected = json.loads((self.root / "SELECTED_MODEL.json").read_text())
        self.config = {int(j): cfg for j, cfg in factor_parameters(selected).items()}
        self.n = len(data["h"])
        self.ti = np.repeat(np.arange(self.n), 10)
        self.cache = {}

    def projection(self, family):
        key = ("projection", family)
        if key not in self.cache:
            b = np.load(self.fit / f"{family}_basis.npz", allow_pickle=False)
            X = self.data["F"][:, b["cols"]].astype(float)
            if family == "ptc":
                X -= X.reshape(self.n, 10, -1).mean(1).repeat(10, axis=0)
            X = (X - b["means"][self.data["g"]]) / b["sd"][self.data["g"]] - b["mu"]
            self.cache[key] = (X @ b["V"].T, b["singular"], int(b["nrows"]))
        return self.cache[key]

    def embedded(self, family, cfg):
        Z, sing, nrows = self.projection(family)
        dim = cfg["dim"]
        return Z[:, :dim] / (np.maximum(sing[:dim] / np.sqrt(nrows), 1e-8) ** cfg["whiten"])

    def regress(self, channel, kernel):
        P = np.zeros((self.n, 20))
        paths = sorted(self.fit.glob(f"channel{channel}_partition*_fit.npz"))
        if not paths:
            raise ValueError(f"Missing frozen factor coefficients: {channel}")
        for path in paths:
            a = np.load(path, allow_pickle=False)
            P[:, a["targets"]] = kernel[:, a["source_ids"]] @ a["coefficients"]
        return P[self.ti, self.data["g"]]

    def evaluate(self, channel):
        if channel in self.cache:
            return self.cache[channel]
        d = self.data
        cfg = self.config.get(channel)
        if channel < 7:
            result = d["base"][:, channel].copy()
            if channel >= 3:
                result = (
                    rankdata(result.reshape(self.n, 10), axis=1, method="average") / 10
                ).ravel()
        elif channel in [7, 8, 9]:
            median = json.loads((self.fit / "COMPLETE.json").read_text())[
                "sequence_median_distance"
            ]
            result = self.regress(channel, np.exp(-d["D"] / (median * cfg["bandwidth"])))
        elif channel in [10, 11]:
            result = np.zeros(self.n * 10)
            for j in range(20):
                ix = np.flatnonzero(d["g"] == j)
                refs = np.flatnonzero(
                    (self.source["cg"] == j)
                    if channel == 10
                    else ((self.source["h"] == self.source["HL"][j]) & (self.source["cg"] != j))
                )
                result[ix] = -d["D"][np.ix_(self.ti[ix], refs)].min(1)
        elif channel == 14:
            W = self.embedded("ptc", cfg)
            a = np.load(self.fit / "channel14_fit.npz")
            result = W @ a["global_coefficients"] + (
                W[:, : cfg["residual_dim"]] * a["residual_coefficients"][d["g"]]
            ).sum(1)
        else:
            family = {12: "mix5", 13: "mix5", 15: "union", 16: "u-lj"}[channel]
            key = ("kernel", family, cfg["dim"], cfg["whiten"])
            if key not in self.cache:
                W = self.embedded(family, cfg)
                b = np.load(self.fit / f"{family}_d{cfg['dim']}_w{cfg['whiten']}_kernel_fit.npz")
                K = np.zeros((self.n, 1000))
                for h in [0, 1]:
                    ids = np.flatnonzero(d["h"] == h)
                    X = (
                        W.reshape(self.n, 10, -1)[ids].reshape(len(ids), 10 * W.shape[1])
                        - b[f"h{h}_center"]
                    )
                    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
                    K[np.ix_(ids, b[f"h{h}_ids"])] = X @ b[f"h{h}_reference"].T
                self.cache[key] = K
            result = self.regress(channel, self.cache[key])
        result = np.asarray(result, dtype=np.float32)
        if not np.isfinite(result).all():
            raise ValueError(f"Nonfinite frozen factor {channel}")
        self.cache[channel] = result
        return result
