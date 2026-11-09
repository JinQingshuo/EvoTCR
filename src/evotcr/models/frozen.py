"""Frozen replay; never learns query statistics or performs global ranking."""

from pathlib import Path
import json
import numpy as np
import torch
from scipy.stats import rankdata
from .head import Model, encode, normalize
from evotcr.parameters import factor_parameters


def factors(root, F, base, D, g, h):
    root = Path(root)
    fit = root / "fitted_factors"
    if not fit.exists():
        fit = root / "final_factors"
    selected = json.loads((root / "SELECTED_MODEL.json").read_text())
    configs = factor_parameters(selected)
    source = np.load(
        root / "train.npz" if (root / "train.npz").exists() else root / "inputs/train.npz"
    )
    n = len(h)
    assert F.shape == (10 * n, 6250) and base.shape == (10 * n, 7) and D.shape == (n, 1000)
    assert np.all(source["HL"][g.reshape(n, 10)] == h[:, None])
    assert np.all(np.diff(g.reshape(n, 10), axis=1) > 0)
    out = np.zeros((n * 10, 17))
    out[:, :7] = base
    out[:, 3:7] = (
        rankdata(base.reshape(n, 10, 7)[:, :, 3:7], axis=1, method="average") / 10
    ).reshape(n * 10, 4)
    ti = np.repeat(np.arange(n), 10)

    def apply_kernel(j, K):
        P = np.zeros((n, 20))
        for p in sorted(fit.glob(f"channel{j}_partition*_fit.npz")):
            a = np.load(p)
            P[:, a["targets"]] = K[:, a["source_ids"]] @ a["coefficients"]
        out[:, j] = P[ti, g]

    median = json.loads((fit / "COMPLETE.json").read_text())["sequence_median_distance"]
    for j in [7, 8, 9]:
        apply_kernel(j, np.exp(-D / (median * configs[str(j)]["bandwidth"])))
    for j in range(20):
        ix = np.flatnonzero(g == j)
        for pos, col in [(True, 10), (False, 11)]:
            refs = np.flatnonzero(
                (source["cg"] == j)
                if pos
                else ((source["h"] == source["HL"][j]) & (source["cg"] != j))
            )
            out[ix, col] = -D[np.ix_(ti[ix], refs)].min(1)
    projected = {}
    for name in ["mix5", "union", "u-lj", "ptc"]:
        b = np.load(fit / (name + "_basis.npz"))
        X = F[:, b["cols"]].astype(float)
        if name == "ptc":
            X -= X.reshape(n, 10, -1).mean(1).repeat(10, axis=0)
        X = (X - b["means"][g]) / b["sd"][g] - b["mu"]
        projected[name] = (X @ b["V"].T, b["singular"], int(b["nrows"]))
    for j, name in [(12, "mix5"), (13, "mix5"), (15, "union"), (16, "u-lj")]:
        cfg = configs[str(j)]
        dim = cfg["dim"]
        whiten = cfg["whiten"]
        Z, sing, nrows = projected[name]
        W = Z[:, :dim] / (np.maximum(sing[:dim] / np.sqrt(nrows), 1e-8) ** whiten)
        frozen = np.load(fit / f"{name}_d{dim}_w{whiten}_kernel_fit.npz")
        K = np.zeros((n, 1000))
        for hh in [0, 1]:
            ids = np.flatnonzero(h == hh)
            X = W.reshape(n, 10, -1)[ids].reshape(len(ids), -1) - frozen[f"h{hh}_center"]
            X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
            K[np.ix_(ids, frozen[f"h{hh}_ids"])] = X @ frozen[f"h{hh}_reference"].T
        apply_kernel(j, K)
    cfg = configs["14"]
    Z, sing, nrows = projected["ptc"]
    dim = cfg["dim"]
    W = Z[:, :dim] / (np.maximum(sing[:dim] / np.sqrt(nrows), 1e-8) ** cfg["whiten"])
    a = np.load(fit / "channel14_fit.npz")
    out[:, 14] = W @ a["global_coefficients"] + (
        W[:, : cfg["residual_dim"]] * a["residual_coefficients"][g]
    ).sum(1)
    # Cross-fitted training factors and the sealed query cache both use float32.
    # Apply the identical representation before the sensitive frozen normalization.
    assert np.isfinite(out).all()
    return out.astype(np.float32)


def score(root, raw, seed=1109):
    root = Path(root)
    selected = json.loads((root / "SELECTED_MODEL.json").read_text())
    cfg = selected["config"]
    out = root / "models" / f"seed{seed}"
    pre = np.load(out / "preprocessing.npz")
    x, _, _ = encode(raw, cfg, pre["mean"], pre["sd"])
    state = dict(np.load(out / "weights.npz"))
    model = Model(np.zeros((2, len(state["mean"])), np.float32), cfg, seed)
    model.load_state_dict({k: torch.tensor(v) for k, v in state.items()})
    model.eval()
    with torch.no_grad():
        rawscore = model(torch.tensor(x)).numpy().astype(float)
    return normalize(rawscore, selected["selection"]["mode"], float(pre["score_sd"]))
