"""Numerically preserved F3 basis, adapted from the audited research builder.

Inputs are validated by the public structure API; no files or labels are read.
"""

from __future__ import annotations

import numpy as np

CH = [0, 1, 2, 3, 4, 5, 11, 14, 21, 24, 25, 28, 29, 30]
CHN = [
    "heavy",
    "bb_bb",
    "bb_sc",
    "sc_bb",
    "sc_sc",
    "C_C",
    "N_O",
    "O_N",
    "polNO",
    "donacc",
    "npC",
    "oppchg",
    "samechg",
    "arom",
]
TR = [
    ("c45", lambda d: 1.0 / (1.0 + np.exp((d - 4.5) / 0.5))),
    ("c60", lambda d: 1.0 / (1.0 + np.exp((d - 6.0) / 1.0))),
    ("lj", lambda d: 1.0 / (1.0 + (d / 5.0) ** 6)),
]
NB = 3


def build(qa, qb, length, cdr3a=None, cdr3b=None, peptides=None, step=500):
    lengths = np.asarray(length, dtype=int)
    n, nch = len(lengths), len(CH)
    lr = np.arange(20)
    cols, names = [], []

    for ci, chain in enumerate("ab"):
        Q = (qa, qb)[ci]
        ln = np.maximum(lengths[:, ci], 1)
        # relative loop coordinate and its box-filter bins, per row
        urel = np.where(
            lr[None, :] < ln[:, None], lr[None, :] / np.maximum(ln - 1, 1)[:, None], 0.0
        ).astype(np.float32)
        bidx = np.minimum((NB * lr[None, :] // ln[:, None]), NB - 1)
        BM = np.zeros((n, NB, 20), np.float32)
        for b in range(NB):
            BM[:, b] = (bidx == b) & (lr[None, :] < ln[:, None])

        grid = {(t, b): np.zeros((n, nch, 9), np.float32) for t, _ in TR for b in range(NB)}
        shp = {
            (t, s): np.zeros((n, nch), np.float32)
            for t, _ in TR
            for s in ("cent", "spread", "lcent", "lspread")
        }

        for i0 in range(0, n, step):
            i1 = min(i0 + step, n)
            q = np.asarray(Q[i0:i1][:, :, :, :, CH])
            bad = q == 255
            d = q.astype(np.float32) * 0.125
            ok = ((lr[None, :] < ln[i0:i1, None])[:, None, :, None, None]) & ~bad
            pp = np.arange(9, dtype=np.float32)[None, :, None]
            uu = urel[i0:i1][:, :, None]
            for t, f in TR:
                vm = np.where(ok, f(d), 0.0).mean(axis=1)  # (b,20,9,nch) seed mean
                for b in range(NB):
                    g = (vm * BM[i0:i1, b][:, :, None, None]).sum(axis=1)  # (b,9,nch)
                    grid[t, b][i0:i1] = g.transpose(0, 2, 1)
                w = vm.sum(axis=1)  # (b,9,nch) peptide profile
                tot = w.sum(axis=1) + 1e-9  # (b,nch)
                c = (w * pp).sum(axis=1) / tot
                shp[t, "cent"][i0:i1] = c
                shp[t, "spread"][i0:i1] = np.sqrt(
                    np.maximum((w * pp * pp).sum(axis=1) / tot - c * c, 0)
                )
                wl = vm.sum(axis=2)  # (b,20,nch) loop profile
                cl = (wl * uu).sum(axis=1) / tot
                shp[t, "lcent"][i0:i1] = cl
                shp[t, "lspread"][i0:i1] = np.sqrt(
                    np.maximum((wl * uu * uu).sum(axis=1) / tot - cl * cl, 0)
                )
        del Q

        for t, _ in TR:
            for k, cn in enumerate(CHN):
                for b in range(NB):
                    cols.append(grid[t, b][:, k, :])
                    names += [f"{chain}|{cn}|{t}|g{b}|p{p}" for p in range(9)]
                    cols.append(grid[t, b][:, k, :].sum(1, keepdims=True))
                    names.append(f"{chain}|{cn}|{t}|g{b}|SUM")
                for s in ("cent", "spread", "lcent", "lspread"):
                    cols.append(shp[t, s][:, k : k + 1])
                    names.append(f"{chain}|{cn}|{t}|shape|{s}")

    F = np.column_stack(cols).astype(np.float32)
    assert F.shape[1] == len(names), (F.shape, len(names))
    return F, names
