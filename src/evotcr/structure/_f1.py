"""Numerically preserved F1 basis, adapted from the audited research builder.

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
FAR = 32.0  # quantisation ceiling; missing and very-far both land here
TR = [
    ("c45", lambda d: 1.0 / (1.0 + np.exp((d - 4.5) / 0.5))),
    ("c60", lambda d: 1.0 / (1.0 + np.exp((d - 6.0) / 1.0))),
    ("lj", lambda d: 1.0 / (1.0 + (d / 5.0) ** 6)),
]
SEG = ["all", "apex"]


def _blocks(n, step):
    for i in range(0, n, step):
        yield i, min(i + step, n)


def build(qa, qb, length, cdr3a=None, cdr3b=None, peptides=None, step=500):
    lengths = np.asarray(length, dtype=int)
    n = len(lengths)
    cols, names = [], []

    for ci, chain in enumerate("ab"):
        Q = (qa, qb)[ci]
        ln = lengths[:, ci]
        # ---- accumulators -------------------------------------------------
        pos = {}  # (tr, seg) -> (n, nch, 9)   sum over CDR3 of transform
        mx = {}  # (tr, seg) -> (n, nch, 9)   max over CDR3 of transform
        mn = {}  # seg       -> (n, nch, 9)   min distance over CDR3
        sd = {}  # seg       -> (n, nch)      across-seed sd of the total contact
        for t, _ in TR:
            for s in SEG:
                pos[t, s] = np.zeros((n, len(CH), 9), np.float32)
                mx[t, s] = np.zeros((n, len(CH), 9), np.float32)
        for s in SEG:
            mn[s] = np.zeros((n, len(CH), 9), np.float32)
            sd[s] = np.zeros((n, len(CH)), np.float32)

        for i0, i1 in _blocks(n, step):
            q = np.asarray(Q[i0:i1][:, :, :, :, CH])  # (b,3,20,9,nch)
            d = q.astype(np.float32)
            bad = q == 255
            d *= 0.125
            dfar = np.where(bad, FAR, d)  # for min pooling
            lrange = np.arange(20)[None, :]
            lo = np.where(ln[i0:i1] >= 8, 3, 0)[:, None]
            hi = (ln[i0:i1] - 3)[:, None]
            segm = {"all": (lrange < ln[i0:i1][:, None]), "apex": (lrange >= lo) & (lrange < hi)}
            for s in SEG:
                M = segm[s][:, None, :, None, None]  # (b,1,20,1,1)
                dm = np.where(M, dfar, FAR)
                mn[s][i0:i1] = (
                    dm.min(axis=2).mean(axis=1).transpose(0, 2, 1)
                )  # min over l, mean over seed
                for t, f in TR:
                    v = np.where(M & ~bad, f(d), 0.0)
                    ps = v.sum(axis=2)  # (b,3,9,nch)
                    pos[t, s][i0:i1] = ps.mean(axis=1).transpose(0, 2, 1)
                    mx[t, s][i0:i1] = v.max(axis=2).mean(axis=1).transpose(0, 2, 1)
                    if t == "c45":
                        sd[s][i0:i1] = ps.sum(axis=2).std(axis=1)
        del Q

        # ---- emit ---------------------------------------------------------
        for t, _ in TR:
            for s in SEG:
                for k, cn in enumerate(CHN):
                    cols.append(pos[t, s][:, k, :])
                    names += [f"{chain}|{cn}|{t}|{s}|p{p}" for p in range(9)]
                    cols.append(pos[t, s][:, k, :].sum(1, keepdims=True))
                    names.append(f"{chain}|{cn}|{t}|{s}|SUM")
                    cols.append(mx[t, s][:, k, :].max(1, keepdims=True))
                    names.append(f"{chain}|{cn}|{t}|{s}|MAX")
        for s in SEG:
            for k, cn in enumerate(CHN):
                cols.append(mn[s][:, k, :])
                names += [f"{chain}|{cn}|dmin|{s}|p{p}" for p in range(9)]
                cols.append(mn[s][:, k, :].min(1, keepdims=True))
                names.append(f"{chain}|{cn}|dmin|{s}|MIN")
                cols.append(sd[s][:, k : k + 1])
                names.append(f"{chain}|{cn}|seedsd|{s}")

    F = np.column_stack(cols).astype(np.float32)
    assert F.shape[1] == len(names), (F.shape, len(names))
    return F, names
