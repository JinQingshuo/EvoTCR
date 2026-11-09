"""Numerically preserved F5 basis, adapted from the audited research builder.

Inputs are validated by the public structure API; no files or labels are read.
"""

from __future__ import annotations

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
AIX = {a: i for i, a in enumerate(AA)}
CH_HEAVY, CH_SCSC = 0, 4  # indices into the 33-channel axis
FAR = 32.0
PROPN = ["hyd", "chg", "vol", "pol", "aro"]
# Kyte-Doolittle hydropathy, formal charge at pH 7, residue volume (A^3),
# Grantham polarity, aromaticity
_P = {
    "A": (1.8, 0, 88.6, 8.1, 0),
    "C": (2.5, 0, 108.5, 5.5, 0),
    "D": (-3.5, -1, 111.1, 13.0, 0),
    "E": (-3.5, -1, 138.4, 12.3, 0),
    "F": (2.8, 0, 189.9, 5.2, 1),
    "G": (-0.4, 0, 60.1, 9.0, 0),
    "H": (-3.2, 0.5, 153.2, 10.4, 1),
    "I": (4.5, 0, 166.7, 5.2, 0),
    "K": (-3.9, 1, 168.6, 11.3, 0),
    "L": (3.8, 0, 166.7, 4.9, 0),
    "M": (1.9, 0, 162.9, 5.7, 0),
    "N": (-3.5, 0, 114.1, 11.6, 0),
    "P": (-1.6, 0, 112.7, 8.0, 0),
    "Q": (-3.5, 0, 143.8, 10.5, 0),
    "R": (-4.5, 1, 173.4, 10.5, 0),
    "S": (-0.8, 0, 89.0, 9.2, 0),
    "T": (-0.7, 0, 116.1, 8.6, 0),
    "V": (4.2, 0, 140.0, 5.9, 0),
    "W": (-0.9, 0, 227.8, 5.4, 1),
    "Y": (-1.3, 0, 193.6, 6.2, 1),
}
PROP = np.array([_P[a] for a in AA], np.float64)
PROP = (PROP - PROP.mean(0)) / PROP.std(0)


def c45(d):
    return 1.0 / (1.0 + np.exp((d - 4.5) / 0.5))


def ljf(d):
    return 1.0 / (1.0 + (d / 5.0) ** 6)


EPS = 1e-6


def onehot(seqs, nslot=20):
    n = len(seqs)
    OH = np.zeros((n, nslot, 20), np.float32)
    PRv = np.zeros((n, nslot, 5), np.float32)
    for r, s in enumerate(seqs):
        for i, a in enumerate(s[:nslot]):
            j = AIX.get(a)
            if j is None:
                continue
            OH[r, i, j] = 1.0
            PRv[r, i] = PROP[j]
    return OH, PRv


def build(qa, qb, length, cdr3a=None, cdr3b=None, peptides=None, step=500):
    lengths = np.asarray(length, dtype=int)
    pep = peptides
    n = len(lengths)
    lr = np.arange(20)
    PQ = np.zeros((n, 5, 9), np.float32)
    for r, p in enumerate(pep):
        for j, a in enumerate(p[:9]):
            if a in AIX:
                PQ[r, :, j] = PROP[AIX[a]]
    cols, names = [], []

    for ci, chain in enumerate("ab"):
        Q = (qa, qb)[ci]
        ln = np.maximum(lengths[:, ci], 1)
        OH, PRv = onehot((cdr3a, cdr3b)[ci])
        A = {
            "pos_aa": np.zeros((n, 20, 9), np.float32),
            "comp_all": np.zeros((n, 20), np.float32),
            "comp_apex": np.zeros((n, 20), np.float32),
            "compn_all": np.zeros((n, 20), np.float32),
            "compn_apex": np.zeros((n, 20), np.float32),
            "prop_h": np.zeros((n, 5, 9), np.float32),
            "prop_sc": np.zeros((n, 5, 9), np.float32),
            "prop_lj": np.zeros((n, 5, 9), np.float32),
            "propmean": np.zeros((n, 5, 9), np.float32),
            "pair": np.zeros((n, 5, 5), np.float32),
        }

        for i0 in range(0, n, step):
            i1 = min(i0 + step, n)
            q = np.asarray(Q[i0:i1][:, :, :, :, [CH_HEAVY, CH_SCSC]]).astype(np.int16)
            ok = q != 255
            cnt = ok.sum(1)
            d = np.where(cnt > 0, np.where(ok, q, 0).sum(1) / np.maximum(cnt, 1) * 0.125, FAR)
            d = d.astype(np.float32)  # (b,20,9,2)
            live = cnt > 0
            res = (lr[None, :] < ln[i0:i1, None])[:, :, None]
            lo = np.where(ln[i0:i1] >= 8, 3, 0)[:, None]
            hi = (ln[i0:i1] - 3)[:, None]
            apx = (res[:, :, 0] & (lr[None, :] >= lo) & (lr[None, :] < hi))[:, :, None]
            m = live & res[:, :, :, None]
            v45 = np.where(m[..., 0], c45(d[..., 0]), 0.0)  # (b,20,9) heavy
            vlj = np.where(m[..., 0], ljf(d[..., 0]), 0.0)
            vsc = np.where(m[..., 1], c45(d[..., 1]), 0.0)
            oh = OH[i0:i1]
            pr = PRv[i0:i1]
            A["pos_aa"][i0:i1] = np.einsum("bip,bia->bap", v45, oh)
            A["prop_h"][i0:i1] = np.einsum("bip,bik->bkp", v45, pr)
            A["prop_sc"][i0:i1] = np.einsum("bip,bik->bkp", vsc, pr)
            A["prop_lj"][i0:i1] = np.einsum("bip,bik->bkp", vlj, pr)
            A["propmean"][i0:i1] = A["prop_h"][i0:i1] / (v45.sum(1)[:, None, :] + EPS)
            A["pair"][i0:i1] = np.einsum("bkp,blp->bkl", A["prop_h"][i0:i1], PQ[i0:i1])
            for seg, msk in (("all", res[:, :, 0]), ("apex", apx[:, :, 0])):
                w = (v45 * msk[:, :, None]).sum(2)  # (b,20) per slot
                c = np.einsum("bi,bia->ba", w, oh)
                A[f"comp_{seg}"][i0:i1] = c
                A[f"compn_{seg}"][i0:i1] = c / (c.sum(1, keepdims=True) + EPS)
            del q, ok, cnt, d
        del Q

        for a in range(20):
            cols.append(A["pos_aa"][:, a, :])
            names += [f"{chain}|aa{AA[a]}|c45|pos|p{p}" for p in range(9)]
        for seg in ("all", "apex"):
            for nm in ("comp", "compn"):
                cols.append(A[f"{nm}_{seg}"])
                names += [f"{chain}|aa{AA[a]}|{nm}|{seg}|w" for a in range(20)]
        for key, tr in (
            ("prop_h", "c45"),
            ("prop_sc", "c45sc"),
            ("prop_lj", "lj"),
            ("propmean", "mean"),
        ):
            for k, pn in enumerate(PROPN):
                cols.append(A[key][:, k, :])
                names += [f"{chain}|{pn}|{tr}|pos|p{p}" for p in range(9)]
        for k, pn in enumerate(PROPN):
            cols.append(A["pair"][:, k, :])
            names += [f"{chain}|{pn}|c45|pair|{q}" for q in PROPN]
        del A

    F = np.column_stack(cols).astype(np.float32)
    assert F.shape[1] == len(names), (F.shape, len(names))
    return F, names
