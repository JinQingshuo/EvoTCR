"""Small pure numerical helpers; no server paths or implicit datasets."""

import numpy as np


def auc(y, s):
    order = np.argsort(-s)
    pos = y.sum()
    neg = len(y) - pos
    if not pos or not neg:
        return np.nan
    fp = 0.0
    tp = 0.0
    area = 0.0
    pf = 0.0
    pt = 0.0
    a = 0
    while a < len(order):
        b = a + 1
        while b < len(order) and s[order[b]] == s[order[a]]:
            b += 1
        for c in range(a, b):
            if y[order[c]]:
                tp += 1
            else:
                fp += 1
        f = fp / neg
        t = tp / pos
        if f >= 0.1:
            te = pt + (t - pt) * (0.1 - pf) / (f - pf) if f > pf else t
            area += (0.1 - pf) * (te + pt) / 2
            break
        area += (f - pf) * (t + pt) / 2
        pf = f
        pt = t
        a = b
    return 0.5 * (1 + (area - 0.005) / 0.095)


def macro(s, y, g):
    z = np.empty(20)
    for j in range(20):
        z[j] = auc(y[g == j], s[g == j])
    return np.mean(z)
