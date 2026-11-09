"""Fixed paired-CDR3 distance used by this release; no fitted calibration."""

import numpy as np

AA = "ARNDCQEGHILKMFPSTWYV"

_B = """4 -1 -2 -2 0 -1 -1 0 -2 -1 -1 -1 -1 -2 -1 1 0 -3 -2 0
-1 5 0 -2 -3 1 0 -2 0 -3 -2 2 -1 -3 -2 -1 -1 -3 -2 -3
-2 0 6 1 -3 0 0 0 1 -3 -3 0 -2 -3 -2 1 0 -4 -2 -3
-2 -2 1 6 -3 0 2 -1 -1 -3 -4 -1 -3 -3 -1 0 -1 -4 -3 -3
0 -3 -3 -3 9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
-1 1 0 0 -3 5 2 -2 0 -3 -2 1 0 -3 -1 0 -1 -2 -1 -2
-1 0 0 2 -4 2 5 -2 0 -3 -3 1 -2 -3 -1 0 -1 -3 -2 -2
0 -2 0 -1 -3 -2 -2 6 -2 -4 -4 -2 -3 -3 -2 0 -2 -2 -3 -3
-2 0 1 -1 -3 0 0 -2 8 -3 -3 -1 -2 -1 -2 -1 -2 -2 2 -3
-1 -3 -3 -3 -1 -3 -3 -4 -3 4 2 -3 1 0 -3 -2 -1 -3 -1 3
-1 -2 -3 -4 -1 -2 -3 -4 -3 2 4 -2 2 0 -3 -2 -1 -2 -1 1
-1 2 0 -1 -3 1 1 -2 -1 -3 -2 5 -1 -3 -1 0 -1 -3 -2 -2
-1 -1 -2 -3 -1 0 -2 -3 -2 1 2 -1 5 0 -2 -1 -1 -1 -1 1
-2 -3 -3 -3 -2 -3 -3 -3 -1 0 0 -3 0 6 -4 -2 -2 1 3 -1
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4 7 -1 -1 -4 -3 -2
1 -1 1 0 -1 0 0 0 -1 -2 -2 0 -1 -2 -1 4 1 -3 -2 -2
0 -1 0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1 1 5 -2 -2 0
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1 1 -4 -3 -2 11 2 -3
-2 -2 -2 -3 -2 -1 -2 -3 2 -1 -1 -2 -1 3 -3 -2 -2 2 7 -1
0 -3 -3 -3 -1 -2 -2 -3 -3 3 1 -2 1 -1 -2 -2 0 -3 -1 4"""

B = np.array([[int(x) for x in r.split()] for r in _B.strip().split("\n")])

IDXA = {a: i for i, a in enumerate(AA)}

GAP = 4.0

D = np.minimum(4.0, 4.0 - B)

np.fill_diagonal(D, 0.0)


def enc(s, L):
    """pad to length L by inserting gaps in the middle (gap coded as -1)."""
    s = [IDXA.get(c, -1) for c in str(s)]
    if len(s) >= L:
        # trim from the middle
        extra = len(s) - L
        h = len(s) // 2
        lo = h - extra // 2
        s = s[:lo] + s[lo + extra :]
        return np.array(s)
    g = L - len(s)
    h = len(s) // 2
    return np.array(s[:h] + [-1] * g + s[h:])


def paired_distances(query_a, query_b, reference_a, reference_b):
    for group in [query_a, query_b, reference_a, reference_b]:
        if any(not 1 <= len(s) <= 20 or not set(s) <= set(AA) for s in group):
            raise ValueError("Canonical CDR3 sequences of length 1..20 are required")
    if len(query_a) != len(query_b) or len(reference_a) != len(reference_b):
        raise ValueError("Unpaired CDR3 arrays")
    out = np.zeros((len(query_a), len(reference_a)), np.float32)
    for q, r in [(query_a, reference_a), (query_b, reference_b)]:
        q = np.stack([enc(s, 20) for s in q])
        r = np.stack([enc(s, 20) for s in r])
        for k in range(20):
            a = q[:, k, None]
            b = r[None, :, k]
            value = np.where(
                (a >= 0) & (b >= 0), D[a.clip(0), b.clip(0)], np.where((a < 0) & (b < 0), 0, GAP)
            )
            out += value.astype(np.float32)
    return out
