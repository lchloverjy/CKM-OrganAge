"""Harrell's C-index in O(n log n), matching lifelines.utils.concordance_index conventions:
pairs (i, j) are comparable when subject i had an event and T_j > T_i, or T_j == T_i with subject j censored;
a pair is concordant when the predicted risk of i exceeds that of j; tied risks count 0.5."""
import numpy as np
from numba import njit


@njit(cache=True)
def _c(T, E, rank, nr):
    n = T.shape[0]
    order = np.argsort(-T, kind='mergesort')
    tree = np.zeros(nr + 1)
    conc = 0.0; tie = 0.0; comp = 0.0; total = 0.0
    i = 0
    while i < n:
        j = i
        t = T[order[i]]
        while j < n and T[order[j]] == t:
            j += 1
        # add censored subjects of this time group first
        for k in range(i, j):
            idx = order[k]
            if E[idx] == 0:
                p = rank[idx] + 1
                while p <= nr:
                    tree[p] += 1.0; p += p & (-p)
                total += 1.0
        for k in range(i, j):
            idx = order[k]
            if E[idx] == 1:
                r = rank[idx]
                # count with rank < r
                lo = 0.0; p = r
                while p > 0:
                    lo += tree[p]; p -= p & (-p)
                le = 0.0; p = r + 1
                while p > 0:
                    le += tree[p]; p -= p & (-p)
                conc += lo; tie += le - lo; comp += total
        for k in range(i, j):
            idx = order[k]
            if E[idx] == 1:
                p = rank[idx] + 1
                while p <= nr:
                    tree[p] += 1.0; p += p & (-p)
                total += 1.0
        i = j
    return (conc + 0.5 * tie) / comp


def cindex(T, risk, E):
    """C-index with higher `risk` meaning earlier event (lifelines: concordance_index(T, -risk, E))."""
    T = np.asarray(T, float); E = np.asarray(E).astype(np.int64)
    u, rank = np.unique(np.asarray(risk, float), return_inverse=True)
    return _c(T, E, rank.astype(np.int64), len(u))
