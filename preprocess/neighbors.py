"""Neighbor processing (v4 §3.10)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def expand_to_second_order(adj_first: dict[int, set[int]], L: int
                           ) -> list[list[int]]:
    """Return per-leaf neighbor lists covering 1st + 2nd order, no self-loop."""
    rows, cols = [], []
    for l, nbrs in adj_first.items():
        for n in nbrs:
            rows.append(l)
            cols.append(n)
    data = np.ones(len(rows), dtype=np.int8)
    A = sp.csr_matrix((data, (rows, cols)), shape=(L, L))
    A2 = A @ A
    combined = (A + A2).astype(bool).tolil()
    combined.setdiag(False)
    combined = combined.tocsr()
    per_leaf: list[list[int]] = []
    for l in range(L):
        row = combined[l].indices.tolist()
        per_leaf.append(row)
    return per_leaf


def pad_to_neighbor_matrix(per_leaf_nbrs: list[list[int]], N_pad: int
                           ) -> np.ndarray:
    """List-of-list neighbors -> [L, N_pad] int32 with -1 padding."""
    L = len(per_leaf_nbrs)
    out = np.full((L, N_pad), -1, dtype=np.int32)
    for l, nbrs in enumerate(per_leaf_nbrs):
        k = min(len(nbrs), N_pad)
        out[l, :k] = nbrs[:k]
    return out


def case_max_degree(per_leaf_nbrs: list[list[int]]) -> int:
    return max(len(n) for n in per_leaf_nbrs) if per_leaf_nbrs else 0
