"""Physical reorder of per-point arrays by (leaf_id, is_surface) (v4 §3.8)."""
from __future__ import annotations

import numpy as np

from .kdtree import L_LEAVES


def build_perm_and_offsets(leaf_intervals: list[np.ndarray],
                           n_vol_keep: int
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct global permutation, leaf_member_offsets, leaf_vol_count."""
    L = L_LEAVES
    pieces_vol, pieces_surf = [], []
    vol_count = np.zeros(L, dtype=np.int32)
    counts = np.zeros(L, dtype=np.int32)
    for l in range(L):
        members = leaf_intervals[l]
        is_surf = members >= n_vol_keep
        vol_m = members[~is_surf]
        surf_m = members[is_surf]
        pieces_vol.append(vol_m)
        pieces_surf.append(surf_m)
        vol_count[l] = vol_m.shape[0]
        counts[l] = members.shape[0]
    perm = np.empty(int(counts.sum()), dtype=np.int64)
    cursor = 0
    for l in range(L):
        v = pieces_vol[l]
        s = pieces_surf[l]
        perm[cursor:cursor + v.shape[0]] = v
        cursor += v.shape[0]
        perm[cursor:cursor + s.shape[0]] = s
        cursor += s.shape[0]
    offsets = np.zeros(L + 1, dtype=np.int32)
    np.cumsum(counts, out=offsets[1:])
    return perm, offsets, vol_count


def apply_perm(arr: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Return arr[perm] (advanced indexing makes a contiguous copy)."""
    return arr[perm]


def split_perm_for_y(perm: np.ndarray, n_vol_keep: int
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Decompose the full perm into volume-local and surface-local perms."""
    is_vol = perm < n_vol_keep
    vol_perm_local = perm[is_vol].astype(np.int64)
    surf_perm_local = (perm[~is_vol] - n_vol_keep).astype(np.int64)
    return vol_perm_local, surf_perm_local
