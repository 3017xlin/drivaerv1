"""KD-tree build with leaf intervals (v4 §3.6).

Iterative max-spread split with strict equal-quality partition
(left = count // 2). Leaf membership comes from the build-time indices,
never from a descent function. Descent helper exists ONLY for test's
non-keep 87.5% points (§3.7).

Invariant: any two leaves' point counts differ by ≤ 1.
For L=2^16 on a keep subset of ~18.25M points, every leaf has ≥ 277
points, comfortably above encoder_k=32. No fallback code path.
"""
from __future__ import annotations

import numpy as np

MAX_DEPTH = 16
L_LEAVES = 1 << MAX_DEPTH                            # 65536
N_INTERNAL = L_LEAVES - 1                            # 65535


class KDTreeResult:
    """Output of build_kdtree."""
    __slots__ = ('split_axes', 'split_values', 'leaf_intervals',
                 'adjacency_first_order')

    def __init__(self, split_axes: np.ndarray, split_values: np.ndarray,
                 leaf_intervals: list[np.ndarray],
                 adjacency_first_order: dict[int, set[int]]):
        self.split_axes = split_axes
        self.split_values = split_values
        self.leaf_intervals = leaf_intervals
        self.adjacency_first_order = adjacency_first_order


def _leaf_id_from_path(node_id: int) -> int:
    """Convert a depth-16 node_id (in 0..2^17-2 heap layout) to leaf id 0..L-1."""
    return node_id - (L_LEAVES - 1)


def build_kdtree(points: np.ndarray) -> KDTreeResult:
    """Build an L=2^16 equal-quality KD-tree.

    Parameters
    ----------
    points : (N, 3) fp32 array of raw coordinates.

    Returns
    -------
    KDTreeResult
    """
    N = points.shape[0]
    if N < L_LEAVES:
        raise ValueError(
            f'KD-tree needs >= L={L_LEAVES} points, got {N}. '
            f'(Subsample ratio likely wrong upstream.)')

    split_axes = np.zeros(N_INTERNAL, dtype=np.int8)
    split_values = np.zeros(N_INTERNAL, dtype=np.float32)
    leaf_intervals: list[np.ndarray | None] = [None] * L_LEAVES
    adj: dict[int, set[int]] = {l: set() for l in range(L_LEAVES)}

    leaf_paths = np.zeros((L_LEAVES, MAX_DEPTH), dtype=np.int8)
    leaf_path_axes = np.zeros((L_LEAVES, MAX_DEPTH), dtype=np.int8)

    stack = [(0, np.arange(N, dtype=np.int32),
              np.zeros(MAX_DEPTH, dtype=np.int8),
              np.zeros(MAX_DEPTH, dtype=np.int8))]
    while stack:
        node_id, idx, path_sides, path_axes = stack.pop()
        depth = _depth_of_heap_node(node_id)
        if depth == MAX_DEPTH:
            leaf_id = _leaf_id_from_path(node_id)
            leaf_intervals[leaf_id] = idx
            leaf_paths[leaf_id] = path_sides
            leaf_path_axes[leaf_id] = path_axes
            continue
        coords = points[idx]
        extent = coords.max(axis=0) - coords.min(axis=0)
        axis = int(extent.argmax())
        order = np.argsort(coords[:, axis], kind='stable')
        sorted_idx = idx[order]
        mid = len(sorted_idx) // 2
        split_value = float(points[sorted_idx[mid], axis])
        split_axes[node_id] = axis
        split_values[node_id] = split_value
        ps_l = path_sides.copy(); pa_l = path_axes.copy()
        ps_r = path_sides.copy(); pa_r = path_axes.copy()
        ps_l[depth] = 0; pa_l[depth] = axis
        ps_r[depth] = 1; pa_r[depth] = axis
        stack.append((2 * node_id + 2, sorted_idx[mid:], ps_r, pa_r))
        stack.append((2 * node_id + 1, sorted_idx[:mid], ps_l, pa_l))

    _add_face_adjacency(leaf_paths, leaf_path_axes, adj)
    _add_knn_adjacency(adj, leaf_intervals, points)

    return KDTreeResult(split_axes=split_axes,
                        split_values=split_values,
                        leaf_intervals=leaf_intervals,
                        adjacency_first_order=adj)


def _depth_of_heap_node(node_id: int) -> int:
    return int(np.floor(np.log2(node_id + 1)))


def _add_face_adjacency(leaf_paths: np.ndarray,
                        leaf_path_axes: np.ndarray,
                        adj: dict[int, set[int]]) -> None:
    L = leaf_paths.shape[0]
    path_codes = np.zeros(L, dtype=np.int64)
    for d in range(MAX_DEPTH):
        path_codes |= (leaf_paths[:, d].astype(np.int64) & 1) << d
        path_codes |= (leaf_path_axes[:, d].astype(np.int64) & 3) << (MAX_DEPTH + 2 * d)
    code_to_leaf: dict[int, int] = {int(c): i for i, c in enumerate(path_codes)}
    for leaf_id in range(L):
        code = int(path_codes[leaf_id])
        for d in range(MAX_DEPTH):
            flipped = code ^ (1 << d)
            j = code_to_leaf.get(flipped)
            if j is not None and j != leaf_id:
                adj[leaf_id].add(j)


def _add_knn_adjacency(adj: dict[int, set[int]],
                       leaf_intervals: list[np.ndarray],
                       points: np.ndarray,
                       k: int = 30) -> None:
    from scipy.spatial import cKDTree
    L = len(leaf_intervals)
    centroids = np.empty((L, 3), dtype=np.float32)
    for i, idx in enumerate(leaf_intervals):
        centroids[i] = points[idx].mean(axis=0)
    tree = cKDTree(centroids)
    _, nbr_idx = tree.query(centroids, k=min(k, L))
    for i in range(L):
        for j_idx in range(1, nbr_idx.shape[1]):
            j = int(nbr_idx[i, j_idx])
            if j != i:
                adj[i].add(j)
                adj[j].add(i)


def descend_to_leaf(points: np.ndarray, split_axes: np.ndarray,
                    split_values: np.ndarray) -> np.ndarray:
    """Assign arbitrary points to leaves via 16-layer comparison.

    Only used for test cases' non-keep 87.5% points (§3.7).
    """
    M = points.shape[0]
    node = np.zeros(M, dtype=np.int32)
    for layer in range(MAX_DEPTH):
        axis = split_axes[node]
        sval = split_values[node]
        go_left = points[np.arange(M), axis] <= sval
        node = np.where(go_left, 2 * node + 1, 2 * node + 2)
    leaf = node - (L_LEAVES - 1)
    return leaf.astype(np.int32)
