"""Baked transient1 + transient2 for curve evaluation (v4 §3.14)."""
from __future__ import annotations

import numpy as np


def bake_transient1(rng: np.random.Generator,
                    offsets: np.ndarray,
                    leaf_vol_count: np.ndarray,
                    point_pos_norm: np.ndarray,
                    point_sdf_zscored: np.ndarray,
                    point_sdf_grad: np.ndarray,
                    point_curv_mean_zscored: np.ndarray,
                    point_curv_gauss_zscored: np.ndarray,
                    leaf_centroid_norm: np.ndarray,
                    encoder_k: int = 32
                    ) -> np.ndarray:
    """Return transient1 [L, 32, 10] bf16-ready fp32 array."""
    L = offsets.shape[0] - 1
    sampled_idx = sample_k_per_leaf(rng, offsets, encoder_k=encoder_k)
    rel = (point_pos_norm[sampled_idx]
           - leaf_centroid_norm[:, None, :])
    sdf = point_sdf_zscored[sampled_idx]
    sdf_g = point_sdf_grad[sampled_idx]
    starts = offsets[:-1].astype(np.int64)
    within_leaf = sampled_idx - starts[:, None]
    surf_flag = (within_leaf >= leaf_vol_count[:, None]).astype(np.float32)
    cm = point_curv_mean_zscored[sampled_idx]
    cg = point_curv_gauss_zscored[sampled_idx]
    out = np.concatenate([
        rel.astype(np.float32),
        sdf[..., None].astype(np.float32),
        sdf_g.astype(np.float32),
        surf_flag[..., None],
        cm[..., None].astype(np.float32),
        cg[..., None].astype(np.float32),
    ], axis=-1)
    return out


def sample_k_per_leaf(rng: np.random.Generator,
                      offsets: np.ndarray,
                      encoder_k: int = 32) -> np.ndarray:
    """Per-leaf random k indices from member intervals."""
    L = offsets.shape[0] - 1
    counts = np.diff(offsets).astype(np.int64)
    max_count = int(counts.max())
    rand_keys = rng.random((L, max_count))
    for i in range(L):
        rand_keys[i, counts[i]:] = 2.0
    sel = np.argpartition(rand_keys, encoder_k, axis=1)[:, :encoder_k]
    starts = offsets[:-1].astype(np.int64)
    out = starts[:, None] + sel
    return out


def bake_transient2(rng: np.random.Generator,
                    vol_reorder_idx: np.ndarray,
                    surf_reorder_idx: np.ndarray,
                    leaf_centroid_norm: np.ndarray,
                    leaf_neighbor_idx: np.ndarray,
                    leaf_id_per_point: np.ndarray,
                    point_pos_norm: np.ndarray,
                    surface_areas: np.ndarray,
                    n_query: int = 500_000,
                    n_query_vol: int = 400_000,
                    surface_area_alpha: float = 1.0,
                    idw_k: int = 8
                    ) -> dict[str, np.ndarray]:
    """Return baked transient2 fields."""
    n_vol_keep = vol_reorder_idx.shape[0]
    n_surf_keep = surf_reorder_idx.shape[0]
    n_query_surf = n_query - n_query_vol
    vol_choice = rng.choice(n_vol_keep, size=n_query_vol, replace=False)
    if surface_area_alpha == 0.0:
        surf_choice = rng.choice(n_surf_keep, size=n_query_surf, replace=False)
    else:
        w = surface_areas.astype(np.float64) ** float(surface_area_alpha)
        w /= w.sum()
        surf_choice = rng.choice(n_surf_keep, size=n_query_surf,
                                 replace=False, p=w)
    query_idx = np.concatenate([
        vol_reorder_idx[vol_choice].astype(np.int32),
        surf_reorder_idx[surf_choice].astype(np.int32),
    ])
    idw_idx, idw_w = compute_idw_numpy(
        query_idx=query_idx,
        leaf_id_per_point=leaf_id_per_point,
        point_pos_norm=point_pos_norm,
        leaf_centroid_norm=leaf_centroid_norm,
        leaf_neighbor_idx=leaf_neighbor_idx,
        idw_k=idw_k,
    )
    return {
        'query_idx': query_idx,
        'idw_idx': idw_idx,
        'idw_w': idw_w.astype(np.float32),
        'n_query_vol': np.int32(n_query_vol),
        'vol_choice': vol_choice.astype(np.int32),
        'surf_choice': surf_choice.astype(np.int32),
    }


def compute_idw_numpy(query_idx: np.ndarray,
                      leaf_id_per_point: np.ndarray,
                      point_pos_norm: np.ndarray,
                      leaf_centroid_norm: np.ndarray,
                      leaf_neighbor_idx: np.ndarray,
                      idw_k: int = 8
                      ) -> tuple[np.ndarray, np.ndarray]:
    """numpy IDW=k neighbors + weights in normalized space."""
    N_q = query_idx.shape[0]
    q_pos = point_pos_norm[query_idx]
    q_leaves = leaf_id_per_point[query_idx]
    cands = leaf_neighbor_idx[q_leaves]
    valid = cands != -1
    safe = np.where(valid, cands, 0)
    cand_c = leaf_centroid_norm[safe]
    diff = q_pos[:, None, :] - cand_c
    d = np.linalg.norm(diff, axis=-1)
    d = np.where(valid, d, np.inf)
    top = np.argpartition(d, kth=idw_k - 1, axis=1)[:, :idw_k]
    rng_n = np.arange(N_q)[:, None]
    top_dists = d[rng_n, top]
    order = np.argsort(top_dists, axis=1)
    top = np.take_along_axis(top, order, axis=1)
    top_dists = np.take_along_axis(top_dists, order, axis=1)
    weights = 1.0 / (top_dists + 1e-8)
    weights = weights / weights.sum(axis=1, keepdims=True)
    idw_idx = np.take_along_axis(cands, top, axis=1).astype(np.int32)
    return idw_idx, weights.astype(np.float32)
