"""Per-step CPU transient1 / transient2 computation (v4 §10.5–§10.6).

Pure numpy — runs in a ProcessPool worker, takes np arrays (or pinned
tensors viewed as np) and returns numpy dicts the main process H2Ds.

Determinism: per (case_id, epoch) RNG so curves and resumes are repeatable.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from preprocess.transient_baked import sample_k_per_leaf
from utils.seed import per_case_epoch_seed, make_rng


def _tensor_to_np(x: torch.Tensor) -> np.ndarray:
    if x.dtype == torch.bfloat16:
        return x.to(torch.float32).numpy()
    return x.numpy()


def build_transient1(pt: dict, epoch: int, encoder_k: int = 32) -> np.ndarray:
    """Compute transient1 [L, 32, 10] fp32 (caller casts to bf16 on GPU)."""
    case_id = int(pt.get('_case_id', 0))
    rng = make_rng(per_case_epoch_seed(case_id, epoch))
    offsets = _tensor_to_np(pt['leaf_member_offsets']).astype(np.int64)
    leaf_vol_count = _tensor_to_np(pt['leaf_vol_count']).astype(np.int64)
    sampled_idx = sample_k_per_leaf(rng, offsets, encoder_k=encoder_k)
    point_pos_norm = _tensor_to_np(pt['point_pos_norm'])
    point_sdf = _tensor_to_np(pt['point_sdf'])
    point_sdf_grad = _tensor_to_np(pt['point_sdf_grad'])
    point_curv_mean = _tensor_to_np(pt['point_curvature_mean'])
    point_curv_gauss = _tensor_to_np(pt['point_curvature_gauss'])
    leaf_centroid_norm = _tensor_to_np(pt['leaf_centroid_norm'])
    rel = (point_pos_norm[sampled_idx]
           - leaf_centroid_norm[:, None, :])
    starts = offsets[:-1]
    within_leaf = sampled_idx - starts[:, None]
    surf_flag = (within_leaf >= leaf_vol_count[:, None]).astype(np.float32)
    sdf = point_sdf[sampled_idx]
    sdf_g = point_sdf_grad[sampled_idx]
    cm = point_curv_mean[sampled_idx]
    cg = point_curv_gauss[sampled_idx]
    t1 = np.concatenate([
        rel.astype(np.float32),
        sdf[..., None].astype(np.float32),
        sdf_g.astype(np.float32),
        surf_flag[..., None],
        cm[..., None].astype(np.float32),
        cg[..., None].astype(np.float32),
    ], axis=-1)
    return t1


def build_transient2(pt: dict, epoch: int,
                     n_query: int = 500_000,
                     n_query_vol: int = 400_000,
                     surface_area_alpha: float = 1.0) -> dict[str, np.ndarray]:
    """Sample N_query queries (80/20 hard, area-weighted surface), gather
    targets. IDW is computed on GPU in model.forward().

    Returns dict with query arrays as numpy (no idw_indices/idw_weights).
    """
    case_id = int(pt.get('_case_id', 0))
    rng = make_rng(per_case_epoch_seed(case_id, epoch) ^ 0xA5A5_A5A5)
    vol_reorder_idx = _tensor_to_np(pt['vol_reorder_idx']).astype(np.int64)
    surf_reorder_idx = _tensor_to_np(pt['surf_reorder_idx']).astype(np.int64)
    n_vol = vol_reorder_idx.shape[0]
    n_surf = surf_reorder_idx.shape[0]
    n_query_vol = min(n_query_vol, n_vol)
    n_query_surf = min(n_query - n_query_vol, n_surf)
    surf_a = _tensor_to_np(pt['surface_areas']).astype(np.float64)
    if surface_area_alpha == 0.0:
        surf_choice = rng.choice(n_surf, size=n_query_surf, replace=False)
    else:
        w = surf_a ** float(surface_area_alpha)
        w /= w.sum()
        surf_choice = rng.choice(n_surf, size=n_query_surf, replace=False, p=w)
    vol_choice = rng.choice(n_vol, size=n_query_vol, replace=False)
    query_idx = np.concatenate([
        vol_reorder_idx[vol_choice],
        surf_reorder_idx[surf_choice],
    ])
    point_pos_norm = _tensor_to_np(pt['point_pos_norm'])
    point_sdf = _tensor_to_np(pt['point_sdf'])
    point_sdf_grad = _tensor_to_np(pt['point_sdf_grad'])
    point_y_volume = _tensor_to_np(pt['point_y_volume'])
    point_y_surface = _tensor_to_np(pt['point_y_surface'])
    leaf_id_per_point = _tensor_to_np(pt['leaf_id_per_point'])

    qpos = point_pos_norm[query_idx].astype(np.float32)
    qsdf = point_sdf[query_idx].astype(np.float32)
    qsdf_g = point_sdf_grad[query_idx].astype(np.float32)
    tgt_vol = point_y_volume[vol_choice].astype(np.float32)
    tgt_surf = point_y_surface[surf_choice].astype(np.float32)
    query_leaf_id = leaf_id_per_point[query_idx].astype(np.int32)

    out = {
        'query_pos_norm': qpos,
        'query_sdf': qsdf,
        'query_sdf_grad': qsdf_g,
        'query_target_volume': tgt_vol,
        'query_target_surface': tgt_surf,
        'query_leaf_id': query_leaf_id,
    }
    if 'nut_log_zscored' in pt:
        nut = _tensor_to_np(pt['nut_log_zscored'])
        out['nut_log_zscored'] = nut[vol_choice].astype(np.float32)
    if 'vort_log_zscored' in pt:
        vort = _tensor_to_np(pt['vort_log_zscored'])
        out['vort_log_zscored'] = vort[vol_choice].astype(np.float32)
    return out
