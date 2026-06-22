"""End-to-end preprocess orchestrator (v4 §3).

Three orchestration stages:

  Stage A — case-parallel Phase 0..7 (per-case worker via ProcessPool).
            Each worker returns:
              - the case's per-point + per-leaf raw arrays
              - its 1st-order adjacency (list of int sets)
              - its case-max degree (for global N_pad computation)
              - its Welford partial state for every Welford field
              - its surface-area sum / count (cheap; not used here yet)
            Outputs are pickled back to the orchestrator.

  Stage B — global reductions (orchestrator, single-threaded):
              * percentile pass for norm_p5 / norm_p95 (np.partition on
                434 cases of pos_world per axis)
              * Chan merge of all Welford partials
              * derive rope_scale_per_axis from norm_p5/p95 extents
              * global N_pad = max per-case-max-degree
              * write coef_norm.pt

  Stage C — case-parallel Phase 10/11 normalization + write (ProcessPool).
            Each worker re-reads its case (raw arrays kept in a temp pickle
            from Stage A; or re-loads step-1 PT and re-runs phases 0..7
            deterministically). Applies z-score, pads neighbors to N_pad,
            torch.save case_<id>.pt (+ log sidecar for train, + baked
            transients for train_eval/val, + cd_true/cl_true for test).

The orchestrator does NOT batch all 484 cases' raw arrays in RAM at once
(would need ~1 TB). Stage A streams partial state out; full raw arrays
are written to per-case scratch pickles which Stage C reads back.
"""
from __future__ import annotations

import gc
import json
import os
import os.path as osp
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from preprocess import welford
from preprocess.geometry import (compute_sdf_and_curvature, compute_vorticity,
                                 subsample_indices)
from preprocess.kdtree import L_LEAVES, build_kdtree, descend_to_leaf
from preprocess.leaf_stats import (compute_leaf_centroid, compute_leaf_stats)
from preprocess.log_sidecar import make_log_sidecar
from preprocess.neighbors import (case_max_degree, expand_to_second_order,
                                  pad_to_neighbor_matrix)
from preprocess.reorder import (apply_perm, build_perm_and_offsets,
                                split_perm_for_y)
from preprocess.transient_baked import (bake_transient1, bake_transient2)


# ---------------------------------------------------------------------------
# Stage A — per-case worker
# ---------------------------------------------------------------------------


def stage_a_worker(case_id: int, step1_dir: str, scratch_dir: str,
                   role: str) -> dict[str, Any]:
    """Run Phase 0..7 + collect Welford partials and pos_world pieces.

    role ∈ {'train', 'val', 'test', 'train_eval'}:
        - train / val / train_eval: subsample 1/8 vol + 1/4 surf
        - test: keep full, but also produce encoder_subset_mask & full
                vorticity. KD-tree is built on encoder-subset only.

    Side effects:
      - Writes scratch_dir/case_<id>_raw.pkl with all raw arrays needed
        by Stage C (later read back to normalize & write final PT).

    Returns:
      A small dict of summaries:
        {'case_id': int, 'role': role, 'n_keep': N_keep, 'n_vol_keep': ...,
         'pos_keep_path': str (scratch file with raw pos_world for percentile
                                 stage if role != 'test'),
         'welford_partials': {field_name: WelfordState},
         'case_max_degree': int}
    """
    step1_pt = torch.load(osp.join(step1_dir, f'run_{case_id}.pt'),
                          map_location='cpu', weights_only=False)
    vol_pos = step1_pt['volume_coords'].numpy().astype(np.float32)
    vol_vel = step1_pt['volume_velocity'].numpy().astype(np.float32)
    vol_p = step1_pt['volume_pressure'].numpy().astype(np.float32)
    vol_nut = step1_pt['volume_nut'].numpy().astype(np.float32)
    surf_pos = step1_pt['surface_coords'].numpy().astype(np.float32)
    surf_p = step1_pt['surface_pressure'].numpy().astype(np.float32)
    surf_wss = step1_pt['surface_wallshearstress'].numpy().astype(np.float32)
    surf_n = step1_pt['surface_normal'].numpy().astype(np.float32)
    surf_a = step1_pt['surface_area'].numpy().astype(np.float32)
    stl_v = step1_pt['stl_vertices'].numpy().astype(np.float32)
    stl_f = step1_pt['stl_faces'].numpy().astype(np.int32)
    ref = step1_pt['reference']
    del step1_pt

    n_vol_full = vol_pos.shape[0]
    n_surf_full = surf_pos.shape[0]
    n_full = n_vol_full + n_surf_full

    # Phase 1: subsample mask
    vol_keep_idx, surf_keep_idx = subsample_indices(n_vol_full, n_surf_full,
                                                    case_id)
    n_vol_keep = vol_keep_idx.shape[0]
    n_surf_keep = surf_keep_idx.shape[0]
    n_keep = n_vol_keep + n_surf_keep
    is_test = (role == 'test')

    # Phase 2: vorticity (keep only for train/val; full for test)
    omega_query_idx = None if is_test else vol_keep_idx
    omega = compute_vorticity(vol_pos, vol_vel, query_indices=omega_query_idx)
    # omega shape: (len(vol_keep_idx), 3) for non-test, (n_vol_full, 3) for test

    # Phase 3: SDF + curvature on the appropriate query set
    if is_test:
        keep_pts = np.concatenate([vol_pos, surf_pos], axis=0)
        # encoder subset for KD-tree build
        enc_idx = np.concatenate([vol_keep_idx,
                                  n_vol_full + surf_keep_idx], axis=0)
        # process full mesh for SDF/curv (test keeps full)
        geom_pts = keep_pts
    else:
        keep_pts = np.concatenate([vol_pos[vol_keep_idx],
                                   surf_pos[surf_keep_idx]], axis=0)
        enc_idx = None
        geom_pts = keep_pts

    geom = compute_sdf_and_curvature(geom_pts, stl_v, stl_f)
    point_sdf_raw = geom['sdf']
    point_sdf_grad = geom['sdf_grad']
    point_curv_mean_raw = geom['curv_mean']
    point_curv_gauss_raw = geom['curv_gauss']

    # Phase 4: assemble y_volume_raw [N_vol_*, 8] and y_surface_raw [N_surf_*, 4]
    if is_test:
        y_volume_raw = np.stack([
            vol_p,
            vol_vel[:, 0], vol_vel[:, 1], vol_vel[:, 2],
            omega[:, 0], omega[:, 1], omega[:, 2],
            vol_nut,
        ], axis=-1).astype(np.float32)                             # (N_vol_full, 8)
        y_surface_raw = np.stack([
            surf_p, surf_wss[:, 0], surf_wss[:, 1], surf_wss[:, 2],
        ], axis=-1).astype(np.float32)                             # (N_surf_full, 4)
    else:
        y_volume_raw = np.stack([
            vol_p[vol_keep_idx],
            vol_vel[vol_keep_idx, 0], vol_vel[vol_keep_idx, 1],
            vol_vel[vol_keep_idx, 2],
            omega[:, 0], omega[:, 1], omega[:, 2],
            vol_nut[vol_keep_idx],
        ], axis=-1).astype(np.float32)
        y_surface_raw = np.stack([
            surf_p[surf_keep_idx],
            surf_wss[surf_keep_idx, 0], surf_wss[surf_keep_idx, 1],
            surf_wss[surf_keep_idx, 2],
        ], axis=-1).astype(np.float32)

    # Phase 5: KD-tree on the encoder/keep set (always 18.25M points)
    if is_test:
        kd_pts = np.concatenate([vol_pos[vol_keep_idx],
                                 surf_pos[surf_keep_idx]], axis=0)
    else:
        kd_pts = keep_pts
    kd = build_kdtree(kd_pts)

    # Phase 5.5 (test only): full leaf assignment via descent
    if is_test:
        leaf_assignment_full = descend_to_leaf(keep_pts, kd.split_axes,
                                               kd.split_values)
    else:
        leaf_assignment_full = None

    # Phase 6: physical reorder + offsets
    perm, offsets, leaf_vol_count = build_perm_and_offsets(
        kd.leaf_intervals, n_vol_keep)
    vol_perm_local, surf_perm_local = split_perm_for_y(perm, n_vol_keep)

    # Reorder per-point arrays. For test, we have separate full and
    # encoder-subset arrays; reordering applies to the encoder-subset
    # views. For non-test, keep_pts/point_sdf are already 18.25M.
    if is_test:
        kd_pos = kd_pts

    # Compute the per-keep / per-encoder-subset geometry arrays that
    # leaf_stats / segment reductions need.
    if is_test:
        # Index-select SDF/curv on encoder subset from full mesh result
        enc_sdf = point_sdf_raw[enc_idx]
        enc_sdf_grad = point_sdf_grad[enc_idx]
        enc_curv_m = point_curv_mean_raw[enc_idx]
        enc_curv_g = point_curv_gauss_raw[enc_idx]
        kd_pos_ord = kd_pos[perm]
        enc_sdf_ord = enc_sdf[perm]
        enc_sdf_grad_ord = enc_sdf_grad[perm]
        enc_curv_m_ord = enc_curv_m[perm]
        enc_curv_g_ord = enc_curv_g[perm]
    else:
        kd_pos_ord = kd_pts[perm]
        enc_sdf_ord = point_sdf_raw[perm]
        enc_sdf_grad_ord = point_sdf_grad[perm]
        enc_curv_m_ord = point_curv_mean_raw[perm]
        enc_curv_g_ord = point_curv_gauss_raw[perm]

    # Also reorder y_volume / y_surface using local perms (these live in
    # separate sub-arrays). For test they remain full N_vol_full / N_surf_full;
    # we do NOT reorder them (decoder addresses them by full point index).
    if not is_test:
        y_volume_raw = y_volume_raw[vol_perm_local]
        y_surface_raw = y_surface_raw[surf_perm_local]
        surf_a_ord = surf_a[surf_keep_idx][surf_perm_local]
        surf_n_ord = surf_n[surf_keep_idx][surf_perm_local]
    else:
        surf_a_ord = surf_a
        surf_n_ord = surf_n

    # Phase 7: leaf-level fields
    centroid_raw, counts = compute_leaf_centroid(kd_pos_ord, offsets)
    leaf_stats_raw = compute_leaf_stats(
        kd_pos_ord, enc_sdf_ord, enc_curv_m_ord, enc_curv_g_ord,
        offsets, leaf_vol_count, centroid_raw,
    )
    # Leaf-level SDF / sdf_grad / curvature computed at centroid positions:
    geom_leaf = compute_sdf_and_curvature(centroid_raw, stl_v, stl_f)
    leaf_sdf_raw = geom_leaf['sdf']
    leaf_sdf_grad = geom_leaf['sdf_grad']
    leaf_curv_mean_raw = geom_leaf['curv_mean']
    leaf_curv_gauss_raw = geom_leaf['curv_gauss']

    # Phase 8: 2nd-order neighbors + per-case max degree
    per_leaf_nbrs = expand_to_second_order(kd.adjacency_first_order, L_LEAVES)
    case_deg = case_max_degree(per_leaf_nbrs)

    # Welford partials (only train+val contribute; test does NOT enter
    # Welford or coordinate percentile passes).
    partials: dict[str, Any] = {}
    if role in ('train', 'val', 'train_eval'):
        # NB: train_eval is a subset of train; the actual train-id case
        # contributes once.
        partials['out_volume'] = welford.update_state(
            welford.init_state(y_volume_raw.shape[1]), y_volume_raw)
        partials['out_surface'] = welford.update_state(
            welford.init_state(y_surface_raw.shape[1]), y_surface_raw)
        partials['sdf'] = welford.update_state(
            welford.init_state(0), enc_sdf_ord)
        partials['curv_mean'] = welford.update_state(
            welford.init_state(0), enc_curv_m_ord)
        partials['curv_gauss'] = welford.update_state(
            welford.init_state(0), enc_curv_g_ord)
        partials['leaf_stats'] = welford.update_state(
            welford.init_state(leaf_stats_raw.shape[1]), leaf_stats_raw)
        # log-domain (only for nut + vorticity from raw target)
        nut_log = np.log(np.maximum(y_volume_raw[:, 7], 1e-6))
        partials['nut_log'] = welford.update_state(
            welford.init_state(0), nut_log)
        vort = y_volume_raw[:, 4:7]
        vort_log = np.sign(vort) * np.log1p(np.abs(vort))
        partials['vort_log'] = welford.update_state(
            welford.init_state(3), vort_log)

    # Persist raw arrays to scratch for Stage C.
    os.makedirs(scratch_dir, exist_ok=True)
    scratch_path = osp.join(scratch_dir, f'case_{case_id}_raw.pkl')
    payload = {
        'role': role,
        'case_id': case_id,
        'n_vol_keep': n_vol_keep, 'n_surf_keep': n_surf_keep,
        'n_keep': n_keep,
        'n_vol_full': n_vol_full, 'n_surf_full': n_surf_full, 'n_full': n_full,
        'kd_pos_ord_raw': kd_pos_ord,
        'enc_sdf_ord_raw': enc_sdf_ord,
        'enc_sdf_grad_ord': enc_sdf_grad_ord,
        'enc_curv_m_ord_raw': enc_curv_m_ord,
        'enc_curv_g_ord_raw': enc_curv_g_ord,
        'y_volume_raw': y_volume_raw,
        'y_surface_raw': y_surface_raw,
        'surf_a_ord': surf_a_ord,
        'surf_n_ord': surf_n_ord,
        'offsets': offsets,
        'leaf_vol_count': leaf_vol_count,
        'centroid_raw': centroid_raw,
        'leaf_stats_raw': leaf_stats_raw,
        'leaf_sdf_raw': leaf_sdf_raw,
        'leaf_sdf_grad': leaf_sdf_grad,
        'leaf_curv_mean_raw': leaf_curv_mean_raw,
        'leaf_curv_gauss_raw': leaf_curv_gauss_raw,
        'per_leaf_nbrs': per_leaf_nbrs,
        'leaf_assignment_full': leaf_assignment_full,
        'encoder_subset_mask_indices': enc_idx,
        'full_pos_raw': keep_pts if is_test else None,
        'full_sdf_raw': point_sdf_raw if is_test else None,
        'full_sdf_grad': point_sdf_grad if is_test else None,
        'stl_vertices': stl_v,
        'stl_faces': stl_f,
        # for test Cd/Cl integration
        'surf_p_full': surf_p if is_test else None,
        'surf_wss_full': surf_wss if is_test else None,
        'surf_n_full': surf_n if is_test else None,
        'surf_a_full': surf_a if is_test else None,
        'a_ref': float(ref['aRef']) if 'aRef' in ref else float(
            ref.get('a_ref', 1.0)),
    }
    with open(scratch_path, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    return {
        'case_id': case_id,
        'role': role,
        'scratch_path': scratch_path,
        'partials': partials,
        'case_max_degree': case_deg,
        'n_vol_keep': n_vol_keep,
        'n_surf_keep': n_surf_keep,
    }




# ---------------------------------------------------------------------------
# Stage B — global reductions
# ---------------------------------------------------------------------------


def stage_b_global(scratch_dir: str, summaries: list[dict[str, Any]],
                   manifest: dict, L: int = L_LEAVES, encoder_k: int = 32
                   ) -> dict[str, Any]:
    """Compute coef_norm fields from per-case summaries.

    Returns coef_norm dict, ready to torch.save.
    """
    train_val_ids = set(manifest['train_ids']) | set(manifest['val_ids'])
    train_val_summaries = [s for s in summaries
                           if s['case_id'] in train_val_ids]

    # Welford merges
    fields = ['out_volume', 'out_surface', 'sdf', 'curv_mean', 'curv_gauss',
              'leaf_stats', 'nut_log', 'vort_log']
    merged = {}
    for f in fields:
        states = [s['partials'][f] for s in train_val_summaries
                  if f in s['partials']]
        merged[f] = welford.reduce(states)

    mean_out_vol, std_out_vol = welford.finalize(merged['out_volume'])
    mean_out_surf, std_out_surf = welford.finalize(merged['out_surface'])
    mean_sdf, std_sdf = welford.finalize(merged['sdf'])
    mean_curv_m, std_curv_m = welford.finalize(merged['curv_mean'])
    mean_curv_g, std_curv_g = welford.finalize(merged['curv_gauss'])
    mean_leaf_stats, std_leaf_stats = welford.finalize(merged['leaf_stats'])
    mean_nut_log, std_nut_log = welford.finalize(merged['nut_log'])
    mean_vort_log, std_vort_log = welford.finalize(merged['vort_log'])

    # Coordinate percentiles via streaming per axis to keep peak RAM bounded
    norm_p5 = np.zeros(3, dtype=np.float32)
    norm_p95 = np.zeros(3, dtype=np.float32)
    for axis in range(3):
        pieces = []
        for s in train_val_summaries:
            with open(s['scratch_path'], 'rb') as f:
                payload = pickle.load(f)
            pieces.append(payload['kd_pos_ord_raw'][:, axis].astype(
                np.float32))
            del payload
        arr = np.concatenate(pieces)
        N = arr.shape[0]
        k_lo, k_hi = int(0.05 * N), int(0.95 * N)
        arr_part = np.partition(arr, [k_lo, k_hi])
        norm_p5[axis] = arr_part[k_lo]
        norm_p95[axis] = arr_part[k_hi]
        del arr, arr_part
        gc.collect()

    extents = (norm_p95 - norm_p5).astype(np.float64)
    geo_mean = float(np.cbrt(np.prod(extents)))
    rope_scale_per_axis = ((L ** (1.0 / 3.0)) * extents / geo_mean
                           ).astype(np.float32)

    # Global N_pad
    N_pad = max(s['case_max_degree'] for s in summaries)

    return {
        'mean_out_volume': mean_out_vol,
        'std_out_volume':  std_out_vol,
        'mean_out_surface': mean_out_surf,
        'std_out_surface':  std_out_surf,
        'mean_nut_log': float(mean_nut_log),
        'std_nut_log':  float(std_nut_log),
        'mean_vort_log': mean_vort_log,
        'std_vort_log':  std_vort_log,
        'mean_sdf': float(mean_sdf), 'std_sdf': float(std_sdf),
        'mean_curv_mean': float(mean_curv_m),
        'std_curv_mean':  float(std_curv_m),
        'mean_curv_gauss': float(mean_curv_g),
        'std_curv_gauss':  float(std_curv_g),
        'mean_leaf_stats': mean_leaf_stats,
        'std_leaf_stats':  std_leaf_stats,
        'norm_p5':  norm_p5, 'norm_p95': norm_p95,
        'rope_scale_per_axis': rope_scale_per_axis,
        'L': int(L), 'encoder_k': int(encoder_k), 'N_pad': int(N_pad),
    }


# ---------------------------------------------------------------------------
# Stage C — normalize + write PT
# ---------------------------------------------------------------------------


def stage_c_worker(summary: dict[str, Any], coef_norm: dict[str, Any],
                   cache_dir: str, train_eval_ids: set[int],
                   val_ids: set[int],
                   q_inf: float, n_query: int, n_query_vol: int,
                   surface_area_alpha: float) -> dict[str, Any]:
    """Apply normalization, pad neighbors to N_pad, write final PT files.

    Returns small dict {case_id, role, written_paths}.
    """
    with open(summary['scratch_path'], 'rb') as f:
        payload = pickle.load(f)
    role = payload['role']
    case_id = payload['case_id']
    n_vol_keep = payload['n_vol_keep']
    n_surf_keep = payload['n_surf_keep']
    is_test = (role == 'test')

    # Coordinate normalization
    norm_p5 = coef_norm['norm_p5']
    norm_p95 = coef_norm['norm_p95']
    span = (norm_p95 - norm_p5).astype(np.float32)
    span = np.where(span > 1e-12, span, 1.0)
    point_pos_norm = (
        (payload['kd_pos_ord_raw'] - norm_p5) / span * 2.0 - 1.0
    ).astype(np.float32)
    leaf_centroid_norm = (
        (payload['centroid_raw'] - norm_p5) / span * 2.0 - 1.0
    ).astype(np.float32)

    # Field z-score (input)
    eps = 1e-8
    point_sdf = ((payload['enc_sdf_ord_raw'] - coef_norm['mean_sdf'])
                 / max(coef_norm['std_sdf'], eps)).astype(np.float32)
    point_curv_mean = ((payload['enc_curv_m_ord_raw'] - coef_norm['mean_curv_mean'])
                       / max(coef_norm['std_curv_mean'], eps)).astype(np.float32)
    point_curv_gauss = ((payload['enc_curv_g_ord_raw'] - coef_norm['mean_curv_gauss'])
                        / max(coef_norm['std_curv_gauss'], eps)).astype(np.float32)
    leaf_sdf = ((payload['leaf_sdf_raw'] - coef_norm['mean_sdf'])
                / max(coef_norm['std_sdf'], eps)).astype(np.float32)
    leaf_curv_mean = ((payload['leaf_curv_mean_raw'] - coef_norm['mean_curv_mean'])
                      / max(coef_norm['std_curv_mean'], eps)).astype(np.float32)
    leaf_curv_gauss = ((payload['leaf_curv_gauss_raw'] - coef_norm['mean_curv_gauss'])
                       / max(coef_norm['std_curv_gauss'], eps)).astype(np.float32)
    leaf_stats = ((payload['leaf_stats_raw'] - coef_norm['mean_leaf_stats'])
                  / np.maximum(coef_norm['std_leaf_stats'], eps)
                  ).astype(np.float32)

    # Targets z-score (train/val/train_eval only; test stays raw)
    if not is_test:
        point_y_volume = ((payload['y_volume_raw'] - coef_norm['mean_out_volume'])
                          / np.maximum(coef_norm['std_out_volume'], eps)
                          ).astype(np.float32)
        point_y_surface = ((payload['y_surface_raw'] - coef_norm['mean_out_surface'])
                           / np.maximum(coef_norm['std_out_surface'], eps)
                           ).astype(np.float32)

    # Pad neighbors
    neighbor_idx = pad_to_neighbor_matrix(payload['per_leaf_nbrs'],
                                          int(coef_norm['N_pad']))

    # For test: normalize full mesh positions and SDF for decoder
    if is_test:
        full_pos_norm = (
            (payload['full_pos_raw'] - norm_p5) / span * 2.0 - 1.0
        ).astype(np.float32)
        full_sdf = ((payload['full_sdf_raw'] - coef_norm['mean_sdf'])
                    / max(coef_norm['std_sdf'], eps)).astype(np.float32)
        full_sdf_grad = payload['full_sdf_grad'].astype(np.float32)

    # Build the case PT dict
    if is_test:
        pt_point_pos = torch.from_numpy(full_pos_norm)
        pt_point_sdf = torch.from_numpy(full_sdf).to(torch.bfloat16)
        pt_point_sdf_grad = torch.from_numpy(full_sdf_grad).to(torch.bfloat16)
    else:
        pt_point_pos = torch.from_numpy(point_pos_norm)
        pt_point_sdf = torch.from_numpy(point_sdf).to(torch.bfloat16)
        pt_point_sdf_grad = torch.from_numpy(
            payload['enc_sdf_grad_ord']).to(torch.bfloat16)

    pt: dict[str, Any] = {
        'point_pos_norm': pt_point_pos,                                    # fp32
        'point_sdf': pt_point_sdf,
        'point_sdf_grad': pt_point_sdf_grad,
        'point_curvature_mean': torch.from_numpy(point_curv_mean).to(
            torch.bfloat16),
        'point_curvature_gauss': torch.from_numpy(point_curv_gauss).to(
            torch.bfloat16),
        'leaf_centroid_norm': torch.from_numpy(leaf_centroid_norm),         # fp32
        'leaf_stats': torch.from_numpy(leaf_stats).to(torch.bfloat16),
        'leaf_sdf': torch.from_numpy(leaf_sdf).to(torch.bfloat16),
        'leaf_sdf_grad': torch.from_numpy(payload['leaf_sdf_grad']).to(
            torch.bfloat16),
        'leaf_curvature_mean': torch.from_numpy(leaf_curv_mean).to(
            torch.bfloat16),
        'leaf_curvature_gauss': torch.from_numpy(leaf_curv_gauss).to(
            torch.bfloat16),
        'leaf_neighbor_idx': torch.from_numpy(neighbor_idx),                # int32
        'leaf_member_offsets': torch.from_numpy(payload['offsets']),        # int32
        'leaf_vol_count': torch.from_numpy(payload['leaf_vol_count']),      # int32
    }
    if is_test:
        pt['point_y_volume_raw'] = torch.from_numpy(
            payload['y_volume_raw']).to(torch.float32)
        pt['point_y_surface_raw'] = torch.from_numpy(
            payload['y_surface_raw']).to(torch.float32)
        pt['surface_normals'] = torch.from_numpy(payload['surf_n_full']
                                                 ).to(torch.float32)
        pt['surface_areas'] = torch.from_numpy(payload['surf_a_full']
                                               ).to(torch.float32)
        pt['point_leaf_assignment'] = torch.from_numpy(
            payload['leaf_assignment_full'])                                # int32
        mask = np.zeros(payload['n_full'], dtype=bool)
        mask[payload['encoder_subset_mask_indices']] = True
        pt['encoder_subset_mask'] = torch.from_numpy(mask)
        pt['stl_vertices'] = torch.from_numpy(payload['stl_vertices'])
        pt['stl_faces'] = torch.from_numpy(payload['stl_faces'])
        # Cd / Cl integration
        force = ((payload['surf_p_full'][:, None] * payload['surf_n_full']
                  + payload['surf_wss_full'])
                 * payload['surf_a_full'][:, None]).sum(axis=0)
        a_ref = payload['a_ref']
        cd_true = float(force[0] / (q_inf * a_ref))
        cl_true = float(force[2] / (q_inf * a_ref))
        pt['cd_true'] = torch.tensor(cd_true, dtype=torch.float32)
        pt['cl_true'] = torch.tensor(cl_true, dtype=torch.float32)
        pt['a_ref'] = torch.tensor(a_ref, dtype=torch.float32)
        pt['N_vol_full'] = int(payload['n_vol_full'])
        pt['N_surf_full'] = int(payload['n_surf_full'])
        pt['N_full'] = int(payload['n_full'])
        pt['N_encoder_subset'] = int(mask.sum())
        # Bake transient1 for test (deterministic seed)
        rng_baked = np.random.default_rng(42 + case_id)
        t1 = bake_transient1(
            rng_baked, payload['offsets'], payload['leaf_vol_count'],
            point_pos_norm, point_sdf, payload['enc_sdf_grad_ord'],
            point_curv_mean, point_curv_gauss,
            leaf_centroid_norm,
        )
        pt['transient1'] = torch.from_numpy(t1).to(torch.bfloat16)
    else:
        pt['point_y_volume'] = torch.from_numpy(point_y_volume).to(
            torch.bfloat16)
        pt['point_y_surface'] = torch.from_numpy(point_y_surface).to(
            torch.bfloat16)
        pt['surface_normals'] = torch.from_numpy(payload['surf_n_ord']
                                                 ).to(torch.float32)
        pt['surface_areas'] = torch.from_numpy(payload['surf_a_ord']
                                               ).to(torch.float32)
        pt['N_vol_keep'] = int(n_vol_keep)
        pt['N_surf_keep'] = int(n_surf_keep)
        pt['N_keep'] = int(payload['n_keep'])

        # Precompute vol/surf reorder indices from interleaved layout
        _offsets = payload['offsets']
        _lvc = payload['leaf_vol_count']
        is_vol_in_reordered = np.zeros(payload['n_keep'], dtype=bool)
        for _l in range(L_LEAVES):
            lo = int(_offsets[_l])
            is_vol_in_reordered[lo:lo + int(_lvc[_l])] = True
        vol_reorder_idx = np.where(is_vol_in_reordered)[0].astype(np.int64)
        surf_reorder_idx = np.where(~is_vol_in_reordered)[0].astype(np.int64)
        pt['vol_reorder_idx'] = torch.from_numpy(vol_reorder_idx)
        pt['surf_reorder_idx'] = torch.from_numpy(surf_reorder_idx)

        leaf_id_per_point = np.repeat(
            np.arange(L_LEAVES, dtype=np.int32),
            np.diff(_offsets).astype(np.int64))
        pt['leaf_id_per_point'] = torch.from_numpy(leaf_id_per_point)

        # Train_eval and val: bake transient1/2
        if case_id in train_eval_ids or case_id in val_ids:
            rng_baked = np.random.default_rng(42 + case_id)
            t1 = bake_transient1(
                rng_baked, _offsets, _lvc,
                point_pos_norm, point_sdf, payload['enc_sdf_grad_ord'],
                point_curv_mean, point_curv_gauss,
                leaf_centroid_norm,
            )
            t2 = bake_transient2(
                rng_baked,
                vol_reorder_idx, surf_reorder_idx,
                leaf_centroid_norm, neighbor_idx, leaf_id_per_point,
                point_pos_norm, payload['surf_a_ord'].astype(np.float32),
                n_query=n_query, n_query_vol=n_query_vol,
                surface_area_alpha=surface_area_alpha,
            )
            pt['transient1'] = torch.from_numpy(t1).to(torch.bfloat16)
            pt['transient2_query_idx'] = torch.from_numpy(t2['query_idx'])
            pt['transient2_idw_idx'] = torch.from_numpy(t2['idw_idx'])
            pt['transient2_idw_w'] = torch.from_numpy(t2['idw_w']).to(
                torch.bfloat16)
            pt['transient2_n_vol'] = int(t2['n_query_vol'])
            pt['transient2_vol_choice'] = torch.from_numpy(t2['vol_choice'])
            pt['transient2_surf_choice'] = torch.from_numpy(t2['surf_choice'])

    # Write the main PT file
    if is_test:
        out_dir = osp.join(cache_dir, 'test')
    else:
        out_dir = cache_dir
    os.makedirs(out_dir, exist_ok=True)
    main_path = osp.join(out_dir, f'case_{case_id}.pt')
    torch.save(pt, main_path)

    written = [main_path]
    if role in ('train', 'train_eval'):
        sidecar = make_log_sidecar(
            payload['y_volume_raw'],
            mean_nut_log=coef_norm['mean_nut_log'],
            std_nut_log=coef_norm['std_nut_log'],
            mean_vort_log=coef_norm['mean_vort_log'],
            std_vort_log=coef_norm['std_vort_log'],
        )
        sidecar_path = osp.join(cache_dir, f'case_{case_id}_log.pt')
        torch.save(sidecar, sidecar_path)
        written.append(sidecar_path)

    # Free scratch memory; keep the per-case raw pickle for now
    # (orchestrator removes scratch after Stage C completes successfully).
    return {'case_id': case_id, 'role': role, 'written': written}


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


U_INF = 38.889
Q_INF = 0.5 * U_INF * U_INF


def run_preprocess(step1_dir: str, cache_dir: str, manifest: dict,
                   max_workers: int = 60, scratch_dir: str | None = None,
                   n_query: int = 500_000, n_query_vol: int = 400_000,
                   surface_area_alpha: float = 1.0) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    if scratch_dir is None:
        scratch_dir = osp.join(cache_dir, '_scratch')
    os.makedirs(scratch_dir, exist_ok=True)

    train_ids = set(manifest['train_ids'])
    val_ids = set(manifest['val_ids'])
    test_ids = set(manifest['test_ids'])
    train_eval_ids = set(manifest['train_eval_ids'])
    all_ids = sorted(train_ids | val_ids | test_ids)

    def role_of(cid: int) -> str:
        if cid in test_ids:
            return 'test'
        if cid in val_ids:
            return 'val'
        if cid in train_eval_ids:
            return 'train_eval'
        return 'train'

    t0 = time.time()
    summaries: list[dict[str, Any]] = []
    print(f'[Stage A] {len(all_ids)} cases on {max_workers} workers …',
          flush=True)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(stage_a_worker, cid, step1_dir, scratch_dir,
                      role_of(cid)): cid
            for cid in all_ids
        }
        for fut in tqdm(as_completed(futures), total=len(futures)):
            cid = futures[fut]
            try:
                summaries.append(fut.result())
            except Exception as e:
                raise RuntimeError(f'Stage A failed on case_id={cid}') from e
    print(f'[Stage A] done in {time.time() - t0:.1f}s', flush=True)

    t1 = time.time()
    coef_norm = stage_b_global(scratch_dir, summaries, manifest)
    torch.save(coef_norm, osp.join(cache_dir, 'coef_norm.pt'))
    print(f'[Stage B] coef_norm written in {time.time() - t1:.1f}s; '
          f'N_pad={coef_norm["N_pad"]}, '
          f'rope_scale_per_axis={coef_norm["rope_scale_per_axis"]}',
          flush=True)

    t2 = time.time()
    print(f'[Stage C] writing case PTs on {max_workers} workers …',
          flush=True)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(stage_c_worker, s, coef_norm, cache_dir,
                      train_eval_ids, val_ids,
                      Q_INF, n_query, n_query_vol, surface_area_alpha): s
            for s in summaries
        }
        for fut in tqdm(as_completed(futures), total=len(futures)):
            _ = fut.result()
    print(f'[Stage C] done in {time.time() - t2:.1f}s', flush=True)

    # Cleanup scratch
    for s in summaries:
        try:
            os.remove(s['scratch_path'])
        except OSError:
            pass
    try:
        os.rmdir(scratch_dir)
    except OSError:
        pass
    print(f'[preprocess] total {time.time() - t0:.1f}s', flush=True)
