"""Streaming test evaluation (v4 §13).

DDP across all ranks. For each rank's shard of 50 test cases:
  1. load case PT (~7 GB) via DataLoader (num_workers=2, prefetch_factor=1)
  2. encoder + ViT once → (enc_feat, vit_feat) on GPU
  3. decoder chunked over 138M points (4M / chunk → ~35 chunks)
  4. denormalize → metrics → Cd/Cl
  5. release case
"""
from __future__ import annotations

import json
import math
import os
import os.path as osp
import pickle
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.loaders import (load_coef_norm, load_manifest,
                              load_val_or_test_streaming, _TestCaseDataset,
                              _no_collate)
from evaluation.denormalize import denormalize_surface, denormalize_volume
from evaluation.metrics import (cd_cl_from_force, integrate_force,
                                relative_l2_scalar, relative_l2_vector)
from models import DrivAer3DModel
from models.idw import gpu_idw


def _decode_chunk(model: DrivAer3DModel, enc_feat: torch.Tensor,
                  vit_feat: torch.Tensor,
                  test_pt: dict, lo: int, hi: int,
                  coef_norm: dict, device: torch.device,
                  idw_k: int = 8) -> dict[str, torch.Tensor]:
    """Run decoder for query indices [lo, hi) on GPU and return raw z-score predictions
    plus volume / surface split metadata for this chunk.
    """
    point_pos_norm = test_pt['point_pos_norm'][lo:hi].to(device,
                                                          non_blocking=True)
    point_sdf = test_pt['point_sdf'][lo:hi].to(device, non_blocking=True)
    point_sdf_grad = test_pt['point_sdf_grad'][lo:hi].to(device,
                                                           non_blocking=True)
    leaf_assignment = test_pt['point_leaf_assignment'][lo:hi].to(device,
                                                                  non_blocking=True)
    leaf_centroid_norm = test_pt['leaf_centroid_norm'].to(device,
                                                            non_blocking=True)
    leaf_neighbor_idx = test_pt['leaf_neighbor_idx'].to(device,
                                                          non_blocking=True)
    idw_idx, idw_w = gpu_idw(point_pos_norm, leaf_centroid_norm,
                             leaf_neighbor_idx, leaf_assignment, idw_k)
    # Split volume / surface by global index: query indices < N_vol_full are vol.
    n_vol_full = int(test_pt['N_vol_full'])
    is_vol = (torch.arange(lo, hi, device=device) < n_vol_full)
    # We need to feed query points sorted volume-then-surface (the
    # decoder uses n_query_vol to split). Within a chunk this means
    # re-permute by is_vol.
    vol_pos = torch.nonzero(is_vol, as_tuple=False).squeeze(-1)
    surf_pos = torch.nonzero(~is_vol, as_tuple=False).squeeze(-1)
    perm = torch.cat([vol_pos, surf_pos], dim=0)
    inv = torch.argsort(perm)
    qpos = point_pos_norm[perm].unsqueeze(0)
    qsdf = point_sdf[perm].unsqueeze(0).to(torch.bfloat16)
    qsdf_g = point_sdf_grad[perm].unsqueeze(0).to(torch.bfloat16)
    idw_idx_p = idw_idx[perm].unsqueeze(0)
    idw_w_p = idw_w[perm].unsqueeze(0).to(torch.bfloat16)
    n_q_vol = int(vol_pos.shape[0])
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        pred_vol, pred_surf = model.decode_chunk(
            enc_feat, vit_feat, qpos, qsdf, qsdf_g, idw_idx_p, idw_w_p,
            n_q_vol)
    return {
        'pred_vol_z':   pred_vol.squeeze(0).float(),
        'pred_surf_z':  pred_surf.squeeze(0).float(),
        'vol_global':   (lo + vol_pos).cpu(),       # original indices (vol)
        'surf_global':  (lo + surf_pos - n_vol_full).cpu(),   # surface-local
        'inv_perm':     inv.cpu(),
    }


def _eval_one_case(model: DrivAer3DModel, cache_dir: str, case_id: int,
                   coef_norm: dict, device: torch.device,
                   chunk_size: int, log_nut: bool, log_vort: bool,
                   keep_arrays: bool = False,
                   pt: dict | None = None
                   ) -> dict[str, Any]:
    if pt is None:
        pt = load_val_or_test_streaming(cache_dir, case_id, is_test=True)
    n_full = int(pt['N_full'])
    n_vol_full = int(pt['N_vol_full'])
    n_surf_full = int(pt['N_surf_full'])

    # Build the encoder/ViT batch (B=1) using the baked transient1
    leaf_keys = ['leaf_centroid_norm', 'leaf_stats', 'leaf_sdf',
                 'leaf_sdf_grad', 'leaf_curvature_mean',
                 'leaf_curvature_gauss', 'leaf_neighbor_idx', 'transient1']
    batch: dict[str, torch.Tensor] = {}
    for k in leaf_keys:
        batch[k] = pt[k].unsqueeze(0).to(device, non_blocking=True)
    # BigBird key_idx for test: take first 142 of N_pad, prepend nothing,
    # append 16 register tokens at L..L+15. (We deliberately use a
    # 158-key window with no random for test — random tokens are only
    # for training regularization.)
    L = batch['leaf_centroid_norm'].shape[1]
    kn = pt['leaf_neighbor_idx']
    K_local = 142
    if kn.shape[-1] < K_local:
        pad = torch.full((kn.shape[0], K_local - kn.shape[-1]), -1,
                         dtype=kn.dtype)
        kn = torch.cat([kn, pad], dim=-1)
    else:
        kn = kn[..., :K_local]
    invalid_mask = (kn == -1)
    kn = kn.clamp(min=0)
    reg = torch.arange(L, L + 16)[None].expand(L, -1)
    key_idx = torch.cat([kn, reg], dim=-1).to(torch.int32).unsqueeze(0).to(
        device, non_blocking=True)
    batch['bigbird_key_idx'] = key_idx
    attn_bias = torch.zeros(1, L, K_local + 16, dtype=torch.float32)
    attn_bias[0, :, :K_local].masked_fill_(invalid_mask, float('-inf'))
    batch['bigbird_attn_bias'] = attn_bias.to(device, non_blocking=True)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        enc_feat, vit_feat = model.encode(batch)

    pred_vol_full_z = torch.zeros(n_vol_full, 8, device='cpu',
                                  dtype=torch.float32)
    pred_surf_full_z = torch.zeros(n_surf_full, 4, device='cpu',
                                   dtype=torch.float32)
    chunk_size = int(chunk_size)
    for lo in range(0, n_full, chunk_size):
        hi = min(lo + chunk_size, n_full)
        out = _decode_chunk(model, enc_feat, vit_feat, pt, lo, hi,
                            coef_norm, device, idw_k=8)
        pred_vol_full_z[out['vol_global']] = out['pred_vol_z'].cpu()
        pred_surf_full_z[out['surf_global']] = out['pred_surf_z'].cpu()

    # Denormalize
    pred_vol_phys = denormalize_volume(
        pred_vol_full_z.to(device), coef_norm, log_nut, log_vort).cpu()
    pred_surf_phys = denormalize_surface(
        pred_surf_full_z.to(device), coef_norm).cpu()
    target_vol = pt['point_y_volume_raw'].to(torch.float32)
    target_surf = pt['point_y_surface_raw'].to(torch.float32)
    surface_normal = pt['surface_normals'].to(torch.float32)
    surface_area = pt['surface_areas'].to(torch.float32)
    a_ref = float(pt['a_ref'].item())
    cd_true = float(pt['cd_true'].item())
    cl_true = float(pt['cl_true'].item())

    metrics = {
        'p_s':    relative_l2_scalar(pred_surf_phys[:, 0], target_surf[:, 0]),
        'tau':    relative_l2_vector(pred_surf_phys[:, 1:4], target_surf[:, 1:4]),
        'p_v':    relative_l2_scalar(pred_vol_phys[:, 0],  target_vol[:, 0]),
        'u':      relative_l2_vector(pred_vol_phys[:, 1:4], target_vol[:, 1:4]),
        'omega':  relative_l2_vector(pred_vol_phys[:, 4:7], target_vol[:, 4:7]),
        # Per-component (Table 2)
        'tau_x':  relative_l2_scalar(pred_surf_phys[:, 1], target_surf[:, 1]),
        'tau_y':  relative_l2_scalar(pred_surf_phys[:, 2], target_surf[:, 2]),
        'tau_z':  relative_l2_scalar(pred_surf_phys[:, 3], target_surf[:, 3]),
        'u_x':    relative_l2_scalar(pred_vol_phys[:, 1], target_vol[:, 1]),
        'u_y':    relative_l2_scalar(pred_vol_phys[:, 2], target_vol[:, 2]),
        'u_z':    relative_l2_scalar(pred_vol_phys[:, 3], target_vol[:, 3]),
        'nut':    relative_l2_scalar(pred_vol_phys[:, 7], target_vol[:, 7]),
    }
    force = integrate_force(pred_surf_phys[:, 0], pred_surf_phys[:, 1:4],
                            surface_normal, surface_area)
    cd_pred, cl_pred = cd_cl_from_force(force, a_ref)
    metrics.update({'cd_pred': cd_pred, 'cl_pred': cl_pred,
                    'cd_true': cd_true, 'cl_true': cl_true})
    if keep_arrays:
        metrics['_pred_vol_phys'] = pred_vol_phys
        metrics['_pred_surf_phys'] = pred_surf_phys
        metrics['_target_vol_phys'] = target_vol
        metrics['_target_surf_phys'] = target_surf
        metrics['_stl_v'] = pt['stl_vertices']
        metrics['_stl_f'] = pt['stl_faces']
        metrics['_pos_norm'] = pt['point_pos_norm']
    return metrics


def run_test_eval(cfg: dict, run_dir: str,
                  viz_case_ids: set[int] | None = None) -> dict[str, Any]:
    """Returns per_case_metrics dict (50 entries) for downstream reporting.

    All DDP ranks participate in inference (test cases sharded across GPUs).
    If viz_case_ids is provided, only those cases retain large arrays for
    visualization; otherwise a two-pass approach finds median cases.
    """
    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local = int(os.environ.get('LOCAL_RANK', '0'))

    cache_dir = cfg['data']['cache_dir']
    manifest = load_manifest(cache_dir)
    coef_norm = load_coef_norm(cache_dir)
    device = (torch.device('cuda', local) if torch.cuda.is_available()
              else torch.device('cpu'))
    model = DrivAer3DModel(cfg).to(device)
    model.vit.rope.set_rope_scale(coef_norm['rope_scale_per_axis'])
    swa_path = osp.join(run_dir, 'swa_model.pt')
    sd = torch.load(swa_path, map_location=device, weights_only=False)
    sd_fp = {k: v.to(torch.float32) if v.is_floating_point() else v
             for k, v in sd.items()}
    model.load_state_dict(sd_fp, strict=False)
    model.eval()
    log_nut = bool(cfg['log_training']['nut'])
    log_vort = bool(cfg['log_training']['vorticity'])
    chunk = int(cfg['evaluation']['test_chunk_size'])

    test_ids = manifest['test_ids']
    my_test_ids = sorted(test_ids[rank::world])

    ds = _TestCaseDataset(cache_dir, my_test_ids)
    dl = DataLoader(ds, batch_size=1, shuffle=False,
                    num_workers=min(2, len(my_test_ids)),
                    prefetch_factor=1, collate_fn=_no_collate,
                    persistent_workers=False, pin_memory=False)

    my_per_case: dict[int, dict[str, Any]] = {}
    it = dl
    if rank == 0:
        it = tqdm(dl, total=len(my_test_ids), desc='test eval (pass 1)')
    for batch in it:
        for cid, pt_cur in batch:
            my_per_case[cid] = _eval_one_case(
                model, cache_dir, cid, coef_norm, device, chunk,
                log_nut, log_vort, keep_arrays=False, pt=pt_cur)
            del pt_cur
            torch.cuda.empty_cache()

    # All-gather scalar metrics to rank 0
    per_case = _allgather_metrics(my_per_case, rank, world, device)

    if rank == 0:
        if viz_case_ids is None:
            from evaluation.viz import _median_case
            vol_median = _median_case(per_case,
                                      ['p_v', 'u_x', 'u_y', 'u_z', 'omega'])
            surf_median = _median_case(per_case,
                                       ['p_s', 'tau_x', 'tau_y', 'tau_z'])
            viz_case_ids = {vol_median, surf_median}

        for cid in tqdm(list(viz_case_ids), desc='test eval (pass 2: viz)'):
            per_case[cid] = _eval_one_case(model, cache_dir, cid, coef_norm,
                                           device, chunk, log_nut, log_vort,
                                           keep_arrays=True)
            torch.cuda.empty_cache()

    if dist.is_initialized():
        dist.barrier()
    return per_case if rank == 0 else {}


def _allgather_metrics(local_metrics: dict[int, dict[str, Any]],
                       rank: int, world: int,
                       device: torch.device) -> dict[int, dict[str, Any]]:
    """Gather scalar per-case metrics from all ranks to rank 0."""
    if world <= 1 or not dist.is_initialized():
        return local_metrics
    serializable = {}
    for cid, m in local_metrics.items():
        serializable[cid] = {k: v for k, v in m.items()
                             if not k.startswith('_')}
    data = pickle.dumps(serializable)
    size_t = torch.tensor([len(data)], dtype=torch.long, device=device)
    sizes = [torch.zeros(1, dtype=torch.long, device=device)
             for _ in range(world)]
    dist.all_gather(sizes, size_t)
    max_size = max(s.item() for s in sizes)
    buf = torch.zeros(max_size, dtype=torch.uint8, device=device)
    buf[:len(data)] = torch.frombuffer(bytearray(data), dtype=torch.uint8).to(device)
    all_bufs = [torch.zeros(max_size, dtype=torch.uint8, device=device)
                for _ in range(world)]
    dist.all_gather(all_bufs, buf)
    merged: dict[int, dict[str, Any]] = {}
    for i in range(world):
        sz = int(sizes[i].item())
        remote_data = bytes(all_bufs[i][:sz].cpu().numpy().tobytes())
        remote_metrics = pickle.loads(remote_data)
        merged.update(remote_metrics)
    return merged
