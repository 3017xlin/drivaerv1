"""Curve evaluation across all saved checkpoints (v4 §12).

  - Train_eval (34) + val (34) cases live in pinned RAM (loaded once).
  - For each ckpt: forward all 68 cases (DDP sharded), gather z-score MSE
    (linear) on rank 0, then delete the ckpt file.
  - Output: train_val_curve.json + train_val_curve.png.
"""
from __future__ import annotations

import gc
import json
import math
import os
import os.path as osp
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

from dataset.loaders import (load_cases_pinned, load_coef_norm, load_manifest)
from evaluation.denormalize import to_linear_zscore_volume
from models import DrivAer3DModel
from training.checkpoint import delete_checkpoint, list_checkpoints
from training.ddp import cleanup_ddp, init_ddp, is_distributed


def _build_curve_batch(case_pts: list[dict],
                       device: torch.device, B: int
                       ) -> list[dict[str, torch.Tensor]]:
    """Group cases into stacked batches of size B (the last may be smaller).

    Curve uses BAKED transient1 / transient2 (seed=42, persisted on disk
    in each train_eval/val PT).
    """
    out = []
    for lo in range(0, len(case_pts), B):
        items = case_pts[lo: lo + B]
        keys = ['leaf_centroid_norm', 'leaf_stats', 'leaf_sdf',
                'leaf_sdf_grad', 'leaf_curvature_mean', 'leaf_curvature_gauss',
                'leaf_neighbor_idx', 'transient1']
        stacked: dict[str, torch.Tensor] = {}
        for k in keys:
            stacked[k] = torch.stack([it[k] for it in items], dim=0).to(
                device, non_blocking=True)

        # transient2 baked
        qpos = torch.stack([it['point_pos_norm'].index_select(
            0, it['transient2_query_idx'].long()) for it in items]).to(device)
        qsdf = torch.stack([it['point_sdf'].index_select(
            0, it['transient2_query_idx'].long()) for it in items]).to(device)
        qsdf_g = torch.stack([it['point_sdf_grad'].index_select(
            0, it['transient2_query_idx'].long()) for it in items]).to(device)
        idw_idx = torch.stack([it['transient2_idw_idx'] for it in items]).to(
            device)
        idw_w = torch.stack([it['transient2_idw_w'] for it in items]).to(
            device)

        n_q_vol = int(items[0]['transient2_n_vol'])
        tgt_vol = torch.stack([it['point_y_volume'].index_select(
            0, it['transient2_query_idx'][:n_q_vol].long()) for it in items]
                              ).to(device)
        tgt_surf = torch.stack([it['point_y_surface'].index_select(
            0, (it['transient2_query_idx'][n_q_vol:].long()
                - it['N_vol_keep']))
            for it in items]).to(device)

        bkey_dummy = None     # curve uses baked transient1 only; ViT path
        # cannot use FlexAttention without keys, so fall back to the dense
        # gather code path with key_idx == all 1+2 order neighbors.
        # (Implementation choice: re-use leaf_neighbor_idx as key_idx by
        # padding to a fixed 158-key window.)
        Lc = stacked['leaf_centroid_norm'].shape[1]
        kn = stacked['leaf_neighbor_idx']                              # (B, L, N_pad)
        # Take first 158 of each row (padded already)
        K = 158
        if kn.shape[-1] < K:
            pad = torch.full((kn.shape[0], kn.shape[1], K - kn.shape[-1]),
                             -1, dtype=kn.dtype, device=kn.device)
            kn = torch.cat([kn, pad], dim=-1)
        else:
            kn = kn[..., :K]
        invalid_mask = (kn == -1)
        kn = kn.clamp(min=0)
        # Add register tokens 65536..65551 to every row
        L_with_reg = Lc + 16
        reg = torch.arange(Lc, L_with_reg, device=kn.device)[None, None].expand(
            kn.shape[0], kn.shape[1], 16)
        kn = torch.cat([kn[:, :, :K - 16], reg], dim=-1)
        attn_bias = torch.zeros(kn.shape[0], kn.shape[1], K,
                                device=kn.device, dtype=torch.float32)
        attn_bias[:, :, :K - 16].masked_fill_(
            invalid_mask[:, :, :K - 16], float('-inf'))

        stacked.update({
            'query_pos_norm': qpos,
            'query_sdf': qsdf,
            'query_sdf_grad': qsdf_g,
            'idw_indices': idw_idx,
            'idw_weights': idw_w,
            'query_target_volume': tgt_vol,
            'query_target_surface': tgt_surf,
            'bigbird_key_idx': kn.to(torch.int32),
            'bigbird_attn_bias': attn_bias,
            'n_query_vol': n_q_vol,
        })
        out.append(stacked)
    return out


def run_curve(cfg: dict, run_dir: str, delete_checkpoints: bool = False
              ) -> None:
    cache_dir = cfg['data']['cache_dir']
    rank, world, local = init_ddp()
    device = (torch.device('cuda', local) if torch.cuda.is_available()
              else torch.device('cpu'))
    manifest = load_manifest(cache_dir)
    coef_norm = load_coef_norm(cache_dir)
    train_eval_ids = manifest['train_eval_ids']
    val_ids = manifest['val_ids']
    case_ids = train_eval_ids + val_ids
    # Load all 68 cases pinned (each ~890MB → ~60GB total)
    all_pt = load_cases_pinned(cache_dir, case_ids,
                               num_workers=int(cfg['training']['num_workers']),
                               with_log_sidecar=(False, False), rank=rank)
    # For curve, we also need 'point_y_volume' / 'point_y_surface' fields
    # (they're in the PT for val; for train_eval they're train-schema PTs
    # which also have these fields).

    # Build the model and prepare for loading state_dicts
    model = DrivAer3DModel(cfg).to(device)
    model.vit.rope.set_rope_scale(coef_norm['rope_scale_per_axis'].to(torch.float32))
    ckpts = list_checkpoints(osp.join(run_dir, 'checkpoints'))
    if rank == 0:
        print(f'[curve] {len(ckpts)} checkpoints to evaluate', flush=True)

    log_nut = bool(cfg['log_training']['nut'])
    log_vort = bool(cfg['log_training']['vorticity'])
    B = int(cfg['evaluation']['curve_batch_size'])

    # DDP shard 68 case ids across ranks
    my_shard = case_ids[rank::world]
    my_pts = [all_pt[cid] for cid in my_shard]
    my_te_pts = [all_pt[cid] for cid in my_shard if cid in train_eval_ids]
    my_val_pts = [all_pt[cid] for cid in my_shard if cid in val_ids]

    curve: dict[str, dict[str, float]] = {}
    for ep, ckpt_path in (tqdm(ckpts, desc='curve') if rank == 0 else ckpts):
        sd = torch.load(ckpt_path, map_location=device)
        # cast bf16 weights back to fp32 for stable loading
        sd_fp = {k: v.to(torch.float32) if v.is_floating_point() else v
                 for k, v in sd.items()}
        model.load_state_dict(sd_fp, strict=False)
        model.eval()
        te_vol, te_surf, n_te = 0.0, 0.0, 0
        v_vol, v_surf, n_v = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in _build_curve_batch(my_te_pts, device, B):
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    pred_vol, pred_surf = model(batch)
                pred_vol_lin = to_linear_zscore_volume(
                    pred_vol.float(), coef_norm, log_nut, log_vort)
                target_vol = batch['query_target_volume'].float()
                target_surf = batch['query_target_surface'].float()
                mse_vol = float(F.mse_loss(pred_vol_lin, target_vol).item())
                mse_surf = float(F.mse_loss(pred_surf.float(), target_surf).item())
                bs = batch['leaf_centroid_norm'].shape[0]
                te_vol += mse_vol * bs
                te_surf += mse_surf * bs
                n_te += bs
            for batch in _build_curve_batch(my_val_pts, device, B):
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    pred_vol, pred_surf = model(batch)
                pred_vol_lin = to_linear_zscore_volume(
                    pred_vol.float(), coef_norm, log_nut, log_vort)
                target_vol = batch['query_target_volume'].float()
                target_surf = batch['query_target_surface'].float()
                mse_vol = float(F.mse_loss(pred_vol_lin, target_vol).item())
                mse_surf = float(F.mse_loss(pred_surf.float(), target_surf).item())
                bs = batch['leaf_centroid_norm'].shape[0]
                v_vol += mse_vol * bs
                v_surf += mse_surf * bs
                n_v += bs
        # all-reduce across ranks
        local_tensors = torch.tensor([te_vol, te_surf, float(n_te),
                                      v_vol, v_surf, float(n_v)],
                                     device=device, dtype=torch.float64)
        if is_distributed():
            dist.all_reduce(local_tensors, op=dist.ReduceOp.SUM)
        v = local_tensors.cpu().tolist()
        denom_te = max(v[2], 1.0)
        denom_v = max(v[5], 1.0)
        if rank == 0:
            curve[str(ep)] = {
                'train_eval_vol_mse': v[0] / denom_te,
                'train_eval_surf_mse': v[1] / denom_te,
                'val_vol_mse':        v[3] / denom_v,
                'val_surf_mse':       v[4] / denom_v,
            }
            with open(osp.join(run_dir, 'train_val_curve.json'), 'w') as f:
                json.dump(curve, f, indent=2)
            if delete_checkpoints:
                delete_checkpoint(ckpt_path)

    if rank == 0:
        _plot_curve(curve, osp.join(run_dir, 'train_val_curve.png'),
                    swa_window_start=cfg['training']['num_epochs']
                    - (cfg['training'].get('swa_window', 100)
                       if cfg['training'].get('swa_window', 'auto') != 'auto'
                       else max(50, cfg['training']['num_epochs'] // 4)))
    cleanup_ddp()


def _plot_curve(curve: dict[str, dict[str, float]], out_png: str,
                swa_window_start: int) -> None:
    import matplotlib.pyplot as plt
    epochs = sorted(int(k) for k in curve)
    te_vol = [curve[str(e)]['train_eval_vol_mse'] for e in epochs]
    te_surf = [curve[str(e)]['train_eval_surf_mse'] for e in epochs]
    v_vol = [curve[str(e)]['val_vol_mse'] for e in epochs]
    v_surf = [curve[str(e)]['val_surf_mse'] for e in epochs]
    fig, (ax_s, ax_v) = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    ax_s.plot(epochs, te_surf, label='train_eval', color='C0', lw=1.5)
    ax_s.plot(epochs, v_surf, label='val',        color='C3', lw=1.5)
    ax_s.set_title('Surface 4d z-score MSE'); ax_s.legend()
    ax_s.axvline(swa_window_start, color='gray', ls='--', alpha=0.6)
    ax_v.plot(epochs, te_vol, label='train_eval', color='C0', lw=1.5)
    ax_v.plot(epochs, v_vol, label='val',        color='C3', lw=1.5)
    ax_v.set_title('Volume 8d z-score MSE'); ax_v.legend()
    ax_v.axvline(swa_window_start, color='gray', ls='--', alpha=0.6)
    for ax in (ax_s, ax_v):
        ax.set_xlabel('epoch'); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=120)
    plt.close(fig)
