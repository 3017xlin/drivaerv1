"""Training loop (v4 §10).

Per-rank flow:
    init_ddp → load+pin 400 train cases → build model → DDP wrap →
    torch.compile(reduce-overhead) → for epoch:
        prefetcher = AsyncPrefetcher(shard_for_this_rank, ...)
        for batch in prefetcher:
            move_to_gpu; forward; loss; backward; opt.step; lr.step
        maybe_snapshot SWA; maybe save checkpoint
    finalize: write swa_model.pt
"""
from __future__ import annotations

import math
import os.path as osp
import time
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from tqdm import tqdm

from dataset.loaders import (load_cases_pinned, load_coef_norm, load_manifest)
from dataset.prefetcher import AsyncPrefetcher
from models import DrivAer3DModel
from training.checkpoint import save_checkpoint, should_checkpoint
from training.ddp import (build_padded_shard, cleanup_ddp, init_ddp,
                          is_distributed)
from training.swa import SWAManager
from training.target_builder import (build_train_target_surf,
                                     build_train_target_vol)
from utils.memory import gpu_peak_gib
from utils.resource_monitor import ResourceMonitor
from utils.seed import set_global_seed


def _resolve_swa_window(spec: int | str, num_epochs: int) -> int:
    if isinstance(spec, str) and spec == 'auto':
        return max(50, num_epochs // 4)
    return int(spec)


def _build_optimizer_and_scheduler(model: nn.Module, cfg: dict,
                                   world: int, B: int
                                   ) -> tuple[torch.optim.Optimizer, Any]:
    base_lr = float(cfg['training']['lr'])
    wd = float(cfg['training']['weight_decay'])
    effective_lr = base_lr * math.sqrt(world * B)
    opt = AdamW(model.parameters(), lr=effective_lr,
                weight_decay=wd, fused=True)
    # 5% warmup + cosine to 10% of peak
    total_steps = max(1, cfg['training']['num_epochs']
                       * cfg['training']['steps_per_epoch_est'])
    warmup_steps = max(1, int(0.05 * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return opt, sched


def _move_batch_to_gpu(batch: dict[str, torch.Tensor], device
                       ) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def train(cfg: dict, run_dir: str, _owns_ddp: bool = True
          ) -> dict[int, dict[str, Any]] | None:
    cache_dir = cfg['data']['cache_dir']
    set_global_seed(int(cfg['seed']))
    rank, world, local = init_ddp()
    monitor = ResourceMonitor()
    if rank == 0:
        monitor.start()
        monitor.mark('init')
    device = (torch.device('cuda', local) if torch.cuda.is_available()
              else torch.device('cpu'))

    manifest = load_manifest(cache_dir)
    coef_norm = load_coef_norm(cache_dir)

    log_nut = bool(cfg['log_training']['nut'])
    log_vort = bool(cfg['log_training']['vorticity'])

    train_ids = manifest['train_ids']
    my_train_ids = sorted(train_ids[rank::world])
    if rank == 0:
        print(f'[train] world={world}, B={cfg["training"]["batch_size"]}, '
              f'log_nut={log_nut}, log_vort={log_vort}', flush=True)
    all_pt_data = load_cases_pinned(
        cache_dir, my_train_ids,
        num_workers=int(cfg['training']['num_workers']),
        with_log_sidecar=(log_nut, log_vort), rank=rank,
    )
    if rank == 0:
        print(f'[train] pinned {len(all_pt_data)} train cases (rank-local)',
              flush=True)
        monitor.mark('load')

    model = DrivAer3DModel(cfg).to(device)
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[local])

    B = int(cfg['training']['batch_size'])
    N = len(my_train_ids)
    steps_per_epoch = (N + B - 1) // B
    cfg['training']['steps_per_epoch_est'] = steps_per_epoch

    opt, sched = _build_optimizer_and_scheduler(model, cfg, world, B)

    inner = model.module if hasattr(model, 'module') else model
    inner.vit.rope.set_rope_scale(coef_norm['rope_scale_per_axis'].to(torch.float32))

    compile_mode = cfg['training']['compile_mode']
    try:
        compiled = torch.compile(model, mode=compile_mode, fullgraph=True)
    except Exception as e:                                                # pragma: no cover
        if rank == 0:
            print(f'[train] torch.compile failed ({e}); proceeding eagerly')
        compiled = model

    num_epochs = int(cfg['training']['num_epochs'])
    swa_window = _resolve_swa_window(
        cfg['training'].get('swa_window', 'auto'), num_epochs)
    swa = SWAManager(swa_window=swa_window, num_epochs=num_epochs,
                     every_epochs=int(cfg['checkpoint']['every_epochs_swa']))
    ckpt_dir = osp.join(run_dir, 'checkpoints')

    sampling = cfg['sampling']
    n_query = int(sampling['N_query'])
    n_query_vol = int(n_query * (1.0 - float(sampling['surface_query_ratio'])))
    surface_area_alpha = float(sampling['surface_area_alpha'])

    if rank == 0:
        monitor.mark('train')
    for epoch in range(num_epochs):
        shard = build_padded_shard(my_train_ids, B, epoch,
                                   seed_offset=int(cfg['seed']))
        prefetcher = AsyncPrefetcher(
            shard, all_pt_data, batch_size=B, epoch=epoch,
            encoder_k=int(cfg['model']['encoder_k']),
            n_query=n_query, n_query_vol=n_query_vol,
            surface_area_alpha=surface_area_alpha,
            bigbird_local=int(cfg['model']['bigbird_local']),
            bigbird_register=int(cfg['model']['bigbird_register']),
            bigbird_random=int(cfg['model']['bigbird_random']),
            num_workers=int(cfg['training']['num_workers']),
            queue_size=int(cfg['training']['prefetch_queue_size']),
        )

        epoch_loss = 0.0
        n_steps = 0
        t_epoch = time.time()
        it = prefetcher
        if rank == 0:
            it = tqdm(prefetcher, desc=f'epoch {epoch:03d}',
                      total=steps_per_epoch)
        for batch_cpu in it:
            batch = _move_batch_to_gpu(batch_cpu, device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred_vol, pred_surf = compiled(batch)
                target_vol = build_train_target_vol(batch, log_nut, log_vort)
                target_surf = build_train_target_surf(batch)
                loss_vol = F.mse_loss(pred_vol, target_vol)
                loss_surf = F.mse_loss(pred_surf, target_surf)
                loss = loss_vol + loss_surf
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg['training']['max_grad_norm']))
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            epoch_loss += float(loss.item())
            n_steps += 1
        prefetcher.close()

        # Sync epoch summary across ranks (rank 0 logs)
        if rank == 0:
            avg = epoch_loss / max(n_steps, 1)
            print(f'[epoch {epoch:03d}] loss={avg:.5f} '
                  f'lr={sched.get_last_lr()[0]:.3e} '
                  f'gpu_peak={gpu_peak_gib():.2f}GiB '
                  f't={time.time() - t_epoch:.1f}s',
                  flush=True)

        # SWA snapshot
        swa.maybe_snapshot(model, epoch)

        # Checkpoint
        if rank == 0:
            if should_checkpoint(
                    epoch, num_epochs, swa_window,
                    int(cfg['checkpoint']['every_epochs_pre_swa']),
                    int(cfg['checkpoint']['every_epochs_swa'])):
                save_checkpoint(model, ckpt_dir, epoch)

        if is_distributed():
            dist.barrier()

    # Finalize: average SWA and save
    if rank == 0 and swa.has_snapshots():
        swa_path = osp.join(run_dir, 'swa_model.pt')
        swa.average_and_save(swa_path)
        print(f'[train] SWA model saved to {swa_path}', flush=True)
    if rank == 0:
        monitor.mark('done')
        monitor.stop()
        monitor.save_png(osp.join(run_dir, 'resource_log.png'))
    if is_distributed():
        dist.barrier()
    if _owns_ddp:
        cleanup_ddp()
    return all_pt_data
