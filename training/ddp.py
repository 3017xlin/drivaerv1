"""DDP utilities: init, padded shard, broadcast (v4 §10.3)."""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.distributed as dist


def init_ddp() -> tuple[int, int, int]:
    """Initialize NCCL DDP. Returns (rank, world_size, local_device_idx).

    Falls back to single-process (rank=0, world=1) if env vars absent.
    """
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(os.environ.get('LOCAL_RANK', rank))
    else:
        rank, world, local = 0, 1, 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    return rank, world, local


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def cleanup_ddp() -> None:
    if is_distributed():
        dist.destroy_process_group()


def build_padded_shard(case_ids: list[int], rank: int, world: int,
                       batch_size: int, epoch: int,
                       seed_offset: int = 0) -> list[int]:
    """Per-rank shard, padded so every rank does the same number of steps.

    Returns a list of case ids assigned to this rank for this epoch,
    of length ``ceil(len(case_ids) / (world * B)) * B``.
    """
    perm_rng = np.random.default_rng(int(epoch) + seed_offset)
    shuffled = list(case_ids)
    perm_rng.shuffle(shuffled)
    total = len(shuffled)
    per_step = world * batch_size
    steps = (total + per_step - 1) // per_step
    padded_total = steps * per_step
    pad = padded_total - total
    if pad > 0:
        shuffled = shuffled + shuffled[:pad]
    # rank i gets steps[i::world]? No — easier: split into ``steps`` chunks
    # of ``per_step`` and let rank pick its window from each chunk.
    out: list[int] = []
    for s in range(steps):
        chunk = shuffled[s * per_step: (s + 1) * per_step]
        out.extend(chunk[rank * batch_size: (rank + 1) * batch_size])
    return out


def broadcast_list_int(lst: list[int], src: int = 0) -> list[int]:
    """Broadcast a python list of ints from rank=src to all ranks."""
    if not is_distributed():
        return lst
    if dist.get_rank() == src:
        t = torch.tensor(lst, dtype=torch.int64, device='cuda')
        size = torch.tensor([len(lst)], dtype=torch.int64, device='cuda')
    else:
        size = torch.tensor([0], dtype=torch.int64, device='cuda')
    dist.broadcast(size, src)
    if dist.get_rank() != src:
        t = torch.empty(int(size.item()), dtype=torch.int64, device='cuda')
    dist.broadcast(t, src)
    return t.cpu().tolist()
