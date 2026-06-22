"""Checkpoint write / delete (v4 §12.1).

Only model weights are persisted (bf16). Optimizer state is NOT saved
(this codebase does not support resume by design).
"""
from __future__ import annotations

import os
import os.path as osp
from collections import OrderedDict

import torch
import torch.nn as nn


def save_checkpoint(model: nn.Module, ckpt_dir: str, epoch: int) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    inner = model.module if hasattr(model, 'module') else model
    sd: OrderedDict[str, torch.Tensor] = OrderedDict()
    for k, v in inner.state_dict().items():
        if v.is_floating_point():
            sd[k] = v.detach().to('cpu', dtype=torch.bfloat16)
        else:
            sd[k] = v.detach().to('cpu')
    path = osp.join(ckpt_dir, f'epoch_{epoch:04d}.pt')
    torch.save(sd, path)
    return path


def delete_checkpoint(ckpt_path: str) -> None:
    try:
        os.remove(ckpt_path)
    except OSError:
        pass


def list_checkpoints(ckpt_dir: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    if not osp.isdir(ckpt_dir):
        return out
    for fn in sorted(os.listdir(ckpt_dir)):
        if fn.startswith('epoch_') and fn.endswith('.pt'):
            ep = int(fn[len('epoch_'):-len('.pt')])
            out.append((ep, osp.join(ckpt_dir, fn)))
    return out


def should_checkpoint(epoch: int, num_epochs: int, swa_window: int,
                      every_pre_swa: int, every_swa: int) -> bool:
    window_start = num_epochs - swa_window
    if epoch < window_start:
        return (epoch + 1) % every_pre_swa == 0
    return (epoch - window_start) % every_swa == 0
