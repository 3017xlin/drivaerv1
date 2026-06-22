"""SWA snapshot manager (v4 §12.1).

Per-5-epoch CPU bf16 snapshot during the last 100 epochs.
After the window closes, average the snapshots, save swa_model.pt,
and immediately clear the buffer + gc.collect().
"""
from __future__ import annotations

import gc
import os.path as osp
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn


def _state_dict_cpu_bf16(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    inner = model.module if hasattr(model, 'module') else model
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    for k, v in inner.state_dict().items():
        if v.is_floating_point():
            out[k] = v.detach().to('cpu', dtype=torch.bfloat16).clone()
        else:
            out[k] = v.detach().to('cpu').clone()
    return out


class SWAManager:
    def __init__(self, swa_window: int, num_epochs: int,
                 every_epochs: int = 5):
        self.swa_window = int(swa_window)
        self.num_epochs = int(num_epochs)
        self.every = int(every_epochs)
        self.snapshots: list[OrderedDict[str, torch.Tensor]] = []
        self.window_start = num_epochs - swa_window

    def maybe_snapshot(self, model: nn.Module, epoch: int) -> bool:
        if epoch < self.window_start:
            return False
        if (epoch - self.window_start) % self.every != 0:
            return False
        self.snapshots.append(_state_dict_cpu_bf16(model))
        return True

    def has_snapshots(self) -> bool:
        return len(self.snapshots) > 0

    def average_and_save(self, path: str) -> str:
        if not self.snapshots:
            raise RuntimeError('No SWA snapshots to average.')
        first = self.snapshots[0]
        averaged: OrderedDict[str, torch.Tensor] = OrderedDict()
        for k, v in first.items():
            if v.is_floating_point():
                averaged[k] = torch.zeros_like(v, dtype=torch.float32)
            else:
                averaged[k] = v.clone()
        for snap in self.snapshots:
            for k, v in snap.items():
                if v.is_floating_point():
                    averaged[k] += v.to(torch.float32)
        n = len(self.snapshots)
        for k, v in averaged.items():
            if v.is_floating_point():
                averaged[k] = (v / n).to(torch.bfloat16)
        torch.save(averaged, path)
        # Free buffer immediately
        self.snapshots.clear()
        gc.collect()
        return path
