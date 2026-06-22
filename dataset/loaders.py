"""Pinned-RAM loaders for train cases (v4 §10.2).

Phase 1 (I/O): DataLoader with multiprocessing workers — pure torch.load,
no pin, no derive. Achieves ~3 GB/s on Lustre.
Phase 2 (pin + derive): sequential in main process (CPU-bound, not I/O).

Optional log sidecar loading (case_<id>_log.pt) is controlled by the
CLI flags; we do not modify pinned tensors in place — log sidecars are
loaded into separate pinned tensors and concatenated only when the
training target is built on GPU.
"""
from __future__ import annotations

import json
import os.path as osp
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def load_manifest(cache_dir: str) -> dict[str, list[int]]:
    with open(osp.join(cache_dir, 'manifest.json'), 'r') as f:
        return json.load(f)


def load_coef_norm(cache_dir: str) -> dict[str, Any]:
    return torch.load(osp.join(cache_dir, 'coef_norm.pt'), map_location='cpu',
                      weights_only=False)


def _derive_leaf_id_per_point(offsets: torch.Tensor) -> torch.Tensor:
    """np.repeat(arange(L), counts) via torch (cheap)."""
    counts = (offsets[1:] - offsets[:-1]).to(torch.int64)
    L = counts.shape[0]
    return torch.repeat_interleave(torch.arange(L, dtype=torch.int32),
                                   counts)


def _derive_vol_surf_reorder_idx(
        offsets: torch.Tensor, leaf_vol_count: torch.Tensor,
        n_keep: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive vol_reorder_idx / surf_reorder_idx from offsets + leaf_vol_count."""
    is_vol = torch.zeros(n_keep, dtype=torch.bool)
    off_np = offsets.numpy()
    lvc_np = leaf_vol_count.numpy()
    L = lvc_np.shape[0]
    for i in range(L):
        lo = int(off_np[i])
        is_vol[lo:lo + int(lvc_np[i])] = True
    vol_idx = torch.nonzero(is_vol, as_tuple=False).squeeze(-1).to(torch.int64)
    surf_idx = torch.nonzero(~is_vol, as_tuple=False).squeeze(-1).to(torch.int64)
    return vol_idx, surf_idx


def _derive_fields(pt: dict[str, Any]) -> None:
    """Derive leaf_id_per_point, vol_reorder_idx, surf_reorder_idx in-place."""
    offsets = pt['leaf_member_offsets']
    if 'leaf_id_per_point' not in pt:
        pt['leaf_id_per_point'] = _derive_leaf_id_per_point(offsets).pin_memory()
    if 'vol_reorder_idx' not in pt and 'N_keep' in pt:
        vol_idx, surf_idx = _derive_vol_surf_reorder_idx(
            offsets, pt['leaf_vol_count'], int(pt['N_keep']))
        pt['vol_reorder_idx'] = vol_idx.pin_memory()
        pt['surf_reorder_idx'] = surf_idx.pin_memory()


def _pin_dict(pt: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pt.items():
        if isinstance(v, torch.Tensor):
            try:
                out[k] = v.pin_memory()
            except RuntimeError:
                out[k] = v
        else:
            out[k] = v
    return out


class _CasePTDataset(Dataset):
    """Dataset that returns (case_id, raw_dict) from torch.load."""

    def __init__(self, cache_dir: str, case_ids: list[int],
                 with_log_sidecar: tuple[bool, bool] = (False, False),
                 subdir: str = ''):
        self.cache_dir = cache_dir
        self.case_ids = case_ids
        self.with_log_sidecar = with_log_sidecar
        self.subdir = subdir

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> tuple[int, dict[str, Any]]:
        cid = self.case_ids[idx]
        path = osp.join(self.cache_dir, self.subdir, f'case_{cid}.pt')
        main = torch.load(path, map_location='cpu', weights_only=False)
        if self.with_log_sidecar[0] or self.with_log_sidecar[1]:
            side = torch.load(
                osp.join(self.cache_dir, f'case_{cid}_log.pt'),
                map_location='cpu', weights_only=False)
            if self.with_log_sidecar[0]:
                main['nut_log_zscored'] = side['nut_log_zscored']
            if self.with_log_sidecar[1]:
                main['vort_log_zscored'] = side['vort_log_zscored']
        return cid, main


def _no_collate(batch: list) -> list:
    """Identity collate — return the list of (cid, dict) as-is."""
    return batch


def load_cases_pinned(cache_dir: str, case_ids: list[int],
                      num_workers: int = 30,
                      with_log_sidecar: tuple[bool, bool] = (False, False),
                      rank: int = 0,
                      ) -> dict[int, dict[str, Any]]:
    """Phase 1: DataLoader I/O.  Phase 2: pin + derive (sequential)."""
    if not case_ids:
        return {}

    ds = _CasePTDataset(cache_dir, case_ids, with_log_sidecar)
    n_workers = min(num_workers, len(case_ids))
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=n_workers,
                    prefetch_factor=1, collate_fn=_no_collate,
                    persistent_workers=False, pin_memory=False)

    raw: dict[int, dict[str, Any]] = {}
    it = dl
    if rank == 0:
        it = tqdm(dl, total=len(case_ids), desc='load (I/O)')
    for batch in it:
        for cid, pt_dict in batch:
            raw[cid] = pt_dict

    # Phase 2: pin + derive (main process, sequential)
    out: dict[int, dict[str, Any]] = {}
    for cid in case_ids:
        pt = _pin_dict(raw[cid])
        _derive_fields(pt)
        out[cid] = pt
    del raw
    return out


def load_cases_dataloader(cache_dir: str, case_ids: list[int],
                          num_workers: int = 30,
                          with_log_sidecar: tuple[bool, bool] = (False, False),
                          rank: int = 0,
                          subdir: str = '',
                          ) -> dict[int, dict[str, Any]]:
    """Load cases via DataLoader (no pin, no derive). For curve val loading."""
    if not case_ids:
        return {}

    ds = _CasePTDataset(cache_dir, case_ids, with_log_sidecar, subdir=subdir)
    n_workers = min(num_workers, len(case_ids))
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=n_workers,
                    prefetch_factor=1, collate_fn=_no_collate,
                    persistent_workers=False, pin_memory=False)

    out: dict[int, dict[str, Any]] = {}
    it = dl
    if rank == 0:
        it = tqdm(dl, total=len(case_ids), desc='load')
    for batch in it:
        for cid, pt_dict in batch:
            out[cid] = _pin_dict(pt_dict)
            _derive_fields(out[cid])
    return out


class _TestCaseDataset(Dataset):
    """Dataset for streaming test cases (one at a time)."""

    def __init__(self, cache_dir: str, case_ids: list[int]):
        self.cache_dir = cache_dir
        self.case_ids = case_ids

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> tuple[int, dict[str, Any]]:
        cid = self.case_ids[idx]
        path = osp.join(self.cache_dir, 'test', f'case_{cid}.pt')
        pt = torch.load(path, map_location='cpu', weights_only=False)
        return cid, pt


def load_val_or_test_streaming(cache_dir: str, case_id: int,
                               is_test: bool = False) -> dict[str, Any]:
    """Single-shot load (NOT pinned) for streaming pipeline (curve / test)."""
    sub = 'test/' if is_test else ''
    return torch.load(osp.join(cache_dir, sub, f'case_{case_id}.pt'),
                      map_location='cpu', weights_only=False)
