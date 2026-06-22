"""Pinned-RAM loaders for train cases (v4 §10.2).

Loads each case_<id>.pt once into pinned host memory; derives the
leaf_id_per_point cache from leaf_member_offsets (NOT in the PT schema,
see v4 §10.2 paragraph 3b).

Optional log sidecar loading (case_<id>_log.pt) is controlled by the
CLI flags; we do not modify pinned tensors in place — log sidecars are
loaded into separate pinned tensors and concatenated only when the
training target is built on GPU.
"""
from __future__ import annotations

import json
import os.path as osp
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


def load_manifest(cache_dir: str) -> dict[str, list[int]]:
    with open(osp.join(cache_dir, 'manifest.json'), 'r') as f:
        return json.load(f)


def load_coef_norm(cache_dir: str) -> dict[str, Any]:
    return torch.load(osp.join(cache_dir, 'coef_norm.pt'), map_location='cpu')


def _derive_leaf_id_per_point(offsets: torch.Tensor) -> torch.Tensor:
    """np.repeat(arange(L), counts) via torch (cheap)."""
    counts = (offsets[1:] - offsets[:-1]).to(torch.int64)
    L = counts.shape[0]
    return torch.repeat_interleave(torch.arange(L, dtype=torch.int32),
                                   counts)


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


def load_one_case(cache_dir: str, case_id: int,
                  with_log_sidecar: tuple[bool, bool] = (False, False)
                  ) -> dict[str, Any]:
    """Load a single case PT, pin tensors, derive leaf_id_per_point.

    with_log_sidecar = (load_nut, load_vort). Sidecar tensors are stored
    under keys ``nut_log_zscored`` / ``vort_log_zscored`` if loaded.
    """
    main = torch.load(osp.join(cache_dir, f'case_{case_id}.pt'),
                      map_location='cpu')
    main = _pin_dict(main)
    # Derive leaf_id_per_point
    main['leaf_id_per_point'] = _derive_leaf_id_per_point(
        main['leaf_member_offsets']).pin_memory()
    if with_log_sidecar[0] or with_log_sidecar[1]:
        side = torch.load(osp.join(cache_dir, f'case_{case_id}_log.pt'),
                          map_location='cpu')
        if with_log_sidecar[0]:
            main['nut_log_zscored'] = side['nut_log_zscored'].pin_memory()
        if with_log_sidecar[1]:
            main['vort_log_zscored'] = side['vort_log_zscored'].pin_memory()
    return main


def load_cases_pinned(cache_dir: str, case_ids: list[int],
                      num_workers: int = 30,
                      with_log_sidecar: tuple[bool, bool] = (False, False),
                      rank: int = 0,
                      ) -> dict[int, dict[str, Any]]:
    """Parallel load + pin every case in ``case_ids``."""
    out: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(load_one_case, cache_dir, cid,
                             with_log_sidecar): cid
                   for cid in case_ids}
        it = as_completed(futures)
        if rank == 0:
            it = tqdm(it, total=len(futures), desc='load+pin')
        for fut in it:
            cid = futures[fut]
            out[cid] = fut.result()
    return out


def load_val_or_test_streaming(cache_dir: str, case_id: int,
                               is_test: bool = False) -> dict[str, Any]:
    """Single-shot load (NOT pinned) for streaming pipeline (curve / test)."""
    sub = 'test/' if is_test else ''
    return torch.load(osp.join(cache_dir, sub, f'case_{case_id}.pt'),
                      map_location='cpu')
