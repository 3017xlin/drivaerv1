"""Async producer-consumer prefetcher (v4 §10.4).

Each rank owns one AsyncPrefetcher. Background ProcessPool builds the
next B-sized batch (CPU transient1 + transient2 + BigBird key_idx +
target tensors). A bounded queue keeps a few batches ready so GPU never
blocks on CPU.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Any, Callable, Iterable

import numpy as np
import torch

from models.bigbird import build_bigbird_index
from training.transient import build_transient1, build_transient2
from utils.seed import per_case_epoch_seed


def _stack(batch_items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Stack a list of single-case dicts into a batched dict.

    Tensor fields with identical shape across the list are stacked along
    a new leading dim B. Scalar fields are stacked into a 1-d tensor.
    """
    out: dict[str, torch.Tensor] = {}
    keys = batch_items[0].keys()
    for k in keys:
        v0 = batch_items[0][k]
        if isinstance(v0, torch.Tensor):
            out[k] = torch.stack([item[k] for item in batch_items], dim=0)
        elif isinstance(v0, np.ndarray):
            out[k] = torch.stack(
                [torch.from_numpy(item[k]) for item in batch_items], dim=0)
        elif isinstance(v0, (int, np.integer)):
            out[k] = torch.tensor([int(item[k]) for item in batch_items],
                                  dtype=torch.int32)
        else:
            out[k] = v0  # constant — keep as-is
    return out


def prepare_one_case(case_pt: dict[str, Any], case_id: int, epoch: int,
                     encoder_k: int,
                     n_query: int, n_query_vol: int,
                     surface_area_alpha: float, idw_k: int,
                     bigbird_local: int, bigbird_register: int,
                     bigbird_random: int) -> dict[str, Any]:
    """Build per-case CPU tensors for one (case, epoch, step) tuple."""
    # Attach case_id so build_transient* can derive RNG
    case_pt['_case_id'] = case_id
    t1 = build_transient1(case_pt, epoch=epoch, encoder_k=encoder_k)
    t2 = build_transient2(case_pt, epoch=epoch,
                          n_query=n_query, n_query_vol=n_query_vol,
                          surface_area_alpha=surface_area_alpha,
                          idw_k=idw_k)
    # BigBird key_idx (per-epoch random tokens)
    seed = per_case_epoch_seed(case_id, epoch) ^ 0xBB17_BB17
    leaf_neighbor_idx = case_pt['leaf_neighbor_idx'].numpy()
    key_idx = build_bigbird_index(
        leaf_neighbor_idx,
        L=int(case_pt['leaf_centroid_norm'].shape[0]),
        n_local=bigbird_local, n_register=bigbird_register,
        n_random=bigbird_random, seed=seed,
    )
    out: dict[str, Any] = {
        # Per-leaf (static, copy from pinned to a stack-able tensor)
        'leaf_centroid_norm': case_pt['leaf_centroid_norm'],
        'leaf_stats': case_pt['leaf_stats'],
        'leaf_sdf': case_pt['leaf_sdf'],
        'leaf_sdf_grad': case_pt['leaf_sdf_grad'],
        'leaf_curvature_mean': case_pt['leaf_curvature_mean'],
        'leaf_curvature_gauss': case_pt['leaf_curvature_gauss'],
        'leaf_neighbor_idx': case_pt['leaf_neighbor_idx'],
        'transient1': torch.from_numpy(t1).to(torch.bfloat16),
        'query_pos_norm': torch.from_numpy(t2['query_pos_norm']),
        'query_sdf': torch.from_numpy(t2['query_sdf']).to(torch.bfloat16),
        'query_sdf_grad': torch.from_numpy(t2['query_sdf_grad']).to(
            torch.bfloat16),
        'idw_indices': torch.from_numpy(t2['idw_indices']),
        'idw_weights': torch.from_numpy(t2['idw_weights']).to(
            torch.bfloat16),
        'query_target_volume': torch.from_numpy(
            t2['query_target_volume']).to(torch.bfloat16),
        'query_target_surface': torch.from_numpy(
            t2['query_target_surface']).to(torch.bfloat16),
        'bigbird_key_idx': torch.from_numpy(key_idx),
        'n_query_vol': int(t2['n_query_vol']),
    }
    if 'nut_log_zscored' in t2:
        out['nut_log_zscored'] = torch.from_numpy(
            t2['nut_log_zscored']).to(torch.bfloat16)
    if 'vort_log_zscored' in t2:
        out['vort_log_zscored'] = torch.from_numpy(
            t2['vort_log_zscored']).to(torch.bfloat16)
    # rope scale broadcast
    out['rope_scale_per_axis'] = case_pt.get('_rope_scale_per_axis')
    return out


class AsyncPrefetcher:
    """Producer-consumer pipeline owned by one rank.

    Maintains a single long-lived ProcessPool. Background thread feeds
    a bounded Queue of ready GPU-shape batches. ``get_next_batch()``
    blocks until one is available.
    """

    def __init__(self, case_id_list: list[int], all_pt_data: dict,
                 batch_size: int, epoch: int,
                 rope_scale_per_axis: torch.Tensor,
                 *, encoder_k: int, n_query: int, n_query_vol: int,
                 surface_area_alpha: float, idw_k: int,
                 bigbird_local: int, bigbird_register: int,
                 bigbird_random: int,
                 num_workers: int = 30, queue_size: int = 4):
        self.case_id_list = list(case_id_list)
        self.all_pt_data = all_pt_data
        self.B = batch_size
        self.epoch = epoch
        self.rope_scale = rope_scale_per_axis      # (3,) fp32
        self.encoder_k = encoder_k
        self.n_query = n_query
        self.n_query_vol = n_query_vol
        self.surface_area_alpha = surface_area_alpha
        self.idw_k = idw_k
        self.bigbird_local = bigbird_local
        self.bigbird_register = bigbird_register
        self.bigbird_random = bigbird_random
        self.queue: Queue = Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._bg = threading.Thread(target=self._run, daemon=True)
        self._bg.start()

    def _build_one(self, case_id: int) -> dict[str, Any]:
        # Note: in-process call (we already paid the ProcessPool cost
        # below). The ProcessPool worker actually calls prepare_one_case
        # with a pickled snapshot of the pinned tensors — that's slow,
        # so for the canonical implementation we keep transient prep
        # in-thread but parallelize across cases within a batch via the
        # executor. For B=1 this is just an in-thread call.
        pt = self.all_pt_data[case_id]
        pt['_rope_scale_per_axis'] = self.rope_scale
        return prepare_one_case(
            pt, case_id, self.epoch,
            encoder_k=self.encoder_k,
            n_query=self.n_query, n_query_vol=self.n_query_vol,
            surface_area_alpha=self.surface_area_alpha,
            idw_k=self.idw_k,
            bigbird_local=self.bigbird_local,
            bigbird_register=self.bigbird_register,
            bigbird_random=self.bigbird_random,
        )

    def _run(self) -> None:
        try:
            pool = ThreadPoolExecutor(max_workers=min(self.B, 8))
            for lo in range(0, len(self.case_id_list), self.B):
                if self._stop.is_set():
                    break
                ids = self.case_id_list[lo: lo + self.B]
                if len(ids) > 1:
                    items = list(pool.map(self._build_one, ids))
                else:
                    items = [self._build_one(ids[0])]
                batch = _stack(items)
                self.queue.put(batch)
            pool.shutdown(wait=False)
        finally:
            self.queue.put(None)                                            # sentinel

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        batch = self.queue.get()
        if batch is None:
            raise StopIteration
        return batch

    def close(self) -> None:
        self._stop.set()
