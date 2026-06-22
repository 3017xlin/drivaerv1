"""Memory reporting helpers."""
import gc

import psutil
import torch


def cpu_rss_gib() -> float:
    return psutil.Process().memory_info().rss / 1024**3


def gpu_peak_gib(device: int = 0) -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024**3


def gpu_alloc_gib(device: int = 0) -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated(device) / 1024**3


def drop_pinned_cases(all_pt_data: dict, keep_ids: set[int]) -> int:
    """Release PT dicts for case ids NOT in keep_ids; return # released."""
    n = 0
    for cid in list(all_pt_data.keys()):
        if cid not in keep_ids:
            del all_pt_data[cid]
            n += 1
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return n
