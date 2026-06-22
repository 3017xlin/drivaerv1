"""Deterministic seed helpers."""
import hashlib
import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def per_case_seed(case_id: int) -> int:
    """Stable seed for per-case subsampling (preprocess time)."""
    return int(hashlib.blake2b(str(case_id).encode(),
                               digest_size=4).hexdigest(), 16)


def per_case_epoch_seed(case_id: int, epoch: int) -> int:
    """Stable seed for per-(case, epoch) sampled_idx / BigBird random tokens."""
    h = hashlib.blake2b(f'{case_id}:{epoch}'.encode(), digest_size=4)
    return int(h.hexdigest(), 16)


def make_rng(seed_int: int) -> np.random.Generator:
    return np.random.default_rng(seed_int)
