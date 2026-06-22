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


Directory structure:
└── 3017xlin-drivaer_initial/
    ├── README.md
    ├── config.yaml
    ├── evaluate.py
    ├── make_manifest.py
    ├── preprocess.py
    ├── requirements.txt
    ├── train.py
    ├── dataset/
    │   ├── __init__.py
    │   ├── loaders.py
    │   ├── prefetcher.py
    │   └── split_ids.py
    ├── evaluation/
    │   ├── __init__.py
    │   ├── curve.py
    │   ├── denormalize.py
    │   ├── metrics.py
    │   ├── test_eval.py
    │   └── viz.py
    ├── models/
    │   ├── __init__.py
    │   ├── bigbird.py
    │   ├── decoder.py
    │   ├── encoder.py
    │   ├── model.py
    │   ├── rope.py
    │   └── vit.py
    ├── preprocess/
    │   ├── __init__.py
    │   ├── geometry.py
    │   ├── kdtree.py
    │   ├── leaf_stats.py
    │   ├── log_sidecar.py
    │   ├── neighbors.py
    │   ├── pipeline.py
    │   ├── reorder.py
    │   ├── transient_baked.py
    │   └── welford.py
    ├── reporting/
    │   ├── __init__.py
    │   ├── summary.py
    │   └── tables.py
    ├── scripts/
    │   ├── run_evaluate.sh
    │   ├── run_preprocess.sh
    │   └── run_train.sh
    ├── tests/
    │   ├── __init__.py
    │   ├── test_kdtree.py
    │   ├── test_rope_scale.py
    │   ├── test_split_ids.py
    │   └── test_welford.py
    ├── training/
    │   ├── __init__.py
    │   ├── checkpoint.py
    │   ├── ddp.py
    │   ├── loop.py
    │   ├── swa.py
    │   ├── target_builder.py
    │   └── transient.py
    └── utils/
        ├── __init__.py
        ├── memory.py
        ├── resource_monitor.py
        └── seed.py
帮我按这个把这些代码内容写入这个repo，非常无脑，直接写
