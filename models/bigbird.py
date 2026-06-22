"""BigBird sparse mask construction via FlexAttention (v4 §10.5).

Per query token: 110 local (from 1+2 order neighbor pool, 1st order
preserved) + 16 register (always) + 32 random (per-case-per-epoch) = 158.
"""
from __future__ import annotations

import numpy as np
import torch


def build_bigbird_index(leaf_neighbor_idx: np.ndarray,
                        L: int, n_local: int = 110, n_register: int = 16,
                        n_random: int = 32, seed: int = 0
                        ) -> np.ndarray:
    """Return key index tensor of shape (L, n_local + n_register + n_random).

    Indices in [0, L+n_register). Register keys are L..L+n_register-1.
    Random keys are drawn from {0..L-1} minus the local set.
    """
    rng = np.random.default_rng(seed)
    K = n_local + n_register + n_random
    out = np.empty((L, K), dtype=np.int32)

    valid_mask = leaf_neighbor_idx != -1
    valid_counts = valid_mask.sum(axis=1)
    local_part = leaf_neighbor_idx[:, :n_local].copy()
    short = valid_counts < n_local
    if short.any():
        for q in np.where(short)[0]:
            vc = int(valid_counts[q])
            local_part[q, :vc] = leaf_neighbor_idx[q, valid_mask[q]][:vc]
            pool = np.setdiff1d(np.arange(L, dtype=np.int32), local_part[q, :vc])
            local_part[q, vc:] = rng.choice(pool, n_local - vc, replace=False)
    out[:, :n_local] = local_part

    out[:, n_local:n_local + n_register] = np.arange(L, L + n_register, dtype=np.int32)

    rand = rng.integers(0, L, size=(L, n_random + 32), dtype=np.int32)
    local_sorted = np.sort(local_part, axis=1)
    pos = np.searchsorted(local_sorted, rand)
    pos = np.clip(pos, 0, n_local - 1)
    hit = np.take_along_axis(local_sorted, pos, 1) == rand
    rand_clean = np.where(hit, L, rand)
    rand_clean.sort(axis=1)
    out[:, n_local + n_register:] = rand_clean[:, :n_random]

    return out


def build_flex_block_mask(key_idx: torch.Tensor, B: int, H: int,
                          L_with_reg: int, BLOCK_SIZE: int = 128):
    """Wrap a (B, L, K) key index tensor as a FlexAttention BlockMask.

    Uses a sorted key lookup to determine valid (q, kv) pairs.
    Register queries (q >= L) attend to all keys; register kv positions
    are always attended to by all queries.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    L = key_idx.shape[-2]
    n_keys = key_idx.shape[-1]

    key_sorted = key_idx.sort(dim=-1).values.contiguous()

    def mask_mod(b, h, q_idx, kv_idx):
        is_reg_q = q_idx >= L
        is_reg_kv = kv_idx >= L
        q_safe = torch.where(q_idx < L, q_idx, torch.zeros_like(q_idx))
        row = key_sorted[b, q_safe]
        pos = torch.searchsorted(row, kv_idx.unsqueeze(-1)).squeeze(-1)
        pos_clamped = pos.clamp(max=n_keys - 1)
        found = row[pos_clamped] == kv_idx
        return is_reg_q | is_reg_kv | found

    block_mask = create_block_mask(
        mask_mod, B=B, H=H, Q_LEN=L_with_reg, KV_LEN=L_with_reg,
        BLOCK_SIZE=BLOCK_SIZE, device=key_idx.device,
    )
    return block_mask


# ---------------------------------------------------------------------------
# Practical fallback used by ViT: gather K/V at the 158 selected positions
# per query, then run dense attention over that 158-key window. This is
# mathematically equivalent to FlexAttention BigBird, costs the same FLOPs,
# and is portable without PyTorch 2.5's experimental APIs.
# ---------------------------------------------------------------------------


def gather_kv_for_bigbird(K: torch.Tensor, V: torch.Tensor,
                          key_idx: torch.Tensor
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather K/V at BigBird key positions.

    K, V         : (B, L_with_reg, H, head_dim)
    key_idx      : (B, L, n_keys) int32 — for each query token q,
                   the L indices that q attends to (register & random).
    Returns:
      K_gather, V_gather : (B, L, n_keys, H, head_dim)
    """
    B, _, H, D = K.shape
    nq, nk = key_idx.shape[1], key_idx.shape[2]
    idx = key_idx.long().clamp(min=0)
    batch_ar = torch.arange(B, device=K.device)[:, None, None]
    K_g = K[batch_ar, idx]                                                 # (B, L, n_keys, H, D)
    V_g = V[batch_ar, idx]
    return K_g, V_g
