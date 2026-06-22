"""Build training target on GPU from batch + optional log sidecar (v4 §9.1).

The pinned tensors are NEVER modified — log sidecar columns are
concatenated into a fresh target tensor on every step.
"""
from __future__ import annotations

import torch


def build_train_target_vol(batch: dict[str, torch.Tensor],
                           use_log_nut: bool, use_log_vort: bool
                           ) -> torch.Tensor:
    """Return (B, N_q_vol, 8) target tensor in current training space."""
    base = batch['query_target_volume']                                    # (B, N_q_vol, 8)
    if not use_log_nut and not use_log_vort:
        return base
    parts: list[torch.Tensor] = []
    # dims 0..3 (p_v, U) always linear z-score
    parts.append(base[..., :4])
    # dims 4:7 (ω)
    if use_log_vort:
        parts.append(batch['vort_log_zscored'])
    else:
        parts.append(base[..., 4:7])
    # dim 7 (nut)
    if use_log_nut:
        parts.append(batch['nut_log_zscored'].unsqueeze(-1))
    else:
        parts.append(base[..., 7:8])
    return torch.cat(parts, dim=-1)


def build_train_target_surf(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return batch['query_target_surface']
