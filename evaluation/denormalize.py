"""z-score → physical denormalization (v4 §13.3) and the inverse log path
used for curve evaluation in linear-z-score space (v4 §12.3).
"""
from __future__ import annotations

import torch


def denormalize_volume(pred_zscore: torch.Tensor,
                       coef_norm: dict,
                       log_nut: bool, log_vort: bool) -> torch.Tensor:
    """(N_vol, 8) z-score → (N_vol, 8) physical."""
    mean_v = torch.as_tensor(coef_norm['mean_out_volume'],
                             device=pred_zscore.device)
    std_v = torch.as_tensor(coef_norm['std_out_volume'],
                            device=pred_zscore.device)
    out = pred_zscore * std_v + mean_v
    if log_vort:
        mean_vl = torch.as_tensor(coef_norm['mean_vort_log'],
                                  device=pred_zscore.device)
        std_vl = torch.as_tensor(coef_norm['std_vort_log'],
                                 device=pred_zscore.device)
        log_phys = pred_zscore[..., 4:7] * std_vl + mean_vl
        out[..., 4:7] = torch.sign(log_phys) * (torch.exp(
            torch.abs(log_phys)) - 1.0)
    if log_nut:
        mean_nl = float(coef_norm['mean_nut_log'])
        std_nl = float(coef_norm['std_nut_log'])
        log_phys = pred_zscore[..., 7] * std_nl + mean_nl
        out[..., 7] = torch.exp(log_phys).clamp(min=0.0)
    return out


def denormalize_surface(pred_zscore: torch.Tensor,
                        coef_norm: dict) -> torch.Tensor:
    """(N_surf, 4) z-score → (N_surf, 4) physical."""
    mean_s = torch.as_tensor(coef_norm['mean_out_surface'],
                             device=pred_zscore.device)
    std_s = torch.as_tensor(coef_norm['std_out_surface'],
                            device=pred_zscore.device)
    return pred_zscore * std_s + mean_s


def to_linear_zscore_volume(pred_zscore: torch.Tensor,
                            coef_norm: dict,
                            log_nut: bool, log_vort: bool
                            ) -> torch.Tensor:
    """For curve MSE: take model output in (possibly) log-z-score space,
    convert log-z dims back to linear-z-score space so all curves are
    comparable across configurations.
    """
    if not log_nut and not log_vort:
        return pred_zscore
    out = pred_zscore.clone()
    if log_vort:
        mean_vl = torch.as_tensor(coef_norm['mean_vort_log'],
                                  device=pred_zscore.device)
        std_vl = torch.as_tensor(coef_norm['std_vort_log'],
                                 device=pred_zscore.device)
        log_phys = pred_zscore[..., 4:7] * std_vl + mean_vl
        phys = torch.sign(log_phys) * (torch.exp(torch.abs(log_phys)) - 1.0)
        lin_mean = torch.as_tensor(coef_norm['mean_out_volume'][4:7],
                                   device=pred_zscore.device)
        lin_std = torch.as_tensor(coef_norm['std_out_volume'][4:7],
                                  device=pred_zscore.device)
        out[..., 4:7] = (phys - lin_mean) / torch.clamp(lin_std, min=1e-8)
    if log_nut:
        mean_nl = float(coef_norm['mean_nut_log'])
        std_nl = float(coef_norm['std_nut_log'])
        log_phys = pred_zscore[..., 7] * std_nl + mean_nl
        phys = torch.exp(log_phys).clamp(min=0.0)
        lin_mean = float(coef_norm['mean_out_volume'][7])
        lin_std = float(coef_norm['std_out_volume'][7])
        out[..., 7] = (phys - lin_mean) / max(lin_std, 1e-8)
    return out
