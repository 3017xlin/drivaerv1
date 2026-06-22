"""Relative L2 and R² helpers (v4 §13.4, §13.5)."""
from __future__ import annotations

import torch


def relative_l2_scalar(pred: torch.Tensor, true: torch.Tensor) -> float:
    num = (pred - true).norm()
    den = true.norm().clamp_min(1e-12)
    return float((num / den).item())


def relative_l2_vector(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Frobenius for shape (N, 3)."""
    num = (pred - true).norm()
    den = true.norm().clamp_min(1e-12)
    return float((num / den).item())


def r_squared(pred_list: list[float], true_list: list[float]) -> float:
    pred = torch.tensor(pred_list, dtype=torch.float64)
    true = torch.tensor(true_list, dtype=torch.float64)
    ss_res = ((pred - true) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum().clamp_min(1e-12)
    return float((1.0 - ss_res / ss_tot).item())


def integrate_force(pred_p_s: torch.Tensor, pred_tau: torch.Tensor,
                    surface_normal: torch.Tensor,
                    surface_area: torch.Tensor) -> torch.Tensor:
    """F = Σ ((p · n + τ) · A)  → (3,) tensor (ρ = 1)."""
    contrib = (pred_p_s[:, None] * surface_normal + pred_tau
               ) * surface_area[:, None]
    return contrib.sum(dim=0)


def cd_cl_from_force(force: torch.Tensor, a_ref: float,
                     u_inf: float = 38.889) -> tuple[float, float]:
    """Convert raw force (3,) to (Cd, Cl).

    Default axis mapping: drag = force[0] (streamwise x), lift = force[2]
    (vertical z). MUST cross-check abupt_codebase tutorial.ipynb's
    Cd/Cl helper at integration time — if AB-UPT uses a different
    convention we adopt theirs.
    """
    q_inf = 0.5 * u_inf * u_inf
    cd = float((force[0] / (q_inf * a_ref)).item())
    cl = float((force[2] / (q_inf * a_ref)).item())
    return cd, cl
