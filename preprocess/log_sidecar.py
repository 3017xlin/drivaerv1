"""Write case_<id>_log.pt with z-scored log-domain nut + vorticity (v4 §3.15)."""
from __future__ import annotations

import numpy as np
import torch


def make_log_sidecar(y_volume_raw: np.ndarray,
                     mean_nut_log: float, std_nut_log: float,
                     mean_vort_log: np.ndarray, std_vort_log: np.ndarray
                     ) -> dict[str, torch.Tensor]:
    """Take raw 8-d volume targets, return dict ready for torch.save."""
    nut_raw = y_volume_raw[:, 7]
    nut_log = np.log(np.maximum(nut_raw, 1e-6))
    nut_log_z = (nut_log - mean_nut_log) / max(std_nut_log, 1e-8)

    vort_raw = y_volume_raw[:, 4:7]
    vort_log = np.sign(vort_raw) * np.log1p(np.abs(vort_raw))
    vort_log_z = (vort_log - mean_vort_log) / np.maximum(std_vort_log, 1e-8)

    return {
        'nut_log_zscored':
            torch.from_numpy(nut_log_z.astype(np.float32)).to(torch.bfloat16),
        'vort_log_zscored':
            torch.from_numpy(vort_log_z.astype(np.float32)).to(torch.bfloat16),
    }
