"""eval_summary.json builder (v4 §15.1)."""
from __future__ import annotations

import json
from typing import Any

from evaluation.metrics import r_squared


def build_eval_summary(per_case: dict[int, dict[str, Any]],
                       timing: dict[str, float],
                       model_info: dict[str, Any]) -> dict[str, Any]:
    cids = sorted(per_case)
    avg = lambda key: float(sum(per_case[c][key]
                                for c in cids) / max(len(cids), 1))
    table1 = {
        'p_s_rel_l2_pct':   100 * avg('p_s'),
        'tau_rel_l2_pct':   100 * avg('tau'),
        'p_v_rel_l2_pct':   100 * avg('p_v'),
        'u_rel_l2_pct':     100 * avg('u'),
        'omega_rel_l2_pct': 100 * avg('omega'),
        'cd_r2': r_squared([per_case[c]['cd_pred'] for c in cids],
                           [per_case[c]['cd_true'] for c in cids]),
        'cl_r2': r_squared([per_case[c]['cl_pred'] for c in cids],
                           [per_case[c]['cl_true'] for c in cids]),
    }
    table2 = {
        'p_s_rel_l2_pct':   100 * avg('p_s'),
        'tau_x_rel_l2_pct': 100 * avg('tau_x'),
        'tau_y_rel_l2_pct': 100 * avg('tau_y'),
        'tau_z_rel_l2_pct': 100 * avg('tau_z'),
        'p_v_rel_l2_pct':   100 * avg('p_v'),
        'u_x_rel_l2_pct':   100 * avg('u_x'),
        'u_y_rel_l2_pct':   100 * avg('u_y'),
        'u_z_rel_l2_pct':   100 * avg('u_z'),
        'nut_rel_l2_pct':   100 * avg('nut'),
    }
    per_case_out = {
        str(c): {
            'p_s': per_case[c]['p_s'],
            'tau': per_case[c]['tau'],
            'p_v': per_case[c]['p_v'],
            'u':   per_case[c]['u'],
            'omega': per_case[c]['omega'],
            'cd_pred': per_case[c]['cd_pred'],
            'cl_pred': per_case[c]['cl_pred'],
            'cd_true': per_case[c]['cd_true'],
            'cl_true': per_case[c]['cl_true'],
        } for c in cids
    }
    return {
        'table1': table1,
        'table2_domino': table2,
        'per_case': per_case_out,
        'timing': timing,
        'model': model_info,
    }


def write_eval_summary(summary: dict[str, Any], path: str) -> None:
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
