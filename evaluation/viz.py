"""Post-train visualizations (v4 §14)."""
from __future__ import annotations

import os.path as osp
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def cdcl_scatter(per_case: dict[int, dict[str, Any]], path: str) -> None:
    cds_t = [m['cd_true'] for m in per_case.values()]
    cds_p = [m['cd_pred'] for m in per_case.values()]
    cls_t = [m['cl_true'] for m in per_case.values()]
    cls_p = [m['cl_pred'] for m in per_case.values()]
    fig, (axd, axl) = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, t, p, name in ((axd, cds_t, cds_p, 'Cd'),
                            (axl, cls_t, cls_p, 'Cl')):
        ax.scatter(t, p, s=20, alpha=0.7)
        lo = min(min(t), min(p))
        hi = max(max(t), max(p))
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5)
        ax.set_xlabel(f'{name} true'); ax.set_ylabel(f'{name} pred')
        ax.set_title(f'{name}'); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120)
    plt.close(fig)


def per_case_error_hist(per_case: dict[int, dict[str, Any]], path: str
                        ) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key in zip(axes, ['p_s', 'u', 'omega']):
        vals = [m[key] * 100 for m in per_case.values()]
        ax.hist(vals, bins=20, alpha=0.8, color='C0', edgecolor='k')
        ax.set_title(f'{key} relative L2 (%)')
        ax.set_xlabel('%'); ax.set_ylabel('# cases')
        ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120)
    plt.close(fig)


def _median_case(per_case: dict[int, dict[str, Any]],
                 component_keys: list[str]) -> int:
    scores = {cid: float(np.mean([m[k] for k in component_keys]))
              for cid, m in per_case.items()}
    sorted_ids = sorted(scores, key=lambda c: scores[c])
    return sorted_ids[len(sorted_ids) // 2]


def vol_slice_velocity_pressure(per_case: dict[int, dict[str, Any]],
                                run_dir: str,
                                y_tol: float = 0.015,
                                pct: tuple[float, float] = (0.5, 99.5)
                                ) -> tuple[str, str]:
    """Tripcolor of y=0 slice for velocity magnitude and pressure."""
    import matplotlib.tri as mtri
    median_id = _median_case(per_case,
                              ['p_v', 'u_x', 'u_y', 'u_z', 'omega'])
    m = per_case[median_id]
    pos = m['_pos_norm'].numpy()
    pred = m['_pred_vol_phys'].numpy()                                     # (N_vol, 8) — full
    target = m['_target_vol_phys'].numpy()
    # Filter |y| < tol; pos here is normalized → tolerance in normalized space
    n_vol = pred.shape[0]
    pos_vol = pos[:n_vol]
    mask = np.abs(pos_vol[:, 1]) < y_tol
    x = pos_vol[mask, 0]; z = pos_vol[mask, 2]
    if x.shape[0] < 100:
        # too few points after slice; relax
        mask = np.abs(pos_vol[:, 1]) < max(y_tol * 2, 0.04)
        x = pos_vol[mask, 0]; z = pos_vol[mask, 2]
    tri = mtri.Triangulation(x, z)
    outs = []
    for name, ch_or_fn in (
            ('vol_slice_velocity.png',
             lambda arr: np.linalg.norm(arr[mask, 1:4], axis=-1)),
            ('vol_slice_pressure.png',
             lambda arr: arr[mask, 0])):
        gt = ch_or_fn(target); pd = ch_or_fn(pred); err = np.abs(pd - gt)
        lo, hi = np.percentile(np.concatenate([gt, pd]), pct)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, arr, title, vlim in (
                (axes[0], gt, 'GT', (lo, hi)),
                (axes[1], pd, 'Pred', (lo, hi)),
                (axes[2], err, 'Abs Error', (0.0, None))):
            tc = ax.tripcolor(tri, arr, shading='gouraud', cmap='RdBu_r',
                              vmin=vlim[0], vmax=vlim[1])
            fig.colorbar(tc, ax=ax, shrink=0.9)
            ax.set_title(title); ax.set_aspect('equal')
            ax.set_xlabel('x (norm)'); ax.set_ylabel('z (norm)')
        fig.suptitle(f'{name} (case {median_id})')
        fig.tight_layout(); out = osp.join(run_dir, name)
        fig.savefig(out, dpi=110); plt.close(fig)
        outs.append(out)
    return tuple(outs)                                                     # type: ignore[return-value]


def surface_field_views(per_case: dict[int, dict[str, Any]], run_dir: str,
                        coef_norm: dict | None = None
                        ) -> tuple[str, str]:
    """3×2 PyVista renders of surface pressure and shear stress magnitude
    on STL mesh: rows (GT, Pred, AbsError) × cols (top, side).
    """
    import pyvista as pv
    from scipy.spatial import cKDTree
    median_id = _median_case(per_case,
                              ['p_s', 'tau_x', 'tau_y', 'tau_z'])
    m = per_case[median_id]
    stl_v = m['_stl_v'].numpy()
    stl_f = m['_stl_f'].numpy()
    faces_pv = np.hstack([np.full((stl_f.shape[0], 1), 3, dtype=np.int64),
                          stl_f.astype(np.int64)]).ravel()
    pmesh = pv.PolyData(stl_v.astype(np.float32), faces_pv)
    n_vol = m['_pred_vol_phys'].shape[0]
    surf_pos = m['_pos_norm'][n_vol:].numpy()
    # Map surface_coords-like positions (normalized) to STL vertices via kNN.
    # For visualization we work directly in normalized space (both arrays
    # come from the same coef_norm), so cKDTree.query on STL renormalized
    # is approximate but visually fine.
    pred_p = m['_pred_surf_phys'][:, 0].numpy()
    target_p = m['_target_surf_phys'][:, 0].numpy()
    pred_tau = np.linalg.norm(m['_pred_surf_phys'][:, 1:4].numpy(), axis=-1)
    target_tau = np.linalg.norm(m['_target_surf_phys'][:, 1:4].numpy(), axis=-1)
    if coef_norm is not None:
        norm_p5 = np.asarray(coef_norm['norm_p5'], dtype=np.float32)
        norm_p95 = np.asarray(coef_norm['norm_p95'], dtype=np.float32)
        span = np.maximum(norm_p95 - norm_p5, 1e-12)
        stl_vn = ((stl_v - norm_p5) / span * 2.0 - 1.0).astype(np.float32)
    else:
        stl_vn = stl_v - stl_v.min(0)
        stl_vn /= np.maximum(stl_vn.max(0), 1e-8)
        stl_vn = stl_vn * 2 - 1
    tree = cKDTree(stl_vn)
    _, idx = tree.query(surf_pos)
    pmesh['gt_p'] = _scatter_to_vertices(target_p, idx, stl_v.shape[0])
    pmesh['pd_p'] = _scatter_to_vertices(pred_p, idx, stl_v.shape[0])
    pmesh['gt_tau'] = _scatter_to_vertices(target_tau, idx, stl_v.shape[0])
    pmesh['pd_tau'] = _scatter_to_vertices(pred_tau, idx, stl_v.shape[0])

    outs = []
    for field_name, fname in (('p', 'surf_pressure.png'),
                                ('tau', 'surf_shearstress.png')):
        gt_arr = pmesh[f'gt_{field_name}']
        pd_arr = pmesh[f'pd_{field_name}']
        err = np.abs(pd_arr - gt_arr)
        fig, axes = plt.subplots(3, 2, figsize=(10, 12))
        for row, (arr, label) in enumerate([(gt_arr, 'GT'),
                                              (pd_arr, 'Pred'),
                                              (err, 'AbsError')]):
            for col, cpos in enumerate([('top-down', 'xy'),
                                          ('side', 'xz')]):
                ax = axes[row, col]
                # Use off-screen PyVista plotter
                pl = pv.Plotter(off_screen=True, window_size=(600, 600))
                pl.add_mesh(pmesh, scalars=arr, cmap='RdBu_r',
                            show_scalar_bar=True)
                pl.camera_position = cpos[1]
                img = pl.screenshot(return_img=True)
                pl.close()
                ax.imshow(img); ax.axis('off')
                ax.set_title(f'{label} - {cpos[0]}')
        fig.suptitle(f'{fname} (case {median_id})')
        fig.tight_layout()
        out = osp.join(run_dir, fname)
        fig.savefig(out, dpi=110); plt.close(fig)
        outs.append(out)
    return tuple(outs)                                                     # type: ignore[return-value]


def _scatter_to_vertices(values: np.ndarray, idx: np.ndarray,
                         n_verts: int) -> np.ndarray:
    """Map per-surf-point values to per-vertex by averaging."""
    out = np.zeros(n_verts, dtype=np.float32)
    cnt = np.zeros(n_verts, dtype=np.float32)
    np.add.at(out, idx, values)
    np.add.at(cnt, idx, 1.0)
    cnt = np.maximum(cnt, 1.0)
    return out / cnt
