"""Per-case geometry passes (v4 §3.2-§3.4): subsample, vorticity, SDF, curvature."""
from __future__ import annotations

import numpy as np

from utils.seed import make_rng, per_case_seed


def subsample_indices(n_vol_full: int, n_surf_full: int, case_id: int,
                      vol_keep_ratio: int = 8, surf_keep_ratio: int = 4
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Return (vol_keep_idx, surf_keep_idx), reproducible per case.

    Both are local indices (vol in [0, n_vol_full), surf in [0, n_surf_full)).
    """
    rng = make_rng(per_case_seed(case_id))
    vol_keep = rng.choice(n_vol_full, size=n_vol_full // vol_keep_ratio,
                          replace=False)
    surf_keep = rng.choice(n_surf_full, size=n_surf_full // surf_keep_ratio,
                           replace=False)
    return (np.sort(vol_keep).astype(np.int64),
            np.sort(surf_keep).astype(np.int64))


def compute_vorticity(volume_coords: np.ndarray,
                      volume_velocity: np.ndarray,
                      query_indices: np.ndarray | None = None,
                      k: int = 10,
                      chunk_size: int = 1_000_000) -> np.ndarray:
    """3D vorticity ω = curl(U) via least-squares velocity gradient (v4 §3.3).

    Parameters
    ----------
    volume_coords : (N_full, 3) fp32 — all volume cell centers
    volume_velocity : (N_full, 3) fp32 — velocity at all cells
    query_indices : (N_query,) int64 or None
        Indices into volume_coords/velocity for which to compute ω.
        If None, compute on every cell (test cases).
    k : kNN size for LS solve
    chunk_size : process this many query points at a time (memory cap)

    Returns
    -------
    omega : (N_query, 3) fp32 (or (N_full, 3) if query_indices is None)

    Notes
    -----
    The cKDTree is built on the FULL mesh, so neighbor accuracy is
    independent of the subsample. Only the LS solve is restricted to
    query_indices, which gives ~8x speedup vs computing on full mesh.
    """
    import scipy.spatial as ss
    n_full = volume_coords.shape[0]
    if query_indices is None:
        query_indices = np.arange(n_full, dtype=np.int64)
    ckdt = ss.cKDTree(volume_coords)

    out = np.empty((query_indices.shape[0], 3), dtype=np.float32)
    for lo in range(0, query_indices.shape[0], chunk_size):
        hi = min(lo + chunk_size, query_indices.shape[0])
        qi = query_indices[lo:hi]
        qpts = volume_coords[qi]
        qvel = volume_velocity[qi]
        _, nbr_idx = ckdt.query(qpts, k=k)                       # (chunk, k)
        dX = volume_coords[nbr_idx] - qpts[:, None, :]            # (chunk, k, 3)
        dU = volume_velocity[nbr_idx] - qvel[:, None, :]          # (chunk, k, 3)
        # Solve J = (dX^T dX)^{-1} dX^T dU per query (vectorized via lstsq)
        # Normal equations are well-conditioned for k=10.
        XtX = np.einsum('nki,nkj->nij', dX, dX)                   # (chunk, 3, 3)
        XtU = np.einsum('nki,nkj->nij', dX, dU)                   # (chunk, 3, 3)
        # Add tiny regularization for numerical robustness on degenerate
        # neighborhoods (e.g., near-coplanar samples). 1e-10 is below
        # float32 epsilon on physical-scale Δx, so it never biases real solves.
        XtX += 1e-10 * np.eye(3, dtype=np.float32)[None]
        try:
            L_chol = np.linalg.cholesky(XtX)
            y = np.linalg.solve(L_chol, XtU)
            J = np.linalg.solve(L_chol.swapaxes(-2, -1), y)
        except np.linalg.LinAlgError:
            J = np.linalg.solve(XtX, XtU)
        # ω = curl: (∂uz/∂y - ∂uy/∂z, ∂ux/∂z - ∂uz/∂x, ∂uy/∂x - ∂ux/∂y)
        out[lo:hi, 0] = J[:, 2, 1] - J[:, 1, 2]
        out[lo:hi, 1] = J[:, 0, 2] - J[:, 2, 0]
        out[lo:hi, 2] = J[:, 1, 0] - J[:, 0, 1]
    return out


def compute_sdf_and_curvature(query_points: np.ndarray,
                              stl_vertices: np.ndarray,
                              stl_faces: np.ndarray
                              ) -> dict[str, np.ndarray]:
    """SDF (signed distance), unit SDF gradient, mean & gaussian curvature at queries.

    Returns dict with keys: 'sdf', 'sdf_grad', 'curv_mean', 'curv_gauss'.
    """
    import open3d as o3d
    import pyvista as pv

    scene = o3d.t.geometry.RaycastingScene()
    mesh = o3d.t.geometry.TriangleMesh()
    mesh.vertex.positions = o3d.core.Tensor(stl_vertices.astype(np.float32))
    mesh.triangle.indices = o3d.core.Tensor(stl_faces.astype(np.int32))
    scene.add_triangles(mesh)

    qpts_t = o3d.core.Tensor(query_points.astype(np.float32))
    result = scene.compute_closest_points(qpts_t)
    closest = result['points'].numpy()
    primitive_ids = result['primitive_ids'].numpy().astype(np.int64)
    sdf = scene.compute_signed_distance(qpts_t).numpy()         # (N,) fp32

    safe_abs = np.maximum(np.abs(sdf), 1e-8)[:, None]
    sdf_grad = ((query_points - closest) / safe_abs).astype(np.float32)

    # Curvature at STL vertices, then barycentric interpolate
    faces_pv = np.hstack([np.full((stl_faces.shape[0], 1), 3, dtype=np.int64),
                          stl_faces.astype(np.int64)]).ravel()
    pmesh = pv.PolyData(stl_vertices.astype(np.float32), faces_pv)
    curv_mean_v = np.asarray(pmesh.curvature('mean'), dtype=np.float32)
    curv_gauss_v = np.asarray(pmesh.curvature('gaussian'), dtype=np.float32)
    curv_mean_q = _barycentric_interp(curv_mean_v, primitive_ids,
                                      query_points, closest, stl_vertices,
                                      stl_faces)
    curv_gauss_q = _barycentric_interp(curv_gauss_v, primitive_ids,
                                       query_points, closest, stl_vertices,
                                       stl_faces)
    return {'sdf': sdf.astype(np.float32),
            'sdf_grad': sdf_grad,
            'curv_mean': curv_mean_q.astype(np.float32),
            'curv_gauss': curv_gauss_q.astype(np.float32)}


def _barycentric_interp(per_vertex: np.ndarray,
                        prim_ids: np.ndarray,
                        query_points: np.ndarray,
                        closest_on_tri: np.ndarray,
                        stl_vertices: np.ndarray,
                        stl_faces: np.ndarray) -> np.ndarray:
    """Interpolate a per-vertex scalar to query points via barycentric coords
    on the closest triangle.
    """
    tris = stl_faces[prim_ids]                                # (N, 3)
    v0 = stl_vertices[tris[:, 0]]                              # (N, 3)
    v1 = stl_vertices[tris[:, 1]]
    v2 = stl_vertices[tris[:, 2]]
    # Compute barycentric weights for closest_on_tri inside triangle (v0,v1,v2).
    # Standard 2D barycentric in the triangle plane.
    e1 = v1 - v0
    e2 = v2 - v0
    p = closest_on_tri - v0
    d00 = np.einsum('ni,ni->n', e1, e1)
    d01 = np.einsum('ni,ni->n', e1, e2)
    d11 = np.einsum('ni,ni->n', e2, e2)
    d20 = np.einsum('ni,ni->n', p, e1)
    d21 = np.einsum('ni,ni->n', p, e2)
    denom = d00 * d11 - d01 * d01
    safe = np.where(np.abs(denom) > 1e-20, denom, 1.0)
    v = (d11 * d20 - d01 * d21) / safe
    w = (d00 * d21 - d01 * d20) / safe
    u = 1.0 - v - w
    return (u * per_vertex[tris[:, 0]]
            + v * per_vertex[tris[:, 1]]
            + w * per_vertex[tris[:, 2]])
