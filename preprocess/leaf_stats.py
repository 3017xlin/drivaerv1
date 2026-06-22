"""22-dim leaf_stats via segment reductions (v4 §3.9).

NO Python loop over 65536 leaves. Everything uses np.add/min/max.reduceat.
Layout (after Phase 6 reorder; offsets[l]..offsets[l+1] contains members
of leaf l, with leaf_vol_count[l] volume members first):

    [0:6]   cov_xx, cov_yy, cov_zz, cov_xy, cov_xz, cov_yz
    [6:9]   dist_mean, dist_std, dist_skew
    [9]     n_valid_norm     = counts / counts.mean()
    [10]    density          = log1p(N / (4/3 π · dist_mean^3 + eps))
    [11]    com_dist         = ||leaf_centroid - leaf_box_center||
    [12:15] sdf_min, sdf_max, sdf_range
    [15:18] mean_dir_x, mean_dir_y, mean_dir_z   (unit-mean of centered/|.|)
    [18]    angular_span     = 1 - ||(mean_dir_x, mean_dir_y, mean_dir_z)||
    [19]    curv_mean_leaf_avg
    [20]    curv_gauss_leaf_avg
    [21]    surf_ratio       = 1.0 - leaf_vol_count / counts
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def compute_leaf_centroid(pos: np.ndarray, offsets: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Return (centroid [L,3], counts [L])."""
    L = offsets.shape[0] - 1
    counts = np.diff(offsets).astype(np.float64)
    sums = np.add.reduceat(pos, offsets[:-1], axis=0)
    centroid = sums / counts[:, None]
    return centroid.astype(np.float32), counts.astype(np.int32)


def compute_leaf_stats(pos: np.ndarray,
                       point_sdf: np.ndarray,
                       point_curv_mean: np.ndarray,
                       point_curv_gauss: np.ndarray,
                       offsets: np.ndarray,
                       leaf_vol_count: np.ndarray,
                       leaf_centroid: np.ndarray
                       ) -> np.ndarray:
    """All 22 stats per leaf."""
    L = offsets.shape[0] - 1
    counts = np.diff(offsets).astype(np.float64)
    counts_safe = np.maximum(counts, 1.0)
    leaf_id_per_point = np.repeat(np.arange(L), counts.astype(np.int64))
    centered = pos - leaf_centroid[leaf_id_per_point]

    out = np.zeros((L, 22), dtype=np.float32)

    cx, cy, cz = centered[:, 0], centered[:, 1], centered[:, 2]
    out[:, 0] = np.add.reduceat(cx * cx, offsets[:-1]) / counts_safe
    out[:, 1] = np.add.reduceat(cy * cy, offsets[:-1]) / counts_safe
    out[:, 2] = np.add.reduceat(cz * cz, offsets[:-1]) / counts_safe
    out[:, 3] = np.add.reduceat(cx * cy, offsets[:-1]) / counts_safe
    out[:, 4] = np.add.reduceat(cx * cz, offsets[:-1]) / counts_safe
    out[:, 5] = np.add.reduceat(cy * cz, offsets[:-1]) / counts_safe

    dist = np.linalg.norm(centered, axis=-1)
    dist_mean = np.add.reduceat(dist, offsets[:-1]) / counts_safe
    centered_dist = dist - dist_mean[leaf_id_per_point]
    dist_var = np.add.reduceat(centered_dist ** 2, offsets[:-1]) / counts_safe
    dist_std = np.sqrt(np.maximum(dist_var, 0.0))
    dist_skew = (np.add.reduceat(centered_dist ** 3, offsets[:-1])
                 / counts_safe / (dist_std ** 3 + EPS))
    out[:, 6] = dist_mean
    out[:, 7] = dist_std
    out[:, 8] = dist_skew

    counts_f = counts.astype(np.float32)
    out[:, 9] = counts_f / max(counts_f.mean(), 1.0)

    radius_cubed = dist_mean ** 3 + EPS
    out[:, 10] = np.log1p(counts_f / ((4.0 / 3.0) * np.pi * radius_cubed))

    pos_min = np.minimum.reduceat(pos, offsets[:-1], axis=0)
    pos_max = np.maximum.reduceat(pos, offsets[:-1], axis=0)
    box_center = 0.5 * (pos_min + pos_max)
    out[:, 11] = np.linalg.norm(leaf_centroid - box_center, axis=-1)

    sdf_min = np.minimum.reduceat(point_sdf, offsets[:-1])
    sdf_max = np.maximum.reduceat(point_sdf, offsets[:-1])
    out[:, 12] = sdf_min
    out[:, 13] = sdf_max
    out[:, 14] = sdf_max - sdf_min

    norm = np.linalg.norm(centered, axis=-1, keepdims=True)
    safe = np.where(norm > 1e-8, norm, 1.0)
    unit = centered / safe
    mean_dir = (np.add.reduceat(unit, offsets[:-1], axis=0)
                / counts_safe[:, None])
    out[:, 15:18] = mean_dir

    out[:, 18] = 1.0 - np.linalg.norm(mean_dir, axis=-1)

    out[:, 19] = (np.add.reduceat(point_curv_mean, offsets[:-1])
                  / counts_safe)
    out[:, 20] = (np.add.reduceat(point_curv_gauss, offsets[:-1])
                  / counts_safe)
    out[:, 21] = 1.0 - leaf_vol_count.astype(np.float32) / counts_f

    return out
