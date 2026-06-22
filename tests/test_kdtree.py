"""KD-tree invariants (v4 §18 item 1)."""
import numpy as np
import pytest

from preprocess.kdtree import L_LEAVES, MAX_DEPTH, build_kdtree


def _make_synthetic_cloud(seed: int = 0, n: int = 200_000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Mix uniform + a tie-prone structured plane to exercise the
    # build-interval rule (no descent function used).
    a = rng.uniform(-1, 1, size=(n // 2, 3)).astype(np.float32)
    b = np.zeros((n // 2, 3), dtype=np.float32)
    b[:, 0] = rng.uniform(-1, 1, size=n // 2)
    b[:, 1] = 0.5                                                          # tie plane on y
    b[:, 2] = rng.uniform(-1, 1, size=n // 2)
    return np.concatenate([a, b], axis=0)


def test_equal_quality_leaves():
    pts = _make_synthetic_cloud(seed=7, n=2 * L_LEAVES + 100)
    res = build_kdtree(pts)
    counts = np.array([li.shape[0] for li in res.leaf_intervals],
                      dtype=np.int64)
    assert counts.sum() == pts.shape[0]
    assert counts.max() - counts.min() <= 1, (
        f'KD-tree leaf occupancy unbalanced: max={counts.max()} '
        f'min={counts.min()}')


def test_encoder_k_invariant_real_size():
    # For the real DrivAerML keep-subset size (~18.25M / 65536 ≥ 277),
    # every leaf must have at least 32 points.
    pts = _make_synthetic_cloud(seed=42, n=18 * L_LEAVES)
    res = build_kdtree(pts)
    counts = np.array([li.shape[0] for li in res.leaf_intervals])
    assert (counts >= 32).all(), 'encoder_k=32 invariant violated'


def test_no_duplicate_indices():
    pts = _make_synthetic_cloud(seed=3, n=L_LEAVES * 3)
    res = build_kdtree(pts)
    all_idx = np.concatenate(res.leaf_intervals)
    assert np.unique(all_idx).shape[0] == pts.shape[0]
