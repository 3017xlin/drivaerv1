"""Chan parallel Welford vs single-worker reference (v4 §18 item 18)."""
import numpy as np

from preprocess import welford


def test_chan_merge_matches_serial_scalar():
    rng = np.random.default_rng(0)
    data = rng.normal(size=10_000)
    # Reference: single pass
    ref_mean, ref_std = data.mean(), data.std()
    # Split into 17 chunks
    chunks = np.array_split(data, 17)
    partials = [
        welford.update_state(welford.init_state(0), c) for c in chunks
    ]
    merged = welford.reduce(partials)
    mean, std = welford.finalize(merged)
    assert np.allclose(mean, ref_mean, rtol=1e-10, atol=1e-12)
    assert np.allclose(std, ref_std, rtol=1e-10, atol=1e-12)


def test_chan_merge_matches_serial_vector():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(10_000, 22))
    ref_mean = data.mean(axis=0)
    ref_std = data.std(axis=0)
    chunks = np.array_split(data, 13, axis=0)
    partials = [
        welford.update_state(welford.init_state(22), c) for c in chunks
    ]
    merged = welford.reduce(partials)
    mean, std = welford.finalize(merged)
    assert np.allclose(mean, ref_mean, rtol=1e-8, atol=1e-10)
    assert np.allclose(std, ref_std, rtol=1e-8, atol=1e-10)
