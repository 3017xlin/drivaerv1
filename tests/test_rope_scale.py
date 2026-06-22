"""rope_scale_per_axis sanity (v4 §18 item 2)."""
import numpy as np


def test_rope_scale_formula():
    # Synthetic DrivAerML-like extents
    extents = np.array([15.0, 4.0, 3.0], dtype=np.float64)
    L = 65536
    geo_mean = float(np.cbrt(np.prod(extents)))
    scale = (L ** (1.0 / 3.0)) * extents / geo_mean
    # Geometric mean should equal L^(1/3)
    assert np.isclose(float(np.cbrt(np.prod(scale))), L ** (1.0 / 3.0),
                      rtol=1e-9)
    # Long axis must get the largest scale
    assert scale[0] > scale[1] > scale[2]
    # No NaN / inf
    assert np.all(np.isfinite(scale))


def test_rope_scale_isotropic_recovers_baseline():
    extents = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    L = 65536
    geo_mean = float(np.cbrt(np.prod(extents)))
    scale = (L ** (1.0 / 3.0)) * extents / geo_mean
    assert np.allclose(scale, L ** (1.0 / 3.0))
