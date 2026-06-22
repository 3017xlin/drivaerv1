"""Chan parallel-merge Welford (v4 §2.4)."""
from __future__ import annotations

import numpy as np

WelfordState = tuple[int, np.ndarray, np.ndarray]


def init_state(shape: tuple[int, ...] | int) -> WelfordState:
    if isinstance(shape, int):
        shape = (shape,) if shape > 0 else ()
    return (0,
            np.zeros(shape, dtype=np.float64),
            np.zeros(shape, dtype=np.float64))


def update_state(state: WelfordState, batch: np.ndarray) -> WelfordState:
    n0, m0, M2_0 = state
    batch = np.asarray(batch, dtype=np.float64)
    if batch.ndim == 1:
        M = batch.shape[0]
        batch_mean = batch.mean()
        batch_var = batch.var()
    else:
        M = batch.shape[0]
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
    n1 = M
    m1 = batch_mean
    M2_1 = batch_var * M
    return merge(state, (n1, m1, M2_1))


def merge(a: WelfordState, b: WelfordState) -> WelfordState:
    n_a, m_a, M2_a = a
    n_b, m_b, M2_b = b
    if n_a == 0:
        return b
    if n_b == 0:
        return a
    n = n_a + n_b
    delta = m_b - m_a
    m = m_a + delta * (n_b / n)
    M2 = M2_a + M2_b + (delta * delta) * (n_a * n_b / n)
    return (n, m, M2)


def reduce(states: list[WelfordState]) -> WelfordState:
    if not states:
        raise ValueError('reduce() called with no states')
    while len(states) > 1:
        merged = []
        for i in range(0, len(states), 2):
            if i + 1 < len(states):
                merged.append(merge(states[i], states[i + 1]))
            else:
                merged.append(states[i])
        states = merged
    return states[0]


def finalize(state: WelfordState, ddof: int = 0
             ) -> tuple[np.ndarray, np.ndarray]:
    n, m, M2 = state
    if n <= ddof:
        raise ValueError(f'Welford finalize: n={n} <= ddof={ddof}')
    var = M2 / (n - ddof)
    std = np.sqrt(np.maximum(var, 0.0))
    return m.astype(np.float32), std.astype(np.float32)
