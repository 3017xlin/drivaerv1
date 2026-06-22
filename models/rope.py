"""3D RoPE with per-axis Nyquist scale (v4 §8).

Positions are leaf_centroid_norm ∈ [-1, 1]^3 (already normalized in
preprocess; no second-pass normalization). rope_scale_per_axis comes
from coef_norm.pt (derived from bounding-box extent / geo mean).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class RotaryEmbedding3D(nn.Module):
    """Compute (cos, sin) tables once per case at first-layer entry.

    Output shape: (B, L+register, head_dim) for cos and sin each.
    """

    def __init__(self, head_dim: int = 64, base: float = 100.0,
                 rope_dims: tuple[int, int, int] = (22, 22, 20),
                 register_tokens: int = 16):
        super().__init__()
        assert sum(rope_dims) == head_dim, (
            f'rope_dims {rope_dims} must sum to head_dim {head_dim}')
        self.head_dim = head_dim
        self.base = base
        self.dims = rope_dims
        self.register_tokens = register_tokens
        self._scale_x = 1.0
        self._scale_y = 1.0
        self._scale_z = 1.0

    def _axis_freqs(self, positions: torch.Tensor, dim: int, scale: float
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        # positions: (B, L) fp32 in normalized [-1, 1] range
        # Returns cos / sin of shape (B, L, dim)
        device = positions.device
        half = dim // 2
        idx = torch.arange(half, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (self.base ** (2.0 * idx / dim))                # (half,)
        theta = positions[..., None] * inv_freq * scale                   # (B, L, half)
        cos = torch.cos(theta).repeat_interleave(2, dim=-1)               # (B, L, dim)
        sin = torch.sin(theta).repeat_interleave(2, dim=-1)
        return cos, sin

    def set_rope_scale(self, rope_scale_per_axis: torch.Tensor) -> None:
        """Cache rope scale as plain floats to avoid graph breaks."""
        self._scale_x = float(rope_scale_per_axis.view(-1)[0])
        self._scale_y = float(rope_scale_per_axis.view(-1)[1])
        self._scale_z = float(rope_scale_per_axis.view(-1)[2])

    def forward(self, leaf_centroid_norm: torch.Tensor,
                rope_scale_per_axis: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """leaf_centroid_norm: (B, L, 3) fp32 in [-1,1].
        rope_scale_per_axis: (B, 3) fp32.
        Returns cos, sin each of shape (B, L+R, head_dim).
        """
        B, L, _ = leaf_centroid_norm.shape
        dx, dy, dz = self.dims
        sx = self._scale_x
        sy = self._scale_y
        sz = self._scale_z
        cos_x, sin_x = self._axis_freqs(leaf_centroid_norm[..., 0], dx, sx)
        cos_y, sin_y = self._axis_freqs(leaf_centroid_norm[..., 1], dy, sy)
        cos_z, sin_z = self._axis_freqs(leaf_centroid_norm[..., 2], dz, sz)
        cos = torch.cat([cos_x, cos_y, cos_z], dim=-1)                    # (B, L, head_dim)
        sin = torch.cat([sin_x, sin_y, sin_z], dim=-1)
        # Register-token identity
        R = self.register_tokens
        reg_cos = torch.ones(B, R, self.head_dim, device=cos.device,
                             dtype=cos.dtype)
        reg_sin = torch.zeros(B, R, self.head_dim, device=sin.device,
                              dtype=sin.dtype)
        cos = torch.cat([cos, reg_cos], dim=1)                            # (B, L+R, head_dim)
        sin = torch.cat([sin, reg_sin], dim=1)
        return cos, sin


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary to (B, H, T, D_head) Q and K, with cos/sin (B, T, D_head)."""
    cos = cos[:, None]                                                    # (B, 1, T, D)
    sin = sin[:, None]
    q_rot = _rotate(q, cos, sin)
    k_rot = _rotate(k, cos, sin)
    return q_rot, k_rot


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
            ) -> torch.Tensor:
    """Pair-wise rotate adjacent dims:  (x0, x1) -> (x0 cos - x1 sin, x1 cos + x0 sin)."""
    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]
    cos_e  = cos[..., 0::2]
    sin_e  = sin[..., 0::2]
    rot_e = x_even * cos_e - x_odd * sin_e
    rot_o = x_odd * cos_e + x_even * sin_e
    # interleave back
    out = torch.stack([rot_e, rot_o], dim=-1).flatten(-2)
    return out
