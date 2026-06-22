"""Encoder cross-attention with MLP Q and MLP K/V (v4 §7.3)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


class EncoderCrossAttention(nn.Module):
    """Single cross-attention layer with MLP projections.

    Q (per leaf):  31-d aggregate → Linear(31, 64) → GELU → Linear(64, 192)
    K (per point): 10-d transient1 → Linear(10, 32) → GELU → Linear(32, 192)
    V same as K (separate MLP).
    """

    def __init__(self, q_in_dim: int = 31, kv_in_dim: int = 10,
                 dim: int = 192, num_heads: int = 3, head_dim: int = 64,
                 q_hidden: int = 64, kv_hidden: int = 32):
        super().__init__()
        assert num_heads * head_dim == dim
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.q_in = nn.Linear(q_in_dim, q_hidden)
        self.q_out = nn.Linear(q_hidden, dim)
        self.k_in = nn.Linear(kv_in_dim, kv_hidden)
        self.k_out = nn.Linear(kv_hidden, dim)
        self.v_in = nn.Linear(kv_in_dim, kv_hidden)
        self.v_out = nn.Linear(kv_hidden, dim)
        self.o = nn.Linear(dim, dim)
        self.norm = RMSNorm(dim)

    def forward(self, leaf_aggr: torch.Tensor, transient1: torch.Tensor
                ) -> torch.Tensor:
        """
        leaf_aggr:  (B, L, 31)    bf16/fp32
        transient1: (B, L, 32, 10) bf16

        Returns leaf_token: (B, L, dim) bf16
        """
        B, L, _ = leaf_aggr.shape
        H, D = self.num_heads, self.head_dim
        leaf_q = self.q_out(F.gelu(self.q_in(leaf_aggr)))                  # (B, L, dim)
        K = self.k_out(F.gelu(self.k_in(transient1)))                       # (B, L, 32, dim)
        V = self.v_out(F.gelu(self.v_in(transient1)))
        # Reshape for multi-head, k=32 keys per leaf (one query per leaf).
        Q = leaf_q.view(B, L, H, D).unsqueeze(-2)                          # (B, L, H, 1, D)
        K = K.view(B, L, 32, H, D).permute(0, 1, 3, 2, 4)                  # (B, L, H, 32, D)
        V = V.view(B, L, 32, H, D).permute(0, 1, 3, 2, 4)
        # Run SDPA over (..., 1, D) Q and (..., 32, D) K/V.
        attn = F.scaled_dot_product_attention(Q, K, V)                     # (B, L, H, 1, D)
        attn = attn.squeeze(-2).reshape(B, L, self.dim)                    # (B, L, dim)
        out = self.o(attn)
        return self.norm(leaf_q + out)


def build_leaf_aggregate(batch: dict) -> torch.Tensor:
    """Concatenate the 31-d per-leaf aggregate from a GPU batch dict.

    [leaf_stats(22) + leaf_sdf(1) + leaf_sdf_grad(3) + leaf_curv_mean(1)
     + leaf_curv_gauss(1) + leaf_centroid_norm(3)] = 31.

    Returns (B, L, 31) bf16.
    """
    parts = [
        batch['leaf_stats'].to(torch.bfloat16),                            # (B, L, 22)
        batch['leaf_sdf'].to(torch.bfloat16).unsqueeze(-1),                # (B, L, 1)
        batch['leaf_sdf_grad'].to(torch.bfloat16),                         # (B, L, 3)
        batch['leaf_curvature_mean'].to(torch.bfloat16).unsqueeze(-1),
        batch['leaf_curvature_gauss'].to(torch.bfloat16).unsqueeze(-1),
        batch['leaf_centroid_norm'].to(torch.bfloat16),                    # (B, L, 3)
    ]
    return torch.cat(parts, dim=-1)
