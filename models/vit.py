"""12-layer ViT with BigBird sparse attention + dense SwiGLU FFN (v4 §7.4).

- Pre-norm RMSNorm + multi-head attention with QK-RMSNorm + 3D RoPE.
- BigBird is implemented as a gathered-K/V dense attention over the
  158-key window per query (mathematically identical to FlexAttention
  BigBird; see models/bigbird.py for the FlexAttention path stub).
- 6 U-Net skip pairs (layer 0↔11, 1↔10, 2↔9, 3↔8, 4↔7, 5↔6).
- 16 register tokens appended after Encoder.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import RMSNorm
from .rope import RotaryEmbedding3D, apply_rope

_FLEX_AVAILABLE = False
try:
    from torch.nn.attention.flex_attention import flex_attention
    _FLEX_AVAILABLE = True
except ImportError:
    pass


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int = 192, hidden: int = 768,
                 dropout: float = 0.1):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden)
        self.w_up = nn.Linear(dim, hidden)
        self.w_down = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class MultiHeadAttention(nn.Module):
    """Multi-head attention with QK-RMSNorm + 3D RoPE + BigBird sparse keys."""

    def __init__(self, dim: int = 192, num_heads: int = 3,
                 attn_dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.attn_dropout = attn_dropout

    def forward(self, x: torch.Tensor, key_idx: torch.Tensor | None,
                cos: torch.Tensor, sin: torch.Tensor,
                flex_mask=None,
                attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        """
        x:        (B, T, dim)         where T = L + R (register tokens included)
        key_idx:  (B, L, n_keys) int32, key positions for non-register queries.
                  If None, runs dense full attention.
        cos, sin: (B, T, head_dim) rotary tables
        flex_mask: optional FlexAttention BlockMask (avoids K/V gather)
        attn_bias: (B, L, n_keys) float — -inf for invalid key positions
        """
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim
        Q = self.q_proj(x).view(B, T, H, D)
        K = self.k_proj(x).view(B, T, H, D)
        V = self.v_proj(x).view(B, T, H, D)
        Q = self.q_norm(Q)
        K = self.k_norm(K)
        Q_t = Q.permute(0, 2, 1, 3)                                       # (B, H, T, D)
        K_t = K.permute(0, 2, 1, 3)
        Q_t, K_t = apply_rope(Q_t, K_t, cos, sin)

        if key_idx is None:
            V_t = V.permute(0, 2, 1, 3)
            attn = F.scaled_dot_product_attention(
                Q_t, K_t, V_t, dropout_p=self.attn_dropout)
            attn = attn.permute(0, 2, 1, 3).reshape(B, T, self.dim)
            return self.o_proj(attn)

        if _FLEX_AVAILABLE and flex_mask is not None:
            V_t = V.permute(0, 2, 1, 3)
            attn = flex_attention(Q_t, K_t, V_t,
                                  block_mask=flex_mask)
            attn = attn.permute(0, 2, 1, 3).reshape(B, T, self.dim)
            return self.o_proj(attn)

        Q = Q_t.permute(0, 2, 1, 3)                                       # (B, T, H, D)
        K = K_t.permute(0, 2, 1, 3)
        L = key_idx.shape[1]
        R = T - L
        idx = key_idx.long().clamp(min=0)                                  # (B, L, n_keys)
        batch_ar = torch.arange(B, device=x.device)[:, None, None]
        K_g = K[batch_ar, idx]                                             # (B, L, n_keys, H, D)
        V_g = V[batch_ar, idx]
        nk = idx.shape[-1]
        Q_main = Q[:, :L].reshape(B * L, 1, H, D).permute(0, 2, 1, 3)
        K_main = K_g.reshape(B * L, nk, H, D).permute(0, 2, 1, 3)
        V_main = V_g.reshape(B * L, nk, H, D).permute(0, 2, 1, 3)
        ab = None
        if attn_bias is not None:
            ab = attn_bias.reshape(B * L, 1, 1, nk)
        attn_main = F.scaled_dot_product_attention(
            Q_main, K_main, V_main, attn_mask=ab,
            dropout_p=self.attn_dropout)
        attn_main = attn_main.squeeze(-2).reshape(B, L, H, D)
        Q_reg = Q[:, L:].permute(0, 2, 1, 3)
        K_all = K.permute(0, 2, 1, 3)
        V_all = V.permute(0, 2, 1, 3)
        attn_reg = F.scaled_dot_product_attention(
            Q_reg, K_all, V_all, dropout_p=self.attn_dropout)
        attn_reg = attn_reg.permute(0, 2, 1, 3)
        attn = torch.cat([attn_main, attn_reg], dim=1).reshape(B, T,
                                                                self.dim)
        return self.o_proj(attn)


class ViTBlock(nn.Module):
    def __init__(self, dim: int = 192, num_heads: int = 3,
                 ffn_hidden: int = 768, ffn_dropout: float = 0.1,
                 attn_dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, attn_dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, ffn_hidden, dropout=ffn_dropout)

    def forward(self, x: torch.Tensor, key_idx: torch.Tensor | None,
                cos: torch.Tensor, sin: torch.Tensor,
                flex_mask=None,
                attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), key_idx, cos, sin,
                          flex_mask, attn_bias)
        x = x + self.ffn(self.norm2(x))
        return x


class ViT(nn.Module):
    """12-layer ViT with 6 U-Net skip pairs."""

    def __init__(self, dim: int = 192, num_layers: int = 12,
                 num_heads: int = 3, ffn_hidden: int = 768,
                 register_tokens: int = 16,
                 head_dim: int = 64,
                 ffn_dropout: float = 0.1, attn_dropout: float = 0.0,
                 rope_base: float = 100.0,
                 rope_dims: tuple[int, int, int] = (22, 22, 20)):
        super().__init__()
        assert num_layers % 2 == 0
        self.num_layers = num_layers
        self.register_tokens = register_tokens
        self.register = nn.Parameter(torch.randn(register_tokens, dim) * 0.02)
        self.blocks = nn.ModuleList([
            ViTBlock(dim, num_heads, ffn_hidden, ffn_dropout, attn_dropout)
            for _ in range(num_layers)
        ])
        # 6 skip projections for the second-half blocks (concat with first-half
        # output, project back to dim).
        self.skip_proj = nn.ModuleList([
            nn.Linear(2 * dim, dim) for _ in range(num_layers // 2)
        ])
        self.rope = RotaryEmbedding3D(head_dim=head_dim, base=rope_base,
                                      rope_dims=rope_dims,
                                      register_tokens=register_tokens)
        self.final_norm = RMSNorm(dim)

    def forward(self, leaf_token: torch.Tensor,
                leaf_centroid_norm: torch.Tensor,
                rope_scale_per_axis: torch.Tensor,
                key_idx: torch.Tensor,
                attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        """
        leaf_token: (B, L, dim) bf16 — output of encoder cross-attention
        leaf_centroid_norm: (B, L, 3) fp32 — for RoPE
        rope_scale_per_axis: (B, 3) fp32
        key_idx: (B, L, n_keys) int32 — BigBird key positions per query
        attn_bias: (B, L, n_keys) float — -inf for invalid key positions

        Returns vit_features: (B, L, dim) bf16
        """
        B, L, dim = leaf_token.shape
        R = self.register_tokens
        reg = self.register[None].expand(B, R, dim).to(leaf_token.dtype)
        x = torch.cat([leaf_token, reg], dim=1)                            # (B, L+R, dim)
        cos, sin = self.rope(leaf_centroid_norm, rope_scale_per_axis)
        cos = cos.to(leaf_token.dtype)
        sin = sin.to(leaf_token.dtype)

        flex_mask = None
        if _FLEX_AVAILABLE:
            try:
                from .bigbird import build_flex_block_mask
                H = self.blocks[0].attn.num_heads
                flex_mask = build_flex_block_mask(key_idx, B, H, L + R)
            except Exception:
                pass

        skips: list[torch.Tensor] = []
        half = self.num_layers // 2
        for i in range(half):
            x = self.blocks[i](x, key_idx, cos, sin,
                               flex_mask, attn_bias)
            skips.append(x)
        for j in range(half):
            i = half + j
            x = self.blocks[i](x, key_idx, cos, sin,
                               flex_mask, attn_bias)
            mirror = skips[half - 1 - j]
            x = self.skip_proj[j](torch.cat([x, mirror], dim=-1))
        x = self.final_norm(x)
        return x[:, :L].contiguous()
