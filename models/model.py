"""Top-level DrivAer3DModel: Encoder → ViT → Decoder (v4 §7)."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import EncoderCrossAttention, build_leaf_aggregate
from .idw import gpu_idw
from .vit import ViT


class DrivAer3DModel(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        m = cfg['model']
        reg = cfg['regularization']
        self.cfg = cfg
        self.encoder = EncoderCrossAttention(
            q_in_dim=31, kv_in_dim=10,
            dim=int(m['latent_dim']),
            num_heads=int(m['num_heads']),
            head_dim=int(m['head_dim']),
        )
        self.vit = ViT(
            dim=int(m['latent_dim']),
            num_layers=int(m['num_layers']),
            num_heads=int(m['num_heads']),
            head_dim=int(m['head_dim']),
            ffn_hidden=int(m['ffn_hidden']),
            register_tokens=int(m['register_tokens']),
            ffn_dropout=float(reg['ffn_dropout']),
            attn_dropout=float(reg['attn_dropout']),
            rope_base=float(m['rope_base']),
            rope_dims=tuple(m['rope_dims']),
        )
        self.decoder = Decoder(
            dim=int(m['latent_dim']),
            idw_k=int(m['decoder_idw_k']),
            fourier_freqs=int(m['decoder_fourier_freqs']),
            pos_hidden=int(m['decoder_pos_hidden']),
            pos_out=int(m['decoder_pos_out']),
            vol_out=8, surf_out=4,
            dropout=float(reg['decoder_dropout']),
        )
        sampling = cfg.get('sampling', {})
        n_query = int(sampling.get('N_query', 500_000))
        self.n_query_vol = int(n_query * (
            1.0 - float(sampling.get('surface_query_ratio', 0.2))))
        self._idw_k = int(m['decoder_idw_k'])

    # ------------------------------------------------------------------
    # Forward (training / curve)
    # ------------------------------------------------------------------

    def forward(self, batch: dict[str, torch.Tensor]
                ) -> tuple[torch.Tensor, torch.Tensor]:
        if 'idw_indices' not in batch:
            idw_idx, idw_w = gpu_idw(
                batch['query_pos_norm'],
                batch['leaf_centroid_norm'],
                batch['leaf_neighbor_idx'],
                batch['query_leaf_id'],
                idw_k=self._idw_k)
            batch['idw_indices'] = idw_idx
            batch['idw_weights'] = idw_w

        leaf_aggr = build_leaf_aggregate(batch)
        leaf_token = self.encoder(leaf_aggr, batch['transient1'])
        vit_feat = self.vit(leaf_token,
                            batch['leaf_centroid_norm'],
                            batch['bigbird_key_idx'],
                            attn_bias=batch.get('bigbird_attn_bias'))
        enc_feat = leaf_token
        pred_vol, pred_surf = self.decoder(
            enc_feat, vit_feat,
            batch['query_pos_norm'], batch['query_sdf'],
            batch['query_sdf_grad'],
            batch['idw_indices'], batch['idw_weights'],
            n_query_vol=self.n_query_vol,
        )
        return pred_vol, pred_surf

    # ------------------------------------------------------------------
    # Test inference helpers (encoder run once; decoder chunked)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, batch: dict[str, torch.Tensor]
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run encoder + ViT once; return (enc_feat, vit_feat) (B, L, D)."""
        leaf_aggr = build_leaf_aggregate(batch)
        leaf_token = self.encoder(leaf_aggr, batch['transient1'])
        vit_feat = self.vit(leaf_token,
                            batch['leaf_centroid_norm'],
                            batch['bigbird_key_idx'],
                            attn_bias=batch.get('bigbird_attn_bias'))
        return leaf_token, vit_feat

    @torch.no_grad()
    def decode_chunk(self, enc_feat: torch.Tensor, vit_feat: torch.Tensor,
                     query_pos_norm: torch.Tensor, query_sdf: torch.Tensor,
                     query_sdf_grad: torch.Tensor,
                     idw_indices: torch.Tensor, idw_weights: torch.Tensor,
                     n_query_vol: int
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decoder(enc_feat, vit_feat, query_pos_norm, query_sdf,
                            query_sdf_grad, idw_indices, idw_weights,
                            n_query_vol)
