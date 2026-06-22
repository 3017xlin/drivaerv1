"""GPU IDW (Inverse Distance Weighting) via torch.topk over neighbor candidates."""
from __future__ import annotations

import torch


def gpu_idw(query_pos_norm: torch.Tensor,
            leaf_centroid_norm: torch.Tensor,
            leaf_neighbor_idx: torch.Tensor,
            leaf_assignment: torch.Tensor,
            idw_k: int = 8
            ) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU IDW=k via torch.topk over each query's 1+2 order neighbors.

    Supports both unbatched and batched inputs:
      unbatched: query_pos_norm (N_q, 3), leaf_assignment (N_q,)
      batched:   query_pos_norm (B, N_q, 3), leaf_assignment (B, N_q)

    Returns (idw_indices, idw_weights) with matching leading dims.
    """
    if query_pos_norm.ndim == 3:
        B = query_pos_norm.shape[0]
        results = []
        for b in range(B):
            idx_b, w_b = _gpu_idw_single(
                query_pos_norm[b], leaf_centroid_norm[b],
                leaf_neighbor_idx[b], leaf_assignment[b], idw_k)
            results.append((idx_b, w_b))
        idw_idx = torch.stack([r[0] for r in results], dim=0)
        idw_w = torch.stack([r[1] for r in results], dim=0)
        return idw_idx, idw_w
    return _gpu_idw_single(query_pos_norm, leaf_centroid_norm,
                           leaf_neighbor_idx, leaf_assignment, idw_k)


def _gpu_idw_single(query_pos_norm: torch.Tensor,
                    leaf_centroid_norm: torch.Tensor,
                    leaf_neighbor_idx: torch.Tensor,
                    leaf_assignment: torch.Tensor,
                    idw_k: int = 8
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-sample GPU IDW."""
    cands = leaf_neighbor_idx[leaf_assignment.long()]
    valid = cands != -1
    safe = cands.clamp(min=0).long()
    cand_c = leaf_centroid_norm[safe]
    diff = query_pos_norm[:, None, :] - cand_c
    d = diff.pow(2).sum(-1).sqrt()
    d = torch.where(valid, d, torch.full_like(d, float('inf')))
    top_d, top = torch.topk(d, k=idw_k, dim=1, largest=False)
    idw_idx = torch.gather(cands, 1, top).to(torch.int32)
    w = 1.0 / (top_d + 1e-8)
    w = w / w.sum(dim=1, keepdim=True)
    return idw_idx, w.to(torch.float32)
