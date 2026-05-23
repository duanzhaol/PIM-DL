import math

import torch


def residual_compensation_channels(in_features: int, ratio: float) -> int:
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError("residual_compensation_ratio must be in [0, 1]")
    if ratio == 0.0:
        return 0
    return min(in_features, max(1, math.ceil(in_features * ratio)))


def input_residual_compensation_correction(
    residual: torch.Tensor,
    weight: torch.Tensor,
    ratio: float,
    metric: str = "abs",
    max_materialized_elements: int = 64_000_000,
) -> torch.Tensor:
    if residual.dim() != 2:
        raise ValueError("residual must be a 2D [tokens, in_features] tensor")
    if weight.dim() != 2:
        raise ValueError("weight must be a 2D [in_features, out_features] tensor")
    if residual.shape[1] != weight.shape[0]:
        raise ValueError("residual feature dimension must match weight input dimension")

    residual_k = residual_compensation_channels(residual.shape[1], ratio)
    if residual_k == 0:
        return residual.new_zeros(residual.shape[0], weight.shape[1])
    if residual_k == residual.shape[1]:
        return residual.matmul(weight)

    scores = residual.abs()
    if metric == "weighted":
        row_norm = weight.detach().to(scores.dtype).square().sum(dim=1).sqrt()
        scores = scores * row_norm.unsqueeze(0)
    elif metric != "abs":
        raise ValueError("residual_compensation_metric must be 'abs' or 'weighted'")

    indices = scores.topk(residual_k, dim=1).indices
    selected_residual = residual.gather(1, indices)
    selected_weight_elements = residual.shape[0] * residual_k * weight.shape[1]
    if selected_weight_elements > max_materialized_elements:
        masked_residual = residual.new_zeros(residual.shape)
        masked_residual.scatter_(1, indices, selected_residual)
        return masked_residual.matmul(weight)

    selected_weight = weight.index_select(0, indices.reshape(-1)).reshape(
        residual.shape[0],
        residual_k,
        weight.shape[1],
    )
    return torch.bmm(selected_residual.unsqueeze(1), selected_weight).squeeze(1)
