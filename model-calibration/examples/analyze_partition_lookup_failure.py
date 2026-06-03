#!/usr/bin/env python
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

REPO_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_PACKAGE_ROOT))

import numpy as np
import torch

from examples.evaluate_single_layer_lut_mse import load_layer_cache


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Measure cross-tile activation coupling and outlier-driven lookup error "
            "from a cached Linear layer."
        )
    )
    parser.add_argument("--layer_cache_path", type=str, required=True)
    parser.add_argument("--centroid_path", type=str, required=True)
    parser.add_argument("--module_name", type=str, default=None)
    parser.add_argument("--vec_len", type=int, required=True)
    parser.add_argument("--ncentroid", type=int, required=True)
    parser.add_argument("--tile_sizes", type=parse_int_list, default=parse_int_list("2,4,8,16,32,64,128"))
    parser.add_argument("--corr_split", choices=["calib", "eval"], default="calib")
    parser.add_argument("--corr_max_tokens", type=int, default=100000)
    parser.add_argument("--corr_slice_start", type=int, default=0)
    parser.add_argument("--corr_slice_width", type=int, default=256)
    parser.add_argument("--error_split", choices=["calib", "eval"], default="eval")
    parser.add_argument("--error_max_tokens", type=int, default=10000)
    parser.add_argument("--error_chunk_tokens", type=int, default=512)
    parser.add_argument("--outlier_quantile", type=float, default=0.99)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested, but torch.cuda.is_available() is false")
    return torch.device(name)


def load_centroids(path: str, module_name: str, vec_len: int, ncentroid: int) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    key = f"{module_name}.centroids.weight"
    if key not in payload:
        available = ", ".join(sorted(payload.keys())[:5])
        raise KeyError(f"centroid key {key!r} not found in {path!r}; first keys: {available}")
    centroids = payload[key].float()
    if centroids.ndim == 2:
        ncodebooks, flat = centroids.shape
        expected_flat = ncentroid * vec_len
        if flat != expected_flat:
            raise ValueError(f"centroid tensor has flat dim {flat}, expected {expected_flat}")
        centroids = centroids.reshape(ncodebooks, ncentroid, vec_len).contiguous()
    elif centroids.ndim == 3:
        if centroids.shape[1:] != (ncentroid, vec_len):
            raise ValueError(f"centroid tensor shape {tuple(centroids.shape)} does not match K/V")
    else:
        raise ValueError(f"unsupported centroid tensor shape: {tuple(centroids.shape)}")
    return centroids


def finite_mean(values: torch.Tensor) -> float:
    values = values[torch.isfinite(values)]
    return float(values.mean().item()) if values.numel() else float("nan")


def compute_abs_correlation_slice(
    activations: torch.Tensor,
    slice_start: int,
    slice_width: int,
    max_tokens: int,
) -> torch.Tensor:
    if slice_start < 0 or slice_width <= 1:
        raise ValueError("correlation slice requires slice_start >= 0 and slice_width > 1")
    feature_end = slice_start + slice_width
    if feature_end > activations.shape[1]:
        raise ValueError(f"correlation slice [{slice_start}, {feature_end}) exceeds feature dim {activations.shape[1]}")
    x = activations[:max_tokens, slice_start:feature_end].float()
    x = x - x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, unbiased=True, keepdim=True).clamp_min(1e-12)
    z = x / std
    corr = z.t().matmul(z) / max(1, z.shape[0] - 1)
    return corr.abs().clamp_(0.0, 1.0)


def summarize_tile_correlation(abs_corr: torch.Tensor, tile_sizes: list[int]) -> list[dict]:
    width = abs_corr.shape[0]
    upper = torch.triu(torch.ones(width, width, dtype=torch.bool), diagonal=1)
    channel_idx = torch.arange(width)
    rows = []
    for tile_size in tile_sizes:
        if tile_size <= 1 or width % tile_size != 0:
            continue
        tile_id = channel_idx // tile_size
        same_tile = tile_id[:, None] == tile_id[None, :]
        adjacent_tile = (tile_id[:, None] - tile_id[None, :]).abs() == 1
        cross_tile = ~same_tile
        within_values = abs_corr[upper & same_tile]
        adjacent_values = abs_corr[upper & adjacent_tile]
        cross_values = abs_corr[upper & cross_tile]
        within_mean = finite_mean(within_values)
        adjacent_cross_mean = finite_mean(adjacent_values)
        all_cross_mean = finite_mean(cross_values)
        rows.append(
            {
                "tile_size": tile_size,
                "within_pair_count": int(within_values.numel()),
                "adjacent_cross_pair_count": int(adjacent_values.numel()),
                "all_cross_pair_count": int(cross_values.numel()),
                "within_abs_corr_mean": within_mean,
                "adjacent_cross_abs_corr_mean": adjacent_cross_mean,
                "all_cross_abs_corr_mean": all_cross_mean,
                "adjacent_cross_over_within": adjacent_cross_mean / within_mean if within_mean > 0 else float("nan"),
                "all_cross_over_within": all_cross_mean / within_mean if within_mean > 0 else float("nan"),
            }
        )
    return rows


def quantization_residual_energy(
    activations: torch.Tensor,
    centroids: torch.Tensor,
    max_tokens: int,
    chunk_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    x_cpu = activations[:max_tokens].float().contiguous()
    tokens, in_features = x_cpu.shape
    ncodebooks, ncentroid, vec_len = centroids.shape
    if in_features != ncodebooks * vec_len:
        raise ValueError(f"activation dim {in_features} does not match centroids {ncodebooks}*{vec_len}")

    centroids_dev = centroids.to(device)
    centroid_norm = centroids_dev.square().sum(dim=-1).unsqueeze(0)
    residual_sum = torch.zeros(in_features, dtype=torch.float64)
    residual_max = torch.zeros(in_features, dtype=torch.float32)

    for start in range(0, tokens, chunk_tokens):
        end = min(start + chunk_tokens, tokens)
        x = x_cpu[start:end].to(device)
        x_tiles = x.reshape(end - start, ncodebooks, vec_len)
        x_norm = x_tiles.square().sum(dim=-1, keepdim=True)
        dot = torch.einsum("bnv,nkv->bnk", x_tiles, centroids_dev)
        indices = (x_norm - 2.0 * dot + centroid_norm).argmin(dim=-1)
        selected = torch.gather(
            centroids_dev.unsqueeze(0).expand(end - start, -1, -1, -1),
            2,
            indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, vec_len),
        ).squeeze(2)
        residual = (x_tiles - selected).reshape(end - start, in_features).float().cpu()
        residual_square = residual.square()
        residual_sum += residual_square.sum(dim=0, dtype=torch.float64)
        residual_max = torch.maximum(residual_max, residual.abs().max(dim=0).values)

    mean_square = (residual_sum / tokens).float()
    return mean_square, residual_max


def channel_weight_norm_square(weight: torch.Tensor, in_features: int) -> torch.Tensor:
    weight = weight.float()
    if weight.shape[0] == in_features:
        return weight.square().sum(dim=1)
    if weight.shape[1] == in_features:
        return weight.square().sum(dim=0)
    raise ValueError(f"weight shape {tuple(weight.shape)} does not contain in_features={in_features}")


def spearman_from_ranks(x: torch.Tensor, y: torch.Tensor) -> float:
    x_order = torch.argsort(x)
    y_order = torch.argsort(y)
    x_rank = torch.empty_like(x_order, dtype=torch.float32)
    y_rank = torch.empty_like(y_order, dtype=torch.float32)
    x_rank[x_order] = torch.arange(x.numel(), dtype=torch.float32)
    y_rank[y_order] = torch.arange(y.numel(), dtype=torch.float32)
    x_rank = x_rank - x_rank.mean()
    y_rank = y_rank - y_rank.mean()
    denom = x_rank.norm() * y_rank.norm()
    return float((x_rank * y_rank).sum().div(denom).item()) if denom > 0 else float("nan")


def top_share(values: torch.Tensor, order: torch.Tensor, fraction: float) -> float:
    count = max(1, math.ceil(values.numel() * fraction))
    total = values.sum().item()
    return float(values[order[:count]].sum().item() / total) if total > 0 else float("nan")


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "summary.json")
    if os.path.exists(summary_path) and not args.overwrite:
        print(f"Skipping existing analysis: {summary_path}")
        return

    cache = load_layer_cache(args.layer_cache_path)
    module_name = args.module_name or cache["module_name"]
    if module_name != cache["module_name"]:
        raise ValueError(f"cache module is {cache['module_name']!r}, requested {module_name!r}")
    in_features = int(cache["in_features"])
    centroids = load_centroids(args.centroid_path, module_name, args.vec_len, args.ncentroid)
    device = resolve_device(args.device)

    corr_activations = cache[f"{args.corr_split}_activations"].float()
    abs_corr = compute_abs_correlation_slice(
        corr_activations,
        args.corr_slice_start,
        args.corr_slice_width,
        args.corr_max_tokens,
    )
    corr_rows = summarize_tile_correlation(abs_corr, args.tile_sizes)
    np.save(os.path.join(args.output_dir, "correlation_slice.npy"), abs_corr.numpy())
    write_csv(os.path.join(args.output_dir, "correlation_metrics.csv"), corr_rows)

    error_activations = cache[f"{args.error_split}_activations"].float()
    error_tokens = min(args.error_max_tokens, error_activations.shape[0])
    outlier_score = torch.quantile(
        error_activations[:error_tokens].abs(),
        args.outlier_quantile,
        dim=0,
    ).float()
    residual_mse, residual_max_abs = quantization_residual_energy(
        error_activations,
        centroids,
        error_tokens,
        args.error_chunk_tokens,
        device,
    )
    weight_norm2 = channel_weight_norm_square(cache["weight"], in_features)
    projected_error = residual_mse * weight_norm2
    outlier_order = torch.argsort(outlier_score, descending=True)
    error_order = torch.argsort(projected_error, descending=True)
    cumulative_error_by_outlier = torch.cumsum(projected_error[outlier_order], dim=0)
    total_projected_error = projected_error.sum().clamp_min(1e-30)
    cumulative_error_by_outlier = cumulative_error_by_outlier / total_projected_error

    channel_rows = []
    max_outlier = outlier_score.max().clamp_min(1e-30)
    max_projected_error = projected_error.max().clamp_min(1e-30)
    for rank, channel in enumerate(outlier_order.tolist(), start=1):
        channel_rows.append(
            {
                "rank_by_outlier": rank,
                "channel": channel,
                "outlier_score": float(outlier_score[channel].item()),
                "outlier_score_norm": float((outlier_score[channel] / max_outlier).item()),
                "residual_mse": float(residual_mse[channel].item()),
                "residual_max_abs": float(residual_max_abs[channel].item()),
                "weight_norm_square": float(weight_norm2[channel].item()),
                "projected_error_contribution": float(projected_error[channel].item()),
                "projected_error_norm": float((projected_error[channel] / max_projected_error).item()),
                "cumulative_projected_error_share": float(cumulative_error_by_outlier[rank - 1].item()),
            }
        )
    write_csv(os.path.join(args.output_dir, "channel_curves.csv"), channel_rows)

    summary = {
        "module_name": module_name,
        "layer_cache_path": args.layer_cache_path,
        "centroid_path": args.centroid_path,
        "in_features": in_features,
        "out_features": int(cache["out_features"]),
        "vec_len": args.vec_len,
        "ncentroid": args.ncentroid,
        "ncodebooks": int(centroids.shape[0]),
        "corr_split": args.corr_split,
        "corr_tokens": int(min(args.corr_max_tokens, corr_activations.shape[0])),
        "corr_slice_start": args.corr_slice_start,
        "corr_slice_width": args.corr_slice_width,
        "correlation_metrics": corr_rows,
        "error_split": args.error_split,
        "error_tokens": int(error_tokens),
        "outlier_quantile": args.outlier_quantile,
        "device": str(device),
        "projected_error_definition": "mean((x_j - q_j)^2) * ||W_j||_2^2, summed over output channels",
        "top1pct_channels": int(max(1, math.ceil(in_features * 0.01))),
        "top5pct_channels": int(max(1, math.ceil(in_features * 0.05))),
        "top1pct_outlier_projected_error_share": top_share(projected_error, outlier_order, 0.01),
        "top5pct_outlier_projected_error_share": top_share(projected_error, outlier_order, 0.05),
        "top1pct_error_channels_projected_error_share": top_share(projected_error, error_order, 0.01),
        "top5pct_error_channels_projected_error_share": top_share(projected_error, error_order, 0.05),
        "spearman_outlier_vs_projected_error": spearman_from_ranks(outlier_score, projected_error),
        "mean_residual_mse": float(residual_mse.mean().item()),
        "mean_projected_error": float(projected_error.mean().item()),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
