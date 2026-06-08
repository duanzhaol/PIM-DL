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

import torch

from examples.evaluate_single_layer_lut_mse import load_layer_cache


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-channel within-tile dependency share for PIM-DL/LUT-NN "
            "partition lookup motivation figures."
        )
    )
    parser.add_argument("--layer_cache_path", type=str, required=True)
    parser.add_argument("--split", choices=["calib", "eval"], default="calib")
    parser.add_argument("--max_tokens", type=int, default=100000)
    parser.add_argument("--tile_sizes", type=parse_int_list, default=parse_int_list("4,8,16,32"))
    parser.add_argument("--random_trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--best_effort", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested, but torch.cuda.is_available() is false")
    return torch.device(name)


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.float(), q).item())


def compute_corr_squared(activations: torch.Tensor, max_tokens: int, device: torch.device) -> torch.Tensor:
    tokens = min(max_tokens, activations.shape[0])
    x = activations[:tokens].to(device=device, dtype=torch.float32)
    x = x - x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, unbiased=True, keepdim=True).clamp_min(1e-12)
    x = x / std
    corr = x.t().matmul(x) / max(1, tokens - 1)
    corr = corr.clamp_(-1.0, 1.0)
    w = corr.square_()
    w.fill_diagonal_(0.0)
    return w.cpu()


def contiguous_groups(num_channels: int, tile_size: int) -> list[torch.Tensor]:
    return [torch.arange(start, start + tile_size) for start in range(0, num_channels, tile_size)]


def share_for_groups(w: torch.Tensor, total_dependency: torch.Tensor, groups: list[torch.Tensor]) -> dict:
    internal = torch.zeros_like(total_dependency)
    tile_id = torch.empty(total_dependency.numel(), dtype=torch.long)
    for group_idx, group in enumerate(groups):
        internal[group] = w[group][:, group].sum(dim=1)
        tile_id[group] = group_idx
    share = internal / total_dependency.clamp_min(1e-30)
    return {
        "internal": internal,
        "share": share,
        "tile_id": tile_id,
    }


def summarize_shares(share: torch.Tensor) -> dict:
    return {
        "mean_share": float(share.mean().item()),
        "median_share": float(share.median().item()),
        "p10_share": percentile(share, 0.10),
        "p90_share": percentile(share, 0.90),
        "min_share": float(share.min().item()),
        "max_share": float(share.max().item()),
    }


def random_groups(num_channels: int, tile_size: int, generator: torch.Generator) -> list[torch.Tensor]:
    perm = torch.randperm(num_channels, generator=generator)
    return [perm[start : start + tile_size] for start in range(0, num_channels, tile_size)]


def greedy_best_effort_groups(w: torch.Tensor, tile_size: int) -> list[torch.Tensor]:
    num_channels = w.shape[0]
    unassigned = torch.ones(num_channels, dtype=torch.bool)
    base_score = w.sum(dim=1)
    groups = []
    for _ in range(num_channels // tile_size):
        seed_scores = base_score.clone()
        seed_scores[~unassigned] = -1.0
        seed = int(seed_scores.argmax().item())
        group = [seed]
        unassigned[seed] = False
        connection = w[seed].clone()
        for _slot in range(1, tile_size):
            candidate_scores = connection.clone()
            candidate_scores[~unassigned] = -1.0
            next_channel = int(candidate_scores.argmax().item())
            group.append(next_channel)
            unassigned[next_channel] = False
            connection += w[next_channel]
        groups.append(torch.tensor(group, dtype=torch.long))
    return groups


def channel_rows_for(tile_size: int, stats: dict, total_dependency: torch.Tensor) -> list[dict]:
    internal = stats["internal"]
    share = stats["share"]
    tile_id = stats["tile_id"]
    rows = []
    for channel in range(share.numel()):
        rows.append(
            {
                "tile_size_v": tile_size,
                "channel": channel,
                "tile_id": int(tile_id[channel].item()),
                "total_dependency_corr2": float(total_dependency[channel].item()),
                "internal_dependency_corr2": float(internal[channel].item()),
                "share": float(share[channel].item()),
                "share_percent": float(share[channel].item() * 100.0),
            }
        )
    return rows


def group_rows_for(tile_size: int, groups: list[torch.Tensor], label: str) -> list[dict]:
    return [
        {
            "tile_size_v": tile_size,
            "tiling": label,
            "tile_id": tile_idx,
            "channels": " ".join(str(int(channel)) for channel in group.tolist()),
        }
        for tile_idx, group in enumerate(groups)
    ]


def write_readme(output_dir: str, summary: dict) -> None:
    path = os.path.join(output_dir, "README.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            f"""# Within-Tile Dependency Share Data

This directory contains the regenerated Figure 4C data for the claim that
partition lookup can only exploit channel dependencies inside each independently
encoded tile.

## Source

- Layer cache: `{summary["layer_cache_path"]}`
- Module: `{summary["module_name"]}`
- Activation split: `{summary["split"]}`
- Tokens used: `{summary["tokens"]}`
- Hidden dimension D: `{summary["in_features"]}`
- Weight definition: squared Pearson correlation, `w(i,j)=corr(i,j)^2`
- Diagonal/self-correlation: excluded by setting `w(i,i)=0`
- Tiling: contiguous channel tiles matching the partition lookup baseline

The cache corresponds to Qwen3.5-4B BF16, a representative middle MLP
projection input (`model.layers.18.mlp.up_proj`). This is the same activation
surface used by the single-layer partition lookup sweep.

## Metric

For each channel `i` in tile `T(i)`:

```text
internal_i = sum_{{j in T(i), j != i}} w(i,j)
total_i    = sum_{{j != i}} w(i,j)
share_i    = internal_i / total_i
S(V)       = mean_i(share_i) * 100%
```

This denominator is local to each channel. It is not normalized by global
correlation energy across unrelated channel pairs.

## Baselines

- `measured`: the actual contiguous tiling used by partition lookup.
- `random`: `{summary["random_trials"]}` random balanced channel permutations.
- `best_effort_greedy`: a deterministic greedy balanced grouping that tries to
  maximize within-tile `corr^2` energy. This is a best-effort upper bound, not an
  exact balanced graph partition optimum.
- `ideal`: 100%, corresponding to a perfectly block-diagonal activation where
  all dependencies of each channel stay inside its tile.

## Files

- `within_tile_dependency_share.csv`: one summary row per tile size.
- `within_tile_dependency_share_by_channel.csv`: measured per-channel shares.
- `random_tiling_trials.csv`: random baseline trials.
- `best_effort_groups.csv`: greedy best-effort channel groups.
- `summary.json`: machine-readable manifest and summary.
"""
        )


def main(argv=None):
    args = parse_args(argv)
    summary_path = os.path.join(args.output_dir, "summary.json")
    if os.path.exists(summary_path) and not args.overwrite:
        print(f"Skipping existing output: {summary_path}")
        return
    os.makedirs(args.output_dir, exist_ok=True)

    cache = load_layer_cache(args.layer_cache_path)
    activations = cache[f"{args.split}_activations"].float()
    tokens = min(args.max_tokens, activations.shape[0])
    num_channels = int(cache["in_features"])
    device = resolve_device(args.device)

    print(
        f"Computing corr^2 for {tokens} tokens x {num_channels} channels on {device}",
        flush=True,
    )
    with torch.no_grad():
        w = compute_corr_squared(activations, args.max_tokens, device)
    total_dependency = w.sum(dim=1)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    summary_rows = []
    measured_channel_rows = []
    random_rows = []
    best_effort_group_rows = []

    for tile_size in args.tile_sizes:
        if tile_size <= 1:
            raise ValueError(f"tile_size must be > 1, got {tile_size}")
        if num_channels % tile_size != 0:
            raise ValueError(f"D={num_channels} must be divisible by tile_size={tile_size}")

        measured_groups = contiguous_groups(num_channels, tile_size)
        measured = share_for_groups(w, total_dependency, measured_groups)
        measured_summary = summarize_shares(measured["share"])
        measured_channel_rows.extend(channel_rows_for(tile_size, measured, total_dependency))

        random_mean_values = []
        random_median_values = []
        for trial in range(args.random_trials):
            groups = random_groups(num_channels, tile_size, generator)
            trial_stats = share_for_groups(w, total_dependency, groups)
            trial_summary = summarize_shares(trial_stats["share"])
            random_mean_values.append(trial_summary["mean_share"])
            random_median_values.append(trial_summary["median_share"])
            random_rows.append(
                {
                    "tile_size_v": tile_size,
                    "trial": trial,
                    "mean_share": trial_summary["mean_share"],
                    "mean_share_percent": trial_summary["mean_share"] * 100.0,
                    "median_share": trial_summary["median_share"],
                    "median_share_percent": trial_summary["median_share"] * 100.0,
                }
            )

        random_mean = torch.tensor(random_mean_values)
        random_median = torch.tensor(random_median_values)

        best_summary = None
        if args.best_effort:
            print(f"Computing greedy best-effort tiling for V={tile_size}", flush=True)
            best_groups = greedy_best_effort_groups(w, tile_size)
            best_stats = share_for_groups(w, total_dependency, best_groups)
            best_summary = summarize_shares(best_stats["share"])
            best_effort_group_rows.extend(group_rows_for(tile_size, best_groups, "best_effort_greedy"))

        pair_baseline = (tile_size - 1) / (num_channels - 1)
        row = {
            "tile_size_v": tile_size,
            "num_tiles": num_channels // tile_size,
            "tokens": tokens,
            "channels_d": num_channels,
            "weight_definition": "corr_squared",
            "measured_mean_share": measured_summary["mean_share"],
            "measured_mean_share_percent": measured_summary["mean_share"] * 100.0,
            "measured_median_share": measured_summary["median_share"],
            "measured_median_share_percent": measured_summary["median_share"] * 100.0,
            "measured_p10_share_percent": measured_summary["p10_share"] * 100.0,
            "measured_p90_share_percent": measured_summary["p90_share"] * 100.0,
            "random_trials": args.random_trials,
            "random_mean_share": float(random_mean.mean().item()),
            "random_mean_share_percent": float(random_mean.mean().item() * 100.0),
            "random_mean_share_std_percent": float(random_mean.std(unbiased=True).item() * 100.0)
            if args.random_trials > 1
            else 0.0,
            "random_median_share": float(random_median.mean().item()),
            "random_median_share_percent": float(random_median.mean().item() * 100.0),
            "random_median_share_std_percent": float(random_median.std(unbiased=True).item() * 100.0)
            if args.random_trials > 1
            else 0.0,
            "theoretical_pair_fraction": pair_baseline,
            "theoretical_pair_fraction_percent": pair_baseline * 100.0,
            "ideal_share_percent": 100.0,
            "best_effort_greedy_mean_share": best_summary["mean_share"] if best_summary else float("nan"),
            "best_effort_greedy_mean_share_percent": best_summary["mean_share"] * 100.0
            if best_summary
            else float("nan"),
            "best_effort_greedy_median_share": best_summary["median_share"] if best_summary else float("nan"),
            "best_effort_greedy_median_share_percent": best_summary["median_share"] * 100.0
            if best_summary
            else float("nan"),
        }
        summary_rows.append(row)
        print(
            "V={tile_size}: measured={measured:.4f}% random={random:.4f}% best={best:.4f}%".format(
                tile_size=tile_size,
                measured=row["measured_mean_share_percent"],
                random=row["random_mean_share_percent"],
                best=row["best_effort_greedy_mean_share_percent"],
            ),
            flush=True,
        )

    write_csv(os.path.join(args.output_dir, "within_tile_dependency_share.csv"), summary_rows)
    write_csv(os.path.join(args.output_dir, "within_tile_dependency_share_by_channel.csv"), measured_channel_rows)
    write_csv(os.path.join(args.output_dir, "random_tiling_trials.csv"), random_rows)
    write_csv(os.path.join(args.output_dir, "best_effort_groups.csv"), best_effort_group_rows)

    summary = {
        "layer_cache_path": args.layer_cache_path,
        "module_name": cache["module_name"],
        "split": args.split,
        "tokens": tokens,
        "in_features": num_channels,
        "out_features": int(cache["out_features"]),
        "tile_sizes": args.tile_sizes,
        "random_trials": args.random_trials,
        "seed": args.seed,
        "device": str(device),
        "metric": "mean_i sum_{j in same tile, j != i} corr(i,j)^2 / sum_{j != i} corr(i,j)^2",
        "rows": summary_rows,
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    write_readme(args.output_dir, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
