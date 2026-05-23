#!/usr/bin/env python
import argparse
import csv
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_PACKAGE_ROOT))

import torch

from LUTNeuro.residual_compensation import (
    input_residual_compensation_correction,
    residual_compensation_channels,
)


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def torch_dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def compute_lut_storage_ratio(ncentroids: int, vec_len: int) -> float:
    return ncentroids / vec_len


def compute_online_read_ratio(
    in_features: int,
    out_features: int,
    ncentroids: int,
    vec_len: int,
) -> float:
    if in_features % vec_len != 0:
        raise ValueError("in_features must be divisible by vec_len")
    dense_reads = in_features * out_features
    centroid_reads = in_features * ncentroids
    lut_reads = (in_features // vec_len) * out_features
    return (centroid_reads + lut_reads) / dense_reads


def precompute_lut(centroids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    ncodebooks, _, vec_len = centroids.shape
    in_features, out_features = weight.shape
    if in_features != ncodebooks * vec_len:
        raise ValueError("weight input dimension must match centroids")
    return torch.bmm(centroids, weight.reshape(ncodebooks, vec_len, out_features))


def lut_cpu_kernel(
    x: torch.Tensor,
    centroids: torch.Tensor,
    lut: torch.Tensor,
    weight: torch.Tensor | None = None,
    residual_compensation_ratio: float = 0.0,
    residual_compensation_metric: str = "abs",
) -> torch.Tensor:
    batch_tokens, in_features = x.shape
    ncodebooks, ncentroids, vec_len = centroids.shape
    if in_features != ncodebooks * vec_len:
        raise ValueError("x input dimension must match centroids")
    if lut.shape[:2] != (ncodebooks, ncentroids):
        raise ValueError("lut first two dimensions must match centroids")

    x_codebooks = x.reshape(batch_tokens, ncodebooks, vec_len).permute(1, 0, 2)
    x_norm = x_codebooks.square().sum(dim=-1, keepdim=True)
    centroid_norm = centroids.square().sum(dim=-1).unsqueeze(1)
    dot = torch.bmm(x_codebooks, centroids.transpose(1, 2))
    indices = (x_norm - 2.0 * dot + centroid_norm).argmin(dim=-1)

    selected = torch.gather(
        lut,
        1,
        indices.unsqueeze(-1).expand(-1, -1, lut.shape[-1]),
    )
    output = selected.sum(dim=0)
    if residual_compensation_ratio <= 0.0:
        return output
    if weight is None:
        raise ValueError("weight is required when residual_compensation_ratio > 0")

    selected_centroids = torch.gather(
        centroids,
        1,
        indices.unsqueeze(-1).expand(-1, -1, vec_len),
    )
    quant_input = selected_centroids.permute(1, 0, 2).reshape(batch_tokens, in_features)
    return output + input_residual_compensation_correction(
        x - quant_input,
        weight,
        residual_compensation_ratio,
        residual_compensation_metric,
    )


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    in_features: int
    out_features: int
    count: int


def qwen3_mlp_specs(hidden_size: int, intermediate_size: int) -> list[ModuleSpec]:
    return [
        ModuleSpec("gate_up_proj", hidden_size, intermediate_size, 2),
        ModuleSpec("down_proj", intermediate_size, hidden_size, 1),
    ]


def tensor_mib(numel: int, dtype: torch.dtype) -> float:
    return numel * torch.empty((), dtype=dtype).element_size() / (1024**2)


def median(values: Iterable[float]) -> float:
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return sorted_values[mid]
    return 0.5 * (sorted_values[mid - 1] + sorted_values[mid])


def time_ms(fn, warmup: int, repeats: int) -> tuple[float, float]:
    for _ in range(warmup):
        out = fn()
        _ = float(out.reshape(-1)[0])
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        _ = float(out.reshape(-1)[0])
        samples.append((time.perf_counter() - start) * 1000.0)
    return median(samples), min(samples)


def make_lut(shape: tuple[int, int, int], dtype: torch.dtype, table_init: str) -> torch.Tensor:
    lut = torch.empty(shape, dtype=dtype)
    if table_init == "fill":
        lut.fill_(0.01)
    elif table_init == "random":
        lut.uniform_(-0.01, 0.01)
    else:
        raise ValueError("table_init must be 'fill' or 'random'")
    return lut


def benchmark_module(
    spec: ModuleSpec,
    ncentroids: int,
    vec_len: int,
    batch_tokens: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
    table_init: str,
    max_table_gib: float,
    residual_compensation_ratio: float,
    residual_compensation_metric: str,
) -> dict:
    if spec.in_features % vec_len != 0:
        raise ValueError(f"{spec.name}: in_features is not divisible by vec_len={vec_len}")

    ncodebooks = spec.in_features // vec_len
    dense_entries = spec.in_features * spec.out_features
    lut_entries = ncodebooks * ncentroids * spec.out_features
    centroid_entries = spec.in_features * ncentroids
    lut_table_mib = tensor_mib(lut_entries, dtype)
    if lut_table_mib / 1024 > max_table_gib:
        raise ValueError(
            f"{spec.name} K={ncentroids} V={vec_len} requires {lut_table_mib / 1024:.2f} GiB LUT, "
            f"above --max_table_gib={max_table_gib}"
        )

    x = torch.randn(batch_tokens, spec.in_features, dtype=dtype)
    weight = torch.randn(spec.in_features, spec.out_features, dtype=dtype)
    centroids = torch.randn(ncodebooks, ncentroids, vec_len, dtype=dtype)
    lut = make_lut((ncodebooks, ncentroids, spec.out_features), dtype, table_init)

    dense_median_ms, dense_min_ms = time_ms(lambda: x.matmul(weight), warmup, repeats)
    lut_median_ms, lut_min_ms = time_ms(
        lambda: lut_cpu_kernel(
            x,
            centroids,
            lut,
            weight=weight,
            residual_compensation_ratio=residual_compensation_ratio,
            residual_compensation_metric=residual_compensation_metric,
        ),
        warmup,
        repeats,
    )

    del x, weight, centroids, lut
    gc.collect()

    online_read_ratio = compute_online_read_ratio(
        spec.in_features,
        spec.out_features,
        ncentroids,
        vec_len,
    )
    storage_ratio = compute_lut_storage_ratio(ncentroids, vec_len)
    residual_channels = residual_compensation_channels(spec.in_features, residual_compensation_ratio)
    compensation_compute_ratio = residual_channels / spec.in_features
    return {
        "module": spec.name,
        "module_count": spec.count,
        "in_features": spec.in_features,
        "out_features": spec.out_features,
        "batch_tokens": batch_tokens,
        "ncentroids": ncentroids,
        "vec_len": vec_len,
        "ncodebooks": ncodebooks,
        "dense_entries": dense_entries,
        "lut_entries": lut_entries,
        "centroid_entries": centroid_entries,
        "lut_storage_ratio": storage_ratio,
        "total_storage_ratio": storage_ratio + ncentroids / spec.out_features,
        "centroid_read_ratio": ncentroids / spec.out_features,
        "lookup_read_ratio": 1 / vec_len,
        "online_read_ratio": online_read_ratio,
        "residual_compensation_ratio": residual_compensation_ratio,
        "residual_compensation_metric": residual_compensation_metric,
        "residual_compensation_channels": residual_channels,
        "compensation_compute_ratio": compensation_compute_ratio,
        "compensated_online_read_ratio": online_read_ratio + compensation_compute_ratio,
        "dense_weight_mib": tensor_mib(dense_entries, dtype),
        "lut_table_mib": lut_table_mib,
        "centroid_mib": tensor_mib(centroid_entries, dtype),
        "dense_median_ms": dense_median_ms,
        "dense_min_ms": dense_min_ms,
        "lut_median_ms": lut_median_ms,
        "lut_min_ms": lut_min_ms,
        "lut_over_dense_median": lut_median_ms / dense_median_ms,
        "lut_over_dense_min": lut_min_ms / dense_min_ms,
    }


def add_weighted_average(rows: list[dict]) -> dict:
    total_weight = sum(row["module_count"] * row["dense_entries"] for row in rows)

    def weighted_mean(key: str) -> float:
        return sum(row[key] * row["module_count"] * row["dense_entries"] for row in rows) / total_weight

    first = rows[0]
    dense_median = weighted_mean("dense_median_ms")
    lut_median = weighted_mean("lut_median_ms")
    dense_min = weighted_mean("dense_min_ms")
    lut_min = weighted_mean("lut_min_ms")
    return {
        "module": "mlp_weighted_avg",
        "module_count": sum(row["module_count"] for row in rows),
        "in_features": "",
        "out_features": "",
        "batch_tokens": first["batch_tokens"],
        "ncentroids": first["ncentroids"],
        "vec_len": first["vec_len"],
        "ncodebooks": "",
        "dense_entries": sum(row["module_count"] * row["dense_entries"] for row in rows),
        "lut_entries": sum(row["module_count"] * row["lut_entries"] for row in rows),
        "centroid_entries": sum(row["module_count"] * row["centroid_entries"] for row in rows),
        "lut_storage_ratio": weighted_mean("lut_storage_ratio"),
        "total_storage_ratio": weighted_mean("total_storage_ratio"),
        "centroid_read_ratio": weighted_mean("centroid_read_ratio"),
        "lookup_read_ratio": weighted_mean("lookup_read_ratio"),
        "online_read_ratio": weighted_mean("online_read_ratio"),
        "residual_compensation_ratio": first["residual_compensation_ratio"],
        "residual_compensation_metric": first["residual_compensation_metric"],
        "residual_compensation_channels": "",
        "compensation_compute_ratio": weighted_mean("compensation_compute_ratio"),
        "compensated_online_read_ratio": weighted_mean("compensated_online_read_ratio"),
        "dense_weight_mib": sum(row["module_count"] * row["dense_weight_mib"] for row in rows),
        "lut_table_mib": sum(row["module_count"] * row["lut_table_mib"] for row in rows),
        "centroid_mib": sum(row["module_count"] * row["centroid_mib"] for row in rows),
        "dense_median_ms": dense_median,
        "dense_min_ms": dense_min,
        "lut_median_ms": lut_median,
        "lut_min_ms": lut_min,
        "lut_over_dense_median": lut_median / dense_median,
        "lut_over_dense_min": lut_min / dense_min,
    }


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark an idealized CPU LUT-NN kernel.")
    parser.add_argument("--hidden_size", type=int, default=2560)
    parser.add_argument("--intermediate_size", type=int, default=9728)
    parser.add_argument("--ncentroids", type=parse_int_list, default=parse_int_list("8,16,32,64"))
    parser.add_argument("--vec_lens", type=parse_int_list, default=parse_int_list("2,4,8,16,32"))
    parser.add_argument("--batch_tokens", type=int, default=1)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--table_init", choices=["fill", "random"], default="fill")
    parser.add_argument("--max_table_gib", type=float, default=8.0)
    parser.add_argument("--residual_compensation_ratio", type=float, default=0.0)
    parser.add_argument("--residual_compensation_metric", choices=["abs", "weighted"], default="abs")
    parser.add_argument("--output_csv", type=str, default="serialization_dir/lut_cpu_kernel_qwen3_mlp.csv")
    parser.add_argument("--output_json", type=str, default="")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    dtype = torch_dtype_from_name(args.dtype)
    specs = qwen3_mlp_specs(args.hidden_size, args.intermediate_size)
    rows = []
    total_cases = len(args.ncentroids) * len(args.vec_lens) * len(specs)
    case_idx = 0
    for ncentroids in args.ncentroids:
        for vec_len in args.vec_lens:
            combo_rows = []
            for spec in specs:
                case_idx += 1
                print(
                    f"[{case_idx}/{total_cases}] module={spec.name} K={ncentroids} V={vec_len}",
                    flush=True,
                )
                row = benchmark_module(
                    spec=spec,
                    ncentroids=ncentroids,
                    vec_len=vec_len,
                    batch_tokens=args.batch_tokens,
                    dtype=dtype,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    table_init=args.table_init,
                    max_table_gib=args.max_table_gib,
                    residual_compensation_ratio=args.residual_compensation_ratio,
                    residual_compensation_metric=args.residual_compensation_metric,
                )
                rows.append(row)
                combo_rows.append(row)
            rows.append(add_weighted_average(combo_rows))

    write_csv(args.output_csv, rows)
    if args.output_json:
        write_json(args.output_json, rows)
    print(f"Wrote {len(rows)} rows to {args.output_csv}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
