#!/usr/bin/env python
import argparse
import csv
import json
import os
import sys
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate per-K/V single-layer LUT MSE JSON files into one CSV.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    return parser.parse_args(argv)


def aggregate_results(input_dir: str, output_csv: str):
    rows = []
    for path in sorted(Path(input_dir).glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            row = json.load(handle)
        row["source_file"] = str(path)
        rows.append(row)

    rows.sort(key=lambda row: (int(row["ncentroid"]), int(row["vec_len"])))
    if not rows:
        raise ValueError(f"no JSON result files found in {input_dir}")

    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "module_name",
        "ncentroid",
        "vec_len",
        "lut_storage_ratio",
        "online_read_ratio",
        "mse",
        "relative_mse",
        "rmse",
        "dense_rms",
        "cosine_similarity",
        "max_abs_error",
        "kmeans_seconds",
        "total_seconds",
        "calib_tokens",
        "eval_tokens",
        "in_features",
        "out_features",
        "ncodebooks",
        "kmeans_iter",
        "source_file",
    ]
    fieldnames = [key for key in preferred if key in fieldnames] + [
        key for key in fieldnames if key not in preferred
    ]

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv=None):
    args = parse_args(argv)
    rows = aggregate_results(args.input_dir, args.output_csv)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main(sys.argv[1:])
