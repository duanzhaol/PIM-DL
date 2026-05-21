#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path

REPO_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_PACKAGE_ROOT))

import torch

from examples.evaluate_single_layer_lut_mse import collect_layer_cache, save_layer_cache


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Collect and cache one Linear layer's activations and weights.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tokenized_dataset_path", type=str, required=True)
    parser.add_argument("--module_name", type=str, default="model.layers.18.mlp.up_proj")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--dataset_seed", type=int, default=42)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--calib_tokens", type=int, default=100000)
    parser.add_argument("--eval_tokens", type=int, default=10000)
    parser.add_argument("--torch_dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if os.path.exists(args.output_path) and not args.overwrite:
        print(f"Skipping existing layer cache: {args.output_path}")
        return

    payload = collect_layer_cache(args)
    save_layer_cache(args.output_path, payload)
    size_gib = os.path.getsize(args.output_path) / (1024**3)
    print(
        f"Saved layer cache to {args.output_path} "
        f"({size_gib:.2f} GiB, calib_tokens={payload['calib_tokens']}, eval_tokens={payload['eval_tokens']})",
        flush=True,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
