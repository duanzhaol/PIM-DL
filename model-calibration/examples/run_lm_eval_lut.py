#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

REPO_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_PACKAGE_ROOT))

import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.run_luterize_causal_lm_no_trainer import (
    apply_lut_replacement,
    load_centroids_if_requested,
    resolve_torch_dtype,
)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Run lm-evaluation-harness with a Qwen-style LUT model.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tasks", type=str, default="mmlu_pro")
    parser.add_argument("--limit", type=float, default=8)
    parser.add_argument("--num_fewshot", type=int, default=None)
    parser.add_argument("--batch_size", type=str, default="1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--bootstrap_iters", type=int, default=0)
    parser.add_argument("--log_samples", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gen_kwargs", type=str, default=None)
    parser.add_argument("--target_modules", choices=["mlp", "attention", "all"], default="mlp")
    parser.add_argument("--vec_len", type=int, default=2)
    parser.add_argument("--ncentroid", type=int, default=16)
    parser.add_argument("--nsharecodebook", type=int, default=1)
    parser.add_argument("--distance_p", type=str, default="2.0")
    parser.add_argument("--centroid_path", type=str, default=None)
    parser.add_argument("--disable_lut", action="store_true")
    parser.add_argument("--torch_dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    return parser.parse_args(input_args)


def parse_tasks(tasks):
    return [task.strip() for task in tasks.split(",") if task.strip()]


def parse_gen_kwargs(gen_kwargs):
    if gen_kwargs is None or gen_kwargs == "":
        return None
    try:
        return json.loads(gen_kwargs)
    except json.JSONDecodeError:
        return gen_kwargs


def make_json_serializable(value):
    if isinstance(value, dict):
        return {key: make_json_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_serializable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def load_lut_or_baseline_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_torch_dtype(args.torch_dtype),
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = True

    if not args.disable_lut:
        model = apply_lut_replacement(model, args)
        loaded_centroids = load_centroids_if_requested(model, args.centroid_path)
        print(f"LUT modules loaded from centroid tensors: {loaded_centroids}", flush=True)

    if args.device is not None:
        model.to(args.device)
    model.eval()
    return model, tokenizer


def run_lm_eval(args):
    model, tokenizer = load_lut_or_baseline_model(args)
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        batch_size=args.batch_size,
        device=args.device,
        dtype=resolve_torch_dtype(args.torch_dtype),
    )
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=parse_tasks(args.tasks),
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
        bootstrap_iters=args.bootstrap_iters,
        log_samples=args.log_samples,
        gen_kwargs=parse_gen_kwargs(args.gen_kwargs),
    )

    if args.output_path is not None:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(make_json_serializable(results), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(json.dumps(make_json_serializable(results.get("results", results)), indent=2, ensure_ascii=False), flush=True)
    return results


def main():
    args = parse_args()
    run_lm_eval(args)


if __name__ == "__main__":
    main()
