#!/usr/bin/env python
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

REPO_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_PACKAGE_ROOT))

import numpy as np
import torch
import torch.nn as nn
from datasets import DatasetDict
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator

from examples.benchmark_lut_cpu_kernel import compute_lut_storage_ratio, compute_online_read_ratio
from examples.run_luterize_causal_lm_no_trainer import (
    TokenizedCausalLMDataset,
    load_tokenized_dataset,
    resolve_torch_dtype,
    sample_dataset,
)
from LUTNeuro.residual_compensation import (
    input_residual_compensation_correction,
    residual_compensation_channels,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate one Linear layer's LUT approximation MSE for one K/V point.")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--tokenized_dataset_path", type=str, default=None)
    parser.add_argument("--layer_cache_path", type=str, default=None)
    parser.add_argument("--module_name", type=str, default="model.layers.18.mlp.up_proj")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--dataset_seed", type=int, default=42)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--calib_tokens", type=int, default=2048)
    parser.add_argument("--eval_tokens", type=int, default=512)
    parser.add_argument("--vec_len", type=int, required=True)
    parser.add_argument("--ncentroid", type=int, required=True)
    parser.add_argument("--kmeans_backend", choices=["sklearn", "faiss-gpu", "torch-gpu"], default="sklearn")
    parser.add_argument("--kmeans_iter", type=int, default=20)
    parser.add_argument("--kmeans_batch_size", type=int, default=0)
    parser.add_argument("--kmeans_codebook_block_size", type=int, default=4)
    parser.add_argument("--torch_dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--eval_chunk_tokens", type=int, default=64)
    parser.add_argument("--eval_device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--codebook_block_size", type=int, default=16)
    parser.add_argument("--residual_compensation_ratio", type=float, default=0.0)
    parser.add_argument("--residual_compensation_metric", choices=["abs", "weighted"], default="abs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


REQUIRED_LAYER_CACHE_KEYS = {
    "module_name",
    "calib_activations",
    "eval_activations",
    "weight",
    "bias",
    "in_features",
    "out_features",
}


def save_layer_cache(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)


def load_layer_cache(path: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    missing = REQUIRED_LAYER_CACHE_KEYS.difference(payload)
    if missing:
        raise ValueError(f"layer cache {path!r} is missing required keys: {sorted(missing)}")
    return payload


def validate_args(args) -> None:
    if args.layer_cache_path is None:
        if args.model_name_or_path is None or args.tokenized_dataset_path is None:
            raise ValueError("--model_name_or_path and --tokenized_dataset_path are required without --layer_cache_path")


def resolve_module(model: nn.Module, module_name: str) -> nn.Module:
    modules = dict(model.named_modules())
    if module_name not in modules:
        raise ValueError(f"module {module_name!r} not found")
    module = modules[module_name]
    if not isinstance(module, nn.Linear):
        raise ValueError(f"module {module_name!r} is {type(module).__name__}, expected nn.Linear")
    return module


def build_tokenized_dataloader(args):
    loaded_dataset = load_tokenized_dataset(args.tokenized_dataset_path)
    source_dataset = loaded_dataset["train"] if isinstance(loaded_dataset, DatasetDict) else loaded_dataset
    sampled_dataset = sample_dataset(source_dataset, args.max_samples, args.dataset_seed)
    lm_dataset = TokenizedCausalLMDataset(sampled_dataset, args.max_seq_length)
    return DataLoader(
        lm_dataset,
        shuffle=False,
        collate_fn=default_data_collator,
        batch_size=args.per_device_batch_size,
    )


def collect_module_activations(model, module, dataloader, device, calib_tokens, eval_tokens):
    storage = {"calib": [], "eval": []}
    counts = {"calib": 0, "eval": 0}

    def hook(_module, inputs, _output):
        flat = inputs[0].detach().float().cpu().reshape(-1, inputs[0].shape[-1])
        offset = 0
        if counts["calib"] < calib_tokens:
            take = min(calib_tokens - counts["calib"], flat.shape[0])
            storage["calib"].append(flat[:take])
            counts["calib"] += take
            offset += take
        if counts["eval"] < eval_tokens and offset < flat.shape[0]:
            take = min(eval_tokens - counts["eval"], flat.shape[0] - offset)
            storage["eval"].append(flat[offset : offset + take])
            counts["eval"] += take

    handle = module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for batch in dataloader:
                if counts["calib"] >= calib_tokens and counts["eval"] >= eval_tokens:
                    break
                batch = {key: value.to(device) for key, value in batch.items()}
                model(**batch)
    finally:
        handle.remove()

    if counts["calib"] < calib_tokens or counts["eval"] < eval_tokens:
        raise ValueError(
            f"collected {counts['calib']} calib and {counts['eval']} eval tokens, "
            f"requested {calib_tokens} and {eval_tokens}"
        )
    return torch.cat(storage["calib"], dim=0), torch.cat(storage["eval"], dim=0)


def fit_centroids_torch_gpu(subvectors, ncentroid, kmeans_iter, codebook_block_size, seed):
    if not torch.cuda.is_available():
        raise RuntimeError("kmeans_backend=torch-gpu requested, but torch.cuda.is_available() is false")

    ncodebooks, tokens, vec_len = subvectors.shape
    centroids = torch.empty(ncodebooks, ncentroid, vec_len, dtype=torch.float32)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    device = torch.device("cuda")
    codebook_block_size = max(1, codebook_block_size)

    for codebook_start in range(0, ncodebooks, codebook_block_size):
        codebook_end = min(codebook_start + codebook_block_size, ncodebooks)
        x_block = subvectors[codebook_start:codebook_end].to(device)
        block_size = codebook_end - codebook_start
        init_indices = torch.stack(
            [torch.randperm(tokens, generator=generator)[:ncentroid] for _ in range(block_size)]
        ).to(device)
        centroids_block = x_block[
            torch.arange(block_size, device=device).unsqueeze(1),
            init_indices,
        ].contiguous()

        for _ in range(kmeans_iter):
            centroid_norm = centroids_block.square().sum(dim=-1).unsqueeze(1)
            scores = torch.bmm(x_block, centroids_block.transpose(1, 2))
            scores.mul_(-2.0).add_(centroid_norm)
            indices = scores.argmin(dim=-1)
            del scores

            next_centroids = torch.empty_like(centroids_block)
            for block_idx in range(block_size):
                sums = torch.zeros(ncentroid, vec_len, dtype=torch.float32, device=device)
                sums.index_add_(0, indices[block_idx], x_block[block_idx])
                counts = torch.bincount(indices[block_idx], minlength=ncentroid)
                nonempty = counts > 0
                updated = centroids_block[block_idx].clone()
                updated[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
                next_centroids[block_idx] = updated
            centroids_block = next_centroids

        centroids[codebook_start:codebook_end] = centroids_block.cpu()
        del x_block, centroids_block
        torch.cuda.empty_cache()

    return centroids


def fit_centroids_from_activations(
    activations,
    vec_len,
    ncentroid,
    kmeans_backend,
    kmeans_iter,
    kmeans_batch_size,
    kmeans_codebook_block_size,
    seed,
):
    tokens, in_features = activations.shape
    if in_features % vec_len != 0:
        raise ValueError(f"in_features={in_features} must be divisible by vec_len={vec_len}")
    if tokens < ncentroid:
        raise ValueError(f"calib_tokens={tokens} must be >= ncentroid={ncentroid}")

    ncodebooks = in_features // vec_len
    subvectors = activations.reshape(tokens, ncodebooks, vec_len).permute(1, 0, 2).contiguous()
    centroids = torch.empty(ncodebooks, ncentroid, vec_len, dtype=torch.float32)
    batch_size = kmeans_batch_size if kmeans_batch_size > 0 else max(ncentroid * 4, 1024)

    if kmeans_backend == "faiss-gpu":
        import faiss

        if faiss.get_num_gpus() <= 0:
            raise RuntimeError("kmeans_backend=faiss-gpu requested, but FAISS reports no available GPU")
        for codebook_idx in range(ncodebooks):
            kmeans = faiss.Kmeans(
                vec_len,
                ncentroid,
                niter=kmeans_iter,
                nredo=1,
                verbose=False,
                gpu=True,
                seed=seed,
                min_points_per_centroid=1,
            )
            kmeans.train(subvectors[codebook_idx].numpy())
            centroids[codebook_idx] = torch.from_numpy(kmeans.centroids)
        return centroids

    if kmeans_backend == "torch-gpu":
        return fit_centroids_torch_gpu(
            subvectors,
            ncentroid,
            kmeans_iter,
            kmeans_codebook_block_size,
            seed,
        )

    for codebook_idx in range(ncodebooks):
        kmeans = MiniBatchKMeans(
            n_clusters=ncentroid,
            batch_size=batch_size,
            max_iter=kmeans_iter,
            n_init=1,
            random_state=seed,
        )
        kmeans.fit(subvectors[codebook_idx].numpy())
        centroids[codebook_idx] = torch.from_numpy(kmeans.cluster_centers_)
    return centroids


def lut_approximate_linear_output(
    activations,
    weight,
    centroids,
    bias=None,
    eval_chunk_tokens=64,
    codebook_block_size=16,
    residual_compensation_ratio=0.0,
    residual_compensation_metric="abs",
):
    tokens, in_features = activations.shape
    ncodebooks, ncentroid, vec_len = centroids.shape
    if in_features != ncodebooks * vec_len:
        raise ValueError("activation feature dimension must match centroids")
    if weight.shape[0] != in_features:
        raise ValueError("weight input dimension must match activations")

    out_features = weight.shape[1]
    weight_blocks = weight.reshape(ncodebooks, vec_len, out_features)
    residual_k = residual_compensation_channels(in_features, residual_compensation_ratio)
    output = torch.zeros(tokens, out_features, dtype=torch.float32, device=activations.device)
    quant_activations = torch.zeros_like(activations) if residual_k > 0 else None

    for codebook_start in range(0, ncodebooks, codebook_block_size):
        codebook_end = min(codebook_start + codebook_block_size, ncodebooks)
        centroid_block = centroids[codebook_start:codebook_end]
        lut_block = torch.bmm(centroid_block, weight_blocks[codebook_start:codebook_end])
        feature_start = codebook_start * vec_len
        feature_end = codebook_end * vec_len

        for token_start in range(0, tokens, eval_chunk_tokens):
            token_end = min(token_start + eval_chunk_tokens, tokens)
            x_block = activations[token_start:token_end, feature_start:feature_end].reshape(
                token_end - token_start,
                codebook_end - codebook_start,
                vec_len,
            ).permute(1, 0, 2)
            x_norm = x_block.square().sum(dim=-1, keepdim=True)
            centroid_norm = centroid_block.square().sum(dim=-1).unsqueeze(1)
            dot = torch.bmm(x_block, centroid_block.transpose(1, 2))
            indices = (x_norm - 2.0 * dot + centroid_norm).argmin(dim=-1)
            selected = torch.gather(
                lut_block,
                1,
                indices.unsqueeze(-1).expand(-1, -1, out_features),
            )
            output[token_start:token_end] += selected.sum(dim=0)
            if quant_activations is not None:
                selected_centroids = torch.gather(
                    centroid_block,
                    1,
                    indices.unsqueeze(-1).expand(-1, -1, vec_len),
                )
                quant_activations[token_start:token_end, feature_start:feature_end] = selected_centroids.permute(
                    1, 0, 2
                ).reshape(
                    token_end - token_start,
                    feature_end - feature_start,
                )

    if quant_activations is not None:
        for token_start in range(0, tokens, eval_chunk_tokens):
            token_end = min(token_start + eval_chunk_tokens, tokens)
            output[token_start:token_end] += input_residual_compensation_correction(
                activations[token_start:token_end] - quant_activations[token_start:token_end],
                weight,
                residual_compensation_ratio,
                residual_compensation_metric,
            )

    if bias is not None:
        output += bias
    return output


def compute_error_metrics(approx, dense):
    diff = approx - dense
    mse = diff.square().mean().item()
    dense_power = dense.square().mean().item()
    return {
        "mse": mse,
        "relative_mse": mse / dense_power if dense_power > 0 else float("inf"),
        "rmse": float(mse**0.5),
        "dense_rms": float(dense_power**0.5),
        "cosine_similarity": torch.nn.functional.cosine_similarity(
            approx.reshape(1, -1),
            dense.reshape(1, -1),
        ).item(),
        "max_abs_error": diff.abs().max().item(),
    }


def dense_linear_output(activations, weight, bias=None):
    output = activations.matmul(weight)
    if bias is not None:
        output += bias
    return output


def write_json(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def collect_layer_cache(args) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_torch_dtype(args.torch_dtype),
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.to(device)
    model.eval()
    module = resolve_module(model, args.module_name)
    dataloader = build_tokenized_dataloader(args)

    print(
        f"Collecting activations for {args.module_name}: "
        f"calib_tokens={args.calib_tokens}, eval_tokens={args.eval_tokens}",
        flush=True,
    )
    calib_activations, eval_activations = collect_module_activations(
        model,
        module,
        dataloader,
        device,
        args.calib_tokens,
        args.eval_tokens,
    )

    payload = {
        "module_name": args.module_name,
        "calib_activations": calib_activations,
        "eval_activations": eval_activations,
        "weight": module.weight.detach().float().cpu().t().contiguous(),
        "bias": module.bias.detach().float().cpu() if module.bias is not None else None,
        "in_features": int(module.in_features),
        "out_features": int(module.out_features),
        "calib_tokens": int(calib_activations.shape[0]),
        "eval_tokens": int(eval_activations.shape[0]),
        "max_seq_length": args.max_seq_length,
        "dataset_seed": args.dataset_seed,
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    if os.path.exists(args.output_json) and not args.overwrite:
        print(f"Skipping existing result: {args.output_json}")
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    start_time = time.perf_counter()
    if args.layer_cache_path is not None:
        print(f"Loading layer cache from {args.layer_cache_path}", flush=True)
        cache = load_layer_cache(args.layer_cache_path)
    else:
        cache = collect_layer_cache(args)

    payload = evaluate_layer_cache(cache, args, start_time)
    write_json(args.output_json, payload)
    print(json.dumps(payload, indent=2), flush=True)


def evaluate_layer_cache(cache: dict, args, start_time: float) -> dict:
    calib_activations = cache["calib_activations"].float()
    eval_activations = cache["eval_activations"].float()
    weight = cache["weight"].float()
    bias = cache["bias"].float() if cache["bias"] is not None else None
    eval_device = args.eval_device
    if eval_device == "auto":
        eval_device = "cuda" if torch.cuda.is_available() else "cpu"
    if eval_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("eval_device=cuda requested, but torch.cuda.is_available() is false")

    print(f"Fitting KMeans: K={args.ncentroid}, V={args.vec_len}", flush=True)
    kmeans_start = time.perf_counter()
    centroids = fit_centroids_from_activations(
        calib_activations,
        args.vec_len,
        args.ncentroid,
        args.kmeans_backend,
        args.kmeans_iter,
        args.kmeans_batch_size,
        args.kmeans_codebook_block_size,
        args.seed,
    )
    kmeans_seconds = time.perf_counter() - kmeans_start

    print("Computing dense and LUT outputs", flush=True)
    if eval_device == "cuda":
        eval_activations = eval_activations.cuda()
        weight = weight.cuda()
        centroids = centroids.cuda()
        if bias is not None:
            bias = bias.cuda()
    dense = dense_linear_output(eval_activations, weight, bias)
    approx = lut_approximate_linear_output(
        eval_activations,
        weight,
        centroids,
        bias=bias,
        eval_chunk_tokens=args.eval_chunk_tokens,
        codebook_block_size=args.codebook_block_size,
        residual_compensation_ratio=args.residual_compensation_ratio,
        residual_compensation_metric=args.residual_compensation_metric,
    )
    metrics = compute_error_metrics(approx, dense)

    return {
        "module_name": cache["module_name"],
        "in_features": int(weight.shape[0]),
        "out_features": int(weight.shape[1]),
        "ncentroid": args.ncentroid,
        "vec_len": args.vec_len,
        "ncodebooks": int(weight.shape[0] // args.vec_len),
        "calib_tokens": int(calib_activations.shape[0]),
        "eval_tokens": int(eval_activations.shape[0]),
        "eval_device": eval_device,
        "kmeans_backend": args.kmeans_backend,
        "kmeans_iter": args.kmeans_iter,
        "kmeans_codebook_block_size": args.kmeans_codebook_block_size,
        "lut_storage_ratio": compute_lut_storage_ratio(args.ncentroid, args.vec_len),
        "online_read_ratio": compute_online_read_ratio(weight.shape[0], weight.shape[1], args.ncentroid, args.vec_len),
        "residual_compensation_ratio": args.residual_compensation_ratio,
        "residual_compensation_metric": args.residual_compensation_metric,
        "residual_compensation_channels": residual_compensation_channels(
            weight.shape[0],
            args.residual_compensation_ratio,
        ),
        "kmeans_seconds": kmeans_seconds,
        "total_seconds": time.perf_counter() - start_time,
        **metrics,
    }


if __name__ == "__main__":
    main(sys.argv[1:])
