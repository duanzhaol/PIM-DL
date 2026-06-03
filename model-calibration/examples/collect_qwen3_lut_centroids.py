#!/usr/bin/env python
import argparse
import logging
import os
import sys
from pathlib import Path

REPO_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_PACKAGE_ROOT))

import torch
from accelerate import Accelerator
from datasets import DatasetDict, load_dataset
from sklearn.cluster import MiniBatchKMeans
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator

from LUTNeuro.module_filter import should_luterize_module
from examples.run_luterize_causal_lm_no_trainer import (
    TokenizedCausalLMDataset,
    group_texts,
    load_tokenized_dataset,
    resolve_torch_dtype,
    sample_dataset,
)


logger = logging.getLogger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Collect Qwen-style LUT centroids from Linear input activations.")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config_name", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--tokenized_dataset_path", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--dataset_seed", type=int, default=42)
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--target_modules", choices=["mlp", "attention", "all"], default="mlp")
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--preprocessing_num_workers", type=int, default=None)
    parser.add_argument("--nsample", type=int, default=32, help="Number of dataloader batches used for collection.")
    parser.add_argument("--max_vectors_per_module", type=int, default=65536)
    parser.add_argument("--vec_len", type=int, default=4)
    parser.add_argument("--ncentroid", type=int, default=16)
    parser.add_argument("--kmeans_backend", choices=["sklearn", "faiss-gpu", "torch-gpu"], default="sklearn")
    parser.add_argument("--kmeans_iter", type=int, default=200)
    parser.add_argument("--kmeans_batch_size", type=int, default=0)
    parser.add_argument("--kmeans_codebook_block_size", type=int, default=4)
    parser.add_argument("--kmeans_seed", type=int, default=0)
    parser.add_argument("--torch_dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument(
        "--activation_cache_path",
        type=str,
        default=None,
        help="Optional .pt cache of raw Linear input activation vectors. Existing caches are reused unless overwritten.",
    )
    parser.add_argument("--overwrite_activation_cache", action="store_true")
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args(input_args)


def centroid_tensor_key(module_name):
    return f"{module_name}.centroids.weight"


def activation_to_codebook_subvectors(activation, vec_len):
    hidden = activation.shape[-1]
    if hidden % vec_len != 0:
        raise ValueError(f"activation hidden size {hidden} must be divisible by vec_len {vec_len}")
    ncodebooks = hidden // vec_len
    flat = activation.reshape(-1, hidden)
    return flat.reshape(-1, ncodebooks, vec_len).permute(1, 0, 2).contiguous()


def activation_to_flat_vectors(activation):
    hidden = activation.shape[-1]
    return activation.reshape(-1, hidden).contiguous()


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


def build_dataloader(args, tokenizer, accelerator):
    if args.tokenized_dataset_path is not None:
        return build_tokenized_dataloader(args)

    raw_dataset = load_dataset(args.dataset_name, args.dataset_config_name, split="train")
    column_names = raw_dataset.column_names
    if args.text_column not in column_names:
        raise ValueError(f"text column {args.text_column!r} not found in dataset columns {column_names}")

    def tokenize_function(examples):
        return tokenizer(examples[args.text_column])

    with accelerator.main_process_first():
        tokenized = raw_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            num_proc=args.preprocessing_num_workers,
            desc="Tokenizing text",
        )
        lm_dataset = tokenized.map(
            lambda examples: group_texts(examples, args.max_seq_length),
            batched=True,
            num_proc=args.preprocessing_num_workers,
            desc=f"Grouping texts into {args.max_seq_length}-token chunks",
        )

    return DataLoader(
        lm_dataset,
        shuffle=False,
        collate_fn=default_data_collator,
        batch_size=args.per_device_batch_size,
    )


def register_activation_hooks(model, args, storage, counts):
    hooks = []

    def make_hook(module_name):
        def hook(_module, inputs, _output):
            if counts[module_name] >= args.max_vectors_per_module:
                return
            activation = activation_to_flat_vectors(inputs[0].detach().float().cpu())
            remaining = args.max_vectors_per_module - counts[module_name]
            clipped = activation[:remaining]
            storage[module_name].append(clipped)
            counts[module_name] += clipped.shape[0]

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and should_luterize_module(name, args.target_modules):
            storage[name] = []
            counts[name] = 0
            hooks.append(module.register_forward_hook(make_hook(name)))
            logger.info("Registered activation hook for %s", name)

    if not hooks:
        raise ValueError(f"no Linear modules matched target_modules={args.target_modules!r}")
    return hooks


def save_activation_cache(cache_path, activations, metadata):
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    torch.save(
        {
            "metadata": metadata,
            "activations": activations,
        },
        cache_path,
    )


def load_activation_cache(cache_path):
    cache = torch.load(cache_path, map_location="cpu")
    if not isinstance(cache, dict) or "activations" not in cache:
        raise ValueError(f"{cache_path} is not a valid activation cache")
    if "metadata" not in cache:
        cache["metadata"] = {}
    return cache


def collected_to_activations(collected):
    if isinstance(collected, torch.Tensor):
        return collected.float().contiguous()
    if not collected:
        raise ValueError("no activations collected")
    return torch.cat(collected, dim=0).float().contiguous()


def fit_centroids_torch_gpu(subvectors, ncentroid, kmeans_iter, codebook_block_size, seed):
    if not torch.cuda.is_available():
        raise RuntimeError("kmeans_backend=torch-gpu requested, but torch.cuda.is_available() is false")

    ncodebooks, tokens, vec_len = subvectors.shape
    centroids = torch.empty((ncodebooks, ncentroid, vec_len), dtype=torch.float32)
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
        raise ValueError(f"activation cache has {tokens} vectors, less than ncentroid={ncentroid}")

    ncodebooks = in_features // vec_len
    codebook_subvectors = activations.reshape(tokens, ncodebooks, vec_len).permute(1, 0, 2).contiguous()
    centroids = torch.empty((ncodebooks, ncentroid, vec_len), dtype=torch.float32)

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
            kmeans.train(codebook_subvectors[codebook_idx].numpy())
            centroids[codebook_idx] = torch.from_numpy(kmeans.centroids)
        return centroids

    if kmeans_backend == "torch-gpu":
        return fit_centroids_torch_gpu(
            codebook_subvectors,
            ncentroid,
            kmeans_iter,
            kmeans_codebook_block_size,
            seed,
        )

    for codebook_idx in range(ncodebooks):
        samples = codebook_subvectors[codebook_idx].numpy()
        kmeans = MiniBatchKMeans(
            n_clusters=ncentroid,
            batch_size=kmeans_batch_size if kmeans_batch_size > 0 else max(ncentroid * 16, 1024),
            max_iter=kmeans_iter,
            n_init=1,
            random_state=seed,
        )
        kmeans.fit(samples)
        centroids[codebook_idx] = torch.from_numpy(kmeans.cluster_centers_)
    return centroids


def fit_centroids_for_module(module_name, collected, args):
    activations = collected_to_activations(collected)
    centroids = fit_centroids_from_activations(
        activations,
        args.vec_len,
        args.ncentroid,
        args.kmeans_backend,
        args.kmeans_iter,
        args.kmeans_batch_size,
        args.kmeans_codebook_block_size,
        args.kmeans_seed,
    )
    ncodebooks = centroids.shape[0]
    return centroids.reshape(ncodebooks, args.ncentroid * args.vec_len)


def save_centroids(tensors, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if output_path.endswith(".safetensors"):
        from safetensors.torch import save_file

        save_file(tensors, output_path)
    else:
        torch.save(tensors, output_path)


def activation_cache_exists(args):
    return args.activation_cache_path is not None and os.path.exists(args.activation_cache_path)


def collect_activations_from_model(args, accelerator):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_torch_dtype(args.torch_dtype),
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.eval()

    storage = {}
    counts = {}
    hooks = register_activation_hooks(model, args, storage, counts)

    dataloader = build_dataloader(args, tokenizer, accelerator)
    model, dataloader = accelerator.prepare(model, dataloader)

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if step >= args.nsample:
                break
            model(**batch)
            accelerator.print(f"collected batch {step + 1}/{args.nsample}")

    for hook in hooks:
        hook.remove()

    activations = {}
    for module_name, collected in storage.items():
        if not collected:
            raise ValueError(f"no activations collected for {module_name}")
        activations[module_name] = collected_to_activations(collected)

    metadata = {
        "model_name_or_path": args.model_name_or_path,
        "tokenized_dataset_path": args.tokenized_dataset_path,
        "dataset_name": args.dataset_name,
        "dataset_config_name": args.dataset_config_name,
        "max_samples": args.max_samples,
        "dataset_seed": args.dataset_seed,
        "target_modules": args.target_modules,
        "max_seq_length": args.max_seq_length,
        "per_device_batch_size": args.per_device_batch_size,
        "nsample": args.nsample,
        "max_vectors_per_module": args.max_vectors_per_module,
        "counts": {name: int(tensor.shape[0]) for name, tensor in activations.items()},
        "hidden_sizes": {name: int(tensor.shape[1]) for name, tensor in activations.items()},
    }
    return activations, metadata


def main():
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    accelerator = Accelerator()

    if activation_cache_exists(args) and not args.overwrite_activation_cache:
        if not accelerator.is_main_process:
            return
        accelerator.print(f"Loading activation cache from {args.activation_cache_path}")
        cache = load_activation_cache(args.activation_cache_path)
        activations = cache["activations"]
    else:
        activations, metadata = collect_activations_from_model(args, accelerator)
        if not accelerator.is_main_process:
            return
        if args.activation_cache_path is not None:
            accelerator.print(f"Saving activation cache to {args.activation_cache_path}")
            save_activation_cache(args.activation_cache_path, activations, metadata)

    if not accelerator.is_main_process:
        return

    centroid_tensors = {}
    for module_name, activation in activations.items():
        centroid_tensors[centroid_tensor_key(module_name)] = fit_centroids_for_module(
            module_name,
            activation,
            args,
        )
        logger.info("Fitted centroids for %s", module_name)

    save_centroids(centroid_tensors, args.output_path)
    logger.info("Saved %d centroid tensors to %s", len(centroid_tensors), args.output_path)


if __name__ == "__main__":
    main()
