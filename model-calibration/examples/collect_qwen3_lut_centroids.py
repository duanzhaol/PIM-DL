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
from datasets import load_dataset
from sklearn.cluster import MiniBatchKMeans
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator

from LUTNeuro.module_filter import should_luterize_module
from examples.run_luterize_causal_lm_no_trainer import group_texts, resolve_torch_dtype


logger = logging.getLogger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Collect Qwen-style LUT centroids from Linear input activations.")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config_name", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--target_modules", choices=["mlp", "attention", "all"], default="mlp")
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--preprocessing_num_workers", type=int, default=None)
    parser.add_argument("--nsample", type=int, default=32, help="Number of dataloader batches used for collection.")
    parser.add_argument("--max_vectors_per_module", type=int, default=65536)
    parser.add_argument("--vec_len", type=int, default=4)
    parser.add_argument("--ncentroid", type=int, default=16)
    parser.add_argument("--kmeans_iter", type=int, default=200)
    parser.add_argument("--torch_dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
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


def build_dataloader(args, tokenizer, accelerator):
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
            activation = inputs[0].detach().float().cpu()
            subvectors = activation_to_codebook_subvectors(activation, args.vec_len)
            remaining = args.max_vectors_per_module - counts[module_name]
            clipped = subvectors[:, :remaining, :]
            storage[module_name].append(clipped)
            counts[module_name] += clipped.shape[1]

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


def fit_centroids_for_module(module_name, collected, args):
    codebook_subvectors = torch.cat(collected, dim=1)
    ncodebooks = codebook_subvectors.shape[0]
    centroids = torch.empty((ncodebooks, args.ncentroid, args.vec_len), dtype=torch.float32)

    for codebook_idx in range(ncodebooks):
        samples = codebook_subvectors[codebook_idx].numpy()
        if samples.shape[0] < args.ncentroid:
            raise ValueError(
                f"{module_name} codebook {codebook_idx} has {samples.shape[0]} samples, "
                f"less than ncentroid={args.ncentroid}"
            )
        kmeans = MiniBatchKMeans(
            n_clusters=args.ncentroid,
            batch_size=max(args.ncentroid * 16, 1024),
            max_iter=args.kmeans_iter,
            n_init="auto",
            random_state=0,
        )
        kmeans.fit(samples)
        centroids[codebook_idx] = torch.from_numpy(kmeans.cluster_centers_)
    return centroids.reshape(ncodebooks, args.ncentroid * args.vec_len)


def save_centroids(tensors, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if output_path.endswith(".safetensors"):
        from safetensors.torch import save_file

        save_file(tensors, output_path)
    else:
        torch.save(tensors, output_path)


def main():
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    accelerator = Accelerator()

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

    if not accelerator.is_main_process:
        return

    centroid_tensors = {}
    for module_name, collected in storage.items():
        if not collected:
            raise ValueError(f"no activations collected for {module_name}")
        centroid_tensors[centroid_tensor_key(module_name)] = fit_centroids_for_module(
            module_name,
            collected,
            args,
        )
        logger.info("Fitted centroids for %s", module_name)

    save_centroids(centroid_tensors, args.output_path)
    logger.info("Saved %d centroid tensors to %s", len(centroid_tensors), args.output_path)


if __name__ == "__main__":
    main()
