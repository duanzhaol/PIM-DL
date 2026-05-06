#!/usr/bin/env python
import argparse
import json
import logging
import math
import os

import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_scheduler,
)

from LUTNeuro.LUTLinear_t import LUTLinear_t
from LUTNeuro.LUTerize import LUTerize


logger = logging.getLogger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Calibrate a Qwen-style causal LM with LUTLinear modules.")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config_name", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--preprocessing_num_workers", type=int, default=None)
    parser.add_argument("--target_modules", choices=["mlp", "attention", "all"], default="mlp")
    parser.add_argument("--vec_len", type=int, default=4)
    parser.add_argument("--ncentroid", type=int, default=16)
    parser.add_argument("--nsharecodebook", type=int, default=1)
    parser.add_argument("--distance_p", type=str, default="2.0")
    parser.add_argument("--reconstruct_rate", type=float, default=1e-3)
    parser.add_argument("--centroid_requires_grad", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--weight_requires_grad", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch_dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--centroid_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="serialization_dir/qwen3_lut")
    return parser.parse_args(input_args)


def resolve_torch_dtype(dtype_name):
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"unsupported torch dtype: {dtype_name}")


def group_texts(examples, max_seq_length):
    concatenated = {}
    for key, sequences in examples.items():
        flattened = []
        for sequence in sequences:
            flattened.extend(sequence)
        concatenated[key] = flattened

    total_length = len(concatenated["input_ids"])
    total_length = (total_length // max_seq_length) * max_seq_length
    result = {
        key: [
            tokens[i : i + max_seq_length]
            for i in range(0, total_length, max_seq_length)
        ]
        for key, tokens in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def sum_lut_loss(model, device):
    losses = [
        module.lut_loss.float()
        for module in model.modules()
        if hasattr(module, "lut_loss")
    ]
    if not losses:
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.stack([loss.to(device) for loss in losses]).sum()


def build_lm_datasets(args, tokenizer, accelerator):
    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    if "validation" not in raw_datasets:
        raw_datasets["validation"] = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split="train[:1%]",
        )
        raw_datasets["train"] = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split="train[1%:]",
        )

    column_names = raw_datasets["train"].column_names
    if args.text_column not in column_names:
        raise ValueError(f"text column {args.text_column!r} not found in dataset columns {column_names}")

    def tokenize_function(examples):
        return tokenizer(examples[args.text_column])

    with accelerator.main_process_first():
        tokenized = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            num_proc=args.preprocessing_num_workers,
            desc="Tokenizing text",
        )
        lm_datasets = tokenized.map(
            lambda examples: group_texts(examples, args.max_seq_length),
            batched=True,
            num_proc=args.preprocessing_num_workers,
            desc=f"Grouping texts into {args.max_seq_length}-token chunks",
        )
    return lm_datasets


def build_dataloaders(args, lm_datasets):
    train_dataloader = DataLoader(
        lm_datasets["train"],
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        lm_datasets["validation"],
        collate_fn=default_data_collator,
        batch_size=args.per_device_eval_batch_size,
    )
    return train_dataloader, eval_dataloader


def apply_lut_replacement(model, args):
    luterizer = LUTerize(
        model=model,
        dataloader=None,
        tokenizer=None,
        logger=logger,
        activation_dir=None,
        centroid_dir=None,
        output_dir=args.output_dir,
        ncentroid=args.ncentroid,
        nsharecodebook=args.nsharecodebook,
        vec_len=args.vec_len,
        init_centroids=False,
        weight_transpose=True,
        fp16=False,
        distance_p=args.distance_p,
        target_modules=args.target_modules,
    )
    luterizer.luterize_model()
    return luterizer.model


def configure_trainable_parameters(model, args):
    for param in model.parameters():
        param.requires_grad = args.weight_requires_grad

    lut_module_count = 0
    trainable_centroid_count = 0
    for module in model.modules():
        if isinstance(module, LUTLinear_t):
            lut_module_count += 1
            module.centroids.weight.requires_grad = args.centroid_requires_grad
            if args.centroid_requires_grad:
                trainable_centroid_count += module.centroids.weight.numel()

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("no trainable parameters; enable centroid or weight training")
    return trainable_params, lut_module_count, trainable_centroid_count


def load_centroids_if_requested(model, centroid_path):
    if centroid_path is None:
        return 0

    try:
        from safetensors.torch import load_file

        tensors = load_file(centroid_path)
    except ValueError:
        tensors = torch.load(centroid_path, map_location="cpu")

    loaded = 0
    for name, module in model.named_modules():
        key = f"{name}.centroids.weight"
        if isinstance(module, LUTLinear_t) and key in tensors:
            module.centroids.weight.data.copy_(tensors[key].to(module.centroids.weight.dtype))
            loaded += 1
    return loaded


def lut_state_dict(model):
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, LUTLinear_t):
            state[f"{name}.centroids.weight"] = module.centroids.weight.detach().cpu()
    return state


def save_lut_state(model, tokenizer, args, accelerator):
    if not accelerator.is_main_process:
        return
    os.makedirs(args.output_dir, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(lut_state_dict(unwrapped_model), os.path.join(args.output_dir, "model_lut_state.pt"))
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "lut_training_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)


@torch.no_grad()
def evaluate(model, eval_dataloader, accelerator, args):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        if args.max_eval_batches is not None and step >= args.max_eval_batches:
            break
        outputs = model(**batch)
        loss = outputs.loss.detach().float().reshape(1)
        losses.append(accelerator.gather_for_metrics(loss))

    if not losses:
        return float("nan"), float("nan")

    eval_loss = torch.cat(losses).mean().item()
    try:
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    model.train()
    return eval_loss, perplexity


def main():
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_torch_dtype(args.torch_dtype),
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model = apply_lut_replacement(model, args)
    loaded_centroids = load_centroids_if_requested(model, args.centroid_path)
    trainable_params, lut_module_count, trainable_centroid_count = configure_trainable_parameters(model, args)

    accelerator.print(
        f"LUT modules: {lut_module_count}, loaded centroid tensors: {loaded_centroids}, "
        f"trainable centroid values: {trainable_centroid_count}"
    )

    lm_datasets = build_lm_datasets(args, tokenizer, accelerator)
    train_dataloader, eval_dataloader = build_dataloaders(args, lm_datasets)

    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    )

    initial_eval_loss, initial_ppl = evaluate(model, eval_dataloader, accelerator, args)
    accelerator.print(f"Initial eval loss: {initial_eval_loss:.6f}, perplexity: {initial_ppl:.6f}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    model.train()
    while completed_steps < args.max_train_steps:
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                model_loss = outputs.loss
                lut_loss = sum_lut_loss(model, accelerator.device)
                loss = model_loss + args.reconstruct_rate * lut_loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                completed_steps += 1
                progress_bar.update(1)
                if completed_steps % args.logging_steps == 0:
                    accelerator.print(
                        f"step {completed_steps}: model_loss={model_loss.detach().float().item():.6f}, "
                        f"lut_loss={lut_loss.detach().float().item():.6f}, "
                        f"total_loss={loss.detach().float().item():.6f}"
                    )
                if args.eval_steps > 0 and completed_steps % args.eval_steps == 0:
                    eval_loss, ppl = evaluate(model, eval_dataloader, accelerator, args)
                    accelerator.print(f"eval step {completed_steps}: loss={eval_loss:.6f}, perplexity={ppl:.6f}")
                if completed_steps >= args.max_train_steps:
                    break

    final_eval_loss, final_ppl = evaluate(model, eval_dataloader, accelerator, args)
    accelerator.print(f"Final eval loss: {final_eval_loss:.6f}, perplexity: {final_ppl:.6f}")
    save_lut_state(model, tokenizer, args, accelerator)


if __name__ == "__main__":
    main()
