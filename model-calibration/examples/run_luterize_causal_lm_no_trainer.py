#!/usr/bin/env python
import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

REPO_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_PACKAGE_ROOT))

import torch
from accelerate import Accelerator
from datasets import DatasetDict, load_dataset, load_from_disk
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
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--validation_file", type=str, default=None)
    parser.add_argument("--tokenized_dataset_path", type=str, default=None)
    parser.add_argument("--validation_tokenized_dataset_path", type=str, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--dataset_seed", type=int, default=42)
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--preprocessing_num_workers", type=int, default=None)
    parser.add_argument("--target_modules", choices=["mlp", "attention", "all"], default="mlp")
    parser.add_argument("--vec_len", type=int, default=4)
    parser.add_argument("--ncentroid", type=int, default=16)
    parser.add_argument("--nsharecodebook", type=int, default=1)
    parser.add_argument("--distance_p", type=str, default="2.0")
    parser.add_argument("--residual_compensation_ratio", type=float, default=0.0)
    parser.add_argument("--residual_compensation_metric", choices=["abs", "weighted"], default="abs")
    parser.add_argument("--activation_topk_only", action="store_true")
    parser.add_argument("--reconstruct_rate", type=float, default=1e-3)
    parser.add_argument("--centroid_requires_grad", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--weight_requires_grad", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--weight_trainable_scope", choices=["all", "lut_modules"], default="all")
    parser.add_argument("--freeze_embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze_lm_head", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch_dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_foreach", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_logging_steps", type=int, default=1)
    parser.add_argument("--microbatch_logging_steps", type=int, default=1)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--baseline_eval_before_lut", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--centroid_path", type=str, default=None)
    parser.add_argument("--resume_from_lut_model", type=str, default=None)
    parser.add_argument("--save_full_lut_model", action=argparse.BooleanOptionalAction, default=False)
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


def should_log_interval(step, interval):
    return interval > 0 and (step == 1 or step % interval == 0)


def format_microbatch_log(microbatch_step, optimizer_step, max_train_steps, model_loss, lut_loss, total_loss, elapsed):
    return (
        f"train microbatch {microbatch_step}: optimizer_step={optimizer_step}/{max_train_steps}, "
        f"model_loss={model_loss:.6f}, lut_loss={lut_loss:.6f}, total_loss={total_loss:.6f}, "
        f"elapsed={elapsed:.1f}s"
    )


class TokenizedCausalLMDataset:
    def __init__(self, dataset, max_seq_length):
        self.dataset = dataset
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        if "input_ids" not in item:
            raise ValueError(f"tokenized dataset item must contain 'input_ids', got columns {list(item)}")

        input_ids = list(item["input_ids"])
        if self.max_seq_length is not None:
            input_ids = input_ids[: self.max_seq_length]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": input_ids.copy(),
        }


def sample_dataset(dataset, max_samples, seed):
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    if max_samples <= 0:
        raise ValueError("sample count must be positive")

    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(dataset)), max_samples))
    return dataset.select(indices)


def load_tokenized_dataset(path):
    dataset = load_from_disk(path)
    if isinstance(dataset, DatasetDict):
        if "train" not in dataset:
            raise ValueError(f"tokenized dataset dict at {path!r} must contain a train split")
    return dataset


def build_tokenized_lm_datasets(args):
    if args.tokenized_dataset_path is None:
        raise ValueError("tokenized_dataset_path is required for tokenized dataset loading")

    loaded_dataset = load_tokenized_dataset(args.tokenized_dataset_path)
    if isinstance(loaded_dataset, DatasetDict):
        train_source = loaded_dataset["train"]
        validation_source = loaded_dataset.get("validation", train_source)
    else:
        train_source = loaded_dataset
        validation_source = loaded_dataset

    if args.validation_tokenized_dataset_path is not None:
        validation_loaded = load_tokenized_dataset(args.validation_tokenized_dataset_path)
        validation_source = validation_loaded["validation"] if isinstance(validation_loaded, DatasetDict) else validation_loaded

    train_dataset = sample_dataset(train_source, args.max_train_samples, args.dataset_seed)
    validation_sample_count = args.max_eval_samples
    if validation_sample_count is None and validation_source is train_source:
        validation_sample_count = min(1024, len(validation_source))
    validation_dataset = sample_dataset(validation_source, validation_sample_count, args.dataset_seed + 1)

    return {
        "train": TokenizedCausalLMDataset(train_dataset, args.max_seq_length),
        "validation": TokenizedCausalLMDataset(validation_dataset, args.max_seq_length),
    }


def load_raw_datasets(args):
    if args.train_file is not None:
        data_files = {"train": args.train_file}
        data_files["validation"] = args.validation_file or args.train_file
        return load_dataset("text", data_files=data_files)

    return load_dataset(args.dataset_name, args.dataset_config_name)


def build_lm_datasets(args, tokenizer, accelerator):
    if args.tokenized_dataset_path is not None:
        return build_tokenized_lm_datasets(args)

    raw_datasets = load_raw_datasets(args)
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


def dataset_length(dataset):
    try:
        return len(dataset)
    except TypeError:
        return "unknown"


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


def enable_gradient_checkpointing_if_requested(model, args):
    if not args.gradient_checkpointing:
        return

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()


def apply_lut_replacement(model, args):
    luterizer = LUTerize(
        model=model,
        dataloader=None,
        tokenizer=None,
        logger=logger,
        activation_dir=None,
        centroid_dir=None,
        output_dir=getattr(args, "output_dir", None),
        ncentroid=args.ncentroid,
        nsharecodebook=args.nsharecodebook,
        vec_len=args.vec_len,
        init_centroids=False,
        weight_transpose=True,
        fp16=False,
        distance_p=args.distance_p,
        target_modules=args.target_modules,
        residual_compensation_ratio=getattr(args, "residual_compensation_ratio", 0.0),
        residual_compensation_metric=getattr(args, "residual_compensation_metric", "abs"),
        activation_topk_only=getattr(args, "activation_topk_only", False),
    )
    luterizer.luterize_model()
    return luterizer.model


def configure_trainable_parameters(model, args):
    for param in model.parameters():
        param.requires_grad = args.weight_requires_grad and args.weight_trainable_scope == "all"

    for name, param in model.named_parameters():
        if args.freeze_embeddings and ("embed" in name.lower() or "embedding" in name.lower()):
            param.requires_grad = False
        if args.freeze_lm_head and name.startswith("lm_head."):
            param.requires_grad = False

    lut_module_count = 0
    trainable_centroid_count = 0
    for module in model.modules():
        if isinstance(module, LUTLinear_t):
            lut_module_count += 1
            if args.weight_trainable_scope == "lut_modules":
                module.weight.requires_grad = args.weight_requires_grad
                if module.bias is not None:
                    module.bias.requires_grad = args.weight_requires_grad
            module.centroids.weight.requires_grad = args.centroid_requires_grad
            if args.centroid_requires_grad:
                trainable_centroid_count += module.centroids.weight.numel()

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("no trainable parameters; enable centroid or weight training")
    return trainable_params, lut_module_count, trainable_centroid_count


def count_trainable_parameters(model):
    counts = {"total": 0, "centroid": 0, "non_centroid": 0}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        numel = param.numel()
        counts["total"] += numel
        if ".centroids.weight" in name or name.endswith("centroids.weight"):
            counts["centroid"] += numel
        else:
            counts["non_centroid"] += numel
    return counts


def load_centroids_if_requested(model, centroid_path):
    if centroid_path is None:
        return 0

    if str(centroid_path).endswith(".safetensors"):
        from safetensors.torch import load_file

        tensors = load_file(centroid_path)
    else:
        tensors = torch.load(centroid_path, map_location="cpu")

    loaded = 0
    for name, module in model.named_modules():
        key = f"{name}.centroids.weight"
        if isinstance(module, LUTLinear_t) and key in tensors:
            module.centroids.weight.data.copy_(tensors[key].to(module.centroids.weight.dtype))
            loaded += 1
    return loaded


def _load_state_dict_file(path):
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)

    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"]
    return state


def _lut_model_state_files(lut_model_path):
    path = Path(lut_model_path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"LUT model checkpoint does not exist: {lut_model_path}")

    full_state_path = path / "full_lut_model_state.pt"
    if full_state_path.exists():
        return [full_state_path]

    index_files = [
        path / "model.safetensors.index.json",
        path / "pytorch_model.bin.index.json",
    ]
    for index_path in index_files:
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            filenames = sorted(set(index["weight_map"].values()))
            return [path / filename for filename in filenames]

    candidates = [
        path / "model.safetensors",
        path / "pytorch_model.bin",
    ]
    for candidate in candidates:
        if candidate.exists():
            return [candidate]

    raise FileNotFoundError(
        f"Could not find model.safetensors, pytorch_model.bin, or shard index in {lut_model_path}"
    )


def _normalize_lut_model_state_dict_for_model(model, state_dict):
    model_keys = set(model.state_dict().keys())
    state_keys = set(state_dict.keys())
    original_overlap = len(model_keys & state_keys)

    replacements = [
        ("model.language_model.language_model.", "model."),
        ("model.language_model.", "model."),
        ("language_model.", "model."),
    ]
    best_state_dict = state_dict
    best_overlap = original_overlap
    for old_prefix, new_prefix in replacements:
        normalized = {
            (new_prefix + key[len(old_prefix) :]) if key.startswith(old_prefix) else key: value
            for key, value in state_dict.items()
        }
        overlap = len(model_keys & set(normalized.keys()))
        if overlap > best_overlap:
            best_state_dict = normalized
            best_overlap = overlap
    return best_state_dict, best_overlap


def load_lut_model_state_if_requested(model, lut_model_path):
    if lut_model_path is None:
        return 0

    loaded_keys = 0
    for state_file in _lut_model_state_files(lut_model_path):
        state_dict = _load_state_dict_file(state_file)
        state_dict, matched_keys = _normalize_lut_model_state_dict_for_model(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
        loaded_keys += matched_keys
        del state_dict
    if loaded_keys == 0:
        raise ValueError(f"No checkpoint keys from {lut_model_path} matched the LUT-converted model")
    return loaded_keys


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


def save_full_lut_model(model, tokenizer, args, accelerator):
    accelerator.wait_for_everyone()
    os.makedirs(args.output_dir, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(unwrapped_model.state_dict(), os.path.join(args.output_dir, "full_lut_model_state.pt"))
    if accelerator.is_main_process:
        unwrapped_model.config.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        with open(os.path.join(args.output_dir, "lut_training_args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)


def max_eval_step_count(eval_dataloader, args):
    dataloader_len = len(eval_dataloader)
    if args.max_eval_batches is None:
        return dataloader_len
    return min(args.max_eval_batches, dataloader_len)


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate(model, eval_dataloader, accelerator, args, label="eval"):
    model.eval()
    losses = []
    total_eval_steps = max_eval_step_count(eval_dataloader, args)
    start_time = time.perf_counter()
    accelerator.print(f"{label}: starting {total_eval_steps} eval batch(es)")
    for step, batch in enumerate(eval_dataloader):
        if args.max_eval_batches is not None and step >= args.max_eval_batches:
            break
        batch = move_batch_to_device(batch, accelerator.device)
        outputs = model(**batch)
        loss = outputs.loss.detach().float().reshape(1)
        losses.append(accelerator.gather_for_metrics(loss))
        current_step = step + 1
        if should_log_interval(current_step, args.eval_logging_steps):
            elapsed = time.perf_counter() - start_time
            accelerator.print(
                f"{label}: finished eval batch {current_step}/{total_eval_steps}, "
                f"loss={loss.item():.6f}, elapsed={elapsed:.1f}s"
            )

    if not losses:
        return float("nan"), float("nan")

    eval_loss = torch.cat(losses).mean().item()
    try:
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    model.train()
    elapsed = time.perf_counter() - start_time
    accelerator.print(f"{label}: completed in {elapsed:.1f}s")
    return eval_loss, perplexity


def load_causal_lm_model(args, accelerator, label="model"):
    accelerator.print(f"Loading {label} from {args.model_name_or_path} with dtype={args.torch_dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_torch_dtype(args.torch_dtype),
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    return model


def main():
    args = parse_args()
    if args.centroid_path is not None and args.resume_from_lut_model is not None:
        raise ValueError("--centroid_path and --resume_from_lut_model are mutually exclusive")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    accelerator.print(f"Loading tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    accelerator.print("Building LM datasets and dataloaders")
    lm_datasets = build_lm_datasets(args, tokenizer, accelerator)
    accelerator.print(
        f"Dataset sizes: train={dataset_length(lm_datasets['train'])}, "
        f"validation={dataset_length(lm_datasets['validation'])}, "
        f"max_seq_length={args.max_seq_length}"
    )
    train_dataloader, eval_dataloader = build_dataloaders(args, lm_datasets)
    accelerator.print(
        f"Dataloader batches: train={len(train_dataloader)}, validation={len(eval_dataloader)}, "
        f"gradient_accumulation_steps={args.gradient_accumulation_steps}, max_train_steps={args.max_train_steps}"
    )

    if args.baseline_eval_before_lut:
        baseline_model = load_causal_lm_model(args, accelerator, label="baseline model")
        baseline_model.to(accelerator.device)
        baseline_eval_loss, baseline_ppl = evaluate(
            baseline_model,
            eval_dataloader,
            accelerator,
            args,
            label="baseline_eval_before_lut",
        )
        accelerator.print(
            f"Baseline eval before LUT replacement: loss={baseline_eval_loss:.6f}, "
            f"perplexity={baseline_ppl:.6f}"
        )
        del baseline_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model = load_causal_lm_model(args, accelerator)
    if args.gradient_checkpointing:
        accelerator.print("Enabling gradient checkpointing")
        enable_gradient_checkpointing_if_requested(model, args)

    accelerator.print(f"Replacing target Linear modules with LUTLinear_t modules: target_modules={args.target_modules}")
    model = apply_lut_replacement(model, args)
    loaded_centroids = load_centroids_if_requested(model, args.centroid_path)
    loaded_lut_model_keys = load_lut_model_state_if_requested(model, args.resume_from_lut_model)
    trainable_params, lut_module_count, trainable_centroid_count = configure_trainable_parameters(model, args)
    trainable_counts = count_trainable_parameters(model)

    accelerator.print(
        f"LUT modules: {lut_module_count}, loaded centroid tensors: {loaded_centroids}, "
        f"loaded full LUT checkpoint keys: {loaded_lut_model_keys}, "
        f"trainable centroid values: {trainable_centroid_count}, "
        f"trainable total params: {trainable_counts['total']}, "
        f"trainable non-centroid params: {trainable_counts['non_centroid']}, "
        f"residual_compensation_ratio={args.residual_compensation_ratio}, "
        f"residual_compensation_metric={args.residual_compensation_metric}, "
        f"activation_topk_only={args.activation_topk_only}"
    )

    if args.eval_only or args.max_train_steps <= 0:
        accelerator.print("Preparing model and eval dataloader with Accelerate")
        model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
        eval_loss, ppl = evaluate(model, eval_dataloader, accelerator, args, label="lut_eval")
        accelerator.print(f"LUT eval loss: {eval_loss:.6f}, perplexity: {ppl:.6f}")
        accelerator.print("Evaluation complete")
        return

    optimizer = AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        foreach=args.adam_foreach,
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    accelerator.print("Preparing model, optimizer, dataloaders, and scheduler with Accelerate")
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    )

    initial_eval_loss, initial_ppl = evaluate(model, eval_dataloader, accelerator, args, label="initial_eval")
    accelerator.print(f"Initial eval loss: {initial_eval_loss:.6f}, perplexity: {initial_ppl:.6f}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    microbatch_steps = 0
    train_start_time = time.perf_counter()
    model.train()
    accelerator.print("Starting training loop")
    while completed_steps < args.max_train_steps:
        for batch in train_dataloader:
            microbatch_steps += 1
            with accelerator.accumulate(model):
                outputs = model(**batch)
                model_loss = outputs.loss
                lut_loss = sum_lut_loss(model, accelerator.device)
                loss = model_loss + args.reconstruct_rate * lut_loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if should_log_interval(microbatch_steps, args.microbatch_logging_steps):
                elapsed = time.perf_counter() - train_start_time
                accelerator.print(
                    format_microbatch_log(
                        microbatch_step=microbatch_steps,
                        optimizer_step=completed_steps + 1,
                        max_train_steps=args.max_train_steps,
                        model_loss=model_loss.detach().float().item(),
                        lut_loss=lut_loss.detach().float().item(),
                        total_loss=loss.detach().float().item(),
                        elapsed=elapsed,
                    )
                )

            if accelerator.sync_gradients:
                completed_steps += 1
                progress_bar.update(1)
                if should_log_interval(completed_steps, args.logging_steps):
                    elapsed = time.perf_counter() - train_start_time
                    accelerator.print(
                        f"step {completed_steps}: model_loss={model_loss.detach().float().item():.6f}, "
                        f"lut_loss={lut_loss.detach().float().item():.6f}, "
                        f"total_loss={loss.detach().float().item():.6f}, "
                        f"elapsed={elapsed:.1f}s"
                    )
                if args.eval_steps > 0 and completed_steps % args.eval_steps == 0:
                    eval_loss, ppl = evaluate(model, eval_dataloader, accelerator, args, label=f"eval_step_{completed_steps}")
                    accelerator.print(f"eval step {completed_steps}: loss={eval_loss:.6f}, perplexity={ppl:.6f}")
                if completed_steps >= args.max_train_steps:
                    break

    final_eval_loss, final_ppl = evaluate(model, eval_dataloader, accelerator, args, label="final_eval")
    accelerator.print(f"Final eval loss: {final_eval_loss:.6f}, perplexity: {final_ppl:.6f}")
    if args.save_full_lut_model:
        accelerator.print(f"Saving full LUT model, LUT state, and tokenizer to {args.output_dir}")
        save_full_lut_model(model, tokenizer, args, accelerator)
    else:
        accelerator.print(f"Saving LUT state and tokenizer to {args.output_dir}")
    save_lut_state(model, tokenizer, args, accelerator)
    accelerator.print("Training complete")


if __name__ == "__main__":
    main()
