#!/usr/bin/env python
import argparse
import json
import random
import re
import sys
from pathlib import Path

REPO_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_PACKAGE_ROOT))

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.run_luterize_causal_lm_no_trainer import (
    apply_lut_replacement,
    load_centroids_if_requested,
    resolve_torch_dtype,
)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Run a small sampled MMLU-Pro evaluation for a Qwen-style LUT model.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="TIGER-Lab/MMLU-Pro")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--dataset_seed", type=int, default=42)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--target_modules", choices=["mlp", "attention", "all"], default="mlp")
    parser.add_argument("--vec_len", type=int, default=2)
    parser.add_argument("--ncentroid", type=int, default=16)
    parser.add_argument("--nsharecodebook", type=int, default=1)
    parser.add_argument("--distance_p", type=str, default="2.0")
    parser.add_argument("--residual_compensation_ratio", type=float, default=0.0)
    parser.add_argument("--residual_compensation_metric", choices=["abs", "weighted"], default="abs")
    parser.add_argument("--activation_topk_only", action="store_true")
    parser.add_argument("--centroid_path", type=str, default=None)
    parser.add_argument("--disable_lut", action="store_true")
    parser.add_argument("--torch_dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--output_jsonl", type=str, default=None)
    return parser.parse_args(input_args)


def option_letters(count):
    if count > 26:
        raise ValueError("at most 26 options are supported")
    return [chr(ord("A") + index) for index in range(count)]


def format_mmlu_pro_prompt(sample):
    letters = option_letters(len(sample["options"]))
    option_lines = "\n".join(
        f"{letter}. {option}"
        for letter, option in zip(letters, sample["options"])
    )
    return (
        "Answer the following multiple-choice question. "
        "Choose the single best answer and finish with the answer letter in parentheses.\n\n"
        f"Question: {sample['question']}\n"
        f"{option_lines}\n\n"
        "The answer is ("
    )


def extract_choice(text, num_options):
    valid_letters = "".join(option_letters(num_options))
    patterns = [
        rf"\(([{valid_letters}])\)",
        rf"\banswer\s*(?:is|:)?\s*\(?([{valid_letters}])\)?",
        rf"\b([{valid_letters}])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def sample_dataset(dataset, max_samples, seed):
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(dataset)), max_samples))
    return dataset.select(indices)


def load_eval_dataset(args):
    dataset = load_dataset(args.dataset_name, split=args.split)
    if args.category is not None:
        dataset = dataset.filter(lambda example: example["category"] == args.category)
    return sample_dataset(dataset, args.max_samples, args.dataset_seed)


def load_model(args):
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

    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return model, tokenizer


@torch.no_grad()
def generate_answer(model, tokenizer, prompt, args):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    do_sample = args.temperature > 0.0
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=do_sample,
        temperature=args.temperature if do_sample else None,
        top_p=args.top_p if do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_ids = outputs[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def evaluate(args):
    dataset = load_eval_dataset(args)
    model, tokenizer = load_model(args)
    records = []
    correct = 0

    output_file = None
    if args.output_jsonl is not None:
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = output_path.open("w", encoding="utf-8")

    try:
        for index, sample in enumerate(dataset, start=1):
            prompt = format_mmlu_pro_prompt(sample)
            completion = generate_answer(model, tokenizer, prompt, args)
            prediction = extract_choice(completion, len(sample["options"]))
            answer = sample["answer"]
            is_correct = prediction == answer
            correct += int(is_correct)
            record = {
                "index": index,
                "question_id": sample.get("question_id"),
                "category": sample.get("category"),
                "answer": answer,
                "prediction": prediction,
                "correct": is_correct,
                "completion": completion,
            }
            records.append(record)
            print(
                f"[{index}/{len(dataset)}] category={record['category']} "
                f"pred={prediction} answer={answer} correct={is_correct} completion={completion!r}",
                flush=True,
            )
            if output_file is not None:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_file.flush()
    finally:
        if output_file is not None:
            output_file.close()

    accuracy = correct / len(dataset) if len(dataset) else float("nan")
    print(f"Accuracy: {correct}/{len(dataset)} = {accuracy:.4f}", flush=True)
    return records, accuracy


def main():
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
