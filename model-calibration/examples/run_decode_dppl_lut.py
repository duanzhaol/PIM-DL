#!/usr/bin/env python
import argparse
import copy
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

REPO_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_PACKAGE_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from LUTNeuro.LUTLinear_t import LUTLinear_t
from examples.run_luterize_causal_lm_no_trainer import (
    apply_lut_replacement,
    load_centroids_if_requested,
    resolve_torch_dtype,
)


@dataclass(frozen=True)
class Window:
    id: str
    tokens: list[int]


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Run decode-stage D-PPL for dense, PIM-DL LUT, or activation-topk-only HF models."
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--windows_path", type=str, required=True)
    parser.add_argument("--prompt_len", type=int, default=512)
    parser.add_argument("--decode_len", type=int, default=512)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--dppl_mode", choices=["isolated", "pathwise"], default="isolated")
    parser.add_argument("--method", choices=["dense", "pimdl_lut", "activation_topk_only"], default="dense")
    parser.add_argument("--source_tokenizer_path", type=str, default=None)
    parser.add_argument("--torch_dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target_modules", choices=["mlp", "attention", "all"], default="mlp")
    parser.add_argument("--vec_len", type=int, default=4)
    parser.add_argument("--ncentroid", type=int, default=16)
    parser.add_argument("--nsharecodebook", type=int, default=1)
    parser.add_argument("--distance_p", type=str, default="2.0")
    parser.add_argument("--centroid_path", type=str, default=None)
    parser.add_argument("--lut_eval_compute_dtype", choices=["float32", "model"], default="float32")
    parser.add_argument("--residual_compensation_ratio", type=float, default=0.0)
    parser.add_argument("--residual_compensation_metric", choices=["abs", "weighted"], default="abs")
    parser.add_argument("--log_steps", type=int, default=64)
    parser.add_argument("--output_path", type=str, default=None)
    return parser.parse_args(input_args)


def _iter_window_records(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return
    if text[0] == "[":
        for record in json.loads(text):
            yield record
    else:
        for line in text.splitlines():
            if line.strip():
                yield json.loads(line)


def load_dppl_windows(path, prompt_len: int, decode_len: int, max_windows: int | None = None) -> list[Window]:
    required_tokens = prompt_len + decode_len + 1
    windows = []
    for record in _iter_window_records(Path(path)):
        if "tokens" not in record:
            raise ValueError(f"window record must contain tokens: {record}")
        tokens = [int(token) for token in record["tokens"]]
        if len(tokens) < required_tokens:
            raise ValueError(
                f"window {record.get('id', len(windows))!r} needs at least {required_tokens} tokens, got {len(tokens)}"
            )
        windows.append(Window(id=str(record.get("id", len(windows))), tokens=tokens[:required_tokens]))
        if max_windows is not None and len(windows) >= max_windows:
            break
    if not windows:
        raise ValueError(f"no D-PPL windows loaded from {path}")
    return windows


def retokenize_windows(windows: list[Window], source_tokenizer, target_tokenizer, required_tokens: int) -> list[Window]:
    retokenized = []
    for window in windows:
        text = source_tokenizer.decode(window.tokens, skip_special_tokens=False)
        tokens = target_tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < required_tokens:
            raise ValueError(
                f"retokenized window {window.id!r} has {len(tokens)} tokens; required {required_tokens}"
            )
        retokenized.append(Window(id=window.id, tokens=[int(token) for token in tokens[:required_tokens]]))
    return retokenized


def compute_next_token_nll(logits: torch.Tensor, gold_token_id: int) -> float:
    logits_2d = logits.reshape(-1, logits.shape[-1]).float()
    log_probs = torch.log_softmax(logits_2d[-1], dim=-1)
    return float(-log_probs[int(gold_token_id)].detach().cpu())


def summarize_nll(total_nll: float, tokens: int) -> dict:
    nll = total_nll / tokens if tokens else float("nan")
    try:
        d_ppl = math.exp(nll)
    except OverflowError:
        d_ppl = float("inf")
    return {"nll": nll, "d_ppl": d_ppl, "tokens": tokens}


def set_lut_eval_compute_dtype(model, compute_dtype: str) -> int:
    updated = 0
    for module in model.modules():
        if isinstance(module, LUTLinear_t):
            module.eval_compute_dtype = compute_dtype
            updated += 1
    return updated


def count_lut_modules(model) -> int:
    return sum(1 for module in model.modules() if isinstance(module, LUTLinear_t))


def move_model_to_device(model, device: torch.device):
    return model.to(device)


def clone_past_key_values(past_key_values):
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "get_seq_length"):
        return copy.deepcopy(past_key_values)
    if hasattr(past_key_values, "to_legacy_cache"):
        return clone_past_key_values(past_key_values.to_legacy_cache())
    if torch.is_tensor(past_key_values):
        return past_key_values.clone()
    if isinstance(past_key_values, tuple):
        return tuple(clone_past_key_values(value) for value in past_key_values)
    if isinstance(past_key_values, list):
        return [clone_past_key_values(value) for value in past_key_values]
    raise TypeError(f"unsupported past_key_values type for isolated D-PPL: {type(past_key_values)!r}")


def make_lut_args(args, activation_topk_only: bool):
    return SimpleNamespace(
        output_dir=None,
        ncentroid=args.ncentroid,
        nsharecodebook=args.nsharecodebook,
        vec_len=args.vec_len,
        distance_p=args.distance_p,
        target_modules=args.target_modules,
        residual_compensation_ratio=args.residual_compensation_ratio,
        residual_compensation_metric=args.residual_compensation_metric,
        activation_topk_only=activation_topk_only,
    )


def load_causal_lm(model_name_or_path: str, args, device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=resolve_torch_dtype(args.torch_dtype),
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = True
    model.eval()
    return model.to(device)


def load_eval_models(args, device: torch.device):
    if args.method == "dense":
        model = load_causal_lm(args.model_name_or_path, args, device)
        return model, model, 0

    dense_model = load_causal_lm(args.model_name_or_path, args, device)
    eval_model = load_causal_lm(args.model_name_or_path, args, device)
    eval_model = apply_lut_replacement(
        eval_model,
        make_lut_args(args, activation_topk_only=args.method == "activation_topk_only"),
    )
    loaded_centroids = load_centroids_if_requested(eval_model, args.centroid_path)
    set_lut_eval_compute_dtype(eval_model, args.lut_eval_compute_dtype)
    eval_model = move_model_to_device(eval_model, device)
    eval_model.eval()
    return dense_model, eval_model, loaded_centroids


@torch.inference_mode()
def prefill_dense(model, tokens: torch.Tensor, prompt_len: int):
    outputs = model(input_ids=tokens[:, :prompt_len], use_cache=True)
    return outputs.past_key_values


@torch.inference_mode()
def run_window_pathwise(dense_model, eval_model, tokens: torch.Tensor, prompt_len: int, decode_len: int):
    past = prefill_dense(dense_model, tokens, prompt_len)
    total_nll = 0.0
    for offset in range(decode_len):
        token_pos = prompt_len + offset
        outputs = eval_model(
            input_ids=tokens[:, token_pos : token_pos + 1],
            past_key_values=past,
            use_cache=True,
        )
        total_nll += compute_next_token_nll(outputs.logits[:, -1, :], int(tokens[0, token_pos + 1]))
        past = outputs.past_key_values
    return total_nll


@torch.inference_mode()
def run_window_isolated(dense_model, eval_model, tokens: torch.Tensor, prompt_len: int, decode_len: int):
    dense_past = prefill_dense(dense_model, tokens, prompt_len)
    total_nll = 0.0
    for offset in range(decode_len):
        token_pos = prompt_len + offset
        current_token = tokens[:, token_pos : token_pos + 1]
        if dense_model is eval_model:
            outputs = dense_model(input_ids=current_token, past_key_values=dense_past, use_cache=True)
            total_nll += compute_next_token_nll(outputs.logits[:, -1, :], int(tokens[0, token_pos + 1]))
            dense_past = outputs.past_key_values
            continue

        anchor_past = clone_past_key_values(dense_past)
        outputs = eval_model(input_ids=current_token, past_key_values=anchor_past, use_cache=True)
        total_nll += compute_next_token_nll(outputs.logits[:, -1, :], int(tokens[0, token_pos + 1]))

        dense_outputs = dense_model(input_ids=current_token, past_key_values=dense_past, use_cache=True)
        dense_past = dense_outputs.past_key_values
    return total_nll


def validate_token_ids(windows: list[Window], vocab_size: int):
    max_token = max(max(window.tokens) for window in windows)
    if max_token >= vocab_size:
        raise ValueError(f"window token id {max_token} exceeds model embedding size {vocab_size}")


def run_decode_dppl(args):
    if args.prompt_len <= 0 or args.decode_len <= 0:
        raise ValueError("prompt_len and decode_len must be positive")

    required_tokens = args.prompt_len + args.decode_len + 1
    windows = load_dppl_windows(args.windows_path, args.prompt_len, args.decode_len, args.max_windows)
    target_tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if args.source_tokenizer_path is not None:
        source_tokenizer = AutoTokenizer.from_pretrained(args.source_tokenizer_path, use_fast=True)
        windows = retokenize_windows(windows, source_tokenizer, target_tokenizer, required_tokens)

    device = torch.device(args.device)
    dense_model, eval_model, loaded_centroids = load_eval_models(args, device)
    vocab_size = eval_model.get_input_embeddings().num_embeddings
    validate_token_ids(windows, vocab_size)

    window_summaries = []
    total_nll = 0.0
    total_tokens = 0
    start_time = time.perf_counter()
    print(
        f"Running D-PPL: method={args.method}, dppl_mode={args.dppl_mode}, "
        f"windows={len(windows)}, P={args.prompt_len}, D={args.decode_len}, "
        f"loaded_centroids={loaded_centroids}, lut_modules={count_lut_modules(eval_model)}"
    )
    for window_index, window in enumerate(windows, start=1):
        tokens = torch.tensor(window.tokens, dtype=torch.long, device=device).unsqueeze(0)
        window_start = time.perf_counter()
        if args.dppl_mode == "isolated":
            window_nll = run_window_isolated(dense_model, eval_model, tokens, args.prompt_len, args.decode_len)
        else:
            window_nll = run_window_pathwise(dense_model, eval_model, tokens, args.prompt_len, args.decode_len)

        total_nll += window_nll
        total_tokens += args.decode_len
        elapsed = time.perf_counter() - window_start
        window_summary = {
            "id": window.id,
            **summarize_nll(window_nll, args.decode_len),
            "elapsed_sec": elapsed,
        }
        window_summaries.append(window_summary)
        if args.log_steps > 0:
            running = summarize_nll(total_nll, total_tokens)
            print(
                f"window {window_index}/{len(windows)} id={window.id}: "
                f"nll={window_summary['nll']:.6f}, d_ppl={window_summary['d_ppl']:.6f}, "
                f"elapsed={elapsed:.1f}s, running_d_ppl={running['d_ppl']:.6f}"
            )

    summary = {
        **summarize_nll(total_nll, total_tokens),
        "elapsed_sec": time.perf_counter() - start_time,
        "method": args.method,
        "dppl_mode": args.dppl_mode,
        "model_name_or_path": args.model_name_or_path,
        "windows_path": args.windows_path,
        "prompt_len": args.prompt_len,
        "decode_len": args.decode_len,
        "max_windows": args.max_windows,
        "source_tokenizer_path": args.source_tokenizer_path,
        "target_modules": args.target_modules,
        "vec_len": args.vec_len,
        "ncentroid": args.ncentroid,
        "centroid_path": args.centroid_path,
        "loaded_centroids": loaded_centroids,
        "lut_modules": count_lut_modules(eval_model),
        "residual_compensation_ratio": args.residual_compensation_ratio,
        "residual_compensation_metric": args.residual_compensation_metric,
        "window_summaries": window_summaries,
    }

    print(
        f"D-PPL summary: nll={summary['nll']:.6f}, "
        f"d_ppl={summary['d_ppl']:.6f}, tokens={summary['tokens']}, "
        f"elapsed={summary['elapsed_sec']:.1f}s"
    )
    if args.output_path is not None:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote summary to {output_path}")
    return summary


def main():
    run_decode_dppl(parse_args())


if __name__ == "__main__":
    main()
