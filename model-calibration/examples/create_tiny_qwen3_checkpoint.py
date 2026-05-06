#!/usr/bin/env python
import argparse
import os

from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Create a tiny randomly initialized Qwen3 checkpoint for smoke tests.")
    parser.add_argument("--source_tokenizer", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--intermediate_size", type=int, default=64)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--num_attention_heads", type=int, default=4)
    parser.add_argument("--num_key_value_heads", type=int, default=2)
    parser.add_argument("--head_dim", type=int, default=8)
    parser.add_argument("--max_position_embeddings", type=int, default=256)
    return parser.parse_args(input_args)


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.source_tokenizer, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        head_dim=args.head_dim,
        max_position_embeddings=args.max_position_embeddings,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
    )
    model = Qwen3ForCausalLM(config)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
