#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

accelerate launch examples/run_luterize_causal_lm_no_trainer.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --target_modules mlp \
  --max_seq_length 128 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_train_steps 2 \
  --eval_steps 1 \
  --max_eval_batches 1 \
  --vec_len 4 \
  --ncentroid 8 \
  --reconstruct_rate 1e-3 \
  --output_dir serialization_dir/qwen3_lut_smoke
