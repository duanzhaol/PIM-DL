#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

SMOKE_ROOT="serialization_dir/qwen3_lut_smoke_assets"
MODEL_DIR="${SMOKE_ROOT}/tiny_qwen3_model"
TRAIN_FILE="${SMOKE_ROOT}/tiny_corpus.txt"

mkdir -p "${SMOKE_ROOT}"
cat > "${TRAIN_FILE}" <<'EOF'
查表近似需要先让质心看到真实的语言模型激活。
Qwen3 的 MLP 投影维度可以被小向量长度整除，因此适合作为 smoke test。
The smoke test checks LUT replacement, causal LM batching, and centroid updates.
Small local corpora keep this validation independent from benchmark downloads.
EOF

python examples/create_tiny_qwen3_checkpoint.py \
  --source_tokenizer Qwen/Qwen3-0.6B \
  --output_dir "${MODEL_DIR}"

accelerate launch examples/run_luterize_causal_lm_no_trainer.py \
  --model_name_or_path "${MODEL_DIR}" \
  --train_file "${TRAIN_FILE}" \
  --validation_file "${TRAIN_FILE}" \
  --target_modules mlp \
  --max_seq_length 32 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_train_steps 2 \
  --eval_steps 1 \
  --max_eval_batches 1 \
  --vec_len 4 \
  --ncentroid 4 \
  --torch_dtype float32 \
  --reconstruct_rate 1e-3 \
  --output_dir serialization_dir/qwen3_lut_smoke
