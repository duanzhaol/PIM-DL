#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/root/models/Qwen3-4B/}"
TOKENIZED_DATASET_PATH="${TOKENIZED_DATASET_PATH:-/root/fineweb-edu/sample/10BT-tokenized-qwen3-2048}"
MODULE_NAME="${MODULE_NAME:-model.layers.18.mlp.up_proj}"
OUTPUT_DIR="${OUTPUT_DIR:-serialization_dir/single_layer_lut_mse}"

KS="${KS:-2 4 8 16 32 64 128 256 512 1024}"
VS="${VS:-2 4 8 16 32 64 128}"

MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
CALIB_TOKENS="${CALIB_TOKENS:-2048}"
EVAL_TOKENS="${EVAL_TOKENS:-512}"
KMEANS_ITER="${KMEANS_ITER:-20}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
EVAL_CHUNK_TOKENS="${EVAL_CHUNK_TOKENS:-64}"
CODEBOOK_BLOCK_SIZE="${CODEBOOK_BLOCK_SIZE:-16}"
SEED="${SEED:-0}"
AGGREGATE_CSV="${AGGREGATE_CSV:-${OUTPUT_DIR}/results.csv}"

mkdir -p "${OUTPUT_DIR}"

module_slug="${MODULE_NAME//./_}"
LAYER_CACHE_PATH="${LAYER_CACHE_PATH:-${OUTPUT_DIR}/${module_slug}_calib${CALIB_TOKENS}_eval${EVAL_TOKENS}_cache.pt}"

if [[ ! -f "${LAYER_CACHE_PATH}" ]]; then
  echo "Preparing layer cache: ${LAYER_CACHE_PATH}"
  cache_cmd=(
    python examples/collect_single_layer_lut_cache.py
    --model_name_or_path "${MODEL_NAME_OR_PATH}"
    --tokenized_dataset_path "${TOKENIZED_DATASET_PATH}"
    --module_name "${MODULE_NAME}"
    --max_seq_length "${MAX_SEQ_LENGTH}"
    --per_device_batch_size "${PER_DEVICE_BATCH_SIZE}"
    --calib_tokens "${CALIB_TOKENS}"
    --eval_tokens "${EVAL_TOKENS}"
    --torch_dtype "${TORCH_DTYPE}"
    --output_path "${LAYER_CACHE_PATH}"
  )

  if [[ -n "${MAX_SAMPLES}" ]]; then
    cache_cmd+=(--max_samples "${MAX_SAMPLES}")
  fi

  "${cache_cmd[@]}"
else
  echo "Using existing layer cache: ${LAYER_CACHE_PATH}"
fi

for k in ${KS}; do
  for v in ${VS}; do
    output_json="${OUTPUT_DIR}/${module_slug}_k${k}_v${v}.json"
    if [[ -f "${output_json}" ]]; then
      echo "Skipping existing ${output_json}"
      continue
    fi

    echo "Running single-layer LUT MSE: module=${MODULE_NAME} K=${k} V=${v}"
    cmd=(
      python examples/evaluate_single_layer_lut_mse.py
      --layer_cache_path "${LAYER_CACHE_PATH}"
      --vec_len "${v}"
      --ncentroid "${k}"
      --kmeans_iter "${KMEANS_ITER}"
      --eval_chunk_tokens "${EVAL_CHUNK_TOKENS}"
      --codebook_block_size "${CODEBOOK_BLOCK_SIZE}"
      --seed "${SEED}"
      --output_json "${output_json}"
    )

    "${cmd[@]}"
  done
done

python examples/aggregate_single_layer_lut_mse.py \
  --input_dir "${OUTPUT_DIR}" \
  --output_csv "${AGGREGATE_CSV}"
