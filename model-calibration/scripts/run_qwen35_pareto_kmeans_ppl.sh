#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ACCELERATE_BIN="${ACCELERATE_BIN:-/root/miniconda3/envs/luturbo/bin/accelerate}"
MODEL_PATH="${MODEL_PATH:-/root/models/Qwen3.5-4B}"
TOKENIZED_DATASET_PATH="${TOKENIZED_DATASET_PATH:-/root/fineweb-edu/sample/10BT-tokenized-qwen35-2048-10k}"

NSAMPLE="${NSAMPLE:-8}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-10000}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-512}"
PPL_MAX_EVAL_BATCHES="${PPL_MAX_EVAL_BATCHES:-50}"
RECONSTRUCT_RATE="${RECONSTRUCT_RATE:-1e-3}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LOG_DIR="${LOG_DIR:-serialization_dir/logs/qwen35_pareto_kmeans_ppl}"
FORCE_PPL="${FORCE_PPL:-0}"

mkdir -p "${LOG_DIR}"

CONFIGS=(
  "fast:4:128"
  "low_middle:64:32"
  "middle:32:8"
  "quality:128:4"
  "expensive:512:4"
)

run_logged() {
  local log_path="$1"
  shift
  echo "[$(date '+%F %T')] $*" | tee "${log_path}"
  "$@" 2>&1 | tee -a "${log_path}"
}

centroid_path_for() {
  local k="$1"
  local v="$2"
  echo "serialization_dir/qwen35_4b_lut_mlp_kmeans_n${NSAMPLE}_v${v}_c${k}.pt"
}

ppl_dir_for() {
  local k="$1"
  local v="$2"
  echo "serialization_dir/qwen35_4b_lut_mlp_ppl_eval_kmeans_v${v}_c${k}"
}

run_kmeans_ppl() {
  local name="$1"
  local k="$2"
  local v="$3"
  local centroid_path
  centroid_path="$(centroid_path_for "${k}" "${v}")"
  local output_dir
  output_dir="$(ppl_dir_for "${k}" "${v}")"
  local log_path="${LOG_DIR}/${name}_v${v}_c${k}_kmeans_ppl.log"

  if [[ ! -f "${centroid_path}" ]]; then
    echo "missing centroid file for ${name} K=${k} V=${v}: ${centroid_path}" >&2
    return 1
  fi

  if [[ "${FORCE_PPL}" != "1" && -f "${log_path}" ]] && grep -q "LUT eval loss" "${log_path}"; then
    echo "skip kmeans ppl ${name} K=${k} V=${v}: ${log_path} already has LUT eval loss"
    return
  fi

  run_logged "${log_path}" \
    "${ACCELERATE_BIN}" launch \
      --num_processes 1 --num_machines 1 --mixed_precision no --dynamo_backend no \
      examples/run_luterize_causal_lm_no_trainer.py \
      --model_name_or_path "${MODEL_PATH}" \
      --tokenized_dataset_path "${TOKENIZED_DATASET_PATH}" \
      --max_train_samples "${MAX_TRAIN_SAMPLES}" \
      --max_eval_samples "${MAX_EVAL_SAMPLES}" \
      --dataset_seed 42 \
      --target_modules mlp \
      --max_seq_length 2048 \
      --per_device_train_batch_size 1 \
      --per_device_eval_batch_size 1 \
      --gradient_accumulation_steps 8 \
      --max_train_steps 0 \
      --eval_steps 0 \
      --eval_logging_steps 1 \
      --max_eval_batches "${PPL_MAX_EVAL_BATCHES}" \
      --vec_len "${v}" \
      --ncentroid "${k}" \
      --torch_dtype bfloat16 \
      --learning_rate "${LEARNING_RATE}" \
      --reconstruct_rate "${RECONSTRUCT_RATE}" \
      --centroid_path "${centroid_path}" \
      --baseline_eval_before_lut \
      --output_dir "${output_dir}"
}

for config in "${CONFIGS[@]}"; do
  IFS=":" read -r name k v <<< "${config}"
  run_kmeans_ppl "${name}" "${k}" "${v}"
done
