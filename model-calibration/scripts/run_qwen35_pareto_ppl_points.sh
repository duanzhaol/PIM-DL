#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ACCELERATE_BIN="${ACCELERATE_BIN:-/root/miniconda3/envs/luturbo/bin/accelerate}"
MODEL_PATH="${MODEL_PATH:-/root/models/Qwen3.5-4B}"
TOKENIZED_DATASET_PATH="${TOKENIZED_DATASET_PATH:-/root/fineweb-edu/sample/10BT-tokenized-qwen35-2048-10k}"

NSAMPLE="${NSAMPLE:-8}"
MAX_VECTORS_PER_MODULE="${MAX_VECTORS_PER_MODULE:-65536}"
KMEANS_BACKEND="${KMEANS_BACKEND:-torch-gpu}"
KMEANS_ITER="${KMEANS_ITER:-20}"
KMEANS_CODEBOOK_BLOCK_SIZE="${KMEANS_CODEBOOK_BLOCK_SIZE:-64}"
KMEANS_SEED="${KMEANS_SEED:-0}"

TRAIN_STEPS="${TRAIN_STEPS:-100}"
TRAIN_LR="${TRAIN_LR:-1e-4}"
TRAIN_LR_TAG="${TRAIN_LR_TAG:-lr1e4}"
TRAIN_WARMUP_STEPS="${TRAIN_WARMUP_STEPS:-5}"
TRAIN_RECONSTRUCT_RATE="${TRAIN_RECONSTRUCT_RATE:-1e-3}"

MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-10000}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-512}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-20}"
PPL_MAX_EVAL_BATCHES="${PPL_MAX_EVAL_BATCHES:-50}"

ACTIVATION_CACHE_PATH="${ACTIVATION_CACHE_PATH:-serialization_dir/qwen35_4b_lut_mlp_activation_cache_n${NSAMPLE}_10k.pt}"
LOG_DIR="${LOG_DIR:-serialization_dir/logs/qwen35_pareto_ppl}"
FORCE_CACHE="${FORCE_CACHE:-0}"
FORCE_CENTROIDS="${FORCE_CENTROIDS:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_PPL="${FORCE_PPL:-0}"
PHASES="${PHASES:-all}"

mkdir -p "${LOG_DIR}"

CONFIGS=(
  "fast:4:128"
  "low_middle:64:32"
  "middle:32:8"
  "quality:128:4"
  "expensive:512:4"
)

phase_enabled() {
  local phase="$1"
  [[ "${PHASES}" == "all" || ",${PHASES}," == *",${phase},"* ]]
}

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

train_dir_for() {
  local k="$1"
  local v="$2"
  echo "serialization_dir/qwen35_4b_lut_mlp_fineweb_kmeans_n${NSAMPLE}_v${v}_c${k}_${TRAIN_LR_TAG}_steps${TRAIN_STEPS}"
}

ppl_dir_for() {
  local k="$1"
  local v="$2"
  echo "serialization_dir/qwen35_4b_lut_mlp_ppl_eval_pareto_v${v}_c${k}"
}

run_centroids() {
  local name="$1"
  local k="$2"
  local v="$3"
  local centroid_path
  centroid_path="$(centroid_path_for "${k}" "${v}")"
  local log_path="${LOG_DIR}/${name}_v${v}_c${k}_centroids.log"

  if [[ "${FORCE_CENTROIDS}" != "1" && -f "${centroid_path}" ]]; then
    echo "skip centroids ${name} K=${k} V=${v}: ${centroid_path} exists"
    return
  fi

  local cache_args=(--activation_cache_path "${ACTIVATION_CACHE_PATH}")
  if [[ "${FORCE_CACHE}" == "1" && ! -f "${ACTIVATION_CACHE_PATH}" ]]; then
    cache_args+=(--overwrite_activation_cache)
  fi

  run_logged "${log_path}" \
    "${ACCELERATE_BIN}" launch \
      --num_processes 1 --num_machines 1 --mixed_precision no --dynamo_backend no \
      examples/collect_qwen3_lut_centroids.py \
      --model_name_or_path "${MODEL_PATH}" \
      --tokenized_dataset_path "${TOKENIZED_DATASET_PATH}" \
      --max_samples "${MAX_TRAIN_SAMPLES}" \
      --dataset_seed 42 \
      --target_modules mlp \
      --max_seq_length 2048 \
      --per_device_batch_size 1 \
      --nsample "${NSAMPLE}" \
      --max_vectors_per_module "${MAX_VECTORS_PER_MODULE}" \
      --vec_len "${v}" \
      --ncentroid "${k}" \
      --kmeans_backend "${KMEANS_BACKEND}" \
      --kmeans_iter "${KMEANS_ITER}" \
      --kmeans_codebook_block_size "${KMEANS_CODEBOOK_BLOCK_SIZE}" \
      --kmeans_seed "${KMEANS_SEED}" \
      --torch_dtype bfloat16 \
      "${cache_args[@]}" \
      --output_path "${centroid_path}"
}

run_train() {
  local name="$1"
  local k="$2"
  local v="$3"
  local centroid_path
  centroid_path="$(centroid_path_for "${k}" "${v}")"
  local output_dir
  output_dir="$(train_dir_for "${k}" "${v}")"
  local state_path="${output_dir}/model_lut_state.pt"
  local log_path="${LOG_DIR}/${name}_v${v}_c${k}_train.log"

  if [[ "${FORCE_TRAIN}" != "1" && -f "${state_path}" ]]; then
    echo "skip train ${name} K=${k} V=${v}: ${state_path} exists"
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
      --max_train_steps "${TRAIN_STEPS}" \
      --eval_steps 20 \
      --logging_steps 10 \
      --eval_logging_steps 1 \
      --microbatch_logging_steps 8 \
      --max_eval_batches "${MAX_EVAL_BATCHES}" \
      --vec_len "${v}" \
      --ncentroid "${k}" \
      --torch_dtype bfloat16 \
      --learning_rate "${TRAIN_LR}" \
      --num_warmup_steps "${TRAIN_WARMUP_STEPS}" \
      --reconstruct_rate "${TRAIN_RECONSTRUCT_RATE}" \
      --centroid_path "${centroid_path}" \
      --output_dir "${output_dir}"
}

run_ppl() {
  local name="$1"
  local k="$2"
  local v="$3"
  local train_dir
  train_dir="$(train_dir_for "${k}" "${v}")"
  local state_path="${train_dir}/model_lut_state.pt"
  local output_dir
  output_dir="$(ppl_dir_for "${k}" "${v}")"
  local log_path="${LOG_DIR}/${name}_v${v}_c${k}_ppl.log"

  if [[ "${FORCE_PPL}" != "1" && -f "${log_path}" ]] && grep -q "LUT eval loss" "${log_path}"; then
    echo "skip ppl ${name} K=${k} V=${v}: ${log_path} already has LUT eval loss"
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
      --learning_rate "${TRAIN_LR}" \
      --reconstruct_rate "${TRAIN_RECONSTRUCT_RATE}" \
      --centroid_path "${state_path}" \
      --baseline_eval_before_lut \
      --output_dir "${output_dir}"
}

for config in "${CONFIGS[@]}"; do
  IFS=":" read -r name k v <<< "${config}"
  if phase_enabled "centroids"; then
    run_centroids "${name}" "${k}" "${v}"
  fi
  if phase_enabled "train"; then
    run_train "${name}" "${k}" "${v}"
  fi
  if phase_enabled "ppl"; then
    run_ppl "${name}" "${k}" "${v}"
  fi
done
