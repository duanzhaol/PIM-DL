#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ACCELERATE_BIN="${ACCELERATE_BIN:-/root/miniconda3/envs/luturbo/bin/accelerate}"
MODEL_PATH="${MODEL_PATH:-/root/models/Qwen3.5-4B}"
TOKENIZED_DATASET_PATH="${TOKENIZED_DATASET_PATH:-/root/fineweb-edu/sample/10BT-tokenized-qwen35-2048-10k}"

NSAMPLE="${NSAMPLE:-8}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-10000}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-512}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-20}"
PPL_MAX_EVAL_BATCHES="${PPL_MAX_EVAL_BATCHES:-50}"

TRAIN_STEPS="${TRAIN_STEPS:-100}"
TRAIN_LR="${TRAIN_LR:-1e-5}"
TRAIN_LR_TAG="${TRAIN_LR_TAG:-lr1e5}"
TRAIN_WARMUP_STEPS="${TRAIN_WARMUP_STEPS:-10}"
TRAIN_RECONSTRUCT_RATE="${TRAIN_RECONSTRUCT_RATE:-1e-3}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
WEIGHT_TRAINABLE_SCOPE="${WEIGHT_TRAINABLE_SCOPE:-lut_modules}"
CENTROID_REQUIRES_GRAD="${CENTROID_REQUIRES_GRAD:-1}"

LOG_DIR="${LOG_DIR:-serialization_dir/logs/qwen35_pareto_weight_finetune}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_PPL="${FORCE_PPL:-0}"
PHASES="${PHASES:-all}"
CONFIG_FILTER="${CONFIG_FILTER:-all}"

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

config_enabled() {
  local name="$1"
  [[ "${CONFIG_FILTER}" == "all" || ",${CONFIG_FILTER}," == *",${name},"* ]]
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
  echo "serialization_dir/qwen35_4b_lut_mlp_weight_ft_n${NSAMPLE}_v${v}_c${k}_${TRAIN_LR_TAG}_steps${TRAIN_STEPS}"
}

ppl_dir_for() {
  local k="$1"
  local v="$2"
  echo "serialization_dir/qwen35_4b_lut_mlp_ppl_eval_weight_ft_v${v}_c${k}"
}

has_full_checkpoint() {
  local output_dir="$1"
  [[ -f "${output_dir}/full_lut_model_state.pt" || -f "${output_dir}/model.safetensors" || -f "${output_dir}/model.safetensors.index.json" || -f "${output_dir}/pytorch_model.bin" || -f "${output_dir}/pytorch_model.bin.index.json" ]]
}

centroid_grad_arg() {
  if [[ "${CENTROID_REQUIRES_GRAD}" == "1" ]]; then
    echo "--centroid_requires_grad"
  else
    echo "--no-centroid_requires_grad"
  fi
}

run_weight_finetune() {
  local name="$1"
  local k="$2"
  local v="$3"
  local centroid_path
  centroid_path="$(centroid_path_for "${k}" "${v}")"
  local output_dir
  output_dir="$(train_dir_for "${k}" "${v}")"
  local log_path="${LOG_DIR}/${name}_v${v}_c${k}_weight_finetune.log"

  if [[ ! -f "${centroid_path}" ]]; then
    echo "missing centroid file for ${name} K=${k} V=${v}: ${centroid_path}" >&2
    return 1
  fi

  if [[ "${FORCE_TRAIN}" != "1" ]] && has_full_checkpoint "${output_dir}"; then
    echo "skip weight finetune ${name} K=${k} V=${v}: ${output_dir} already has a full checkpoint"
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
      --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
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
      --weight_requires_grad \
      --weight_trainable_scope "${WEIGHT_TRAINABLE_SCOPE}" \
      "$(centroid_grad_arg)" \
      --save_full_lut_model \
      --output_dir "${output_dir}"
}

run_weight_finetuned_ppl() {
  local name="$1"
  local k="$2"
  local v="$3"
  local train_dir
  train_dir="$(train_dir_for "${k}" "${v}")"
  local output_dir
  output_dir="$(ppl_dir_for "${k}" "${v}")"
  local log_path="${LOG_DIR}/${name}_v${v}_c${k}_weight_finetuned_ppl.log"

  if ! has_full_checkpoint "${train_dir}"; then
    echo "missing full checkpoint for ${name} K=${k} V=${v}: ${train_dir}" >&2
    return 1
  fi

  if [[ "${FORCE_PPL}" != "1" && -f "${log_path}" ]] && grep -q "LUT eval loss" "${log_path}"; then
    echo "skip weight-finetuned ppl ${name} K=${k} V=${v}: ${log_path} already has LUT eval loss"
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
      --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
      --max_train_steps 0 \
      --eval_steps 0 \
      --eval_logging_steps 1 \
      --max_eval_batches "${PPL_MAX_EVAL_BATCHES}" \
      --vec_len "${v}" \
      --ncentroid "${k}" \
      --torch_dtype bfloat16 \
      --learning_rate "${TRAIN_LR}" \
      --reconstruct_rate "${TRAIN_RECONSTRUCT_RATE}" \
      --resume_from_lut_model "${train_dir}" \
      --baseline_eval_before_lut \
      --output_dir "${output_dir}"
}

for config in "${CONFIGS[@]}"; do
  IFS=":" read -r name k v <<< "${config}"
  if ! config_enabled "${name}"; then
    echo "skip config ${name}: CONFIG_FILTER=${CONFIG_FILTER}"
    continue
  fi
  if phase_enabled "train"; then
    run_weight_finetune "${name}" "${k}" "${v}"
  fi
  if phase_enabled "ppl"; then
    run_weight_finetuned_ppl "${name}" "${k}" "${v}"
  fi
done
