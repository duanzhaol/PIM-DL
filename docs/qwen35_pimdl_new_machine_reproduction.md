# Qwen3.5 PIM-DL New Machine Reproduction Guide

本文档说明如何在一台新机器上复现当前 Qwen3.5-4B PIM-DL 实验流程，包括环境配置、数据准备、activation/centroid 收集、centroid 微调、不同在线补偿比例下的 decode-stage D-PPL，以及下游任务 accuracy 测试。

默认示例配置为：

```text
model:          /root/models/Qwen3.5-4B
target_modules: mlp
K / ncentroid:  64
V / vec_len:    32
dataset:        /root/fineweb-edu/sample/10BT-tokenized-qwen35-2048-10k
dppl windows:   /root/PIM-DL-ASPLOS/wikipedia-1k.json
```

如果要换成其他 K/V，只需要改文中的 `K` 和 `V` 环境变量，以及对应输出目录名。

## 1. 代码和路径

推荐目录布局：

```bash
/root/PIM-DL-ASPLOS
/root/PIM-DL-ASPLOS/model-calibration
/root/models/Qwen3.5-4B
/root/fineweb-edu/sample/10BT-tokenized-qwen35-2048-10k
/root/PIM-DL-ASPLOS/wikipedia-1k.json
```

进入工作目录：

```bash
cd /root/PIM-DL-ASPLOS/model-calibration
```

关键脚本：

```text
examples/collect_qwen3_lut_centroids.py
examples/run_luterize_causal_lm_no_trainer.py
examples/run_decode_dppl_lut.py
examples/run_lm_eval_lut.py
```

## 2. Conda 环境

当前实验使用 Conda 环境 `luturbo`。新机器上建议按下面方式创建：

```bash
conda create -n luturbo python=3.12 -y
conda activate luturbo
```

安装 PyTorch。CUDA 版本需要按新机器实际驱动选择，下面是 CUDA 12.6 示例：

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

安装本项目需要的 Python 包：

```bash
python -m pip install \
  accelerate datasets evaluate scikit-learn scipy numpy pandas tqdm \
  transformers huggingface-hub tokenizers safetensors \
  lm-eval
```

如果国内网络不稳定，可以使用镜像源：

```bash
python -m pip install -U transformers huggingface-hub tokenizers safetensors \
  -i https://mirrors.aliyun.com/pypi/simple
```

当前 Qwen3.5 实验机器上可用的版本是：

```text
transformers:      5.9.0
huggingface-hub:   1.16.1
tokenizers:        0.22.2
accelerate:        1.13.0
```

注意：`transformers==5.9.0` 会和某些包的固定依赖冲突，例如 `sglang 0.5.6.post2` 要求 `transformers==4.57.1`。如果新机器还要跑 sglang，建议为 PIM-DL 单独建环境。

验证环境：

```bash
python - <<'PY'
import torch, transformers, accelerate, datasets
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
print("datasets", datasets.__version__)
PY
```

## 3. 数据准备

最稳妥的方法是直接从旧机器拷贝已经 tokenized 好的数据：

```text
/root/fineweb-edu/sample/10BT-tokenized-qwen35-2048-10k
```

这个数据集的口径是：

```text
train:      10000 chunks
validation:   512 chunks
seq_len:     2048
tokenizer:   Qwen3.5-4B tokenizer
```

如果需要在新机器重新生成 tokenized dataset，可以用下面脚本模板。假设原始 parquet 在：

```text
/root/fineweb-edu/sample/10BT/000_00000.parquet
```

生成命令：

```bash
python - <<'PY'
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

model_path = "/root/models/Qwen3.5-4B"
input_file = "/root/fineweb-edu/sample/10BT/000_00000.parquet"
output_dir = "/root/fineweb-edu/sample/10BT-tokenized-qwen35-2048-10k"
seq_len = 2048
target_train = 10000
target_val = 512

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
raw = load_dataset("parquet", data_files=input_file, split="train")

all_ids = []
for row in raw:
    text = row.get("text") or ""
    if not text:
        continue
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    all_ids.extend(ids)
    needed = (target_train + target_val) * seq_len
    if len(all_ids) >= needed:
        break

chunks = [
    {"input_ids": all_ids[i:i + seq_len]}
    for i in range(0, len(all_ids) - seq_len + 1, seq_len)
]
chunks = chunks[: target_train + target_val]
ds = DatasetDict({
    "train": Dataset.from_list(chunks[:target_train]),
    "validation": Dataset.from_list(chunks[target_train:target_train + target_val]),
})
ds.save_to_disk(output_dir)
print(ds)
PY
```

## 4. 公共变量

后续命令都使用这些变量。默认是 `K=64,V=32`：

```bash
cd /root/PIM-DL-ASPLOS/model-calibration
conda activate luturbo

export MODEL_PATH=/root/models/Qwen3.5-4B
export TOKENIZED_DATASET_PATH=/root/fineweb-edu/sample/10BT-tokenized-qwen35-2048-10k
export WINDOWS_PATH=/root/PIM-DL-ASPLOS/wikipedia-1k.json

export K=64
export V=32
export NSAMPLE=8

export ACCELERATE_BIN=/root/miniconda3/envs/luturbo/bin/accelerate
export PYTHON_BIN=/root/miniconda3/envs/luturbo/bin/python

export ACT_CACHE=serialization_dir/qwen35_4b_lut_mlp_activation_cache_n${NSAMPLE}_10k.pt
export CENTROID_PATH=serialization_dir/qwen35_4b_lut_mlp_kmeans_n${NSAMPLE}_v${V}_c${K}.pt
export TRAIN_DIR=serialization_dir/qwen35_4b_lut_mlp_fineweb_kmeans_n${NSAMPLE}_v${V}_c${K}_lr1e4_steps100
export LOG_DIR=serialization_dir/logs/qwen35_new_machine_v${V}_c${K}
mkdir -p "$LOG_DIR"
```

## 5. 收集 activation 并做 KMeans centroid

`collect_qwen3_lut_centroids.py` 会先通过 forward hook 收集目标 Linear 的输入 activation，然后按 `vec_len` 切块，对每个 codebook 做 KMeans，最后保存 centroid 文件。

推荐使用缓存路径 `--activation_cache_path`。第一次运行会生成 cache；后续换 K/V 时可以复用同一个 cache，不需要重新跑模型收集 activation。

```bash
"$ACCELERATE_BIN" launch \
  --num_processes 1 --num_machines 1 --mixed_precision no --dynamo_backend no \
  examples/collect_qwen3_lut_centroids.py \
  --model_name_or_path "$MODEL_PATH" \
  --tokenized_dataset_path "$TOKENIZED_DATASET_PATH" \
  --max_samples 10000 \
  --dataset_seed 42 \
  --target_modules mlp \
  --max_seq_length 2048 \
  --per_device_batch_size 1 \
  --nsample "$NSAMPLE" \
  --max_vectors_per_module 65536 \
  --vec_len "$V" \
  --ncentroid "$K" \
  --kmeans_backend torch-gpu \
  --kmeans_iter 20 \
  --kmeans_codebook_block_size 64 \
  --kmeans_seed 0 \
  --torch_dtype bfloat16 \
  --activation_cache_path "$ACT_CACHE" \
  --output_path "$CENTROID_PATH" \
  2>&1 | tee "$LOG_DIR/collect_centroids.log"
```

如果希望强制重新收集 activation cache，加：

```bash
--overwrite_activation_cache
```

成功后应看到类似：

```text
Saved 96 centroid tensors to serialization_dir/qwen35_4b_lut_mlp_kmeans_n8_v32_c64.pt
```

## 6. Centroid 微调训练

这里训练的是 LUT centroid 参数，不是 full dense model weight。命令如下：

```bash
"$ACCELERATE_BIN" launch \
  --num_processes 1 --num_machines 1 --mixed_precision no --dynamo_backend no \
  examples/run_luterize_causal_lm_no_trainer.py \
  --model_name_or_path "$MODEL_PATH" \
  --tokenized_dataset_path "$TOKENIZED_DATASET_PATH" \
  --max_train_samples 10000 \
  --max_eval_samples 512 \
  --dataset_seed 42 \
  --target_modules mlp \
  --max_seq_length 2048 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_train_steps 100 \
  --eval_steps 20 \
  --logging_steps 10 \
  --eval_logging_steps 1 \
  --microbatch_logging_steps 8 \
  --max_eval_batches 20 \
  --vec_len "$V" \
  --ncentroid "$K" \
  --torch_dtype bfloat16 \
  --learning_rate 1e-4 \
  --num_warmup_steps 5 \
  --reconstruct_rate 1e-3 \
  --centroid_path "$CENTROID_PATH" \
  --output_dir "$TRAIN_DIR" \
  2>&1 | tee "$LOG_DIR/train_centroids.log"
```

成功完成后应有：

```text
$TRAIN_DIR/model_lut_state.pt
$TRAIN_DIR/lut_training_args.json
$TRAIN_DIR/tokenizer.json
```

注意：如果训练中断，没有 `Training complete` 或没有 `model_lut_state.pt`，就不要把该配置视为完整训练过。

## 7. Full-forward PPL sanity

先跑一次 full-forward validation PPL，用于快速 sanity check，不是最终论文里的 decode-stage D-PPL：

```bash
"$ACCELERATE_BIN" launch \
  --num_processes 1 --num_machines 1 --mixed_precision no --dynamo_backend no \
  examples/run_luterize_causal_lm_no_trainer.py \
  --model_name_or_path "$MODEL_PATH" \
  --tokenized_dataset_path "$TOKENIZED_DATASET_PATH" \
  --max_train_samples 10000 \
  --max_eval_samples 512 \
  --dataset_seed 42 \
  --target_modules mlp \
  --max_seq_length 2048 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_train_steps 0 \
  --eval_steps 0 \
  --eval_logging_steps 1 \
  --max_eval_batches 50 \
  --vec_len "$V" \
  --ncentroid "$K" \
  --torch_dtype bfloat16 \
  --learning_rate 1e-4 \
  --reconstruct_rate 1e-3 \
  --centroid_path "$TRAIN_DIR/model_lut_state.pt" \
  --baseline_eval_before_lut \
  --output_dir "serialization_dir/qwen35_4b_lut_mlp_ppl_eval_v${V}_c${K}" \
  2>&1 | tee "$LOG_DIR/full_forward_ppl.log"
```

如果只想评估 KMeans 初始化、不加载训练后 state，把 `--centroid_path` 改成：

```bash
--centroid_path "$CENTROID_PATH"
```

## 8. Decode-stage D-PPL：不同补偿比例

论文里建议使用 decode-stage D-PPL，而不是 full-forward PPL。当前实现支持两种模式：

```text
isolated: 每个 decode token 从 dense anchor state 出发，不传播误差
pathwise: 沿 LUT decode path 前进，误差会传播到后续 KV/cache
```

先跑 dense baseline：

```bash
for MODE in isolated pathwise; do
  "$PYTHON_BIN" examples/run_decode_dppl_lut.py \
    --model_name_or_path "$MODEL_PATH" \
    --windows_path "$WINDOWS_PATH" \
    --prompt_len 512 \
    --decode_len 512 \
    --max_windows 2 \
    --dppl_mode "$MODE" \
    --method dense \
    --torch_dtype bfloat16 \
    --device cuda \
    --output_path "serialization_dir/dppl_dense_qwen35_${MODE}_w2.json" \
    2>&1 | tee "$LOG_DIR/dppl_dense_${MODE}.log"
done
```

再跑 PIM-DL + 在线补偿。ratio 建议固定为：

```text
0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50
```

命令：

```bash
for RATIO in 0 0.02 0.05 0.10 0.20 0.30 0.40 0.50; do
  RTAG=$(printf "%02d" "$(python - <<PY
r=float("$RATIO")
print(round(r*100))
PY
)")
  for MODE in isolated pathwise; do
    "$PYTHON_BIN" examples/run_decode_dppl_lut.py \
      --model_name_or_path "$MODEL_PATH" \
      --windows_path "$WINDOWS_PATH" \
      --prompt_len 512 \
      --decode_len 512 \
      --max_windows 2 \
      --dppl_mode "$MODE" \
      --method pimdl_lut \
      --target_modules mlp \
      --vec_len "$V" \
      --ncentroid "$K" \
      --centroid_path "$TRAIN_DIR/model_lut_state.pt" \
      --lut_eval_compute_dtype model \
      --residual_compensation_ratio "$RATIO" \
      --residual_compensation_metric abs \
      --torch_dtype bfloat16 \
      --device cuda \
      --output_path "serialization_dir/dppl_qwen35_v${V}_c${K}_comp${RTAG}_${MODE}_w2.json" \
      2>&1 | tee "$LOG_DIR/dppl_v${V}_c${K}_comp${RTAG}_${MODE}.log"
  done
done
```

如需对比“只有在线补偿、无 LUT 查表贡献”，把 `--method pimdl_lut` 改为：

```bash
--method activation_topk_only
```

同时输出文件名建议改成：

```text
dppl_qwen35_topk_only_comp${RTAG}_${MODE}_w2.json
```

## 9. 汇总 D-PPL 结果

每个 D-PPL JSON 里重点看：

```json
{
  "nll": ...,
  "d_ppl": ...,
  "tokens": ...
}
```

可以用下面脚本汇总：

```bash
python - <<'PY'
import csv, json
from pathlib import Path

dense_path = Path("serialization_dir/dppl_dense_qwen35_isolated_w2.json")
dense = json.load(open(dense_path))
dense_nll = dense["nll"]
dense_dppl = dense["d_ppl"]

rows = []
for path in sorted(Path("serialization_dir").glob("dppl_qwen35_v*_c*_comp*_*.json")):
    data = json.load(open(path))
    name = path.stem
    rows.append({
        "file": path.name,
        "nll": data["nll"],
        "d_ppl": data["d_ppl"],
        "tokens": data["tokens"],
        "dense_nll": dense_nll,
        "dense_d_ppl": dense_dppl,
        "delta_nll": data["nll"] - dense_nll,
        "d_ppl_increase_pct": (data["d_ppl"] / dense_dppl - 1.0) * 100.0,
    })

out = Path("serialization_dir/dppl_qwen35_summary.csv")
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(out)
PY
```

## 10. 下游任务 accuracy：lm-eval / MMLU-Pro

正式 accuracy 建议使用 `lm-evaluation-harness`，入口是：

```text
examples/run_lm_eval_lut.py
```

先跑 dense baseline：

```bash
"$PYTHON_BIN" examples/run_lm_eval_lut.py \
  --model_name_or_path "$MODEL_PATH" \
  --tasks mmlu_pro \
  --limit 10 \
  --batch_size 1 \
  --disable_lut \
  --torch_dtype bfloat16 \
  --gen_kwargs '{"max_gen_toks":128,"until":["Question:"],"do_sample":false}' \
  --output_path "serialization_dir/lm_eval_mmlu_pro_qwen35_dense_limit10.json" \
  2>&1 | tee "$LOG_DIR/lm_eval_dense_mmlu_pro_limit10.log"
```

再跑 PIM-DL + 在线补偿。为了和 D-PPL 对齐，建议同样扫：

```text
0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50
```

Smoke 测试时可以先设 `LIMIT=2` 或 `LIMIT=5`；正式表格再把 `LIMIT` 提高或去掉。

```bash
export LIMIT=10

for RATIO in 0 0.02 0.05 0.10 0.20 0.30 0.40 0.50; do
  RTAG=$(printf "%02d" "$(python - <<PY
r=float("$RATIO")
print(round(r*100))
PY
)")
  "$PYTHON_BIN" examples/run_lm_eval_lut.py \
    --model_name_or_path "$MODEL_PATH" \
    --tasks mmlu_pro \
    --limit "$LIMIT" \
    --batch_size 1 \
    --target_modules mlp \
    --vec_len "$V" \
    --ncentroid "$K" \
    --centroid_path "$TRAIN_DIR/model_lut_state.pt" \
    --lut_eval_compute_dtype model \
    --residual_compensation_ratio "$RATIO" \
    --residual_compensation_metric abs \
    --torch_dtype bfloat16 \
    --gen_kwargs '{"max_gen_toks":128,"until":["Question:"],"do_sample":false}' \
    --output_path "serialization_dir/lm_eval_mmlu_pro_qwen35_v${V}_c${K}_comp${RTAG}_limit${LIMIT}.json" \
    2>&1 | tee "$LOG_DIR/lm_eval_v${V}_c${K}_comp${RTAG}_mmlu_pro_limit${LIMIT}.log"
done
```

注意：

- `mmlu_pro` 是多个子任务组成的 group，`--limit 10` 不是总共 10 条，而是每个子任务 10 条。
- baseline 和 PIM-DL 必须使用相同 `limit`、`num_fewshot`、`gen_kwargs`。
- 如果要正式报告，请把 `--limit` 去掉或设为足够大；如果只是 smoke，可先用 `--limit 2` 或 `--limit 5`。
- 当前 PyTorch LUT 模拟端到端生成非常慢，建议先用小 limit 验证流程。

结果 JSON 里重点看：

```text
results.mmlu_pro["exact_match,custom-extract"]
```

也可以快速打印：

```bash
python - <<'PY'
import csv, json
from pathlib import Path

rows = []
for p in Path("serialization_dir").glob("lm_eval_mmlu_pro_qwen35*.json"):
    data = json.load(open(p))
    result = data.get("groups", {}).get("mmlu_pro") or data.get("results", {}).get("mmlu_pro")
    acc = result.get("exact_match,custom-extract") if result else None
    print(p.name, acc if acc is not None else "missing")
    rows.append({"file": p.name, "mmlu_pro_exact_match_custom_extract": acc})

out = Path("serialization_dir/lm_eval_mmlu_pro_qwen35_summary.csv")
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["file", "mmlu_pro_exact_match_custom_extract"])
    writer.writeheader()
    writer.writerows(rows)
print(out)
PY
```

## 11. 推荐复现顺序

在新机器上建议按这个顺序执行：

1. 配好 Conda 环境，确认 `torch.cuda.is_available()` 为 `True`。
2. 准备 Qwen3.5-4B 模型和 tokenized dataset。
3. 运行第 5 节，生成 activation cache 和 `K=64,V=32` centroid。
4. 运行第 6 节，训练 centroid，确认 `model_lut_state.pt` 存在。
5. 运行第 7 节，做 full-forward PPL sanity。
6. 运行第 8 节，做 isolated/pathwise D-PPL ratio sweep。
7. 运行第 10 节，做 lm-eval accuracy。
8. 运行第 9 节和第 10 节末尾脚本，汇总 CSV/accuracy。

## 12. 常见问题

### `.pt` centroid 不能用 safetensors 读取

`.pt` 文件应通过 `torch.load` 读取。`run_luterize_causal_lm_no_trainer.py` 当前会按扩展名区分 `.pt` 和 `.safetensors`。

### Qwen3.5 加载时提示 fast path unavailable

可能看到：

```text
The fast path is not available ... Falling back to torch implementation
```

这是 Qwen3.5 linear attention 相关优化库缺失导致，影响速度，不影响实验口径。

### 训练中断

如果日志没有 `Training complete`，或者输出目录没有 `model_lut_state.pt`，不要把该配置当成训练完成。可以重新运行第 6 节命令覆盖输出。

### D-PPL 与 full-forward PPL 不一致

这是正常的。论文里用于 decode 质量分析的是第 8 节的 decode-stage D-PPL；第 7 节只是快速 sanity。
