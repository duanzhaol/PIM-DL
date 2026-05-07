# Qwen3 LUT 校准与评测记录

本文档记录当前 Qwen3 LUT 校准路径、主要脚本、复现实验命令和最新 sanity 结果。当前实现是 PyTorch-only 模拟，不依赖 PIM 硬件。

## 当前代码入口

- `model-calibration/LUTNeuro/LUTLinear_t.py`
  - Qwen 兼容的 LUT linear 层。
  - 支持 causal LM 中常见的 2D 和 3D activation 输入。
  - `distance_p=2.0` 时使用 squared-L2 最近 centroid fast path。
  - `model.eval()` 时跳过训练专用的 `soft_output` 和 `lut_loss` 计算。
  - 支持 `eval_compute_dtype="model"`，用于 bf16/fp16 的更快评测模拟。
- `model-calibration/LUTNeuro/module_filter.py`
  - 按 Qwen 模块名筛选 `mlp`、`attention`、`all`。
  - 默认排除 `lm_head`。
- `model-calibration/examples/collect_qwen3_lut_centroids.py`
  - 从 raw 或 tokenized LM 数据中收集 activation，并做 KMeans centroid 初始化。
- `model-calibration/examples/run_luterize_causal_lm_no_trainer.py`
  - causal LM 的 LUT 校准和 PPL 评测脚本。
  - 支持 `--eval_only`；`--max_train_steps 0` 也会在一次 LUT eval 后退出。
  - `--baseline_eval_before_lut` 会在 LUT 替换前报告原模型 loss/PPL。
- `model-calibration/examples/run_lm_eval_lut.py`
  - 将内存中的 LUT 模型包装成 lm-eval `HFLM`。
  - 支持 `--disable_lut` 跑 baseline。
  - 支持 `--lut_eval_compute_dtype model` 做更快的 LUT eval 模拟。

## 当前主配置

目前 sanity 结果最好的配置是：

```text
model: /root/models/Qwen3-4B
dataset: /root/fineweb-edu/sample/10BT-tokenized-qwen3-2048
target_modules: mlp
vec_len: 2
ncentroid: 32
centroid_path: serialization_dir/qwen3_4b_lut_mlp_kmeans_n8_v2_c32.pt
torch_dtype: bfloat16
```

## 收集 KMeans Centroid

```bash
cd /root/PIM-DL-ASPLOS/model-calibration

accelerate launch examples/collect_qwen3_lut_centroids.py \
  --model_name_or_path ~/models/Qwen3-4B/ \
  --tokenized_dataset_path /root/fineweb-edu/sample/10BT-tokenized-qwen3-2048 \
  --max_samples 8 \
  --dataset_seed 42 \
  --target_modules mlp \
  --max_seq_length 2048 \
  --per_device_batch_size 1 \
  --nsample 8 \
  --max_vectors_per_module 16384 \
  --vec_len 2 \
  --ncentroid 32 \
  --kmeans_iter 20 \
  --torch_dtype bfloat16 \
  --output_path serialization_dir/qwen3_4b_lut_mlp_kmeans_n8_v2_c32.pt
```

## FineWeb-Edu PPL 评测

使用 `--eval_only` 可以只跑一次 baseline eval 和一次 LUT eval：

```bash
cd /root/PIM-DL-ASPLOS/model-calibration

accelerate launch examples/run_luterize_causal_lm_no_trainer.py \
  --model_name_or_path ~/models/Qwen3-4B/ \
  --tokenized_dataset_path /root/fineweb-edu/sample/10BT-tokenized-qwen3-2048 \
  --max_train_samples 10000 \
  --max_eval_samples 512 \
  --dataset_seed 42 \
  --target_modules mlp \
  --max_seq_length 2048 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --eval_only \
  --eval_logging_steps 1 \
  --max_eval_batches 50 \
  --vec_len 2 \
  --ncentroid 32 \
  --torch_dtype bfloat16 \
  --learning_rate 1e-3 \
  --reconstruct_rate 1e-3 \
  --centroid_path serialization_dir/qwen3_4b_lut_mlp_kmeans_n8_v2_c32.pt \
  --baseline_eval_before_lut \
  --output_dir serialization_dir/qwen3_4b_lut_mlp_ppl_eval_v2_c32
```

50 个 eval batch 上的最新结果：

| 模型 | Loss | PPL |
| --- | ---: | ---: |
| Qwen3-4B baseline | 2.628071 | 13.847037 |
| LUT MLP, vec_len=2, ncentroid=32 | 4.660872 | 105.728235 |

派生指标：

```text
loss delta = 2.032801
PPL ratio = 7.64x
```

## MMLU-Pro 评测

Baseline 命令：

```bash
cd /root/PIM-DL-ASPLOS/model-calibration

python examples/run_lm_eval_lut.py \
  --model_name_or_path ~/models/Qwen3-4B/ \
  --tasks mmlu_pro \
  --limit 10 \
  --batch_size 1 \
  --disable_lut \
  --torch_dtype bfloat16 \
  --output_path serialization_dir/lm_eval_mmlu_pro_baseline_limit10.json
```

可对标的 LUT 命令：

```bash
python examples/run_lm_eval_lut.py \
  --model_name_or_path ~/models/Qwen3-4B/ \
  --tasks mmlu_pro \
  --limit 10 \
  --batch_size 1 \
  --target_modules mlp \
  --vec_len 2 \
  --ncentroid 32 \
  --centroid_path serialization_dir/qwen3_4b_lut_mlp_kmeans_n8_v2_c32.pt \
  --torch_dtype bfloat16 \
  --lut_eval_compute_dtype model \
  --output_path serialization_dir/lm_eval_mmlu_pro_lut_v2_c32_fast_limit10.json
```

最新 140 request 结果：

| 模型 | MMLU-Pro score |
| --- | ---: |
| Qwen3-4B baseline | 0.5785714285714286 |
| LUT MLP | 0.04285714285714286 |

`mmlu_pro` 是 14 个子任务组成的 group，因此 `--limit 10` 实际是 140 个请求。默认 MMLU-Pro 是 5-shot，并且 `max_gen_toks=2048`，在当前 PyTorch LUT 模拟下会很慢。

如果只是做诊断，可以在 baseline 和 LUT 命令中同时加入相同的生成长度限制：

```bash
--gen_kwargs '{"max_gen_toks":128,"until":["Question:"],"do_sample":false}'
```

## 当前判断

当前 LUT 代码已经能完成 KMeans 初始化、FineWeb-Edu PPL 评测和 lm-eval MMLU-Pro 评测，但 MLP-only LUT 的质量还明显不够。FineWeb-Edu PPL 从 13.85 劣化到 105.73，能够解释 MMLU-Pro 从 0.5786 掉到 0.0429 的现象。

后续应优先解决精度问题，而不是扩大评测规模：

- 增强 centroid 初始化或增加有效容量。
- 检查 centroid selection 的梯度路径。
- 在 MLP 质量改善前，暂缓扩大到 attention/MLP 混合替换。
- 正式报告时同时记录 fp32 eval simulation 和 `--lut_eval_compute_dtype model` fast eval 的差异。

