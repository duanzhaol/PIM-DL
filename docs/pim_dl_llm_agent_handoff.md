# PIM-DL LLM Adaptation Agent Handoff

本文档面向后续接手本仓库的 Agent，目标是快速说明当前分支在原始 PIM-DL 代码上的大模型适配工作、环境配置方式、关键脚本入口和复现实验命令。

## 1. 当前工作概览

原始 PIM-DL 仓库主要围绕 DNN/Transformer encoder 场景，例如 BERT、ViT、GLUE/MLM/image classification，核心思路是把 Linear/GEMM 转换为 centroid + lookup table 的 LUT-NN 形式。我们在此基础上做了 Qwen3 causal LM 适配、lm-eval 评测、KMeans centroid 初始化、单层精度/性能扫描和若干调试工具。

当前主要工作目录是：

```bash
/root/PIM-DL-ASPLOS/model-calibration
```

当前主要模型和数据路径是：

```text
model:   /root/models/Qwen3-4B/
dataset: /root/fineweb-edu/sample/10BT-tokenized-qwen3-2048
```

当前实现仍是 PyTorch 模拟，不是 PIM 硬件 kernel。尤其要注意：`LUTLinear_t.forward()` 当前在运行时选择最近 centroid 后，仍然把量化后的 activation 与原始权重做 `matmul`；它不是完整在线查表 kernel。真正预计算 LUT 并查表的逻辑目前只在 CPU benchmark 和单层 MSE 分析脚本里实现。

## 2. 我们实现了什么

### 2.1 Qwen3/Causal LM 训练与评测入口

新增/扩展入口：

```text
model-calibration/examples/run_luterize_causal_lm_no_trainer.py
```

能力：

- 支持 `AutoModelForCausalLM`，用于 Qwen3-4B 这类 decoder-only LM。
- 支持 tokenized dataset from disk：`--tokenized_dataset_path`。
- 支持 `target_modules=mlp|attention|all`，默认只替换 MLP。
- 支持 `--baseline_eval_before_lut`，在 LUT 替换前先评测原模型 PPL。
- 支持 `--eval_only` 或 `--max_train_steps 0`，只跑 LUT eval，不训练。
- 支持 `--centroid_path` 加载预先 KMeans 得到的 centroid。
- 支持 `--eval_logging_steps` 和 `--microbatch_logging_steps`，便于长任务中观察每个 eval batch 和 microbatch loss。
- 支持 gradient checkpointing，且使用 `use_reentrant=False` 并调用 `enable_input_require_grads()`，避免冻结权重时 centroid 收不到梯度。
- 保存 LUT 状态到 `output_dir/model_lut_state.pt`，并保存 tokenizer 与 `lut_training_args.json`。

训练目标目前是：

```text
total_loss = model_loss + reconstruct_rate * lut_loss
```

其中 `model_loss` 是 LM loss，`lut_loss` 是 soft output 与 quantized output 的重构项。当前训练效果有限，完整模型上还未达到可用精度。

### 2.2 LUT Linear 层适配

核心文件：

```text
model-calibration/LUTNeuro/LUTLinear_t.py
```

当前能力：

- 支持 2D 输入 `[batch, hidden]` 和 3D 输入 `[batch, seq, hidden]`。
- Qwen causal LM 的 activation 形状可以正常通过。
- 支持 `distance_p=2.0` 的 squared-L2 fast path，避免直接 `torch.cdist` 的额外开销。
- eval 时可用 `eval_compute_dtype="model"`，让 eval 使用模型 dtype，例如 bf16，以便加速模拟。
- train 时使用 STE 形式：

```python
quant_output = soft_output + (quant_output - soft_output).detach()
```

重要限制：

- 当前 eval forward 不是论文里的预计算 LUT 查表，它是 nearest-centroid quantization 后再 matmul dense weight。
- 如果后续目标是端到端在线 LUT 推理，需要把 `LUTLinear_t.forward()` 改为预计算/加载 LUT，并执行 table gather + accumulation。

### 2.3 模块筛选

核心文件：

```text
model-calibration/LUTNeuro/module_filter.py
```

能力：

- 按模块名选择 Qwen attention 或 MLP。
- 默认排除 `lm_head`。
- 配合 `LUTerize` 替换 `nn.Linear`。

常用配置：

```text
--target_modules mlp
```

### 2.4 KMeans centroid 收集

核心文件：

```text
model-calibration/examples/collect_qwen3_lut_centroids.py
```

能力：

- 对目标 Linear 注册 forward hook。
- 收集 Linear 输入 activation。
- 按 `vec_len` 切成 sub-vector。
- 对每个 codebook 用 `MiniBatchKMeans` 得到 `ncentroid` 个 centroid。
- 输出 `.pt` 或 `.safetensors`。

输出 key 形如：

```text
model.layers.18.mlp.up_proj.centroids.weight
```

注意：

- `.pt` 文件必须用 `torch.load` 读，`.safetensors` 才用 `safetensors.torch.load_file`。
- 曾经把 `.pt` 当 safetensors 读会报 `SafetensorError: header too large`，现在脚本按扩展名区分。

### 2.5 lm-eval 统一评测入口

核心文件：

```text
model-calibration/examples/run_lm_eval_lut.py
```

能力：

- 加载 baseline Qwen3 或 LUT 替换后的 Qwen3。
- 包成 lm-evaluation-harness 的 `HFLM`。
- 支持 `--disable_lut` 跑 baseline。
- 支持 `--tasks mmlu_pro`。
- 支持 `--gen_kwargs` 限制生成长度，避免 MMLU-Pro generation 太慢或复读。
- 支持 `--lut_eval_compute_dtype model` 加速 LUT eval 模拟。

当前 MMLU-Pro 是 14 个子任务组成的 group，所以 `--limit 10` 实际会产生约 140 个请求。

### 2.6 CPU LUT kernel 性能基准

核心文件：

```text
model-calibration/examples/benchmark_lut_cpu_kernel.py
```

能力：

- 实现一个理想化 CPU LUT-NN kernel：nearest-centroid search + LUT gather + accumulation。
- 默认使用 Qwen3-4B MLP 形状：
  - `hidden_size=2560`
  - `intermediate_size=9728`
  - `gate_proj/up_proj`: `2560 -> 9728`，计数 2
  - `down_proj`: `9728 -> 2560`，计数 1
- 输出每个 module 和 `mlp_weighted_avg` 的性能。
- 记录理论 proxy：
  - `lut_storage_ratio`
  - `online_read_ratio`
  - `lut_table_mib`
  - `lut_median_ms`
  - `dense_median_ms`

完整 sweep 结果路径：

```text
model-calibration/serialization_dir/lut_cpu_kernel_qwen3_mlp_k2_to_1024_v2_to_128_bt1_f32.csv
model-calibration/serialization_dir/lut_cpu_kernel_qwen3_mlp_weighted_avg_k2_to_1024_v2_to_128_bt1_f32.csv
```

### 2.7 单层 LUT 精度扫描

核心文件：

```text
model-calibration/examples/collect_single_layer_lut_cache.py
model-calibration/examples/evaluate_single_layer_lut_mse.py
model-calibration/examples/aggregate_single_layer_lut_mse.py
model-calibration/scripts/run_single_layer_lut_mse_sweep.sh
```

用途：

- 不跑完整模型，只选一个 Linear 层。
- 收集该层 activation 和 weight。
- 对不同 `K=ncentroid`、`V=vec_len` 做 KMeans。
- 预计算 LUT。
- 对比 dense output 与 LUT output 的 MSE/relative MSE。

当前默认层：

```text
model.layers.18.mlp.up_proj
```

当前完整 100k calibration token sweep 结果：

```text
model-calibration/serialization_dir/single_layer_lut_mse_up_proj_100k/results.csv
```

cache 大小约：

```text
1.2G model-calibration/serialization_dir/single_layer_lut_mse_up_proj_100k
```

### 2.8 测试覆盖

新增/扩展测试：

```text
model-calibration/tests/test_causal_lm_script.py
model-calibration/tests/test_centroid_collection.py
model-calibration/tests/test_lm_eval_lut.py
model-calibration/tests/test_lut_cpu_kernel_benchmark.py
model-calibration/tests/test_single_layer_lut_mse.py
model-calibration/tests/test_aggregate_single_layer_lut_mse.py
```

覆盖范围包括：

- Qwen3 causal LM script helper。
- centroid `.pt` 加载。
- gradient checkpointing 下 centroid 梯度。
- lm-eval wrapper 参数与 baseline/LUT 分支。
- CPU LUT kernel correctness smoke。
- 单层 MSE cache/eval/aggregate smoke。
- 所有 example script 的 `--help` 可直接从 `model-calibration` 目录运行。

最近一次完整回归结果为：

```text
PYTHONPATH=. pytest tests -q
42 passed
```

## 3. 关键概念和公式

令：

```text
I = Linear 输入维度
O = Linear 输出维度
K = ncentroid
V = vec_len
ncodebooks = I / V
```

PIM-DL/LUT-NN 转换中，每个 codebook 有 `K` 个 centroid，每个 centroid 是长度 `V` 的 activation sub-vector。

预计算 LUT 表大小：

```text
LUT entries = ncodebooks * K * O = (I / V) * K * O
Dense weight entries = I * O
LUT table / dense weight = K / V
```

centroid 存储大小：

```text
centroid entries = ncodebooks * K * V = I * K
centroid / dense weight = K / O
```

粗略在线读数 proxy：

```text
online_read_ratio = centroid_read_ratio + lookup_read_ratio
                  = K / O + 1 / V
```

解释：

- `V` 越小，codebook 数 `I/V` 越多，所以 LUT 表越大。
- 每条 centroid 对应的 LUT 输出向量长度都是 `O`，所以 `V` 小会让 LUT 列数变多。
- nearest-centroid search 对每个 tile 比较 `K` 个 centroid。按点乘乘加数量粗略看，总搜索计算约与 `I*K` 相关，和 `V` 不呈简单 `1/V` 关系。
- 真实性能还受 gather、cache、向量化和内存布局影响，不能只看公式。

## 4. 环境配置

当前已验证环境是 Conda，不是 uv：

```bash
conda activate luturbo
cd /root/PIM-DL-ASPLOS/model-calibration
```

确认过的环境信息：

```text
python:      /root/miniconda3/envs/luturbo/bin/python
Python:      3.12.12
torch:       2.9.1+cu128
transformers 4.57.1
accelerate:  1.13.0
datasets:    4.4.2
scikit-learn 1.8.0
lm_eval:     0.4.11
```

关键 pip 包版本：

```text
accelerate==1.13.0
datasets==4.4.2
evaluate==0.4.6
huggingface-hub==0.36.0
lm_eval==0.4.11
numpy==2.4.0
pandas==2.3.3
protobuf==6.33.2
safetensors==0.7.0
scikit-learn==1.8.0
scipy==1.16.3
sentencepiece==0.2.1
tokenizers==0.22.1
torch==2.9.1
tqdm==4.67.1
transformers==4.57.1
```

从新机器重建环境时，可按以下方式开始：

```bash
conda create -n luturbo python=3.12 -y
conda activate luturbo

pip install \
  torch==2.9.1 \
  transformers==4.57.1 \
  accelerate==1.13.0 \
  datasets==4.4.2 \
  scikit-learn==1.8.0 \
  safetensors==0.7.0 \
  lm_eval==0.4.11 \
  evaluate==0.4.6 \
  sentencepiece==0.2.1 \
  protobuf==6.33.2 \
  pandas==2.3.3 \
  tqdm==4.67.1

cd /root/PIM-DL-ASPLOS/model-calibration
pip install -e .
```

如果不执行 `pip install -e .`，也可以在运行测试时显式设置：

```bash
PYTHONPATH=. pytest tests -q
```

注意：

- `accelerate launch` 如果没有配置，会打印 `num_processes/mixed_precision/dynamo_backend` 默认值警告，这不影响单卡运行。
- 当前大模型实验依赖本地模型和本地 tokenized dataset；这些不在 git 中。
- `serialization_dir/` 里保存了大量中间结果，一般不要提交。

## 5. 如何运行

以下命令默认从 `model-calibration` 目录执行：

```bash
cd /root/PIM-DL-ASPLOS/model-calibration
conda activate luturbo
```

### 5.1 基础回归测试

```bash
PYTHONPATH=. pytest tests -q
```

### 5.2 收集 Qwen3 MLP centroid

推荐先用较小 `nsample` 做 smoke，再扩大：

```bash
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

当前已有 centroid：

```text
serialization_dir/qwen3_4b_lut_mlp_kmeans_n8_v2_c32.pt
```

文件大小约 66 MiB。

### 5.3 FineWeb-Edu PPL baseline + LUT eval

只评测，不训练：

```bash
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

当前 50 eval batch 结果：

| Model | Loss | PPL |
| --- | ---: | ---: |
| Qwen3-4B baseline | 2.628071 | 13.847037 |
| LUT MLP, V=2, K=32 | 4.660872 | 105.728235 |

### 5.4 短训练/校准

示例：

```bash
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
  --max_train_steps 40 \
  --eval_steps 10 \
  --logging_steps 5 \
  --eval_logging_steps 1 \
  --microbatch_logging_steps 8 \
  --max_eval_batches 20 \
  --vec_len 2 \
  --ncentroid 32 \
  --torch_dtype bfloat16 \
  --learning_rate 1e-4 \
  --num_warmup_steps 5 \
  --reconstruct_rate 1e-3 \
  --centroid_path serialization_dir/qwen3_4b_lut_mlp_kmeans_n8_v2_c32.pt \
  --output_dir serialization_dir/qwen3_4b_lut_mlp_fineweb_kmeans_v2_c32_lr1e4_steps40
```

实际观察：短训练没有显著修复 PPL/MMLU 退化。后续若继续训练，应优先检查梯度路径、目标函数和真正 LUT 查表实现，而不是盲目增加 step。

### 5.5 lm-eval MMLU-Pro baseline

```bash
python examples/run_lm_eval_lut.py \
  --model_name_or_path ~/models/Qwen3-4B/ \
  --tasks mmlu_pro \
  --limit 10 \
  --batch_size 1 \
  --disable_lut \
  --torch_dtype bfloat16 \
  --output_path serialization_dir/lm_eval_mmlu_pro_baseline_limit10.json
```

### 5.6 lm-eval MMLU-Pro LUT

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

如果 generation 很慢或复读，baseline 和 LUT 都使用相同 `gen_kwargs`：

```bash
--gen_kwargs '{"max_gen_toks":128,"until":["Question:"],"do_sample":false}'
```

当前 140 request 结果：

| Model | MMLU-Pro score |
| --- | ---: |
| Qwen3-4B baseline | 0.5785714285714286 |
| LUT MLP, V=2, K=32 | 0.04285714285714286 |

### 5.7 CPU kernel 性能 sweep

完整 K/V sweep：

```bash
python examples/benchmark_lut_cpu_kernel.py \
  --ncentroids 2,4,8,16,32,64,128,256,512,1024 \
  --vec_lens 2,4,8,16,32,64,128 \
  --batch_tokens 1 \
  --dtype float32 \
  --warmup 1 \
  --repeats 3 \
  --threads 0 \
  --max_table_gib 64 \
  --output_csv serialization_dir/lut_cpu_kernel_qwen3_mlp_k2_to_1024_v2_to_128_bt1_f32.csv
```

然后从输出 CSV 中筛选 `module=mlp_weighted_avg`，或直接使用已有：

```text
serialization_dir/lut_cpu_kernel_qwen3_mlp_weighted_avg_k2_to_1024_v2_to_128_bt1_f32.csv
```

### 5.8 单层 MSE 精度 sweep

建议先构建 cache，再逐个 K/V 跑。脚本已经做了这件事，并且每个 K/V 都是单独 Python 调用，便于中断后续跑：

```bash
CALIB_TOKENS=100000 \
EVAL_TOKENS=10000 \
OUTPUT_DIR=serialization_dir/single_layer_lut_mse_up_proj_100k \
bash scripts/run_single_layer_lut_mse_sweep.sh
```

如需后台运行：

```bash
mkdir -p logs
tmux new-session -d -s single_layer_mse_100k \
  'cd /root/PIM-DL-ASPLOS/model-calibration && CALIB_TOKENS=100000 EVAL_TOKENS=10000 OUTPUT_DIR=serialization_dir/single_layer_lut_mse_up_proj_100k bash scripts/run_single_layer_lut_mse_sweep.sh >> logs/single_layer_lut_mse_up_proj_100k.log 2>&1'
```

查看进度：

```bash
tail -n 120 logs/single_layer_lut_mse_up_proj_100k.log
find serialization_dir/single_layer_lut_mse_up_proj_100k -maxdepth 1 -name '*.json' | wc -l
```

最终结果：

```text
serialization_dir/single_layer_lut_mse_up_proj_100k/results.csv
```

## 6. 当前主要实验结论

### 6.1 完整模型质量

当前完整 Qwen3-4B MLP-only LUT 的质量不够：

- FineWeb-Edu PPL 从 `13.85` 变为 `105.73`。
- MMLU-Pro score 从 `0.5786` 变为 `0.0429`。

这说明当前方法作为大模型在线推理替换还不可用。

### 6.2 单层精度 tradeoff

单层 `model.layers.18.mlp.up_proj` 上，`relative_mse` 结果显示：

| K | V=2 | V=4 | V=8 | V=16 | V=32 | V=64 | V=128 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.5337 | 0.6754 | 0.7572 | 0.7996 | 0.8266 | 0.8399 | 0.8467 |
| 4 | 0.2836 | 0.5018 | 0.6529 | 0.7370 | 0.7845 | 0.8069 | 0.8207 |
| 8 | 0.1562 | 0.3639 | 0.5556 | 0.6763 | 0.7429 | 0.7779 | 0.7962 |
| 16 | 0.0841 | 0.2664 | 0.4717 | 0.6197 | 0.7052 | 0.7529 | 0.7757 |
| 32 | 0.0444 | 0.1938 | 0.3996 | 0.5678 | 0.6696 | 0.7269 | 0.7554 |
| 64 | 0.0232 | 0.1409 | 0.3392 | 0.5207 | 0.6361 | 0.7009 | 0.7376 |
| 128 | 0.0121 | 0.1020 | 0.2881 | 0.4774 | 0.6055 | 0.6803 | 0.7281 |
| 256 | 0.0066 | 0.0738 | 0.2450 | 0.4378 | 0.5766 | 0.6620 | 0.7317 |
| 512 | 0.0034 | 0.0532 | 0.2079 | 0.4020 | 0.5506 | 0.6459 | 0.7238 |
| 1024 | 0.0018 | 0.0384 | 0.1767 | 0.3693 | 0.5259 | 0.6298 | 0.7680 |

结论：

- `V` 对精度影响极强。
- 小 `V` 精度明显更好，但 LUT 表大小按 `K/V` 增长。
- 单纯增大 `K` 能改善精度，但在大 `V` 下改善有限。

### 6.3 CPU kernel 性能 tradeoff

Qwen3 MLP weighted average 的 `lut_median_ms` 表：

| K | V=2 | V=4 | V=8 | V=16 | V=32 | V=64 | V=128 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 7.005 | 1.809 | 0.996 | 0.645 | 0.413 | 0.316 | 0.254 |
| 4 | 6.891 | 1.800 | 1.016 | 0.642 | 0.374 | 0.329 | 0.230 |
| 8 | 6.993 | 1.920 | 1.049 | 0.655 | 0.472 | 0.317 | 0.248 |
| 16 | 6.953 | 1.836 | 1.038 | 0.691 | 0.416 | 0.325 | 0.252 |
| 32 | 7.098 | 1.910 | 1.119 | 0.720 | 0.474 | 0.338 | 0.277 |
| 64 | 7.203 | 1.995 | 1.128 | 0.736 | 0.456 | 0.372 | 0.272 |
| 128 | 7.506 | 2.173 | 1.224 | 0.843 | 0.538 | 0.435 | 0.332 |
| 256 | 7.983 | 2.539 | 1.493 | 0.962 | 0.622 | 0.488 | 0.442 |
| 512 | 10.510 | 4.368 | 1.793 | 1.255 | 0.790 | 0.655 | 0.561 |
| 1024 | 14.385 | 5.502 | 3.628 | 2.700 | 2.345 | 2.121 | 2.073 |

同一 benchmark 中 dense baseline 约 `0.88 ms`。该 benchmark 是 CPU 上的理想化 PyTorch kernel，不等价于最终 PIM/CPU/CUDA 产品级 kernel。

## 7. 常见问题和排查

### 7.1 `SafetensorError: header too large`

原因通常是把 `.pt` 当作 safetensors 加载。当前 `load_centroids_if_requested()` 已按扩展名判断。确保：

```text
.pt          -> torch.load
.safetensors -> safetensors.torch.load_file
```

### 7.2 centroid 文件找不到

先确认路径：

```bash
ls -lh serialization_dir/qwen3_4b_lut_mlp_kmeans_n8_v2_c32.pt
```

注意 `accelerate launch` 会把 `~/models/...` 展开为 `/root/models/...`，但相对路径仍以当前工作目录为准。推荐始终从 `model-calibration` 目录运行。

### 7.3 MMLU-Pro 太慢

原因：

- MMLU-Pro 是 group，`--limit N` 会乘以子任务数。
- 默认 generation 长度可能很长。
- 当前 LUT PyTorch 模拟慢。

处理：

```bash
--limit 2
--gen_kwargs '{"max_gen_toks":128,"until":["Question:"],"do_sample":false}'
--lut_eval_compute_dtype model
```

baseline 和 LUT 必须使用同样的 `limit`、`num_fewshot`、`gen_kwargs` 才能对比。

### 7.4 cosine similarity 大于 1

单层 MSE 脚本里 cosine 是对很大的 flatten tensor 做 fp32 reduction，偶尔会因为数值误差略大于 1。报告时可以 clamp 到 1；主要指标看 `relative_mse`。

### 7.5 训练 loss 看起来不收敛

已观察到：

- 随机 centroid 初始化下 PPL 极差。
- KMeans 初始化显著改善 initial PPL，但仍远差于 baseline。
- 短训练对 eval loss 改善有限。
- 增大 `K` 或减小 `V` 可以改善单层 MSE，但完整模型质量仍可能不够。

优先排查方向：

1. 当前 `LUTLinear_t.forward()` 与论文在线查表路径不一致。
2. centroid 选择是 hard argmin，`model_loss` 到 centroid 的梯度路径有限。
3. MLP-only 替换已经造成明显累积误差，需先在单层或少层上验证。
4. `V` 太大时单层相对 MSE 已经很差。

## 8. 建议后续路线

优先级从高到低：

1. 把 `LUTLinear_t` eval 路径改成真正 LUT gather + accumulation，并与单层脚本数值对齐。
2. 做少层/单层替换的完整 PPL，定位误差主要来自哪些 layer/module。
3. 分别比较 `gate_proj`、`up_proj`、`down_proj` 的单层 MSE，不只看 `model.layers.18.mlp.up_proj`。
4. 尝试更强 centroid 初始化或更大的 calibration set，但同时记录 LUT 表大小。
5. 如果继续训练，明确区分：
   - 只训练 centroid。
   - 训练 weight + centroid。
   - STE/hard assignment/soft assignment 的差异。
6. 正式画图时建议横轴使用 `lut_storage_ratio=K/V` 或 `online_read_ratio=K/O+1/V`，纵轴同时画：
   - 单层 `relative_mse`。
   - CPU benchmark `lut_median_ms`。
   - dense baseline 水平线。

## 9. 当前重要文件索引

代码：

```text
model-calibration/LUTNeuro/LUTLinear_t.py
model-calibration/LUTNeuro/module_filter.py
model-calibration/examples/run_luterize_causal_lm_no_trainer.py
model-calibration/examples/collect_qwen3_lut_centroids.py
model-calibration/examples/run_lm_eval_lut.py
model-calibration/examples/benchmark_lut_cpu_kernel.py
model-calibration/examples/collect_single_layer_lut_cache.py
model-calibration/examples/evaluate_single_layer_lut_mse.py
model-calibration/examples/aggregate_single_layer_lut_mse.py
model-calibration/scripts/run_single_layer_lut_mse_sweep.sh
```

结果：

```text
model-calibration/serialization_dir/qwen3_4b_lut_mlp_kmeans_n8_v2_c32.pt
model-calibration/serialization_dir/single_layer_lut_mse_up_proj_100k/results.csv
model-calibration/serialization_dir/lut_cpu_kernel_qwen3_mlp_weighted_avg_k2_to_1024_v2_to_128_bt1_f32.csv
model-calibration/serialization_dir/qwen3_4b_lut_mlp_ppl_eval_v2_c32/
```

已有补充文档：

```text
docs/qwen3_lut_calibration.md
```
