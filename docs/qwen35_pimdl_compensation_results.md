# Qwen3.5 PIM-DL 与在线补偿 PPL 测试记录

本文档记录当前在 Qwen3.5-4B 上对 PIM-DL 查表近似、PIM-DL + 在线补偿、以及仅在线补偿方案的测试方法和结果。目标是让后续 Agent 能够复现实验，并清楚理解不同补偿比例下 PPL 的绝对变化和相对 Dense baseline 的增加比例。

## 1. 测试口径

本轮结果使用的是 full-forward validation PPL，不是 decode-stage D-PPL。

具体口径：

- 模型：`/root/models/Qwen3.5-4B`
- 数据源：`/root/fineweb-edu/sample/10BT/000_00000.parquet`
- Tokenizer：Qwen3.5-4B tokenizer
- Tokenized 数据：`/root/fineweb-edu/sample/10BT-tokenized-qwen35-2048-10k`
- Train split：10000 条，每条 2048 tokens
- Validation split：512 条，每条 2048 tokens
- PPL 评测：`max_eval_batches=20`
- 替换模块：`target_modules=mlp`
- Dense baseline：原始 Qwen3.5-4B，不做 LUT 替换

所有相对增幅都按下面公式计算：

```text
PPL 相对增加百分比 = (method_ppl / dense_ppl - 1) * 100%
```

Dense baseline 为：

| 模型 | Loss | PPL |
| --- | ---: | ---: |
| Qwen3.5-4B dense | 2.303579 | 10.009945 |

## 2. 环境与注意事项

实验环境是 Conda 环境 `luturbo`：

```bash
conda activate luturbo
cd /root/PIM-DL-ASPLOS/model-calibration
```

当前为了支持 Qwen3.5，环境中的 Transformers 已升级到：

```text
transformers: 5.9.0
huggingface-hub: 1.16.1
tokenizers: 0.22.2
accelerate: 1.13.0
```

注意：该升级会和 `sglang 0.5.6.post2` 的 `transformers==4.57.1` 依赖要求冲突。如果后续要跑 sglang，应另建环境或回滚 Transformers。

Qwen3.5 加载时会看到如下警告：

```text
The fast path is not available ... Falling back to torch implementation
```

这是 Qwen3.5 线性 attention 相关依赖缺失导致的性能回退，不影响本轮 PPL 口径。

## 3. 数据准备

当前使用了一个本地生成的 Qwen3.5 tokenized 子集：

```text
/root/fineweb-edu/sample/10BT-tokenized-qwen35-2048-10k
```

该数据由 FineWeb-Edu parquet 的 `text` 列经过 Qwen3.5 tokenizer 得到，按 2048 tokens 切块：

```text
train:      10000 chunks
validation:   512 chunks
seq_len:     2048
```

这个子集是为了避免复用旧的 Qwen3 tokenizer 数据。旧路径：

```text
/root/fineweb-edu/sample/10BT-tokenized-qwen3-2048
```

不应用于 Qwen3.5 的正式 PPL 测试。

## 4. PIM-DL 配置

本轮 Qwen3.5 PIM-DL 配置：

```text
target_modules: mlp
vec_len: 16
ncentroid: 128
nsample: 8
max_vectors_per_module: 65536
kmeans_iter: 200
torch_dtype: bfloat16
```

收集得到的 KMeans centroid 文件：

```text
model-calibration/serialization_dir/qwen35_4b_lut_mlp_kmeans_n8_v16_c128.pt
```

文件大小约 `225M`，包含 96 个 MLP centroid tensor：

```text
model.layers.{0..31}.mlp.{gate_proj,up_proj,down_proj}.centroids.weight
```

其中典型 shape：

```text
gate_proj/up_proj: (160, 2048)
down_proj:         (576, 2048)
```

这里 `2048 = ncentroid * vec_len = 128 * 16`。

## 5. PIM-DL Centroid 微调

训练命令核心参数：

```bash
accelerate launch --num_processes 1 --num_machines 1 --mixed_precision no --dynamo_backend no \
  examples/run_luterize_causal_lm_no_trainer.py \
  --model_name_or_path /root/models/Qwen3.5-4B \
  --tokenized_dataset_path /root/fineweb-edu/sample/10BT-tokenized-qwen35-2048-10k \
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
  --vec_len 16 \
  --ncentroid 128 \
  --torch_dtype bfloat16 \
  --learning_rate 1e-4 \
  --num_warmup_steps 5 \
  --reconstruct_rate 1e-3 \
  --centroid_path serialization_dir/qwen35_4b_lut_mlp_kmeans_n8_v16_c128.pt \
  --output_dir serialization_dir/qwen35_4b_lut_mlp_fineweb_kmeans_n8_v16_c128_lr1e4_steps100
```

训练输出：

```text
model-calibration/serialization_dir/qwen35_4b_lut_mlp_fineweb_kmeans_n8_v16_c128_lr1e4_steps100/model_lut_state.pt
```

训练日志：

```text
model-calibration/serialization_dir/logs/qwen35_v16_c128_train.log
```

训练过程中的 PPL 相对 Dense baseline 仍然非常高：

| 阶段 | Loss | PPL | PPL 相对 Dense 增加 |
| --- | ---: | ---: | ---: |
| initial eval | 8.577521 | 5310.925135 | 52956.49% |
| step 20 | 8.560401 | 5220.774089 | 52055.87% |
| step 40 | 8.555043 | 5192.877326 | 51777.18% |
| step 60 | 8.559466 | 5215.897029 | 52007.15% |
| step 80 | 8.554168 | 5188.333091 | 51731.78% |
| step 100 / final | 8.550246 | 5168.026839 | 51528.92% |

结论：在当前实现和训练目标下，100 step centroid 微调只带来很小改善，不能从根本上修复 PIM-DL MLP-only 近似带来的 PPL 退化。

## 6. 在线补偿对比实验

本轮对比两种方法：

1. `PIM-DL + 在线补偿`
   - 先用 PIM-DL LUT 近似输出。
   - 再对选中的 input residual channels 做真实 dense correction。
   - 加载训练后的 `model_lut_state.pt`。

2. `仅在线补偿`
   - 启用 `--activation_topk_only`。
   - 不使用 LUT 查表近似贡献，只计算被选中 input channels 的真实 dense contribution。
   - 这个实验用于检验“补偿 50% 时，PIM-DL 查表部分是否仍有正贡献”。

补偿比例：

```text
0%, 2%, 5%, 10%, 20%, 30%, 40%, 50%
```

metric 使用默认：

```text
--residual_compensation_metric abs
```

Sweep 日志：

```text
model-calibration/serialization_dir/logs/qwen35_comp_ppl_sweep/sweep.log
```

### 6.1 结果表

| 在线补偿比例 | PIM-DL + 补偿 Loss | PIM-DL + 补偿 PPL | PIM-DL + 补偿 PPL 相对 Dense 增加 | 仅在线补偿 Loss | 仅在线补偿 PPL | 仅在线补偿 PPL 相对 Dense 增加 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 8.550246 | 5168.026839 | 51528.92% | 9.597405 | 14726.523014 | 147018.92% |
| 2% | 6.955024 | 1048.403953 | 10373.62% | 9.757753 | 17287.748876 | 172605.73% |
| 5% | 5.754991 | 315.762572 | 3054.49% | 10.579422 | 39317.382272 | 392683.20% |
| 10% | 4.368393 | 78.916706 | 688.38% | 8.344403 | 4206.571679 | 41923.92% |
| 20% | 3.089761 | 21.971837 | 119.50% | 4.815109 | 123.360230 | 1132.38% |
| 30% | 2.651379 | 14.173575 | 41.59% | 3.110654 | 22.435715 | 124.13% |
| 40% | 2.469344 | 11.814693 | 18.03% | 2.617130 | 13.696359 | 36.83% |
| 50% | 2.383571 | 10.843555 | 8.33% | 2.434010 | 11.404523 | 13.93% |

### 6.2 主要观察

PIM-DL + 在线补偿显著优于仅在线补偿。尤其在 20%-40% 区间，PIM-DL 查表近似仍提供明显正贡献：

| 在线补偿比例 | 仅在线补偿 PPL 相比 PIM-DL + 补偿额外增加 |
| ---: | ---: |
| 20% | 461.45% |
| 30% | 58.29% |
| 40% | 15.93% |
| 50% | 5.17% |

50% 补偿时：

```text
Dense PPL:                10.009945
PIM-DL + 50% compensation: 10.843555, 相对 Dense 增加 8.33%
TopK-only 50%:             11.404523, 相对 Dense 增加 13.93%
```

因此，即使补偿比例达到 50%，PIM-DL 的 LUT 近似部分仍然不是无效的；它相比“只算最大激活通道贡献”仍然降低了 PPL。

低补偿比例下，仅在线补偿表现很差。0%-5% 时甚至比 PIM-DL 无补偿更差，这是合理的：仅在线补偿会丢弃未选中 input channels 的全部贡献，而 PIM-DL 至少给所有 input tiles 提供了一个近似输出。

## 7. 复现实验命令

### 7.1 Dense baseline

```bash
cd /root/PIM-DL-ASPLOS/model-calibration

PYTHONUNBUFFERED=1 /root/miniconda3/envs/luturbo/bin/python \
  examples/run_luterize_causal_lm_no_trainer.py \
  --model_name_or_path /root/models/Qwen3.5-4B \
  --tokenized_dataset_path /root/fineweb-edu/sample/10BT-tokenized-qwen35-2048-10k \
  --max_train_samples 10000 \
  --max_eval_samples 512 \
  --dataset_seed 42 \
  --target_modules mlp \
  --max_seq_length 2048 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_train_steps 0 \
  --eval_only \
  --eval_logging_steps 0 \
  --max_eval_batches 20 \
  --vec_len 16 \
  --ncentroid 128 \
  --torch_dtype bfloat16 \
  --centroid_path serialization_dir/qwen35_4b_lut_mlp_fineweb_kmeans_n8_v16_c128_lr1e4_steps100/model_lut_state.pt \
  --baseline_eval_before_lut \
  --output_dir serialization_dir/qwen35_comp_ppl_sweep/dense_ref
```

看日志中的：

```text
Baseline eval before LUT replacement
```

### 7.2 PIM-DL + 在线补偿

以 30% 补偿为例：

```bash
cd /root/PIM-DL-ASPLOS/model-calibration

PYTHONUNBUFFERED=1 /root/miniconda3/envs/luturbo/bin/python \
  examples/run_luterize_causal_lm_no_trainer.py \
  --model_name_or_path /root/models/Qwen3.5-4B \
  --tokenized_dataset_path /root/fineweb-edu/sample/10BT-tokenized-qwen35-2048-10k \
  --max_train_samples 10000 \
  --max_eval_samples 512 \
  --dataset_seed 42 \
  --target_modules mlp \
  --max_seq_length 2048 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_train_steps 0 \
  --eval_only \
  --eval_logging_steps 0 \
  --max_eval_batches 20 \
  --vec_len 16 \
  --ncentroid 128 \
  --torch_dtype bfloat16 \
  --centroid_path serialization_dir/qwen35_4b_lut_mlp_fineweb_kmeans_n8_v16_c128_lr1e4_steps100/model_lut_state.pt \
  --residual_compensation_ratio 0.30 \
  --output_dir serialization_dir/qwen35_comp_ppl_sweep/pimdl_r0p30
```

### 7.3 仅在线补偿

以 30% 补偿为例：

```bash
cd /root/PIM-DL-ASPLOS/model-calibration

PYTHONUNBUFFERED=1 /root/miniconda3/envs/luturbo/bin/python \
  examples/run_luterize_causal_lm_no_trainer.py \
  --model_name_or_path /root/models/Qwen3.5-4B \
  --tokenized_dataset_path /root/fineweb-edu/sample/10BT-tokenized-qwen35-2048-10k \
  --max_train_samples 10000 \
  --max_eval_samples 512 \
  --dataset_seed 42 \
  --target_modules mlp \
  --max_seq_length 2048 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_train_steps 0 \
  --eval_only \
  --eval_logging_steps 0 \
  --max_eval_batches 20 \
  --vec_len 16 \
  --ncentroid 128 \
  --torch_dtype bfloat16 \
  --residual_compensation_ratio 0.30 \
  --activation_topk_only \
  --output_dir serialization_dir/qwen35_comp_ppl_sweep/topk_r0p30
```

## 8. 当前结论

当前 Qwen3.5 上最重要的结论是：

1. 单独 PIM-DL MLP LUT 近似质量不足。训练后 full-forward PPL 仍为 `5168.03`，相对 Dense 增加 `51528.92%`。
2. 在线补偿非常有效。PIM-DL + 30% 补偿已经把 PPL 降到 `14.17`，相对 Dense 增加 `41.59%`。
3. PIM-DL + 50% 补偿达到 `10.84` PPL，相对 Dense 只增加 `8.33%`。
4. 仅在线补偿在同等比例下通常更差。50% 时 PPL 为 `11.40`，相对 Dense 增加 `13.93%`。
5. 因此，在当前 Qwen3.5 设置下，LUT 近似和在线补偿是互补的；在线补偿不是完全替代 LUT 的方法。

## 9. Decode-stage D-PPL Smoke

在 full-forward PPL 之后，又按 decode-stage D-PPL 口径做了一组 smoke 测试。这个结果和上面的 full-forward PPL 不同，不应直接混用。

本轮 D-PPL 口径：

```text
windows_path: /root/PIM-DL-ASPLOS/wikipedia-1k.json
max_windows: 2
prompt_len: 512
decode_len: 512
tokens: 1024 decode loss tokens
```

测试方法：

- `isolated`：每个 decode token 都从 dense anchor KV/cache 出发，只测当前位置 LUT/补偿替换 dense 的局部误差。
- `pathwise`：先 dense prefill，然后沿 eval model 的 decode path 逐 token 前进，误差会随 KV/cache 传播。

本轮 D-PPL sweep 使用的 PIM-DL 配置：

```text
PIM-DL: vec_len=16, ncentroid=128, target_modules=mlp
centroid/state: serialization_dir/qwen35_4b_lut_mlp_fineweb_kmeans_n8_v16_c128_lr1e4_steps100/model_lut_state.pt
residual_compensation_ratio: 0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50
residual_compensation_metric: abs
```

结果文件：

```text
model-calibration/serialization_dir/qwen35_dppl_results/
model-calibration/serialization_dir/qwen35_dppl_ratio_sweep/
```

Dense baseline：

| 方法 | D-PPL 模式 | NLL | D-PPL | 相对 Dense D-PPL 增加 | Tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| Dense | isolated | 1.969976 | 7.170503 | 0.00% | 1024 |
| Dense | pathwise | 1.969976 | 7.170503 | 0.00% | 1024 |

PIM-DL + 在线补偿 ratio sweep 结果。`Delta D-PPL` 和 `Delta %` 都是相对同一 D-PPL 模式下的 Dense baseline 计算：

| 补偿比例 | isolated NLL | isolated D-PPL | isolated Delta D-PPL | isolated Delta % | pathwise NLL | pathwise D-PPL | pathwise Delta D-PPL | pathwise Delta % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 6.212615 | 499.004205 | 491.833702 | 6859.12% | 7.847349 | 2558.942431 | 2551.771928 | 35587.07% |
| 2% | 4.491347 | 89.241577 | 82.071074 | 1144.57% | 5.654056 | 285.447015 | 278.276512 | 3880.85% |
| 5% | 3.651434 | 38.529894 | 31.359391 | 437.34% | 4.687168 | 108.545315 | 101.374812 | 1413.78% |
| 10% | 3.016779 | 20.425400 | 13.254897 | 184.85% | 3.682840 | 39.759161 | 32.588658 | 454.48% |
| 20% | 2.454008 | 11.634882 | 4.464378 | 62.26% | 2.635066 | 13.944236 | 6.773733 | 94.47% |
| 30% | 2.196330 | 8.991950 | 1.821447 | 25.40% | 2.268932 | 9.669065 | 2.498562 | 34.85% |
| 40% | 2.097812 | 8.148318 | 0.977815 | 13.64% | 2.101361 | 8.177296 | 1.006792 | 14.04% |
| 50% | 2.021803 | 7.551927 | 0.381424 | 5.32% | 2.035641 | 7.657156 | 0.486653 | 6.79% |

仅在线补偿 ratio sweep 结果。这里启用 `--activation_topk_only`，不使用 LUT 查表近似输出，只计算被选中 input channels 的真实 dense contribution：

```text
result_dir: model-calibration/serialization_dir/qwen35_dppl_topk_only_ratio_sweep/
```

为了便于后续 Agent 直接读取，PIM-DL + 在线补偿、仅在线补偿、Dense baseline 的 D-PPL sweep 汇总表另存为 CSV：

```text
docs/qwen35_dppl_ratio_sweep_results.csv
```

| 补偿比例 | isolated D-PPL | isolated Delta D-PPL | isolated Delta % | pathwise D-PPL | pathwise Delta D-PPL | pathwise Delta % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 13724.475072 | 13717.304569 | 191301.84% | 20806.982609 | 20799.812106 | 290074.65% |
| 2% | 7126.425997 | 7119.255493 | 99285.30% | 12556.693181 | 12549.522678 | 175015.93% |
| 5% | 2264.754263 | 2257.583760 | 31484.31% | 11480.770001 | 11473.599498 | 160011.08% |
| 10% | 260.926532 | 253.756029 | 3538.89% | 1305.046004 | 1297.875501 | 18100.20% |
| 20% | 21.193093 | 14.022589 | 195.56% | 54.814131 | 47.643627 | 664.44% |
| 30% | 11.856860 | 4.686357 | 65.36% | 13.734978 | 6.564475 | 91.55% |
| 40% | 8.572114 | 1.401611 | 19.55% | 9.212027 | 2.041524 | 28.47% |
| 50% | 7.768470 | 0.597966 | 8.34% | 7.919116 | 0.748613 | 10.44% |

仅在线补偿相比 PIM-DL + 在线补偿的额外 D-PPL 退化：

| 补偿比例 | isolated 额外 D-PPL | isolated 相对 PIM-DL 额外增加 | pathwise 额外 D-PPL | pathwise 相对 PIM-DL 额外增加 |
| ---: | ---: | ---: | ---: | ---: |
| 0% | 13225.470867 | 2650.37% | 18248.040178 | 713.11% |
| 2% | 7037.184419 | 7885.54% | 12271.246166 | 4298.96% |
| 5% | 2226.224369 | 5777.91% | 11372.224686 | 10476.94% |
| 10% | 240.501132 | 1177.46% | 1265.286843 | 3182.38% |
| 20% | 9.558211 | 82.15% | 40.869895 | 293.10% |
| 30% | 2.864910 | 31.86% | 4.065913 | 42.05% |
| 40% | 0.423796 | 5.20% | 1.034732 | 12.65% |
| 50% | 0.216543 | 2.87% | 0.261960 | 3.42% |

D-PPL sweep 的主要观察：

1. Dense 的 isolated 和 pathwise 结果一致，说明脚本在 dense 口径下没有引入模式差异。
2. 低补偿比例下 pathwise 远差于 isolated，说明 decode error 会沿 KV/cache 传播并快速放大。
3. 补偿比例越高，isolated 和 pathwise 的差距越小。40% 时二者已经接近，50% 时 pathwise 相比 isolated 只多增加约 `1.47` 个百分点。
4. PIM-DL + 50% 补偿在 isolated 下 D-PPL 相对 Dense 增加 `5.32%`，pathwise 下增加 `6.79%`。
5. PIM-DL + 在线补偿在所有 ratio 下都优于仅在线补偿。50% 时差距缩小，但仅在线补偿仍比 PIM-DL + 在线补偿多退化 `0.216543` isolated D-PPL 和 `0.261960` pathwise D-PPL。
6. 这和 full-forward PPL 的结论一致：LUT 近似部分仍有正贡献，不只是在线补偿本身在起作用。

后续建议：

- 对补偿 metric 继续比较 `abs` 与 `weighted`。
- 做 per-layer/per-module ratio table，而不是全层统一 ratio；当前结果说明高 ratio 有效，但 uniform 50% 计算成本较高。
