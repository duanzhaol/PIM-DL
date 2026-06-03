# Qwen3.5 PIM-DL V=64 K=128 Decode D-PPL 记录

本文档记录 Qwen3.5-4B 上 `vec_len=64`、`ncentroid=128` 的 PIM-DL centroid 采集、centroid 微调、以及 decode-stage D-PPL 测试结果。测试目标是和前一轮 `V=16,K=128` 保持相同口径，观察更大 `V` 下 PIM-DL + input residual 在线补偿的 D-PPL 增幅。

## 1. 测试口径

本轮测试使用 decode-stage D-PPL，不是 full-forward PPL。

口径如下：

- 模型：`/root/models/Qwen3.5-4B`
- 测试窗口：`/root/PIM-DL-ASPLOS/wikipedia-1k.json`
- Prompt 长度：`512`
- Decode loss tokens：`512`
- 每个窗口 token 数：`512 + 512 + 1 = 1025`
- 测试窗口数：`2`
- 总计 loss tokens：`1024`
- Dense prefill：使用 dense Qwen3.5 建 KV/cache，prefill logits 不计 loss
- Decode：teacher-forced one-token decode，每步 `n_tokens == 1`
- D-PPL：`exp(mean next-token NLL)`

测试两种 D-PPL 模式：

- `isolated`：每个 decode token 都从 dense anchor state 出发，只测当前位置 LUT 替换 dense 的局部误差，不让误差传播。
- `pathwise`：沿着 LUT decode path 一步步前进，误差会进入后续 KV/cache，更接近真实推理路径。

Dense baseline 在两个模式下相同：

| mode | NLL | D-PPL | tokens |
| --- | ---: | ---: | ---: |
| isolated | 1.969975833 | 7.170503193 | 1024 |
| pathwise | 1.969975833 | 7.170503193 | 1024 |

相对增幅按下面公式计算：

```text
D-PPL % = (method_d_ppl / dense_d_ppl - 1) * 100%
```

## 2. 配置与产物

PIM-DL 配置：

```text
target_modules: mlp
vec_len: 64
ncentroid: 128
nsample: 8
max_vectors_per_module: 65536
kmeans_iter: 200
torch_dtype: bfloat16
training_lr: 1e-4
training_warmup_steps: 5
reconstruct_rate: 1e-3
training_steps: 100
```

关键产物：

```text
Centroid:
model-calibration/serialization_dir/qwen35_4b_lut_mlp_kmeans_n8_v64_c128.pt

Trained LUT state:
model-calibration/serialization_dir/qwen35_4b_lut_mlp_fineweb_kmeans_n8_v64_c128_lr1e4_steps100/model_lut_state.pt

D-PPL sweep:
model-calibration/serialization_dir/qwen35_dppl_ratio_sweep_v64_c128/
model-calibration/serialization_dir/qwen35_dppl_ratio_sweep_v64_c128/summary.csv
```

Centroid 文件大小约 `225M`，包含 96 个 MLP centroid tensor：

```text
model.layers.{0..31}.mlp.{gate_proj,up_proj,down_proj}.centroids.weight
```

典型 tensor shape：

```text
gate_proj/up_proj: (40, 8192)
down_proj:         (144, 8192)
```

其中 `8192 = ncentroid * vec_len = 128 * 64`。

## 3. 质心采集命令

```bash
cd /root/PIM-DL-ASPLOS/model-calibration

/root/miniconda3/envs/luturbo/bin/accelerate launch \
  --num_processes 1 --num_machines 1 --mixed_precision no --dynamo_backend no \
  examples/collect_qwen3_lut_centroids.py \
  --model_name_or_path /root/models/Qwen3.5-4B \
  --tokenized_dataset_path /root/fineweb-edu/sample/10BT-tokenized-qwen35-2048-10k \
  --max_samples 10000 \
  --dataset_seed 42 \
  --target_modules mlp \
  --max_seq_length 2048 \
  --per_device_batch_size 1 \
  --nsample 8 \
  --max_vectors_per_module 65536 \
  --vec_len 64 \
  --ncentroid 128 \
  --kmeans_iter 200 \
  --torch_dtype bfloat16 \
  --output_path serialization_dir/qwen35_4b_lut_mlp_kmeans_n8_v64_c128.pt
```

采集完成日志要点：

```text
Saved 96 centroid tensors to serialization_dir/qwen35_4b_lut_mlp_kmeans_n8_v64_c128.pt
```

## 4. Centroid 微调命令

```bash
cd /root/PIM-DL-ASPLOS/model-calibration

/root/miniconda3/envs/luturbo/bin/accelerate launch \
  --num_processes 1 --num_machines 1 --mixed_precision no --dynamo_backend no \
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
  --vec_len 64 \
  --ncentroid 128 \
  --torch_dtype bfloat16 \
  --learning_rate 1e-4 \
  --num_warmup_steps 5 \
  --reconstruct_rate 1e-3 \
  --centroid_path serialization_dir/qwen35_4b_lut_mlp_kmeans_n8_v64_c128.pt \
  --output_dir serialization_dir/qwen35_4b_lut_mlp_fineweb_kmeans_n8_v64_c128_lr1e4_steps100
```

训练结果：

| 阶段 | Loss | PPL |
| --- | ---: | ---: |
| initial eval | 9.735214 | 16902.456298 |
| step 20 | 9.767260 | 17452.873731 |
| step 40 | 9.795534 | 17953.388203 |
| step 60 | 9.807086 | 18161.985700 |
| step 80 | 9.815249 | 18310.857027 |
| step 100 / final | 9.798809 | 18012.280451 |

观察：`V=64,K=128` 在 full-forward validation PPL 上没有随着 100 step 微调改善，最终 eval 反而比 initial eval 略差。

## 5. D-PPL Sweep 代码

本轮 sweep 使用了一次性脚本，避免每个 ratio 重新加载两份 4B 模型：

```text
model-calibration/serialization_dir/run_qwen35_v64_c128_dppl_sweep.py
```

脚本逻辑：

1. 加载 dense Qwen3.5 和 PIM-DL LUT Qwen3.5。
2. 加载训练后的 `model_lut_state.pt`。
3. 先跑 dense `isolated` 和 `pathwise` baseline。
4. 对每个补偿比例设置所有 `LUTLinear_t.residual_compensation_ratio`。
5. 依次跑 `isolated` 和 `pathwise`。
6. 写出每个 ratio/mode 的 JSON，以及汇总 CSV。

关键代码片段：

```python
def set_ratio(model, ratio: float) -> int:
    updated = 0
    for module in model.modules():
        if isinstance(module, LUTLinear_t):
            module.residual_compensation_ratio = ratio
            updated += 1
    return updated
```

核心依赖函数来自：

```text
model-calibration/examples/run_decode_dppl_lut.py
```

其中：

- `run_window_isolated(...)` 实现 dense anchor 的 isolated D-PPL。
- `run_window_pathwise(...)` 实现沿 LUT path 传播的 pathwise D-PPL。
- `load_eval_models(...)` 负责加载 dense/eval model 并应用 PIM-DL 替换。
- `load_dppl_windows(...)` 读取 `/root/PIM-DL-ASPLOS/wikipedia-1k.json`。

运行命令：

```bash
cd /root/PIM-DL-ASPLOS/model-calibration

/root/miniconda3/envs/luturbo/bin/python \
  serialization_dir/run_qwen35_v64_c128_dppl_sweep.py
```

## 6. D-PPL 结果

Dense baseline D-PPL：`7.170503193`

| 在线补偿比例 | isolated D-PPL | isolated ΔD-PPL | isolated +% | pathwise D-PPL | pathwise ΔD-PPL | pathwise +% |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 2913.934532 | 2906.764028 | 40537.80% | 14223.743155 | 14216.572652 | 198264.64% |
| 2% | 693.933859 | 686.763356 | 9577.62% | 4811.108435 | 4803.937931 | 66995.83% |
| 5% | 219.455115 | 212.284612 | 2960.53% | 1151.881253 | 1144.710750 | 15964.16% |
| 10% | 58.604453 | 51.433950 | 717.30% | 171.649518 | 164.479015 | 2293.83% |
| 20% | 17.786153 | 10.615649 | 148.05% | 26.050695 | 18.880192 | 263.30% |
| 30% | 10.831438 | 3.660935 | 51.06% | 12.220060 | 5.049557 | 70.42% |
| 40% | 8.552711 | 1.382208 | 19.28% | 8.946311 | 1.775808 | 24.77% |
| 50% | 7.757009 | 0.586506 | 8.18% | 7.773677 | 0.603174 | 8.41% |

## 7. 和 V=16,K=128 的对比

同样 D-PPL 口径下，前一轮 `V=16,K=128` 的关键结果保存在：

```text
docs/qwen35_dppl_ratio_sweep_results.csv
```

`V=16,K=128` 与 `V=64,K=128` 对比：

| 在线补偿比例 | V=16 isolated +% | V=64 isolated +% | V=16 pathwise +% | V=64 pathwise +% |
| ---: | ---: | ---: | ---: | ---: |
| 0% | 6859.12% | 40537.80% | 35587.07% | 198264.64% |
| 10% | 184.85% | 717.30% | 454.48% | 2293.83% |
| 20% | 62.26% | 148.05% | 94.47% | 263.30% |
| 30% | 25.40% | 51.06% | 34.85% | 70.42% |
| 40% | 13.64% | 19.28% | 14.04% | 24.77% |
| 50% | 5.32% | 8.18% | 6.79% | 8.41% |

结论：`V=64,K=128` 明显弱于 `V=16,K=128`，尤其在低补偿比例下差距非常大。到 50% input residual 在线补偿时，两者差距缩小，但 `V=64,K=128` 仍比 `V=16,K=128` 更差。

## 8. 结论

本轮结果说明：

- 仅增大 `V` 到 64 并保持 `K=128`，不能提升 PIM-DL 的模型质量。
- `V=64,K=128` 的裸 PIM-DL D-PPL 很高，pathwise 误差传播后更严重。
- 在线补偿仍然非常有效，50% 补偿时 D-PPL 增幅降到 isolated `8.18%`、pathwise `8.41%`。
- 但同样 50% 补偿下，`V=64,K=128` 仍不如 `V=16,K=128`。

因此，如果目标是质量优先，当前结果更支持使用较小 `V`，例如 `V=16,K=128`。如果目标是推理性能优先，`V=64,K=128` 可能仍有讨论价值，但需要结合 CPU/GPU kernel 性能和补偿开销一起评估。
