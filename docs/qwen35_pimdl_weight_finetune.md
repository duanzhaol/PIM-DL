# Qwen3.5 PIM-DL Weight Finetuning

This note records the current implementation needed to run a PIM-DL-style weight finetuning stage after KMeans centroid initialization.

## Motivation

The previous Qwen/Qwen3.5 experiments mostly used KMeans centroids plus centroid-only training. That differs from the original PIM-DL flow, where the converted LUT model can also be finetuned on task loss with model weights trainable. For large causal LMs this distinction matters: centroid-only updates are too weak to recover the quality loss introduced by independent tile lookup.

The new flow keeps the existing KMeans centroid files as initialization, then finetunes the LUT-converted model with both LUT centroids and model weights trainable.

## Code Changes

Main training script:

```text
model-calibration/examples/run_luterize_causal_lm_no_trainer.py
```

New options:

- `--save_full_lut_model`: saves a full LUT-converted Hugging Face checkpoint to `output_dir`.
- `--resume_from_lut_model PATH`: restores a full LUT-converted checkpoint after replacing target Linear layers with `LUTLinear_t`.
- `--weight_requires_grad`: enables model weight finetuning.
- `--weight_trainable_scope {all,lut_modules}`: `all` follows the original broad weight-finetune switch; `lut_modules` trains only weights inside replaced `LUTLinear_t` modules. The Qwen3.5 Pareto script defaults to `lut_modules` because single-card full-model AdamW does not leave enough room for the large online centroid-search buffers.
- `--centroid_requires_grad`: enables centroid finetuning.
- `--freeze_embeddings` / `--freeze_lm_head`: enabled by default. This follows the original MLM script's embedding-freeze behavior and avoids spending optimizer memory on the large token embedding/output head during LUT calibration.
- `--adam_foreach`: disabled by default. The foreach AdamW path materializes temporary tensor lists and caused an additional OOM at the first optimizer step on A100 80GB.

Important behavior:

- `--centroid_path` and `--resume_from_lut_model` are mutually exclusive.
- The old `model_lut_state.pt` path still exists and stores only LUT centroid tensors.
- Weight-finetuned experiments must use `--save_full_lut_model`; otherwise the updated weights are not recoverable. New checkpoints save the raw LUT-converted model state as `full_lut_model_state.pt`.
- Older smoke checkpoints written through `save_pretrained()` may contain keys such as `model.language_model.layers.*`, while a freshly LUT-converted model expects `model.layers.*`. The loader normalizes this old prefix and raises an error if no checkpoint key matches, so a PPL run cannot silently evaluate an untrained LUT model.
- Logs now print trainable total, centroid, and non-centroid parameter counts to make the training mode explicit.
- A first attempt with all model parameters trainable reached `trainable non-centroid params=4.205B` and OOMed at the first AdamW step on an A100 80GB while allocating optimizer state. Freezing embedding and `lm_head` reduces this to `3.570B`, and disabling AdamW foreach lets the first optimizer step finish. The next forward still OOMed for `K=128,V=4` because optimizer state left too little room for centroid-search buffers. The single-card smoke therefore uses `--weight_trainable_scope lut_modules`.

Workflow script:

```text
model-calibration/scripts/run_qwen35_pareto_weight_finetune.sh
```

This script consumes already collected KMeans centroid files for the Pareto points:

| name | K | V |
| --- | ---: | ---: |
| fast | 4 | 128 |
| low_middle | 64 | 32 |
| middle | 32 | 8 |
| quality | 128 | 4 |
| expensive | 512 | 4 |

The script supports:

- `CONFIG_FILTER=quality` to run one point.
- `PHASES=train,ppl` to run finetuning and then PPL evaluation.
- `TRAIN_STEPS`, `TRAIN_LR`, and `TRAIN_RECONSTRUCT_RATE` for quick sweeps.
- `CENTROID_REQUIRES_GRAD=0` to freeze centroids and train only the selected weight scope.

## Smoke Command

Run one point first:

```bash
cd /root/PIM-DL-ASPLOS/model-calibration

CONFIG_FILTER=quality \
PHASES=train,ppl \
TRAIN_STEPS=20 \
TRAIN_LR=1e-5 \
TRAIN_LR_TAG=lr1e5 \
TRAIN_WARMUP_STEPS=2 \
WEIGHT_TRAINABLE_SCOPE=lut_modules \
bash scripts/run_qwen35_pareto_weight_finetune.sh
```

This uses:

```text
K=128
V=4
centroid_path=serialization_dir/qwen35_4b_lut_mlp_kmeans_n8_v4_c128.pt
```

Expected training output directory:

```text
serialization_dir/qwen35_4b_lut_mlp_weight_ft_n8_v4_c128_lr1e5_steps20
```

Expected logs:

```text
serialization_dir/logs/qwen35_pareto_weight_finetune/quality_v4_c128_weight_finetune.log
serialization_dir/logs/qwen35_pareto_weight_finetune/quality_v4_c128_weight_finetuned_ppl.log
```

## How To Judge The Smoke

First check that finetuning really trains weights:

```text
trainable non-centroid params > 0
```

Then compare:

- initial eval loss before finetuning
- final eval loss after finetuning
- PPL eval after loading the saved full LUT checkpoint

If final eval loss is lower than initial eval loss, the weight finetuning path is functioning and has a positive short-run training signal. If the full-checkpoint PPL evaluation matches the trained model behavior, the save/load path is also working.

Observed `quality` smoke on one A100 80GB with `K=128,V=4`, `TRAIN_STEPS=20`, `TRAIN_LR=1e-5`, and `WEIGHT_TRAINABLE_SCOPE=lut_modules`:

| metric | value |
| --- | ---: |
| trainable centroid params | 58,720,256 |
| trainable non-centroid params | 2,264,924,160 |
| dense baseline eval loss / PPL, 20 batches | 2.303579 / 10.009945 |
| initial eval loss / PPL, 20 batches | 3.193043 / 24.362455 |
| final eval loss / PPL, 20 batches | 2.906930 / 18.300533 |
| reload eval loss / PPL, 20 batches | 2.906930 / 18.300533 |

This smoke shows a positive in-process finetuning signal, and the eval-only reload reproduces the trained model result exactly on the same 20 validation batches. It does not yet show that the quality is acceptable relative to dense; it only verifies that the weight-finetuning and checkpoint save/load path is functioning.

An additional continuation run loaded the 20-step checkpoint and trained further. It was stopped manually before completion, but the first continuation eval point still improved:

| metric | loss / PPL |
| --- | ---: |
| continuation start, same as 20-step checkpoint | 2.906930 / 18.300533 |
| continuation step 20 | 2.863921 / 17.530123 |

This indicates the short smoke had not fully plateaued.

## Expensive Point Smoke

The `expensive` point uses:

```text
K=512
V=4
centroid_path=serialization_dir/qwen35_4b_lut_mlp_kmeans_n8_v4_c512.pt
```

KMeans-only initial eval is better than the `K=128,V=4` point:

| metric | value |
| --- | ---: |
| trainable centroid params, if enabled | 234,881,024 |
| trainable non-centroid params with `lut_modules` scope | 2,264,924,160 |
| initial eval loss / PPL, 20 batches | 2.826471 / 16.885771 |

However, the current Python LUT training path OOMs on an A100 80GB at the first backward pass for this point. The failing allocation is in `LUTLinear_t._nearest_centroid_indices()`:

```text
torch.OutOfMemoryError: Tried to allocate 9.00 GiB
```

This happens both when training weights plus centroids and when freezing centroids with `CENTROID_REQUIRES_GRAD=0`. The immediate bottleneck is the online centroid-search distance tensor for `K=512,V=4` during backward/checkpoint recomputation, not a missing centroid file or optimizer foreach overhead. To train this point with the current implementation, reduce activation length, use multi-GPU/offload, or implement a chunked centroid-search kernel that does not materialize the full distance tensor at once.

## Full Pareto Run

After smoke succeeds:

```bash
cd /root/PIM-DL-ASPLOS/model-calibration

PHASES=train,ppl \
TRAIN_STEPS=100 \
TRAIN_LR=1e-5 \
TRAIN_LR_TAG=lr1e5 \
bash scripts/run_qwen35_pareto_weight_finetune.sh
```

This is significantly heavier than centroid-only training because AdamW now tracks model weights. On Qwen3.5-4B, run on an 80GB GPU first and watch for OOM.
