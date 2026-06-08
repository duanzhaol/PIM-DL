# Activation Lookup Reconstruction Error Samples

This directory contains 100 real token activations and their activation-space reconstruction error after PIM-DL lookup quantization. The error is computed before the Linear projection, so it is not output/projection error.

## Source

- Module: `model.layers.18.mlp.up_proj`
- Cache: `serialization_dir/single_layer_lut_mse_up_proj_100k/model_layers_18_mlp_up_proj_calib100000_eval10000_cache.pt`
- Activation split: `eval`
- Selected samples: 100 random eval activations with seed 42
- Hidden dimension: 2560
- Lookup config: `K=128, V=4`
- Centroid file: `serialization_dir/qwen35_4b_lut_mlp_kmeans_n8_v4_c128.pt`

## Definition

For each selected activation vector `x`, we split it into `V=4`-dimensional tiles, find the nearest KMeans centroid for each tile, reconstruct `q(x)`, and export:

```text
reconstruction_error = x - q(x)
abs_reconstruction_error = |x - q(x)|
```

No Linear weight is applied in this export.

## Summary

- Global activation RMS: 0.430026
- Global reconstruction RMSE: 0.289934
- Global MAE: 0.182403
- Max absolute reconstruction error: 8.761663
- Relative activation MSE: 0.454577

## Files

- `activation_reconstruction_samples.npz`: compressed arrays for plotting 100 traces.
- `activation_reconstruction_error_long.csv`: long-form CSV with one row per sample/channel.
- `per_sample_summary.csv`: one row per activation.
- `per_channel_summary.csv`: one row per channel across these 100 activations.
- `top_abs_error_points.csv`: top 1000 absolute reconstruction error points.
- `summary.json`: metadata and scalar summary.
