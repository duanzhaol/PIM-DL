# Partition Lookup Failure Figure Data

This directory contains the final exported CSV files for the motivation figure
about why PIM-DL/LUT-NN style partition lookup is a poor fit for LLM
activations.

The data comes from a representative Qwen3.5-4B MLP layer:

- Module: `model.layers.18.mlp.up_proj`
- Input channels: 2560
- Output channels: 9728
- Correlation split: 100k calibration activations
- Lookup-error split: 10k held-out evaluation activations
- Source cache:
  `model-calibration/serialization_dir/single_layer_lut_mse_up_proj_100k/model_layers_18_mlp_up_proj_calib100000_eval10000_cache.pt`

The figure supports two claims:

1. Fixed partition lookup hides most channel co-occurrence from tile-local
   encoding.
2. Lookup error is long-tailed across channels and concentrates on a small set
   of high-error or high-outlier dimensions.

## Panel A: Strict Per-Tile Co-Occurrence Coverage

PIM-DL partitions an activation vector into fixed-size tiles and encodes each
tile independently. If two channels co-occur but are placed in different tiles,
that co-occurrence is invisible to tile-local lookup.

We define channel co-occurrence by thresholded absolute correlation:

```text
co-occurrence(i, j) = 1 if |Corr(i, j)| >= threshold
```

For each tile `t`, the strict local coverage is:

```text
StrictCoverage(t) =
  co-occurrence strength inside tile t
  ----------------------------------------------------
  co-occurrence strength of all pairs touching tile t
```

The denominator only includes pairs involving at least one channel in the
current tile. It does not include unrelated pairs elsewhere in the layer.

The exported files include several thresholds:

```text
|Corr(i, j)| >= 0, 0.01, 0.02, 0.05, 0.10
```

Use `abs_corr_threshold = 0.05` as the default plotting row unless you want a
threshold sensitivity plot.

Recommended y-axis:

```text
weighted_tile_internal_cooccurrence_energy_share_percent
```

This column uses squared-correlation energy (`Corr(i,j)^2`) as co-occurrence
strength. It is preferred over raw pair count because high thresholds can make
small-tile pair counts sparse and noisy.

Main file:

- `tile_strict_cooccurrence_coverage_full2560.csv`

Extended file:

- `tile_strict_cooccurrence_coverage_slice2048.csv`
  - Use this only if the plot must include `V=1024`.
  - `V=1024` is not an even tiling of the full 2560-channel layer.

Full 2560-channel result at `|Corr| >= 0.05`:

| Tile size V | Internal co-occurrence energy share |
| ---: | ---: |
| 4 | 0.15% |
| 8 | 0.37% |
| 16 | 0.74% |
| 32 | 1.52% |
| 64 | 2.99% |
| 128 | 5.89% |
| 256 | 10.79% |
| 512 | 20.65% |

Interpretation:

> Among co-occurrences that involve the current tile's channels, only a small
> fraction remains inside the tile for practical small tile sizes. Therefore,
> independent per-tile lookup hides most channel co-occurrence from tile-local
> encoding.

## Panel B: Outlier-Driven Lookup Error

For each K/V configuration, the activation tile is replaced by its nearest
KMeans centroid. Let:

```text
r = x - q(x)
```

where `q(x)` is the quantized activation. For Linear output `y = xW`, the
per-channel projected lookup error is:

```text
e_j = E[r_j^2] * ||W_j||_2^2
```

The activation outlier score is:

```text
s_j = p99(|x_j|)
```

Use these files:

- `outlier_error_summary_by_kv.csv`
  - One-row summary per K/V configuration.
- `outlier_error_cdf_by_channel.csv`
  - CDF data for per-channel projected lookup error.
- `outlier_error_pareto_by_channel.csv`
  - Pareto data showing how much total error is contributed by top-error
    channels.
- `outlier_error_by_activation_outlier_rank.csv`
  - Per-channel rows sorted by activation outlier score.

Recommended CDF plot:

- filter one or more `label` values
- x-axis: `projected_error_norm_by_mean`
- y-axis: `channel_cdf`
- use log scale on x-axis

Recommended Pareto plot:

- x-axis: `top_channel_percent`
- y-axis: `cumulative_projected_error_share_descending`

Outlier summary:

| Label | K | V | Mean residual MSE | Mean projected error | Top 1% outlier error share | Top 5% outlier error share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fast | 4 | 128 | 0.4307 | 1.2410 | 10.61% | 26.79% |
| low_middle | 64 | 32 | 0.3347 | 1.1837 | 9.86% | 26.91% |
| middle | 32 | 8 | 0.2609 | 0.9139 | 8.95% | 21.25% |
| quality | 128 | 4 | 0.0842 | 0.2810 | 14.01% | 26.19% |
| expensive | 512 | 4 | 0.0533 | 0.1504 | 19.47% | 32.33% |

The `top 1% outlier error share` column means: sort channels by activation
outlier score `p99(|x_j|)`, take the top 1% channels, and measure their share
of total projected lookup error.

## Final Files

- `tile_strict_cooccurrence_coverage_full2560.csv`
  - Final data for Panel A on the full 2560-channel layer.
- `tile_strict_cooccurrence_coverage_slice2048.csv`
  - Optional Panel A data including `V=1024`.
- `outlier_error_summary_by_kv.csv`
  - Panel B K/V summary.
- `outlier_error_cdf_by_channel.csv`
  - Panel B CDF source.
- `outlier_error_pareto_by_channel.csv`
  - Panel B Pareto source.
- `outlier_error_by_activation_outlier_rank.csv`
  - Panel B activation-outlier-ranked source.
- `source_manifest.csv`
  - Raw source paths for traceability.

## K/V Configurations

The outlier-error tables include five representative PIM-DL points:

| Label | K | V | Interpretation |
| --- | ---: | ---: | --- |
| fast | 4 | 128 | small table, low compute, poor accuracy |
| low_middle | 64 | 32 | moderate cost |
| middle | 32 | 8 | middle point |
| quality | 128 | 4 | higher quality, higher cost |
| expensive | 512 | 4 | near the expensive end tested here |

## Reproducibility

The raw statistics were generated from:

```bash
/root/miniconda3/envs/luturbo/bin/python \
  model-calibration/examples/analyze_partition_lookup_failure.py \
  --layer_cache_path model-calibration/serialization_dir/single_layer_lut_mse_up_proj_100k/model_layers_18_mlp_up_proj_calib100000_eval10000_cache.pt \
  --centroid_path <centroid_path_for_KV> \
  --vec_len <V> \
  --ncentroid <K> \
  --corr_split calib \
  --corr_max_tokens 100000 \
  --error_split eval \
  --error_max_tokens 10000 \
  --device cuda \
  --output_dir <raw_output_dir>
```

The strict co-occurrence CSVs were derived from the same cached activations by
computing the full channel correlation matrix and applying the local denominator
defined above.
