# Within-Tile Dependency Share Data

This directory contains the regenerated Figure 4C data for the claim that
partition lookup can only exploit channel dependencies inside each independently
encoded tile.

## Source

- Layer cache: `serialization_dir/single_layer_lut_mse_up_proj_100k/model_layers_18_mlp_up_proj_calib100000_eval10000_cache.pt`
- Module: `model.layers.18.mlp.up_proj`
- Activation split: `calib`
- Tokens used: `100000`
- Hidden dimension D: `2560`
- Weight definition: squared Pearson correlation, `w(i,j)=corr(i,j)^2`
- Diagonal/self-correlation: excluded by setting `w(i,i)=0`
- Tiling: contiguous channel tiles matching the partition lookup baseline

The cache corresponds to Qwen3.5-4B BF16, a representative middle MLP
projection input (`model.layers.18.mlp.up_proj`). This is the same activation
surface used by the single-layer partition lookup sweep.

## Metric

For each channel `i` in tile `T(i)`:

```text
internal_i = sum_{j in T(i), j != i} w(i,j)
total_i    = sum_{j != i} w(i,j)
share_i    = internal_i / total_i
S(V)       = mean_i(share_i) * 100%
```

This denominator is local to each channel. It is not normalized by global
correlation energy across unrelated channel pairs.

## Baselines

- `measured`: the actual contiguous tiling used by partition lookup.
- `random`: `100` random balanced channel permutations.
- `best_effort_greedy`: a deterministic greedy balanced grouping that tries to
  maximize within-tile `corr^2` energy. This is a best-effort upper bound, not an
  exact balanced graph partition optimum.
- `ideal`: 100%, corresponding to a perfectly block-diagonal activation where
  all dependencies of each channel stay inside its tile.

## Results

All values are `S(V)=mean_i(share_i)*100%`.

| Tile size V | Measured contiguous | Random baseline mean +/- std | Theoretical pair fraction | Greedy best-effort | Ideal |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.1456% | 0.1171% +/- 0.0032% | 0.1172% | 0.8767% | 100% |
| 8 | 0.3418% | 0.2738% +/- 0.0048% | 0.2735% | 1.3514% | 100% |
| 16 | 0.7167% | 0.5858% +/- 0.0062% | 0.5862% | 1.9770% | 100% |
| 32 | 1.4740% | 1.2121% +/- 0.0086% | 1.2114% | 2.9881% | 100% |

The measured contiguous tiling is only slightly above random grouping and is
far from the 100% block-diagonal ideal. Even the greedy best-effort balanced
tiling captures less than 3% of the per-channel squared-correlation dependency
for `V <= 32`.

## Files

- `within_tile_dependency_share.csv`: one summary row per tile size.
- `within_tile_dependency_share_by_channel.csv`: measured per-channel shares.
- `random_tiling_trials.csv`: random baseline trials.
- `best_effort_groups.csv`: greedy best-effort channel groups.
- `summary.json`: machine-readable manifest and summary.
