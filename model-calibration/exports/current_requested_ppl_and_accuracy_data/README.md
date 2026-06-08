# Current requested PPL and downstream accuracy data

This folder collects existing local results for two requested groups.

## Files

- `representative_dppl_and_downstream_accuracy.csv`: compact representative datapoints.
- `qwen35_pimdl_v16_c128_dppl_ratio_sweep.csv`: Qwen3.5-4B PIM-DL + online compensation, K=128,V=16, ratios 0/2/5/10/20/30/40/50%, isolated and pathwise decode-stage D-PPL.
- `qwen35_pimdl_v64_c128_dppl_ratio_sweep.csv`: same for K=128,V=64.
- `qwen35_topk_only_dppl_ratio_sweep.csv`: online compensation only, no LUT contribution, ratios 0/2/5/10/20/30/40/50%, isolated and pathwise decode-stage D-PPL.

## Important caveat

The decode-stage D-PPL results are for Qwen3.5-4B on `/root/PIM-DL-ASPLOS/wikipedia-1k.json` with prompt_len=512, decode_len=512, max_windows=2.

The available downstream lm-eval/MMLU-Pro accuracy data are older Qwen3-4B smoke results, not Qwen3.5-4B. They are included because they are the only local downstream task accuracy artifacts currently present. For a paper-quality paired result, rerun lm-eval on Qwen3.5 with the selected representative PIM-DL checkpoint/config.

## Dense D-PPL baseline

- Qwen3.5-4B dense D-PPL: 7.170503193478727
- Dense NLL: 1.969975832550496
- Tokens: 1024
