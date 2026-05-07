import torch.nn as nn

from LUTNeuro.LUTLinear_t import LUTLinear_t
from examples.run_lm_eval_lut import parse_args, parse_tasks, set_lut_eval_compute_dtype


def test_parse_tasks_splits_comma_separated_task_names():
    assert parse_tasks("mmlu_pro,hellaswag") == ["mmlu_pro", "hellaswag"]


def test_parse_args_defaults_to_mmlu_pro_sample():
    args = parse_args(["--model_name_or_path", "/tmp/model"])

    assert args.tasks == "mmlu_pro"
    assert args.limit == 8
    assert args.batch_size == "1"
    assert args.bootstrap_iters == 0
    assert args.output_dir is None
    assert args.lut_eval_compute_dtype == "float32"


def test_set_lut_eval_compute_dtype_updates_lut_modules_only():
    model = nn.Sequential(
        LUTLinear_t(8, 4, bias=False, ncentroids=2, vec_len=2),
        nn.Linear(4, 2),
    )

    updated = set_lut_eval_compute_dtype(model, "model")

    assert updated == 1
    assert model[0].eval_compute_dtype == "model"
    assert not hasattr(model[1], "eval_compute_dtype")
