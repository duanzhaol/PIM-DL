from examples.run_lm_eval_lut import parse_args, parse_tasks


def test_parse_tasks_splits_comma_separated_task_names():
    assert parse_tasks("mmlu_pro,hellaswag") == ["mmlu_pro", "hellaswag"]


def test_parse_args_defaults_to_mmlu_pro_sample():
    args = parse_args(["--model_name_or_path", "/tmp/model"])

    assert args.tasks == "mmlu_pro"
    assert args.limit == 8
    assert args.batch_size == "1"
    assert args.bootstrap_iters == 0
