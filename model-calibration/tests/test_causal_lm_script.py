import os
import subprocess
import sys

import torch
import torch.nn as nn

from examples.run_luterize_causal_lm_no_trainer import enable_gradient_checkpointing_if_requested, group_texts, sum_lut_loss
from examples.run_luterize_causal_lm_no_trainer import load_raw_datasets, parse_args


def test_group_texts_splits_flat_tokens_and_drops_remainder():
    examples = {
        "input_ids": [[1, 2, 3], [4, 5, 6, 7]],
        "attention_mask": [[1, 1, 1], [1, 1, 1, 1]],
    }

    grouped = group_texts(examples, max_seq_length=3)

    assert grouped["input_ids"] == [[1, 2, 3], [4, 5, 6]]
    assert grouped["attention_mask"] == [[1, 1, 1], [1, 1, 1]]
    assert grouped["labels"] == [[1, 2, 3], [4, 5, 6]]


class ModuleWithLutLoss(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.lut_loss = torch.tensor(value, dtype=torch.float32)


def test_sum_lut_loss_adds_modules_with_lut_loss():
    model = nn.Sequential(ModuleWithLutLoss(1.25), ModuleWithLutLoss(2.75))

    loss = sum_lut_loss(model, device=torch.device("cpu"))

    assert loss.item() == 4.0


class DummyCheckpointModel:
    def __init__(self):
        self.gradient_checkpointing_enabled = False
        self.input_require_grads_enabled = False

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True

    def enable_input_require_grads(self):
        self.input_require_grads_enabled = True


def test_gradient_checkpointing_enables_input_require_grads_for_frozen_lut_training():
    model = DummyCheckpointModel()
    args = parse_args([])

    enable_gradient_checkpointing_if_requested(model, args)

    assert model.gradient_checkpointing_enabled
    assert model.input_require_grads_enabled


def test_example_scripts_help_run_from_examples_path_without_pythonpath():
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    for script_name in [
        "run_luterize_causal_lm_no_trainer.py",
        "collect_qwen3_lut_centroids.py",
        "create_tiny_qwen3_checkpoint.py",
    ]:
        result = subprocess.run(
            [sys.executable, f"examples/{script_name}", "--help"],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr


def test_load_raw_datasets_from_local_text_files(tmp_path):
    train_file = tmp_path / "train.txt"
    valid_file = tmp_path / "valid.txt"
    train_file.write_text("hello world\nsecond sample\n", encoding="utf-8")
    valid_file.write_text("validation sample\n", encoding="utf-8")
    args = parse_args(
        [
            "--train_file",
            str(train_file),
            "--validation_file",
            str(valid_file),
        ]
    )

    raw_datasets = load_raw_datasets(args)

    assert "train" in raw_datasets
    assert "validation" in raw_datasets
    assert raw_datasets["train"][0]["text"] == "hello world"
    assert raw_datasets["validation"][0]["text"] == "validation sample"
