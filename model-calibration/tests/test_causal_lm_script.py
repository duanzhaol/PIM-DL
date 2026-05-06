import os
import subprocess
import sys

import torch
import torch.nn as nn
from datasets import Dataset
from transformers import Qwen3Config, Qwen3ForCausalLM

from examples.run_luterize_causal_lm_no_trainer import apply_lut_replacement, build_tokenized_lm_datasets, configure_trainable_parameters, enable_gradient_checkpointing_if_requested, group_texts, should_log_interval, sum_lut_loss
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


def test_progress_log_interval_logs_first_and_requested_interval():
    assert should_log_interval(1, 10)
    assert not should_log_interval(9, 10)
    assert should_log_interval(10, 10)
    assert not should_log_interval(1, 0)


def test_progress_logging_defaults_are_chatty_enough_for_long_initial_eval():
    args = parse_args([])

    assert args.eval_logging_steps == 1
    assert args.microbatch_logging_steps == 1


class DummyCheckpointModel:
    def __init__(self):
        self.gradient_checkpointing_enabled = False
        self.gradient_checkpointing_kwargs = None
        self.input_require_grads_enabled = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing_enabled = True
        self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs

    def enable_input_require_grads(self):
        self.input_require_grads_enabled = True


def test_gradient_checkpointing_enables_input_require_grads_for_frozen_lut_training():
    model = DummyCheckpointModel()
    args = parse_args([])

    enable_gradient_checkpointing_if_requested(model, args)

    assert model.gradient_checkpointing_enabled
    assert model.gradient_checkpointing_kwargs == {"use_reentrant": False}
    assert model.input_require_grads_enabled


def test_lut_centroids_receive_gradients_with_gradient_checkpointing():
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    model = Qwen3ForCausalLM(config)
    model.config.use_cache = False
    args = parse_args(
        [
            "--target_modules",
            "mlp",
            "--vec_len",
            "4",
            "--ncentroid",
            "4",
            "--torch_dtype",
            "float32",
        ]
    )
    enable_gradient_checkpointing_if_requested(model, args)
    model = apply_lut_replacement(model, args)
    trainable_params, _, _ = configure_trainable_parameters(model, args)
    batch = {
        "input_ids": torch.randint(0, 128, (1, 16)),
        "attention_mask": torch.ones(1, 16, dtype=torch.long),
        "labels": torch.randint(0, 128, (1, 16)),
    }

    outputs = model(**batch)
    loss = outputs.loss + args.reconstruct_rate * sum_lut_loss(model, torch.device("cpu"))
    loss.backward()
    centroid_grad_norm = sum(
        param.grad.abs().sum().item()
        for param in trainable_params
        if param.grad is not None
    )

    assert centroid_grad_norm > 0


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


def test_build_tokenized_lm_datasets_samples_saved_input_ids_and_adds_labels(tmp_path):
    dataset_path = tmp_path / "tokenized"
    Dataset.from_dict(
        {
            "input_ids": [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
                [17, 18, 19, 20],
                [21, 22, 23, 24],
            ],
            "overflow_to_sample_mapping": [0, 1, 2, 3, 4, 5],
        }
    ).save_to_disk(dataset_path)
    args = parse_args(
        [
            "--tokenized_dataset_path",
            str(dataset_path),
            "--max_train_samples",
            "3",
            "--max_eval_samples",
            "2",
            "--dataset_seed",
            "7",
        ]
    )

    lm_datasets = build_tokenized_lm_datasets(args)

    assert len(lm_datasets["train"]) == 3
    assert len(lm_datasets["validation"]) == 2
    sample = lm_datasets["train"][0]
    assert "overflow_to_sample_mapping" not in sample
    assert sample["attention_mask"] == [1, 1, 1, 1]
    assert sample["labels"] == sample["input_ids"]
