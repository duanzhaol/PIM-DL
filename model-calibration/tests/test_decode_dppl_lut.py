import json
import math

import torch
import torch.nn as nn

from examples.run_decode_dppl_lut import (
    Window,
    clone_past_key_values,
    compute_next_token_nll,
    load_dppl_windows,
    move_model_to_device,
    summarize_nll,
)


def test_load_dppl_windows_requires_prompt_decode_plus_one_tokens(tmp_path):
    windows_path = tmp_path / "windows.jsonl"
    windows_path.write_text(
        json.dumps({"id": "ok", "tokens": [1, 2, 3, 4, 5]}) + "\n",
        encoding="utf-8",
    )

    windows = load_dppl_windows(windows_path, prompt_len=2, decode_len=2)

    assert windows == [Window(id="ok", tokens=[1, 2, 3, 4, 5])]


def test_load_dppl_windows_rejects_short_windows(tmp_path):
    windows_path = tmp_path / "windows.jsonl"
    windows_path.write_text(
        json.dumps({"id": "short", "tokens": [1, 2, 3, 4]}) + "\n",
        encoding="utf-8",
    )

    try:
        load_dppl_windows(windows_path, prompt_len=2, decode_len=2)
    except ValueError as exc:
        assert "needs at least 5 tokens" in str(exc)
    else:
        raise AssertionError("short D-PPL windows should be rejected")


def test_compute_next_token_nll_uses_gold_next_token():
    logits = torch.tensor([[-1.0, 2.0, 0.0]], dtype=torch.float32)

    nll = compute_next_token_nll(logits, gold_token_id=1)

    expected = -torch.log_softmax(logits, dim=-1)[0, 1].item()
    assert nll == expected


def test_summarize_nll_reports_dppl():
    summary = summarize_nll(total_nll=math.log(4.0) * 3, tokens=3)

    assert summary["nll"] == math.log(4.0)
    assert summary["d_ppl"] == 4.0
    assert summary["tokens"] == 3


class FakeCache:
    def __init__(self):
        self.tensor = torch.tensor([1.0])

    def get_seq_length(self):
        return 1


def test_clone_past_key_values_preserves_cache_objects():
    cache = FakeCache()

    cloned = clone_past_key_values(cache)

    assert isinstance(cloned, FakeCache)
    assert cloned is not cache
    assert cloned.tensor is not cache.tensor
    cloned.tensor.add_(1.0)
    assert cache.tensor.item() == 1.0


def test_move_model_to_device_returns_model_on_requested_device():
    model = nn.Linear(2, 2)

    moved = move_model_to_device(model, torch.device("cpu"))

    assert moved is model
    assert next(model.parameters()).device.type == "cpu"
