import torch
import torch.nn as nn

from examples.run_luterize_causal_lm_no_trainer import group_texts, sum_lut_loss


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
