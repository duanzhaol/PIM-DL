import torch
from datasets import Dataset

from examples.collect_qwen3_lut_centroids import (
    activation_to_codebook_subvectors,
    build_dataloader,
    centroid_tensor_key,
    parse_args,
)


def test_activation_to_codebook_subvectors_returns_codebook_major_tensor():
    activation = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)

    subvectors = activation_to_codebook_subvectors(activation, vec_len=4)

    assert subvectors.shape == (2, 6, 4)
    assert torch.equal(subvectors[0, 0], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert torch.equal(subvectors[1, 0], torch.tensor([4.0, 5.0, 6.0, 7.0]))
    assert torch.equal(subvectors[0, 1], torch.tensor([8.0, 9.0, 10.0, 11.0]))


def test_centroid_tensor_key_matches_training_loader():
    assert (
        centroid_tensor_key("model.layers.0.mlp.gate_proj")
        == "model.layers.0.mlp.gate_proj.centroids.weight"
    )


def test_build_dataloader_loads_saved_tokenized_dataset(tmp_path):
    dataset_path = tmp_path / "tokenized"
    Dataset.from_dict(
        {
            "input_ids": [
                [1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
                [13, 14, 15, 16, 17, 18],
            ],
            "overflow_to_sample_mapping": [0, 1, 2],
        }
    ).save_to_disk(dataset_path)
    args = parse_args(
        [
            "--tokenized_dataset_path",
            str(dataset_path),
            "--max_samples",
            "2",
            "--max_seq_length",
            "4",
            "--per_device_batch_size",
            "2",
            "--output_path",
            str(tmp_path / "centroids.pt"),
        ]
    )

    dataloader = build_dataloader(args, tokenizer=None, accelerator=None)
    batch = next(iter(dataloader))

    assert len(dataloader.dataset) == 2
    assert batch["input_ids"].shape == (2, 4)
    assert torch.equal(batch["attention_mask"], torch.ones_like(batch["input_ids"]))
    assert torch.equal(batch["labels"], batch["input_ids"])
