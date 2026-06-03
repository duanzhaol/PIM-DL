import torch
from datasets import Dataset

from examples.collect_qwen3_lut_centroids import (
    activation_to_flat_vectors,
    activation_to_codebook_subvectors,
    build_dataloader,
    centroid_tensor_key,
    fit_centroids_for_module,
    load_activation_cache,
    parse_args,
    save_activation_cache,
)


def test_activation_to_codebook_subvectors_returns_codebook_major_tensor():
    activation = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)

    subvectors = activation_to_codebook_subvectors(activation, vec_len=4)

    assert subvectors.shape == (2, 6, 4)
    assert torch.equal(subvectors[0, 0], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert torch.equal(subvectors[1, 0], torch.tensor([4.0, 5.0, 6.0, 7.0]))
    assert torch.equal(subvectors[0, 1], torch.tensor([8.0, 9.0, 10.0, 11.0]))


def test_activation_to_flat_vectors_flattens_batch_and_sequence_dims():
    activation = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)

    vectors = activation_to_flat_vectors(activation)

    assert vectors.shape == (6, 4)
    assert torch.equal(vectors[0], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert torch.equal(vectors[5], torch.tensor([20.0, 21.0, 22.0, 23.0]))


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


def test_activation_cache_round_trip_preserves_module_activations(tmp_path):
    cache_path = tmp_path / "activation_cache.pt"
    activations = {
        "model.layers.0.mlp.up_proj": torch.randn(3, 8),
        "model.layers.0.mlp.down_proj": torch.randn(4, 16),
    }
    metadata = {"target_modules": "mlp", "nsample": 2}

    save_activation_cache(str(cache_path), activations, metadata)
    loaded = load_activation_cache(str(cache_path))

    assert loaded["metadata"] == metadata
    assert torch.equal(loaded["activations"]["model.layers.0.mlp.up_proj"], activations["model.layers.0.mlp.up_proj"])
    assert torch.equal(loaded["activations"]["model.layers.0.mlp.down_proj"], activations["model.layers.0.mlp.down_proj"])


def test_fit_centroids_for_module_reuses_raw_cache_for_different_vec_lens(tmp_path):
    activations = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [0.1, 0.0, 10.1, 10.0],
            [5.0, 5.0, 20.0, 20.0],
            [5.1, 5.0, 20.1, 20.0],
        ],
        dtype=torch.float32,
    )
    args_v2 = parse_args(
        [
            "--vec_len",
            "2",
            "--ncentroid",
            "2",
            "--kmeans_iter",
            "2",
            "--output_path",
            str(tmp_path / "centroids_v2.pt"),
        ]
    )
    args_v4 = parse_args(
        [
            "--vec_len",
            "4",
            "--ncentroid",
            "2",
            "--kmeans_iter",
            "2",
            "--output_path",
            str(tmp_path / "centroids_v4.pt"),
        ]
    )

    centroids_v2 = fit_centroids_for_module("toy", activations, args_v2)
    centroids_v4 = fit_centroids_for_module("toy", activations, args_v4)

    assert centroids_v2.shape == (2, 4)
    assert centroids_v4.shape == (1, 8)


def test_parse_args_accepts_activation_cache_and_gpu_kmeans_flags(tmp_path):
    args = parse_args(
        [
            "--activation_cache_path",
            str(tmp_path / "cache.pt"),
            "--overwrite_activation_cache",
            "--kmeans_backend",
            "torch-gpu",
            "--kmeans_codebook_block_size",
            "8",
            "--output_path",
            str(tmp_path / "centroids.pt"),
        ]
    )

    assert args.activation_cache_path == str(tmp_path / "cache.pt")
    assert args.overwrite_activation_cache is True
    assert args.kmeans_backend == "torch-gpu"
    assert args.kmeans_codebook_block_size == 8
