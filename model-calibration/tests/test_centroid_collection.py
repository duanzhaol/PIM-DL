import torch

from examples.collect_qwen3_lut_centroids import (
    activation_to_codebook_subvectors,
    centroid_tensor_key,
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
