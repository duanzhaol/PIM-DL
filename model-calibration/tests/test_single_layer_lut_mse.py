import torch
import torch.nn as nn

from examples.evaluate_single_layer_lut_mse import (
    compute_error_metrics,
    dense_linear_output,
    load_layer_cache,
    lut_approximate_linear_output,
    resolve_module,
    save_layer_cache,
)


def test_resolve_module_returns_named_child():
    model = nn.Sequential(nn.Linear(3, 4), nn.Sequential(nn.Linear(4, 5)))

    module = resolve_module(model, "1.0")

    assert isinstance(module, nn.Linear)
    assert module.out_features == 5


def test_lut_approximate_linear_output_matches_reference_lookup_sum():
    activations = torch.tensor(
        [
            [0.1, 0.2, 10.2, 10.1],
            [1.1, 1.2, 20.2, 20.1],
        ],
        dtype=torch.float32,
    )
    weight = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    centroids = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 1.0]],
            [[10.0, 10.0], [20.0, 20.0]],
        ],
        dtype=torch.float32,
    )

    actual = lut_approximate_linear_output(activations, weight, centroids)

    lut = torch.bmm(centroids, weight.reshape(2, 2, 3))
    expected = torch.stack([lut[0, 0] + lut[1, 0], lut[0, 1] + lut[1, 1]])
    assert torch.allclose(actual, expected)


def test_lut_approximate_linear_output_full_residual_compensation_matches_dense():
    torch.manual_seed(0)
    activations = torch.randn(5, 8)
    weight = torch.randn(8, 3)
    bias = torch.randn(3)
    centroids = torch.zeros(4, 2, 2)

    actual = lut_approximate_linear_output(
        activations,
        weight,
        centroids,
        bias=bias,
        residual_compensation_ratio=1.0,
    )
    expected = dense_linear_output(activations, weight, bias)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_compute_error_metrics_reports_relative_mse_and_cosine():
    dense = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    approx = torch.tensor([[1.0, 1.0], [5.0, 4.0]])

    metrics = compute_error_metrics(approx, dense)

    assert metrics["mse"] == 1.25
    assert metrics["relative_mse"] == 1.25 / 7.5
    assert 0.0 < metrics["cosine_similarity"] <= 1.0
    assert metrics["max_abs_error"] == 2.0


def test_layer_cache_round_trip_preserves_tensors(tmp_path):
    cache_path = tmp_path / "layer_cache.pt"
    payload = {
        "module_name": "model.layers.0.mlp.up_proj",
        "calib_activations": torch.randn(4, 8),
        "eval_activations": torch.randn(2, 8),
        "weight": torch.randn(8, 3),
        "bias": torch.randn(3),
        "in_features": 8,
        "out_features": 3,
    }

    save_layer_cache(str(cache_path), payload)
    loaded = load_layer_cache(str(cache_path))

    assert loaded["module_name"] == payload["module_name"]
    assert torch.equal(loaded["calib_activations"], payload["calib_activations"])
    assert torch.equal(loaded["eval_activations"], payload["eval_activations"])
    assert torch.equal(loaded["weight"], payload["weight"])
    assert torch.equal(loaded["bias"], payload["bias"])
