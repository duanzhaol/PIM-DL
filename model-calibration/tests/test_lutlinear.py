import torch

from LUTNeuro.LUTLinear_t import LUTLinear_t
from LUTNeuro.residual_compensation import input_residual_compensation_correction


def test_lutlinear_forward_preserves_3d_shape_and_dtype():
    layer = LUTLinear_t(
        in_features=16,
        out_features=12,
        ncentroids=4,
        vec_len=4,
        bias=False,
        dtype=torch.bfloat16,
        distance_p="2.0",
    )
    x = torch.randn(2, 5, 16, dtype=torch.bfloat16)

    y = layer(x)

    assert y.shape == (2, 5, 12)
    assert y.dtype == torch.bfloat16
    assert layer.lut_loss.shape == ()
    assert layer.lut_loss.dtype == torch.float32


def test_lutlinear_forward_preserves_2d_shape():
    layer = LUTLinear_t(
        in_features=8,
        out_features=3,
        ncentroids=4,
        vec_len=2,
        bias=True,
        dtype=torch.float32,
        distance_p="2.0",
    )
    x = torch.randn(7, 8)

    y = layer(x)

    assert y.shape == (7, 3)
    assert layer.lut_loss.isfinite()


def test_lutlinear_forward_does_not_gather_output_sized_lut(monkeypatch):
    layer = LUTLinear_t(
        in_features=16,
        out_features=64,
        ncentroids=4,
        vec_len=4,
        bias=False,
        dtype=torch.float32,
        distance_p="2.0",
    )
    x = torch.randn(2, 16, 16)
    max_centroid_gather_indices = layer.ncodebooks * x.shape[0] * x.shape[1] * layer.vec_len
    original_gather = torch.gather

    def guarded_gather(input, dim, index, *args, **kwargs):
        assert index.numel() <= max_centroid_gather_indices
        return original_gather(input, dim, index, *args, **kwargs)

    monkeypatch.setattr(torch, "gather", guarded_gather)

    layer(x)


def test_lutlinear_forward_matches_reference_lut_output():
    torch.manual_seed(0)
    layer = LUTLinear_t(
        in_features=8,
        out_features=5,
        ncentroids=4,
        vec_len=2,
        bias=False,
        dtype=torch.float32,
        distance_p="2.0",
    )
    x = torch.randn(2, 3, 8)

    actual = layer(x)

    tokens = x.reshape(6, 8)
    x_codebooks = tokens.reshape(6, layer.ncodebooks, layer.vec_len).permute(1, 0, 2)
    weight_flat = layer.weight.reshape(layer.ncodebooks, layer.vec_len, layer.out_features)
    centroids = layer.centroids.weight.reshape(layer.ncodebooks, layer.ncentroids, layer.vec_len)
    min_indices = torch.cdist(x_codebooks, centroids, p=2.0).argmin(dim=-1)
    lut = torch.bmm(centroids, weight_flat)
    selected_lut = torch.gather(
        lut,
        1,
        min_indices.unsqueeze(-1).expand(-1, -1, layer.out_features),
    )
    expected = selected_lut.sum(0).reshape(2, 3, 5)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_lutlinear_input_residual_compensation_restores_dense_output_at_full_ratio():
    torch.manual_seed(0)
    layer = LUTLinear_t(
        in_features=8,
        out_features=5,
        ncentroids=4,
        vec_len=2,
        bias=True,
        dtype=torch.float32,
        distance_p="2.0",
        residual_compensation_ratio=1.0,
    )
    layer.eval()
    x = torch.randn(2, 3, 8)

    actual = layer(x)
    expected = x.reshape(6, 8).matmul(layer.weight).reshape(2, 3, 5) + layer.bias

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_lutlinear_input_residual_compensation_selects_largest_residual_channel():
    layer = LUTLinear_t(
        in_features=4,
        out_features=2,
        ncentroids=1,
        vec_len=2,
        bias=False,
        dtype=torch.float32,
        distance_p="2.0",
        residual_compensation_ratio=0.25,
    )
    layer.eval()
    layer.centroids.weight.data.zero_()
    layer.weight.data.copy_(
        torch.tensor(
            [
                [10.0, 0.0],
                [0.0, 20.0],
                [30.0, 0.0],
                [0.0, 40.0],
            ]
        )
    )
    x = torch.tensor([[0.1, 3.0, -0.2, 1.0]])

    actual = layer(x)

    expected = torch.tensor([[0.0, 60.0]])
    assert torch.allclose(actual, expected)


def test_lutlinear_activation_topk_only_selects_largest_activation_channel():
    layer = LUTLinear_t(
        in_features=4,
        out_features=2,
        ncentroids=1,
        vec_len=2,
        bias=False,
        dtype=torch.float32,
        distance_p="2.0",
        residual_compensation_ratio=0.25,
        activation_topk_only=True,
    )
    layer.eval()
    layer.centroids.weight.data.fill_(1.0)
    layer.weight.data.copy_(
        torch.tensor(
            [
                [10.0, 0.0],
                [0.0, 20.0],
                [30.0, 0.0],
                [0.0, 40.0],
            ]
        )
    )
    x = torch.tensor([[0.1, 3.0, -0.2, 1.0]])

    actual = layer(x)

    expected = torch.tensor([[0.0, 60.0]])
    assert torch.allclose(actual, expected)


def test_lutlinear_activation_topk_only_full_ratio_matches_dense_output():
    torch.manual_seed(0)
    layer = LUTLinear_t(
        in_features=8,
        out_features=5,
        ncentroids=4,
        vec_len=2,
        bias=True,
        dtype=torch.float32,
        distance_p="2.0",
        residual_compensation_ratio=1.0,
        activation_topk_only=True,
    )
    layer.eval()
    x = torch.randn(2, 3, 8)

    actual = layer(x)
    expected = x.reshape(6, 8).matmul(layer.weight).reshape(2, 3, 5) + layer.bias

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_input_residual_compensation_memory_safe_fallback_matches_sparse_path():
    torch.manual_seed(0)
    residual = torch.randn(4, 8)
    weight = torch.randn(8, 6)

    sparse_path = input_residual_compensation_correction(
        residual,
        weight,
        ratio=0.5,
        max_materialized_elements=10_000,
    )
    fallback_path = input_residual_compensation_correction(
        residual,
        weight,
        ratio=0.5,
        max_materialized_elements=1,
    )

    assert torch.allclose(fallback_path, sparse_path, atol=1e-6, rtol=1e-6)


def test_lutlinear_l2_forward_does_not_call_cdist(monkeypatch):
    layer = LUTLinear_t(
        in_features=8,
        out_features=5,
        ncentroids=4,
        vec_len=2,
        bias=False,
        dtype=torch.float32,
        distance_p="2.0",
    )
    x = torch.randn(2, 3, 8)

    def fail_cdist(*args, **kwargs):
        raise AssertionError("L2 LUT lookup should use the squared-distance fast path")

    monkeypatch.setattr(torch, "cdist", fail_cdist)

    y = layer(x)

    assert y.shape == (2, 3, 5)


def test_lutlinear_eval_skips_soft_output_matmul(monkeypatch):
    layer = LUTLinear_t(
        in_features=8,
        out_features=5,
        ncentroids=4,
        vec_len=2,
        bias=False,
        dtype=torch.float32,
        distance_p="2.0",
    )
    layer.eval()
    x = torch.randn(2, 3, 8)
    original_matmul = torch.Tensor.matmul
    matmul_calls = 0

    def counting_matmul(self, other):
        nonlocal matmul_calls
        matmul_calls += 1
        return original_matmul(self, other)

    monkeypatch.setattr(torch.Tensor, "matmul", counting_matmul)

    layer(x)

    assert matmul_calls == 1


def test_lutlinear_eval_model_dtype_uses_weight_dtype_matmul(monkeypatch):
    layer = LUTLinear_t(
        in_features=8,
        out_features=5,
        ncentroids=4,
        vec_len=2,
        bias=False,
        dtype=torch.bfloat16,
        distance_p="2.0",
    )
    layer.eval()
    layer.eval_compute_dtype = "model"
    x = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    original_matmul = torch.Tensor.matmul
    matmul_dtypes = []

    def recording_matmul(self, other):
        matmul_dtypes.append((self.dtype, other.dtype))
        return original_matmul(self, other)

    monkeypatch.setattr(torch.Tensor, "matmul", recording_matmul)

    layer(x)

    assert matmul_dtypes == [(torch.bfloat16, torch.bfloat16)]


def test_lutlinear_rejects_bad_vec_len():
    try:
        LUTLinear_t(in_features=10, out_features=4, ncentroids=4, vec_len=4)
    except ValueError as exc:
        assert "in_features must be divisible" in str(exc)
    else:
        raise AssertionError("expected ValueError")
