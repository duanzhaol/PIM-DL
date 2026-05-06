import torch

from LUTNeuro.LUTLinear_t import LUTLinear_t


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


def test_lutlinear_rejects_bad_vec_len():
    try:
        LUTLinear_t(in_features=10, out_features=4, ncentroids=4, vec_len=4)
    except ValueError as exc:
        assert "in_features must be divisible" in str(exc)
    else:
        raise AssertionError("expected ValueError")
