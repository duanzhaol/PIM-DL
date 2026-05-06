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


def test_lutlinear_rejects_bad_vec_len():
    try:
        LUTLinear_t(in_features=10, out_features=4, ncentroids=4, vec_len=4)
    except ValueError as exc:
        assert "in_features must be divisible" in str(exc)
    else:
        raise AssertionError("expected ValueError")
