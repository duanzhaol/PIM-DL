import torch

from examples.benchmark_lut_cpu_kernel import (
    compute_lut_storage_ratio,
    compute_online_read_ratio,
    lut_cpu_kernel,
    precompute_lut,
)


def test_lut_cost_ratios_match_closed_forms():
    assert compute_lut_storage_ratio(ncentroids=32, vec_len=2) == 16.0
    assert compute_online_read_ratio(in_features=2560, out_features=9728, ncentroids=32, vec_len=2) == (
        32 / 9728 + 1 / 2
    )


def test_precompute_lut_matches_codebook_weight_products():
    centroids = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)
    weight = torch.arange(20, dtype=torch.float32).reshape(4, 5)

    lut = precompute_lut(centroids, weight)

    expected = torch.bmm(centroids, weight.reshape(2, 2, 5))
    assert torch.equal(lut, expected)


def test_lut_cpu_kernel_matches_reference_lookup_sum():
    torch.manual_seed(0)
    x = torch.randn(3, 8)
    centroids = torch.randn(4, 5, 2)
    weight = torch.randn(8, 6)
    lut = precompute_lut(centroids, weight)

    actual = lut_cpu_kernel(x, centroids, lut)

    x_codebooks = x.reshape(3, 4, 2).permute(1, 0, 2)
    indices = torch.cdist(x_codebooks, centroids, p=2.0).argmin(dim=-1)
    selected = torch.gather(lut, 1, indices.unsqueeze(-1).expand(-1, -1, 6))
    expected = selected.sum(dim=0)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_lut_cpu_kernel_full_residual_compensation_matches_dense():
    torch.manual_seed(0)
    x = torch.randn(3, 8)
    centroids = torch.zeros(4, 2, 2)
    weight = torch.randn(8, 6)
    lut = precompute_lut(centroids, weight)

    actual = lut_cpu_kernel(
        x,
        centroids,
        lut,
        weight=weight,
        residual_compensation_ratio=1.0,
    )
    expected = x.matmul(weight)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
