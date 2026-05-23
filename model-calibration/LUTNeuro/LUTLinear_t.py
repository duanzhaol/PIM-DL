import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter, UninitializedParameter

from LUTNeuro.residual_compensation import input_residual_compensation_correction


class LUTLinear_t(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool=True,
        device=None,
        dtype=None,
        nsharecodebooks: int=1,
        ncentroids: int=16,
        vec_len: int=16,
        fp16=False,
        debug=False,
        distance_p="inf",
        eval_compute_dtype="float32",
        residual_compensation_ratio=0.0,
        residual_compensation_metric="abs",
        activation_topk_only=False,
        **factory_kwargs,
    ):
        super().__init__()

        if vec_len <= 0:
            raise ValueError("vec_len must be positive")
        if nsharecodebooks <= 0:
            raise ValueError("nsharecodebooks must be positive")
        if in_features % (vec_len * nsharecodebooks) != 0:
            raise ValueError("in_features must be divisible by vec_len * nsharecodebooks")

        self.ncodebooks = in_features // vec_len // nsharecodebooks
        self.in_features = in_features
        self.out_features = out_features
        self.ncentroids = ncentroids

        assert self.in_features % self.ncodebooks == 0
        self.vec_len = vec_len
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.centroids = nn.Embedding(self.ncodebooks, self.ncentroids * self.vec_len, **factory_kwargs)
        self.weight = Parameter(torch.empty((in_features, out_features), **factory_kwargs, requires_grad=False))

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.debug = False

        self.fp16 = fp16
        self.distance_p = str(distance_p).lower()
        self.eval_compute_dtype = eval_compute_dtype
        self.residual_compensation_ratio = float(residual_compensation_ratio)
        self.residual_compensation_metric = residual_compensation_metric
        self.activation_topk_only = bool(activation_topk_only)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5e-2)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def _uses_l2_distance(self):
        try:
            return float(self.distance_p) == 2.0
        except ValueError:
            return False

    def _forward_compute_dtype(self):
        if self.training or self.eval_compute_dtype == "float32":
            return torch.float32
        if self.eval_compute_dtype == "model":
            return self.weight.dtype
        raise ValueError("eval_compute_dtype must be 'float32' or 'model'")

    def _nearest_centroid_indices(self, x_codebooks, centroids):
        with torch.no_grad():
            if self._uses_l2_distance():
                x_norm = x_codebooks.square().sum(dim=-1, keepdim=True)
                centroid_norm = centroids.square().sum(dim=-1).unsqueeze(1)
                dot = torch.bmm(x_codebooks, centroids.transpose(1, 2))
                return (x_norm - 2.0 * dot + centroid_norm).argmin(dim=-1)

            cdist_x = x_codebooks if x_codebooks.dtype == torch.float32 else x_codebooks.to(torch.float32)
            cdist_centroids = centroids if centroids.dtype == torch.float32 else centroids.to(torch.float32)
            dist = torch.cdist(cdist_x, cdist_centroids, p=float(self.distance_p))
            return dist.argmin(dim=-1)

    def forward(self, x):
        original_dim = x.dim()
        if original_dim == 2:
            batch, in_features = x.shape
            seq_len = 1
        elif original_dim == 3:
            batch, seq_len, in_features = x.shape
        else:
            raise ValueError("LUTLinear_t only supports 2D or 3D inputs")
        if in_features != self.in_features:
            raise ValueError(f"expected input feature size {self.in_features}, got {in_features}")

        print('all shape: ', x.shape, self.centroids.shape, self.weight.shape) if self.debug else None

        compute_dtype = self._forward_compute_dtype()
        x_tokens = x.reshape(batch * seq_len, self.in_features).to(compute_dtype)
        weight = self.weight.to(compute_dtype)
        if self.activation_topk_only:
            quant_output = input_residual_compensation_correction(
                x_tokens,
                weight,
                self.residual_compensation_ratio,
                self.residual_compensation_metric,
            )
            if self.training:
                self.lut_loss = torch.zeros((), dtype=torch.float32, device=x.device)
            output = (
                quant_output.reshape(batch, seq_len, self.out_features)
                if original_dim == 3
                else quant_output.reshape(batch, self.out_features)
            )
            output = output.to(x.dtype)
            if self.bias is not None:
                output = output + self.bias.to(output.dtype)
            return output

        x_codebooks = x_tokens.reshape(batch * seq_len, self.ncodebooks, self.vec_len).permute(1, 0, 2)
        centroids = self.centroids.weight.reshape(self.ncodebooks, self.ncentroids, self.vec_len).to(compute_dtype)

        min_indices = self._nearest_centroid_indices(x_codebooks, centroids)

        selected_centroids = torch.gather(
            centroids,
            1,
            min_indices.unsqueeze(-1).expand(-1, -1, self.vec_len),
        )
        quant_input = selected_centroids.permute(1, 0, 2).reshape(batch * seq_len, self.in_features)
        quant_output = quant_input.matmul(weight)
        if self.residual_compensation_ratio > 0.0:
            residual = x_tokens - quant_input
            quant_output = quant_output + input_residual_compensation_correction(
                residual,
                weight,
                self.residual_compensation_ratio,
                self.residual_compensation_metric,
            )

        if self.training:
            soft_output = x_tokens.matmul(weight)
            self.lut_loss = (torch.mean((quant_output.detach() - soft_output) ** 2) + torch.mean((quant_output - soft_output.detach()) ** 2))

            quant_output = soft_output + (quant_output - soft_output).detach()
        output = quant_output.reshape(batch, seq_len, self.out_features) if original_dim == 3 else quant_output.reshape(batch, self.out_features)
        output = output.to(x.dtype)

        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output
