import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter, UninitializedParameter


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

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5e-2)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

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

        x_flat = x.reshape(batch * seq_len, self.ncodebooks, self.vec_len).permute(1, 0, 2).to(torch.float32)
        weight_flat = self.weight.reshape(self.ncodebooks, self.vec_len, self.out_features).to(torch.float32)
        centroids = self.centroids.weight.reshape(self.ncodebooks, self.ncentroids, self.vec_len).to(torch.float32)
        soft_output = torch.bmm(x_flat, weight_flat).sum(0)

        dist = torch.cdist(x_flat, centroids, p=float(self.distance_p))
        min_indices = dist.argmin(dim=-1)

        lut = torch.bmm(centroids, weight_flat)
        selected_lut = torch.gather(
            lut,
            1,
            min_indices.unsqueeze(-1).expand(-1, -1, self.out_features),
        )
        quant_output = selected_lut.sum(0)

        self.lut_loss = (torch.mean((quant_output.detach() - soft_output) ** 2) + torch.mean((quant_output - soft_output.detach()) ** 2))

        quant_output = soft_output + (quant_output - soft_output).detach()
        output = quant_output.reshape(batch, seq_len, self.out_features) if original_dim == 3 else quant_output.reshape(batch, self.out_features)
        output = output.to(x.dtype)

        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output
