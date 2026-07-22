"""Standalone PyTorch reference for l2n98_matmul_avgpool_gelu_scale_max (debug mirror)."""

import torch
import torch.nn.functional as F


def run(x, weight, bias, pool_kernel_size, scale_factor):
    x = F.linear(x, weight, bias)
    x = F.avg_pool1d(x.unsqueeze(1), int(pool_kernel_size)).squeeze(1)
    x = F.gelu(x)
    x = x * scale_factor
    return torch.max(x, dim=1).values
