"""Standalone PyTorch reference for l2n55_matmul_maxpool_sum_scale (debug mirror)."""

import torch
import torch.nn.functional as F


def run(x, weight, bias, kernel_size, scale_factor):
    x = F.linear(x, weight, bias)
    x = F.max_pool1d(x.unsqueeze(1), int(kernel_size)).squeeze(1)
    x = torch.sum(x, dim=1)
    return x * scale_factor
