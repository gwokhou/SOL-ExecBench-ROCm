"""Standalone PyTorch reference for l2n40_matmul_scaling_residual (debug mirror)."""

import torch.nn.functional as F


def run(x, weight, bias, scaling_factor):
    x = F.linear(x, weight, bias)
    return x * scaling_factor + x
