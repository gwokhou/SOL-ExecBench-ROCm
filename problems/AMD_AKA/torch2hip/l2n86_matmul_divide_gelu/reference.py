"""Standalone PyTorch reference for l2n86_matmul_divide_gelu (debug mirror)."""

import torch.nn.functional as F


def run(x, weight, bias, divisor):
    x = F.linear(x, weight, bias)
    x = x / divisor
    return F.gelu(x)
