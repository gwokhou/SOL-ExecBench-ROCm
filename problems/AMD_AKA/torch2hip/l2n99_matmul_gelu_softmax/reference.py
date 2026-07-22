"""Standalone PyTorch reference for l2n99_matmul_gelu_softmax (debug mirror)."""

import torch.nn.functional as F


def run(x, weight, bias):
    x = F.linear(x, weight, bias)
    x = F.gelu(x)
    return F.softmax(x, dim=1)
