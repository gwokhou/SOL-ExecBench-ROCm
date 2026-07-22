"""Standalone PyTorch reference for gpumode_gelu_4d (debug mirror)."""

import torch.nn.functional as F


def run(x):
    return F.gelu(x)
