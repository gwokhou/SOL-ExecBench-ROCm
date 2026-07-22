"""Standalone PyTorch reference for l1n26_gelu (debug mirror)."""

import torch.nn.functional as F


def run(x):
    return F.gelu(x)
