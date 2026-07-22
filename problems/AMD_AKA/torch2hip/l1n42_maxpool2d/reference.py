"""Standalone PyTorch reference for l1n42_maxpool2d (debug mirror)."""

import torch.nn.functional as F


def run(x):
    return F.max_pool2d(x, kernel_size=2, stride=2)
