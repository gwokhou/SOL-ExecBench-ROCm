"""Standalone PyTorch reference for gpumode_fused_leaky_relu (debug mirror)."""

import torch.nn.functional as F


def run(x, bias, negative_slope, scale):
    x = x + bias.reshape(1, -1, 1, 1)[:, : x.shape[1]]
    x = F.leaky_relu(x, negative_slope=negative_slope)
    return x * scale
