"""Standalone PyTorch reference for gpumode_silu (debug mirror)."""

import torch


def run(x):
    return x * torch.sigmoid(x)
