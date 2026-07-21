"""Standalone PyTorch reference for l1n36_rmsnorm (debug mirror)."""

import torch


def run(x, eps):
    rms = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + eps)
    return x / rms
