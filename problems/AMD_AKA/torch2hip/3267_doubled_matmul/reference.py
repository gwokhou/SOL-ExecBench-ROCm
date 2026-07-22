"""Standalone PyTorch reference for 3267_doubled_matmul (debug mirror)."""

import torch


def run(a, b):
    return torch.matmul(a, b + b)
