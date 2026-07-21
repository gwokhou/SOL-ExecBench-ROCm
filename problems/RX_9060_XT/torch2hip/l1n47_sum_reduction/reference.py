"""Standalone PyTorch reference for l1n47_sum_reduction (debug mirror)."""

import torch


def run(x):
    return torch.sum(x, dim=-1, keepdim=True)
