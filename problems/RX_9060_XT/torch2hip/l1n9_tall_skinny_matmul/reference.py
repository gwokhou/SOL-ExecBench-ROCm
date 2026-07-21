"""Standalone PyTorch reference for l1n9_tall_skinny_matmul (debug mirror)."""

import torch


def run(A, B):
    return torch.matmul(A, B)
