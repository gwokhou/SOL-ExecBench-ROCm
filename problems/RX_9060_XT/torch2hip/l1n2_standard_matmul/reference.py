"""Standalone PyTorch reference for l1n2_standard_matmul (debug mirror)."""

import torch


def run(A, B):
    return torch.matmul(A, B)
