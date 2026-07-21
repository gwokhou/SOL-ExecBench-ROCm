"""Standalone PyTorch reference for l1n1_square_matmul (debug mirror)."""

import torch


def run(A, B):
    return torch.matmul(A, B)
