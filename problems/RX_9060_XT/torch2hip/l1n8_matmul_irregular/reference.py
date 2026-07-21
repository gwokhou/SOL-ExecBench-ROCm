"""Standalone PyTorch reference for l1n8_matmul_irregular (debug mirror)."""

import torch


def run(A, B):
    return torch.matmul(A, B)
