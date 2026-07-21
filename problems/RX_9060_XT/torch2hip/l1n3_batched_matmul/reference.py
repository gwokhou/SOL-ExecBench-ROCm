"""Standalone PyTorch reference for l1n3_batched_matmul (debug mirror)."""

import torch


def run(A, B):
    return torch.bmm(A, B)
