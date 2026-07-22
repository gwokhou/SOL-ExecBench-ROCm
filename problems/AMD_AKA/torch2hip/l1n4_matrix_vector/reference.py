"""Standalone PyTorch reference for l1n4_matrix_vector (debug mirror)."""

import torch


def run(A, B):
    return torch.matmul(A, B)
