"""Frozen gemm_p2 intervention for numerical judge diagnostics."""

import torch


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Apply only the preregistered candidate intervention."""
    y = torch.matmul(a, b)
    return torch.where((y.abs() > 0) & (y.abs() < 2.0**-12), 0, y)
