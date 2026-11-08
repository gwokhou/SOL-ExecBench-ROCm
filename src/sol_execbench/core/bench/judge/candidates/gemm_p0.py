"""Frozen gemm_p0 intervention for numerical judge diagnostics."""

import torch


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Apply only the preregistered candidate intervention."""
    return torch.matmul(a, b)
