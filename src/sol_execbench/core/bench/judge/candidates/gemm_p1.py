"""Frozen gemm_p1 intervention for numerical judge diagnostics."""

import torch


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Apply only the preregistered candidate intervention."""
    return torch.matmul(a[:, :-1], b[:-1])
