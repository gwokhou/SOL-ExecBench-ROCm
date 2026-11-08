"""Frozen sum_p0 intervention for numerical judge diagnostics."""

import torch


def run(x: torch.Tensor) -> torch.Tensor:
    """Apply only the preregistered candidate intervention."""
    return torch.sum(x, dim=-1, keepdim=True)
