"""Frozen sum_p2 intervention for numerical judge diagnostics."""

import torch


def run(x: torch.Tensor) -> torch.Tensor:
    """Apply only the preregistered candidate intervention."""
    return torch.sum(
        torch.where(x.abs() < 2.0**-12, 0, x), dim=-1, keepdim=True
    )
