"""Frozen sum_p1 intervention for numerical judge diagnostics."""

import torch


def run(x: torch.Tensor) -> torch.Tensor:
    """Apply only the preregistered candidate intervention."""
    return torch.sum(x[:, :-1], dim=-1, keepdim=True)
