"""Frozen layernorm_p2 intervention for numerical judge diagnostics."""

import torch


def run(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: torch.Tensor
) -> torch.Tensor:
    """Apply only the preregistered candidate intervention."""
    mean = x.mean(dim=-1, keepdim=True)
    variance = (x * x).mean(dim=-1, keepdim=True) - mean * mean
    return (x - mean) / torch.sqrt(variance + eps) * weight + bias
