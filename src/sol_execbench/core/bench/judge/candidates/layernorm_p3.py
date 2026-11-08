"""Frozen layernorm_p3 intervention for numerical judge diagnostics."""

import torch


def run(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: torch.Tensor
) -> torch.Tensor:
    """Apply only the preregistered candidate intervention."""
    value = x.double()
    mean = value.mean(dim=-1, keepdim=True)
    variance = ((value - mean) ** 2).mean(dim=-1, keepdim=True)
    result = (value - mean) / torch.sqrt(variance + float(eps))
    return (result * weight.double() + bias.double()).to(x.dtype)
