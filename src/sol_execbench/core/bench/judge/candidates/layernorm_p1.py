"""Frozen layernorm_p1 intervention for numerical judge diagnostics."""

import torch


def run(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: torch.Tensor
) -> torch.Tensor:
    """Apply only the preregistered candidate intervention."""
    return torch.nn.functional.layer_norm(
        x, (x.shape[-1],), None, None, float(eps)
    )
