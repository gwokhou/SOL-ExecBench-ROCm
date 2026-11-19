"""Centered two-pass FP64 LayerNorm for the E2 research reference."""

import torch


def run(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float | torch.Tensor,
) -> torch.Tensor:
    """Evaluate the FP32 problem with centered FP64 statistics and affine."""
    value = x.double()
    shifted = value - value[..., :1]
    centered = shifted - shifted.mean(dim=-1, keepdim=True)
    variance = centered.square().mean(dim=-1, keepdim=True)
    normalized = centered / torch.sqrt(variance + float(eps))
    return (normalized * weight.double() + bias.double()).float()
