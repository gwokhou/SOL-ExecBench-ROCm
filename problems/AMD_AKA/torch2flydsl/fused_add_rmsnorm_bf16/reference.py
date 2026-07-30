"""Standalone PyTorch reference for fused_add_rmsnorm_bf16 (debug mirror)."""

import torch


def run(input, weight, residual, eps):  # noqa: A002 - benchmark ABI
    residual_out = input + residual
    value = residual_out.float()
    rstd = torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + eps)
    output = value * rstd * weight.float()
    return output.to(input.dtype), residual_out.to(input.dtype)
