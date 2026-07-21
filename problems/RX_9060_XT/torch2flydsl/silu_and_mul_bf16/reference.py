"""Standalone PyTorch reference for silu_and_mul_bf16 (debug mirror)."""

import torch
import torch.nn.functional as F


def run(input):
    d = input.shape[-1] // 2
    x, y = input.split([d, d], dim=-1)
    return (F.silu(x.float()) * y.float()).to(torch.bfloat16)
