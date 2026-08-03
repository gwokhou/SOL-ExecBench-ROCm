"""Standalone PyTorch reference for per_token_i8_quant (debug mirror)."""

import torch


def run(input):  # noqa: A002 - benchmark ABI
    value = input.float()
    scale = value.abs().amax(dim=-1, keepdim=True) / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    codes = torch.clamp(value / scale, -128.0, 127.0).to(torch.int8)
    return codes, scale.float()
