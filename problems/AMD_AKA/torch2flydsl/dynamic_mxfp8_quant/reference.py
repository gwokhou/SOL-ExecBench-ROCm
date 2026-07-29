"""Standalone PyTorch reference for dynamic_mxfp8_quant (debug mirror)."""

import torch


def run(input):
    value = input.float()
    m, k = value.shape
    blocks = value.reshape(m, k // 32, 32)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    bits = (amax.contiguous().view(torch.int32) + 0x200000) & -8388608
    power = bits.view(torch.float32)
    exponent = torch.clamp(power.log2().floor() - 8, -127, 127)
    scale = (exponent.to(torch.int32) + 127).to(torch.uint8)
    codes = (blocks * torch.exp2(-exponent)).reshape(m, k)
    return codes.to(torch.float8_e4m3fn), scale.reshape(m, k // 32)
