"""Standalone PyTorch reference for fp8_cast_sentinel (debug mirror)."""

import torch


def run(x):
    return x.to(torch.float8_e4m3fn)
