"""Standalone PyTorch reference for gpumode_sigmoid (debug mirror)."""

import torch


def run(v, a, max):  # noqa: A002 - benchmark ABI
    return torch.sigmoid(a * v) * max
