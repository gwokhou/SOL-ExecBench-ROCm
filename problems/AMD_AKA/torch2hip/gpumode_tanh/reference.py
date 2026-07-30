"""Standalone PyTorch reference for gpumode_tanh (debug mirror)."""

import torch


def run(v, a, max):  # noqa: A002 - benchmark ABI
    return torch.tanh(a * v) * max
