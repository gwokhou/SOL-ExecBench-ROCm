"""Standalone PyTorch reference for gpumode_tanh (debug mirror)."""

import torch


def run(v, a, max):
    return torch.tanh(a * v) * max
