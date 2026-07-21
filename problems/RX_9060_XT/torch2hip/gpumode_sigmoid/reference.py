"""Standalone PyTorch reference for gpumode_sigmoid (debug mirror)."""

import torch


def run(v, a, max):
    return torch.sigmoid(a * v) * max
