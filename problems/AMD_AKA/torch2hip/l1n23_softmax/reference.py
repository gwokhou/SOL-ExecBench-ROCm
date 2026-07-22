"""Standalone PyTorch reference for l1n23_softmax (debug mirror)."""

import torch


def run(x):
    return torch.softmax(x, dim=-1)
