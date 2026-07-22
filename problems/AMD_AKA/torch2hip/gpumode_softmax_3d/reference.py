"""Standalone PyTorch reference for gpumode_softmax_3d (debug mirror)."""

import torch.nn.functional as F


def run(v, axis):
    return F.softmax(v, dim=int(axis))
