"""Standalone PyTorch reference for l1n95_cross_entropy (debug mirror)."""

import torch.nn.functional as F


def run(predictions, targets):
    return F.cross_entropy(predictions, targets)
