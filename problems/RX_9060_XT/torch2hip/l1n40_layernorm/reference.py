"""Standalone PyTorch reference for l1n40_layernorm (debug mirror)."""

import torch.nn.functional as F


def run(x, weight, bias, eps):
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
