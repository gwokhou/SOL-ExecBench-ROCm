"""Standalone PyTorch reference for l2n37_groupnorm_fused (debug mirror)."""

import torch
import torch.nn.functional as F


def run(x, weight, bias, extra_bias, gn_weight, gn_bias, num_groups):
    x = F.linear(x, weight, bias)
    x = torch.sigmoid(x) * x
    x = x + extra_bias
    return F.group_norm(x, int(num_groups), gn_weight, gn_bias)
