"""Standalone PyTorch reference for l2n17_instancenorm_fused (debug mirror)."""

import torch.nn.functional as F


def run(x, conv_weight, conv_bias, divide_by):
    x = F.conv2d(x, conv_weight, conv_bias)
    x = F.instance_norm(x)
    return x / divide_by
