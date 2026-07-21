"""Standalone PyTorch reference for l1n63_conv2d (debug mirror)."""

import torch.nn.functional as F


def run(x, weight, bias):
    return F.conv2d(x, weight, bias, stride=1, padding=0, dilation=1, groups=1)
