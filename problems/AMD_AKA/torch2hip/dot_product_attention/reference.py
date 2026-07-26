"""Standalone PyTorch reference for dot_product_attention (debug mirror)."""

import torch
import torch.nn.functional as F


def run(
    x,
    query_weight,
    query_bias,
    key_weight,
    key_bias,
    value_weight,
    value_bias,
    gamma_weight,
    gamma_bias,
):
    b, c, h, w = x.shape
    query = F.conv2d(x, query_weight, query_bias)
    query = query.view(b, -1, h * w).permute(0, 2, 1)
    key = F.conv2d(x, key_weight, key_bias).view(b, -1, h * w)
    energy = torch.bmm(query, key)
    energy = F.elu(energy) / (h * w)
    value = F.conv2d(x, value_weight, value_bias).view(b, c, h * w)
    out = torch.bmm(value, energy).view(b, c, h, w)
    return F.conv2d(out, gamma_weight, gamma_bias)
