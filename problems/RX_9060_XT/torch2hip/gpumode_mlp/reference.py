"""Standalone PyTorch reference for gpumode_mlp (debug mirror)."""

import torch.nn.functional as F


def run(
    xb,
    linear1_weight,
    linear1_bias,
    linear2_weight,
    linear2_bias,
    linear3_weight,
    linear3_bias,
    linear4_weight,
    linear4_bias,
    linear5_weight,
    linear5_bias,
    linear6_weight,
    linear6_bias,
    linear7_weight,
    linear7_bias,
):
    xb = xb.view(xb.size(0), -1)
    out = F.relu(F.linear(xb, linear1_weight, linear1_bias))
    out = F.relu(F.linear(out, linear2_weight, linear2_bias))
    out = F.relu(F.linear(out, linear3_weight, linear3_bias))
    out = F.relu(F.linear(out, linear4_weight, linear4_bias))
    out = F.relu(F.linear(out, linear5_weight, linear5_bias))
    out = F.relu(F.linear(out, linear6_weight, linear6_bias))
    return F.linear(out, linear7_weight, linear7_bias)
