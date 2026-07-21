"""Standalone PyTorch reference for gpumode_feedforward (debug mirror)."""

import torch
import torch.nn.functional as F


def run(x, y, fc1_weight, fc1_bias, fc2_weight, fc2_bias):
    inp = torch.vstack([x, y])
    hidden = F.linear(inp, fc1_weight, fc1_bias)
    relu = F.relu(hidden)
    output = F.linear(relu, fc2_weight, fc2_bias)
    return torch.sigmoid(output)
