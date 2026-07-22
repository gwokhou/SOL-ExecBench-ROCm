"""Standalone PyTorch reference for sdp_attention (debug mirror)."""

import math
import torch
import torch.nn.functional as F


def run(q, k, v):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    return torch.matmul(F.softmax(scores, dim=-1), v)
