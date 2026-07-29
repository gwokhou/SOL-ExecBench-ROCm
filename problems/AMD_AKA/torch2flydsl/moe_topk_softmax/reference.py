"""Standalone PyTorch reference for moe_topk_softmax (debug mirror)."""

import torch


def gen_gating(values, device):
    experts, tokens = values["E"], values["T"]
    base = torch.arange(-1, 1, 2.0 / experts, device=device)[:experts]
    gating = base.repeat(tokens, 1).to(torch.bfloat16)
    perm = torch.argsort(torch.rand(gating.shape, device=device), dim=-1)
    return {"gating": torch.gather(gating, -1, perm).contiguous()}


def run(gating, bias, topk, route_scale):
    scores = torch.softmax(gating.float(), dim=-1)
    ids = (scores + bias.float()).topk(topk, dim=-1, sorted=False).indices
    weights = scores.gather(1, ids) * route_scale
    return weights.float(), ids.to(torch.int32)
