"""Standalone PyTorch reference for rope_thd_fwd_bf16 (debug mirror)."""

import torch


def gen_structured_inputs(values, device):
    cases = {
        (1024, 6): [0, 100, 228, 484, 712, 1024],
        (1024, 5): [0, 233, 456, 711, 1024],
        (1024, 9): [0, 100, 102, 128, 233, 456, 460, 711, 1024],
        (2048, 4): [0, 512, 1024, 2048],
    }
    cu = torch.tensor(
        cases[(values["T"], values["S1"])], dtype=torch.int32, device=device
    )
    half = values["D"] // 2
    inv = 1.0 / (10000 ** (torch.arange(half, device=device).float() / half))
    pos = torch.arange(values["T"], device=device).float()
    freqs = (
        torch.einsum("i,j->ij", pos, inv)
        .view(values["T"], 1, 1, half)
        .to(torch.bfloat16)
    )
    return {"cu_seqlens": cu, "freqs": freqs}


def run(input, cu_seqlens, freqs):
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    outputs = []
    for value in torch.split(input, lengths):
        x = value.float().unsqueeze(1)
        angles = freqs[: value.shape[0]].float().repeat(1, 1, 1, 2)
        first, second = x.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        outputs.append((x * angles.cos() + rotated * angles.sin()).squeeze(1))
    return torch.cat(outputs).to(input.dtype)
