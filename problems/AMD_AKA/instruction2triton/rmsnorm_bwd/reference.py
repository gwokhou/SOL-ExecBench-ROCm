"""Standalone PyTorch reference for rmsnorm_bwd (debug mirror)."""

import torch


def run(x, g, grad_output):
    xr = x.clone().detach().requires_grad_()
    gr = g.clone().detach().requires_grad_()
    rms = torch.sqrt(
        torch.sum(xr.float() ** 2, dim=-1, keepdim=True) * (1.0 / xr.shape[-1])
    )
    y = (xr.float() / rms * gr.float()).to(x.dtype)
    y.backward(grad_output)
    return xr.grad.to(x.dtype), gr.grad.to(x.dtype)
