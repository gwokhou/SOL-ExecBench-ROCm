"""Standalone PyTorch reference for 14007_kd_loss (debug mirror)."""

import torch.nn.functional as F


def run(input, target, temperature):  # noqa: A002 - benchmark ABI
    log_p = F.log_softmax(input / temperature, dim=1)
    return (
        F.kl_div(log_p, target, reduction="sum")
        * (temperature * temperature)
        / input.size(0)
    )
