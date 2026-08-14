# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Output allocation and collection at the benchmark callable boundary."""

from typing import Any

import torch

from sol_execbench.core.bench.io import allocate_outputs, normalize_outputs


def call_and_collect_outputs(
    fn: Any,
    inputs: list[Any],
    *,
    destination_passing_style: bool,
    definition: Any,
    resolved_axes: dict[str, Any],
    device: str,
    output_names: list[str],
    output_dtypes: dict[str, torch.dtype],
) -> list[torch.Tensor]:
    """Call a benchmark function and normalize its outputs."""
    torch_device = torch.device(device)
    if destination_passing_style:
        outputs = allocate_outputs(definition, resolved_axes, device)
        fn(*inputs, *outputs)
        if torch_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        return outputs

    result = fn(*inputs)
    if torch_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    out_dict = normalize_outputs(
        result,
        device=torch_device,
        output_names=output_names,
        output_dtypes=output_dtypes,
    )
    return [out_dict[name] for name in output_names]


__all__ = ["call_and_collect_outputs"]
