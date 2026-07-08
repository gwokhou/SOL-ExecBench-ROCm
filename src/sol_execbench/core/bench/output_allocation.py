# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Output normalization and allocation helpers."""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from sol_execbench.core.data.definition import Definition


def normalize_outputs(
    out: Any,
    *,
    device: torch.device,
    output_names: List[str],
    output_dtypes: Dict[str, torch.dtype],
) -> Dict[str, torch.Tensor]:
    def to_tensor(name: str, v: Any) -> torch.Tensor:
        if isinstance(v, torch.Tensor):
            return v.to(device) if v.device != device else v
        dtype = output_dtypes[name]
        return torch.tensor(v, dtype=dtype, device=device)

    if isinstance(out, dict):
        return {k: to_tensor(k, v) for k, v in out.items() if k in output_dtypes}

    if isinstance(out, torch.Tensor):
        if len(output_names) != 1:
            raise RuntimeError(
                "Single Tensor returned but multiple outputs are defined"
            )
        name = output_names[0]
        return {name: to_tensor(name, out)}

    if isinstance(out, (int, float, bool)):
        if len(output_names) != 1:
            raise RuntimeError("Scalar returned but multiple outputs are defined")
        name = output_names[0]
        return {name: to_tensor(name, out)}

    if isinstance(out, (tuple, list)):
        if len(out) != len(output_names):
            raise RuntimeError(
                f"Tuple/list has {len(out)} elements but {len(output_names)} outputs expected"
            )
        return {name: to_tensor(name, val) for name, val in zip(output_names, out)}

    raise RuntimeError(
        "Unexpected return type; must be Tensor, scalar, or dict[name -> Tensor/scalar]"
    )


def allocate_outputs(
    definition: Definition, resolved_axes: dict[str, int], device: str
) -> list[torch.Tensor]:
    """Allocate output tensors based on definition and resolved axis values.

    Parameters
    ----------
    definition : Definition
        The kernel definition specifying output tensor specs.
    resolved_axes : dict[str, int]
        Concrete values for all variable axes (as returned by
        ``definition.get_resolved_axes_values(workload.axes)``).
    device : str
        The device to allocate tensors on (e.g., a PyTorch ROCm GPU device).

    Returns
    -------
    list[torch.Tensor]
        List of allocated (uninitialized) output tensors in definition order.
    """
    output_shapes = list(definition.get_output_shapes(resolved_axes).values())

    dtypes = definition.torch_output_dtypes
    return [
        torch.zeros([] if shape is None else shape, dtype=dtype, device=device)
        for shape, dtype in zip(output_shapes, dtypes)
    ]
