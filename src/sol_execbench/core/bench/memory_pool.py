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

"""Iteration-shifting tensor memory pool allocator."""

from __future__ import annotations

from typing import Any, Dict, List

import torch


class ShiftingMemoryPoolAllocator:
    """Pre-allocated memory pool that provides unique ``data_ptr`` per iteration.

    Allocates a buffer only slightly larger than the input tensors
    (overhead ≈ ``total_iterations × 256`` bytes per tensor).  The source
    data is retained and copied into an advancing offset of the pool on
    each call to :meth:`get_unique_args`, so every iteration sees a
    distinct ``data_ptr`` while VRAM usage stays near 1× input size.

    Stride patterns (including stride-0 broadcasts from ``expand()``) are
    preserved so that the pool only stores the physical storage footprint,
    not the full logical numel.

    Parameters
    ----------
    inputs : list[Any]
        Initial input values (tensors and scalars) as returned by
        :func:`gen_inputs`.  Tensor data is stored as the copy source;
        scalars are returned as-is on every call.
    outputs : list[torch.Tensor]
        Output tensors for destination-passing-style kernels, as returned
        by :func:`allocate_outputs`.  These are zero-filled at each offset
        on every call.  Pass an empty list for non-DPS kernels.
    total_iterations : int
        Total number of :meth:`get_unique_args` calls expected
        (warmup + timed iterations).
    """

    # Tensor alignment in bytes – shift data_ptr by this much each iteration
    _POOL_ALIGNMENT = 256

    def __init__(
        self,
        inputs: List[Any],
        outputs: List[torch.Tensor],
        total_iterations: int,
    ) -> None:
        self._call_idx = 0
        self._total_iterations = total_iterations
        self._input_entries: List[Dict[str, Any]] = []
        self._output_entries: List[Dict[str, Any]] = []

        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                self._input_entries.append({"scalar": inp})
                continue

            self._input_entries.append(self._make_pool_entry(inp, total_iterations))

        for out in outputs:
            self._output_entries.append(self._make_pool_entry(out, total_iterations))

    @staticmethod
    def _storage_span(tensor: torch.Tensor) -> int:
        """Number of contiguous storage elements spanned by *tensor*.

        For a standard contiguous tensor this equals ``numel()``.  For
        broadcast/expanded tensors (stride 0) this is much smaller — only
        the physically stored elements are counted.
        """
        if tensor.numel() == 0:
            return 0
        span = 1
        for s, st in zip(tensor.shape, tensor.stride()):
            if s > 1:
                span += (s - 1) * st
        return span

    @classmethod
    def _make_pool_entry(
        cls, tensor: torch.Tensor, total_iterations: int
    ) -> Dict[str, Any]:
        # Negative strides (e.g. from flip()) make storage span math
        # ambiguous — materialise to contiguous up front.
        if any(st < 0 for st in tensor.stride()):
            tensor = tensor.contiguous()

        shape = tuple(tensor.shape)
        strides = tensor.stride()
        storage_span = cls._storage_span(tensor)
        elem_size = tensor.element_size()

        # Shift data_ptr by _POOL_ALIGNMENT bytes each iteration.
        stride_numel = max(1, cls._POOL_ALIGNMENT // elem_size)

        # Pool only needs storage_span + (iters-1)*stride extra elements.
        pool_numel = storage_span + (total_iterations - 1) * stride_numel
        pool = torch.empty(pool_numel, dtype=tensor.dtype, device=tensor.device)

        # Flat 1D view of the physical storage this tensor spans.
        source = tensor.as_strided((storage_span,), (1,))

        return {
            "pool": pool,
            "source": source,
            "shape": shape,
            "strides": strides,
            "storage_span": storage_span,
            "stride_numel": stride_numel,
        }

    def get_unique_args(self) -> List[Any]:
        """Copy source data into the next pool offset and return views.

        Returns inputs followed by zero-filled outputs (for DPS kernels).
        Each call advances the internal offset so every returned tensor has
        a unique ``data_ptr``.  Memory between consecutive calls overlaps —
        only the starting address differs.
        """
        if self._call_idx >= self._total_iterations:
            raise RuntimeError(
                f"ShiftingMemoryPoolAllocator exhausted: called {self._call_idx + 1} "
                f"times but was allocated for {self._total_iterations} iterations"
            )

        result: List[Any] = []
        idx = self._call_idx

        for entry in self._input_entries:
            if "scalar" in entry:
                result.append(entry["scalar"])
                continue

            start = idx * entry["stride_numel"]
            entry["pool"].narrow(0, start, entry["storage_span"]).copy_(entry["source"])
            result.append(
                entry["pool"].as_strided(entry["shape"], entry["strides"], start)
            )

        for entry in self._output_entries:
            start = idx * entry["stride_numel"]
            entry["pool"].narrow(0, start, entry["storage_span"]).zero_()
            result.append(
                entry["pool"].as_strided(entry["shape"], entry["strides"], start)
            )

        self._call_idx += 1
        return result
