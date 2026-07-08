# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Output aliasing helpers for eval driver correctness checks."""

from __future__ import annotations

from typing import Any

import torch


def tensor_storage_id(value: Any) -> tuple[str, int | None, int] | None:
    """Return a stable storage identity for tensor alias checks."""
    if not isinstance(value, torch.Tensor):
        return None
    try:
        return (
            value.device.type,
            value.device.index,
            value.untyped_storage().data_ptr(),
        )
    except Exception:
        return None


def tensor_aliases_any(value: Any, candidates: list[Any]) -> bool:
    """Whether value aliases any tensor storage in candidates."""
    storage_id = tensor_storage_id(value)
    if storage_id is None:
        return False
    return any(storage_id == tensor_storage_id(candidate) for candidate in candidates)


def stable_reference_outputs(outputs: list[torch.Tensor], inputs: list[Any]) -> list[torch.Tensor]:
    """Clone reference outputs that alias inputs so user code cannot mutate them."""
    stable = []
    for output in outputs:
        detached = output.detach()
        if tensor_aliases_any(detached, inputs):
            stable.append(detached.clone())
        else:
            stable.append(detached)
    return stable
