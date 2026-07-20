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

"""Custom input generation support for benchmark execution."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

import torch

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.dtypes import dtype_str_to_torch_dtype
from sol_execbench.core.data.workload import Workload


GEN_INPUTS_ERROR = "gen_inputs_error"
GEN_INPUTS_OOM_BLOCKED = "gen_inputs_oom_blocked"
GEN_INPUTS_TIMEOUT = "gen_inputs_timeout"
GEN_INPUTS_SCHEMA_MISMATCH = "gen_inputs_schema_mismatch"
GEN_INPUTS_DEVICE_MISMATCH = "gen_inputs_device_mismatch"


@dataclass(frozen=True)
class CustomInputProvenance:
    entrypoint: str | None
    seed: int
    workload_uuid: str | None
    row_index: int | None
    generated_keys: tuple[str, ...] = ()
    failure_class: str | None = None

    def log_text(self) -> str:
        parts = [
            f"entrypoint={self.entrypoint or '<none>'}",
            f"seed={self.seed}",
            f"workload_uuid={self.workload_uuid or '<none>'}",
            f"row_index={self.row_index if self.row_index is not None else '<none>'}",
            f"generated_keys={list(self.generated_keys)!r}",
        ]
        if self.failure_class:
            parts.append(f"failure_class={self.failure_class}")
        return "custom_inputs " + " ".join(parts)


class CustomInputGenerationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        failure_class: str,
        provenance: CustomInputProvenance,
    ) -> None:
        super().__init__(message)
        self.failure_class = failure_class
        self.provenance = provenance


def derive_custom_input_seed(
    definition: Definition,
    workload: Workload,
    *,
    row_index: int | None = None,
    base_seed: int | None = None,
    round_index: int | None = None,
) -> int:
    """Derive a stable per-workload and per-round seed."""
    parts = [
        definition.name,
        getattr(workload, "uuid", "") or "",
        "" if row_index is None else str(row_index),
    ]
    if base_seed is not None or round_index is not None:
        parts.extend(
            (
                "" if base_seed is None else str(base_seed),
                "" if round_index is None else str(round_index),
            )
        )
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


@contextmanager
def isolated_torch_rng(seed: int):
    cpu_state = torch.random.get_rng_state()
    cuda_states = None
    if torch.cuda.is_available():
        try:
            cuda_states = torch.cuda.get_rng_state_all()
        except Exception:
            cuda_states = None
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            try:
                torch.cuda.manual_seed_all(seed)
            except Exception:
                pass
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            try:
                torch.cuda.set_rng_state_all(cuda_states)
            except Exception:
                pass


def _custom_input_provenance(
    definition: Definition,
    workload: Workload,
    *,
    row_index: int | None,
    seed: int,
    generated_keys: Sequence[str] = (),
    failure_class: str | None = None,
) -> CustomInputProvenance:
    return CustomInputProvenance(
        entrypoint=definition.custom_inputs_entrypoint,
        seed=seed,
        workload_uuid=getattr(workload, "uuid", None),
        row_index=row_index,
        generated_keys=tuple(sorted(generated_keys)),
        failure_class=failure_class,
    )


def _raise_custom_input_error(
    message: str,
    *,
    failure_class: str,
    provenance: CustomInputProvenance,
) -> None:
    raise CustomInputGenerationError(
        message,
        failure_class=failure_class,
        provenance=replace(provenance, failure_class=failure_class),
    )


def _classify_custom_generation_exception(exc: BaseException) -> str:
    text = str(exc).lower()
    name = type(exc).__name__.lower()
    if isinstance(exc, TimeoutError) or "timeout" in text or "timed out" in text:
        return GEN_INPUTS_TIMEOUT
    if (
        isinstance(exc, torch.cuda.OutOfMemoryError)
        or "outofmemory" in name
        or "out of memory" in text
        or "hip out of memory" in text
        or "cuda out of memory" in text
    ):
        return GEN_INPUTS_OOM_BLOCKED
    return GEN_INPUTS_ERROR


def _validate_custom_tensors(
    definition: Definition,
    workload: Workload,
    generated: Any,
    *,
    device: torch.device,
    provenance: CustomInputProvenance,
) -> dict[str, Any]:
    if not isinstance(generated, Mapping):
        _raise_custom_input_error(
            "custom_inputs_entrypoint must return a mapping of input names to values",
            failure_class=GEN_INPUTS_SCHEMA_MISMATCH,
            provenance=provenance,
        )

    expected_names = set(definition.inputs.keys())
    generated_names = set(generated.keys())
    missing = sorted(expected_names - generated_names)
    if missing:
        _raise_custom_input_error(
            f"custom_inputs_entrypoint missing required input keys: {missing}",
            failure_class=GEN_INPUTS_SCHEMA_MISMATCH,
            provenance=provenance,
        )
    unexpected = sorted(generated_names - expected_names)
    if unexpected:
        _raise_custom_input_error(
            f"custom_inputs_entrypoint returned unexpected input keys: {unexpected}",
            failure_class=GEN_INPUTS_SCHEMA_MISMATCH,
            provenance=provenance,
        )

    shapes = definition.get_input_shapes(workload.axes)
    validated: dict[str, Any] = {}
    for name, spec in definition.inputs.items():
        value = generated[name]
        expected_shape = shapes[name]
        expected_dtype = dtype_str_to_torch_dtype(spec.dtype)
        if expected_shape is None:
            if isinstance(value, torch.Tensor):
                _raise_custom_input_error(
                    f"'{name}' expected scalar, got tensor",
                    failure_class=GEN_INPUTS_SCHEMA_MISMATCH,
                    provenance=provenance,
                )
            if not isinstance(value, (int, float, bool)):
                _raise_custom_input_error(
                    f"'{name}' expected scalar, got {type(value).__name__}",
                    failure_class=GEN_INPUTS_SCHEMA_MISMATCH,
                    provenance=provenance,
                )
            validated[name] = value
            continue

        if not isinstance(value, torch.Tensor):
            _raise_custom_input_error(
                f"'{name}' expected tensor, got {type(value).__name__}",
                failure_class=GEN_INPUTS_SCHEMA_MISMATCH,
                provenance=provenance,
            )
        if tuple(value.shape) != expected_shape:
            _raise_custom_input_error(
                f"'{name}' expected shape {expected_shape}, got {tuple(value.shape)}",
                failure_class=GEN_INPUTS_SCHEMA_MISMATCH,
                provenance=provenance,
            )
        if value.dtype != expected_dtype:
            _raise_custom_input_error(
                f"'{name}' expected dtype {expected_dtype}, got {value.dtype}",
                failure_class=GEN_INPUTS_SCHEMA_MISMATCH,
                provenance=provenance,
            )
        if value.device != device:
            _raise_custom_input_error(
                f"'{name}' expected device {device}, got {value.device}",
                failure_class=GEN_INPUTS_DEVICE_MISMATCH,
                provenance=provenance,
            )
        validated[name] = value
    return validated


def gen_custom_inputs(
    definition: Definition,
    workload: Workload,
    device: torch.device,
    custom_inputs_fn: Any,
    *,
    row_index: int | None = None,
    seed: int | None = None,
) -> tuple[dict[str, Any], CustomInputProvenance]:
    seed = (
        derive_custom_input_seed(definition, workload, row_index=row_index)
        if seed is None
        else int(seed)
    )
    provenance = _custom_input_provenance(
        definition,
        workload,
        row_index=row_index,
        seed=seed,
    )
    axes_and_scalars = {
        **definition.get_resolved_axes_values(workload.axes),
        **workload.get_scalar_inputs(),
    }
    try:
        with isolated_torch_rng(seed):
            generated = custom_inputs_fn(axes_and_scalars, device)
    except Exception as exc:
        failure_class = _classify_custom_generation_exception(exc)
        err = CustomInputGenerationError(
            f"custom_inputs_entrypoint failed: {exc}",
            failure_class=failure_class,
            provenance=replace(provenance, failure_class=failure_class),
        )
        raise err from exc
    generated_keys = tuple(generated.keys()) if isinstance(generated, Mapping) else ()
    provenance = _custom_input_provenance(
        definition,
        workload,
        row_index=row_index,
        seed=seed,
        generated_keys=generated_keys,
    )
    return (
        _validate_custom_tensors(
            definition,
            workload,
            generated,
            device=device,
            provenance=provenance,
        ),
        provenance,
    )
