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

"""Safetensors loading and FlashInfer trace environment helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path

import torch

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.dtypes import dtype_str_to_torch_dtype
from sol_execbench.core.data.workload import SafetensorsInput, Workload
from sol_execbench.core.integrity import validate_relative_artifact_path

FLASHINFER_TRACE_ENV = "FLASHINFER_TRACE_DIR"


def _load_safetensors_file(path: str) -> dict[str, torch.Tensor]:
    """Load one file through the optional native dependency boundary."""
    try:
        from safetensors.torch import load_file
    except Exception as exc:
        raise RuntimeError(
            "safetensors is not available in the current environment",
        ) from exc
    return load_file(path)


def flashinfer_safetensors_env(
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return subprocess env with local FlashInfer safetensors roots enabled.

    FlashInfer workloads store safetensors paths as repo-relative
    ``data/flashinfer-trace/...`` locations. Evaluation runs from a temporary staging
    directory, so subprocesses need an additional root when the local trace checkout is
    available. User-provided ``FLASHINFER_TRACE_DIR`` always wins.
    """
    env = dict(os.environ if base_env is None else base_env)
    if env.get(FLASHINFER_TRACE_ENV):
        return env

    repo_root = Path(__file__).resolve().parents[4]
    if (repo_root / "data" / "flashinfer-trace" / "blob").exists():
        env[FLASHINFER_TRACE_ENV] = str(repo_root)
    return env


def _candidate_within_blob_root(
    root: Path,
    relative_parts: tuple[str, ...],
) -> Path | None:
    if root.is_symlink():
        raise ValueError(f"Safetensors blob root must not be a symlink: {root}")
    resolved_root = root.resolve()
    current = root
    for part in relative_parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Safetensors path traverses a symlink: {current}")
    if not current.exists():
        return None
    if current.is_symlink() or not current.is_file():
        raise ValueError(f"Safetensors blob is not a regular file: {current}")
    resolved_candidate = current.resolve()
    if (
        not resolved_candidate.is_relative_to(resolved_root)
        or resolved_candidate == resolved_root
    ):
        raise ValueError(f"Safetensors blob escapes blob root: {current}")
    return current


def _resolve_blob_path(rel: Path, blob_roots: list[Path]) -> Path | None:
    """Resolve a validated relative path against blob roots."""
    raw = str(rel)
    if "\\" in raw:
        raise ValueError(f"safetensors path must not use backslashes: {raw}")
    safe_rel = validate_relative_artifact_path(
        rel.as_posix(),
        "safetensors path",
    )
    parts = Path(safe_rel).parts
    for root in blob_roots:
        for start in range(len(parts)):
            matched = _candidate_within_blob_root(root, parts[start:])
            if matched is not None:
                return matched
    return None


def _resolve_safetensors_input_path(
    raw_path: str,
    blob_roots: list[Path] | None,
) -> str:
    if "\\" in raw_path or "\x00" in raw_path:
        raise ValueError(
            "safetensors path must be a normalized relative artifact path"
        )
    safe_rel = validate_relative_artifact_path(raw_path, "safetensors path")
    if blob_roots is not None:
        resolved = _resolve_blob_path(Path(safe_rel), blob_roots)
        if resolved is None:
            raise FileNotFoundError(
                f"Safetensors file not found under blob roots: {safe_rel}",
            )
        return str(resolved.resolve())
    candidate = Path(safe_rel)
    current = Path()
    for part in candidate.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(
                f"Safetensors path traverses a symlink: {raw_path}"
            )
    return str(candidate.resolve())


def load_safetensors(
    definition: Definition,
    workload: Workload,
    blob_roots: list[Path] | None = None,
) -> dict[str, torch.Tensor]:
    """Load safetensors inputs for a workload."""
    expected = definition.get_input_shapes(workload.axes)

    safe_tensors: dict[str, torch.Tensor] = {}
    loaded_files: dict[str, dict[str, torch.Tensor]] = {}
    for name, input_spec in workload.inputs.items():
        if not isinstance(input_spec, SafetensorsInput):
            continue

        path = _resolve_safetensors_input_path(input_spec.path, blob_roots)

        if path not in loaded_files:
            loaded_files[path] = _load_safetensors_file(path)
        tensors = loaded_files[path]
        if input_spec.tensor_key not in tensors:
            raise ValueError(
                f"Missing key '{input_spec.tensor_key}' in '{path}'",
            )
        t = tensors[input_spec.tensor_key]
        if tuple(t.shape) != expected[name]:
            raise ValueError(
                f"'{name}' expected {expected[name]}, got {list(t.shape)}",
            )
        expect_dtype = dtype_str_to_torch_dtype(definition.inputs[name].dtype)
        if t.dtype != expect_dtype:
            raise ValueError(f"'{name}' expected {expect_dtype}, got {t.dtype}")

        t = t.contiguous()
        if torch.cuda.is_available():
            with suppress(RuntimeError):
                t = t.pin_memory()
        safe_tensors[name] = t
    return safe_tensors


# ── Input generation ─────────────────────────────────────────────────────────
