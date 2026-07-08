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
from pathlib import Path
from typing import Dict, List, Optional

import torch

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.dtypes import dtype_str_to_torch_dtype
from sol_execbench.core.data.workload import SafetensorsInput, Workload


FLASHINFER_TRACE_ENV = "FLASHINFER_TRACE_DIR"


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


def _resolve_blob_path(rel: Path, blob_roots: List[Path]) -> Optional[Path]:
    """Resolve a relative path against blob roots, handling partial overlap.

    Tries ``root / rel`` first, then progressively strips leading components
    from *rel* so that paths like ``foo/bar/data.safetensors`` can match a
    blob root that already contains the ``foo/`` prefix.
    """
    parts = rel.parts
    for root in blob_roots:
        for start in range(len(parts)):
            candidate = root / Path(*parts[start:])
            if candidate.exists():
                return candidate
    return None


def load_safetensors(
    definition: Definition,
    workload: Workload,
    blob_roots: Optional[List[Path]] = None,
) -> Dict[str, torch.Tensor]:
    """Load safetensors inputs for a workload.

    Safetensors blobs are resolved from the staging directory (passed via
    ``blob_roots``).  The first root whose ``root / path`` exists is used.
    If none match, the path is passed as-is to safetensors (which will raise
    a FileNotFoundError with a clear message).
    """
    try:
        import safetensors.torch as st
    except Exception as e:
        raise RuntimeError(
            "safetensors is not available in the current environment"
        ) from e

    expected = definition.get_input_shapes(workload.axes)

    safe_tensors: Dict[str, torch.Tensor] = {}
    loaded_files: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, input_spec in workload.inputs.items():
        if not isinstance(input_spec, SafetensorsInput):
            continue

        path = input_spec.path
        if not Path(path).is_absolute() and blob_roots:
            resolved = _resolve_blob_path(Path(path), blob_roots)
            if resolved is not None:
                path = str(resolved)

        # Resolve to canonical absolute path so different representations
        # of the same file (symlinks, ".." components) share a cache entry.
        path = str(Path(path).resolve())

        if path not in loaded_files:
            loaded_files[path] = st.load_file(path)
        tensors = loaded_files[path]
        if input_spec.tensor_key not in tensors:
            raise ValueError(f"Missing key '{input_spec.tensor_key}' in '{path}'")
        t = tensors[input_spec.tensor_key]
        if tuple(t.shape) != expected[name]:
            raise ValueError(f"'{name}' expected {expected[name]}, got {list(t.shape)}")
        expect_dtype = dtype_str_to_torch_dtype(definition.inputs[name].dtype)
        if t.dtype != expect_dtype:
            raise ValueError(f"'{name}' expected {expect_dtype}, got {t.dtype}")

        try:
            t = t.contiguous().pin_memory()
        except Exception:
            t = t.contiguous()
        safe_tensors[name] = t
    return safe_tensors


# ── Input generation ─────────────────────────────────────────────────────────
