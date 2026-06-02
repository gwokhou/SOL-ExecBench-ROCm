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

"""Utility functions for benchmark execution."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import torch

from sol_execbench.core.bench.io import allocate_outputs, normalize_outputs
from sol_execbench.core.data import (
    Correctness,
    Evaluation,
    EvaluationStatus,
    Performance,
)
from sol_execbench.core.utils import env_snapshot, flush_stdio_streams


_MAX_EMBEDDED_LOG_BYTES = 5 * 1024 * 1024


def _read_log_file(
    log_path: Optional[str], *, limit: int = _MAX_EMBEDDED_LOG_BYTES
) -> Optional[str]:
    if not log_path:
        return None

    flush_stdio_streams()

    try:
        with open(log_path, "rb") as fh:
            data = fh.read(limit + 1)
    except FileNotFoundError:
        return None
    except OSError:
        return None

    truncated = len(data) > limit
    if truncated:
        data = data[:limit]

    text = data.decode("utf-8", errors="replace")
    if truncated:
        text += "\n\n[log truncated]\n"
    return text


def make_eval(
    status: EvaluationStatus,
    device: str,
    log_path: Optional[str],
    correctness: Optional[Correctness] = None,
    performance: Optional[Performance] = None,
    extra_msg: Optional[str] = None,
) -> Evaluation:
    log_text = _read_log_file(log_path) or ""
    if extra_msg:
        log_text = log_text + "\n" + extra_msg if log_text else extra_msg
    return Evaluation(
        status=status,
        log=log_text,
        environment=env_snapshot(device),
        timestamp=datetime.now().isoformat(),
        correctness=correctness,
        performance=performance,
    )


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
    if destination_passing_style:
        outputs = allocate_outputs(definition, resolved_axes, device)
        fn(*inputs, *outputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        return outputs

    result = fn(*inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    out_dict = normalize_outputs(
        result,
        device=torch.device(device),
        output_names=output_names,
        output_dtypes=output_dtypes,
    )
    return [out_dict[name] for name in output_names]
