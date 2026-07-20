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

"""Reward hack defenses for SOL ExecBench evaluation.

Provides detection functions for four common reward-hacking patterns.
The identity of torch.cuda.Event.elapsed_time is captured at module load
time — before any user code is imported — so patching after the fact is
detected.
"""

from __future__ import annotations

from typing import Any, List

import torch

from sol_execbench.core.bench.reward_hack.models import RewardHackDetected


_ELAPSED_TIME_ADDR: int | None = None

try:
    import torch.cuda as _tc_init

    _ELAPSED_TIME_ADDR = id(_tc_init.Event.elapsed_time)
except Exception:
    pass


def check_monkey_patch() -> None:
    """Detect if torch.cuda.Event.elapsed_time has been patched.

    Compares the current function identity against the address captured at
    module load time.  Must be called before the timed section.

    Raises:
        RewardHackDetected: If the timing function has been replaced.
    """
    try:
        import torch.cuda as _tc

        if (
            _ELAPSED_TIME_ADDR is not None
            and id(_tc.Event.elapsed_time) != _ELAPSED_TIME_ADDR
        ):
            raise RewardHackDetected(
                "torch.cuda.Event.elapsed_time has been monkey-patched"
            )
    except RewardHackDetected:
        raise
    except Exception:
        pass


def check_thread_injection(threads_before: int, threads_after: int) -> None:
    """Detect if user code spawned background threads.

    Capture ``threading.active_count()`` before and after the user call and
    pass both values here.

    Raises:
        RewardHackDetected: If the thread count increased.
    """
    if threads_after > threads_before:
        raise RewardHackDetected(
            f"Thread injection detected: "
            f"{threads_after} threads after call vs {threads_before} before"
        )


def check_lazy_outputs(outputs: List[Any]) -> None:
    """Detect lazy/proxy tensors in the user output.

    Uses strict ``type()`` equality — not ``isinstance`` — so any subclass
    (including ``FakeTensor``) is rejected.

    Raises:
        RewardHackDetected: If any output is not exactly ``torch.Tensor``.
    """
    for t in outputs:
        if type(t) is not torch.Tensor:
            raise RewardHackDetected(
                f"Lazy evaluation detected: output is {type(t).__name__}, not torch.Tensor"
            )


def snapshot_critical_functions(namespace: dict, names: List[str]) -> dict[str, int]:
    """Capture ``id()`` of named functions from a namespace.

    Call this **before** user code is imported.  Pass the returned dict to
    :func:`check_eval_integrity` after user code runs.

    Args:
        namespace: The globals dict to snapshot (typically ``globals()``).
        names: Function names to capture.

    Returns:
        Mapping of name → ``id()`` for each name present in *namespace*.
    """
    return {name: id(namespace[name]) for name in names if name in namespace}


def check_eval_integrity(snapshot: dict[str, int], namespace: dict) -> None:
    """Verify that critical eval-driver functions have not been replaced.

    Compares the current ``id()`` of each snapshotted name against the
    value captured before user code was imported.

    Args:
        snapshot: The dict returned by :func:`snapshot_critical_functions`.
        namespace: The current globals dict to check.

    Raises:
        RewardHackDetected: If any function identity has changed.
    """
    for name, expected_id in snapshot.items():
        current = namespace.get(name)
        if current is None or id(current) != expected_id:
            raise RewardHackDetected(
                f"Eval driver integrity violated: '{name}' has been monkey-patched"
            )


def _runtime_integrity_namespace(driver_namespace: dict[str, Any]) -> dict[str, Any]:
    import importlib
    import sys

    namespace = {
        name: value
        for name, value in driver_namespace.items()
        if callable(value) and not name.startswith("_")
    }
    modules = (
        ("eval_correctness", "sol_execbench.core.bench.eval_correctness"),
        ("eval_runtime", "sol_execbench.core.bench.eval_runtime"),
        ("eval_timing", "sol_execbench.core.bench.eval_timing"),
        ("eval_execution", "sol_execbench.core.bench.eval_workload_execution"),
        ("timing", "sol_execbench.core.bench.timing"),
    )
    for prefix, module_name in modules:
        module = sys.modules.get(module_name)
        if module is None:
            module = importlib.import_module(module_name)
        for name, value in vars(module).items():
            if callable(value) and not name.startswith("__"):
                namespace[f"{prefix}.{name}"] = value
    return namespace


def snapshot_runtime_integrity(driver_namespace: dict[str, Any]) -> dict[str, int]:
    """Snapshot driver and cross-module timing/correctness functions."""
    namespace = _runtime_integrity_namespace(driver_namespace)
    return snapshot_critical_functions(namespace, list(namespace))


def check_runtime_integrity(
    snapshot: dict[str, int], driver_namespace: dict[str, Any]
) -> None:
    """Detect monkey-patching across the complete evaluation call graph."""
    check_eval_integrity(snapshot, _runtime_integrity_namespace(driver_namespace))
