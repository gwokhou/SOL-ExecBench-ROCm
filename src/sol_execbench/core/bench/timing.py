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

"""GPU timing helpers backed by PyTorch device events."""

from __future__ import annotations

import statistics
from collections.abc import Callable
from typing import Any, Literal

import torch

from sol_execbench.core.bench.io import ShiftingMemoryPoolAllocator
from sol_execbench.core.bench.reward_hack import RewardHackError
from sol_execbench.core.platform.runtime import (
    CacheClearPolicy,
    cache_clear_policy_for_device,
)


def _summarize_statistics(
    times: list[float],
    return_mode: Literal["mean", "median", "all"],
) -> float | list[float]:
    """Summarize timing statistics based on return mode."""
    if return_mode == "all":
        return times
    if return_mode == "mean":
        return statistics.mean(times)
    if return_mode == "median":
        return statistics.median(times)
    raise ValueError(f"Unknown return_mode: {return_mode}")


def _get_empty_cache_for_benchmark(
    device: str,
    policy: CacheClearPolicy | None = None,
) -> torch.Tensor:
    """Create the target-derived L2 cache-clearing buffer."""
    effective = policy or cache_clear_policy_for_device(device)
    return torch.empty(
        effective.clear_buffer_bytes,
        dtype=torch.int8,
        device=device,
    )


def _clear_cache(cache: torch.Tensor) -> None:
    """Clear the cache buffer by zeroing it."""
    cache.zero_()


def _measurement_budget_reached(
    times_ms: list[float],
    min_measurement_time_seconds: float | None,
) -> bool:
    """Return whether the configured aggregate measurement budget is satisfied."""
    return (
        min_measurement_time_seconds is not None
        and sum(times_ms) >= min_measurement_time_seconds * 1000.0
    )


def _timing_device_index(device: str) -> int | None:
    """Resolve the CUDA device index a timed callable must remain pinned to."""
    if not torch.cuda.is_available():
        return None
    resolved = torch.device(device).index
    return resolved if resolved is not None else torch.cuda.current_device()


def _assert_timing_device_unchanged(target_index: int | None) -> None:
    """Fail closed if the candidate switched the active CUDA device mid-timing."""
    if target_index is not None and torch.cuda.current_device() != target_index:
        raise RewardHackError(
            "candidate switched the active CUDA device during timed execution",
        )


def _pin_timing_device(target_index: int | None) -> None:
    """Re-pin the active device before a timed iteration on multi-GPU hosts.

    Single-GPU targets skip the call entirely so legitimate kernels traverse a
    zero-overhead path and frozen timing evidence is byte-for-byte unaffected.
    """
    if target_index is not None and torch.cuda.device_count() > 1:
        torch.cuda.set_device(target_index)


def clone_args(args: list[Any]) -> list[Any]:
    """Clone tensor arguments to prevent cross-iteration data contamination.

    Returns fresh copies of all tensor arguments so each benchmark iteration
    starts with independent data. Non-tensor arguments are passed through.
    """
    return [
        arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in args
    ]


def bench_time_with_device_events(
    fn: Callable[..., Any],
    warmup: int = 10,
    rep: int = 50,
    setup: Callable[[], Any] | None = None,
    device: str = "cuda",
    min_measurement_time_seconds: float | None = None,
    validator: Callable[[Any, Any], None] | None = None,
    cache_clear_policy: CacheClearPolicy | None = None,
) -> list[float]:
    """Benchmark a GPU callable using PyTorch device events.

    PyTorch exposes HIP-backed ROCm event timing through the historical
    ``torch.cuda.Event`` API, so this remains the ROCm-compatible timing path
    for both PyTorch ROCm and Triton ROCm kernels.

    Setup time is excluded from measurements. The setup callback must not
    explicitly synchronize; it may enqueue default-stream work that should be
    excluded by the synchronization before each start event.
    """
    if warmup < 0 or rep <= 0:
        raise ValueError("warmup must be >= 0 and rep must be > 0")
    if (
        min_measurement_time_seconds is not None
        and min_measurement_time_seconds <= 0
    ):
        raise ValueError("min_measurement_time_seconds must be > 0 or None")
    cache = _get_empty_cache_for_benchmark(device, cache_clear_policy)
    target_device_index = _timing_device_index(device)
    torch.cuda.synchronize()

    if setup is None:
        _fn = fn

        def fn(_: Any) -> Any:
            return _fn()

        def setup() -> Any:
            return None

    for _ in range(warmup):
        args = setup()
        _clear_cache(cache)
        fn(args)
        _assert_timing_device_unchanged(target_device_index)
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(rep):
        args = setup()
        _clear_cache(cache)
        torch.cuda.synchronize()
        _pin_timing_device(target_device_index)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        result = fn(args)
        _assert_timing_device_unchanged(target_device_index)
        torch.cuda.synchronize()
        end_event.record()
        end_event.synchronize()
        times.append(start_event.elapsed_time(end_event))
        if validator is not None:
            validator(args, result)
        if _measurement_budget_reached(times, min_measurement_time_seconds):
            break

    return times


def time_runnable(
    fn: Any,
    inputs: list[Any],
    outputs: list[Any],
    device: str,
    warmup: int = 10,
    rep: int = 50,
    min_measurement_time_seconds: float | None = None,
    return_mode: Literal["mean", "median", "all"] = "mean",
    methodology: Literal["events"] = "events",
    validator: Callable[[list[Any], Any], None] | None = None,
    cache_clear_policy: CacheClearPolicy | None = None,
) -> float | list[float]:
    """Time a callable using ROCm-compatible PyTorch device events.

    Creates a :class:`ShiftingMemoryPoolAllocator` from *inputs* and *outputs*
    so each timed iteration receives arguments with a unique ``data_ptr``.
    Allocator setup time is excluded from measurements, and tensors are
    pre-allocated before the benchmark loop.
    """
    total_iterations = warmup + rep
    allocator = ShiftingMemoryPoolAllocator(inputs, outputs, total_iterations)
    with torch.cuda.device(device):
        if methodology != "events":
            raise ValueError(f"Unknown timing methodology: {methodology}")
        times = bench_time_with_device_events(
            fn=lambda args: fn(*args),
            warmup=warmup,
            rep=rep,
            setup=allocator.get_unique_args,
            device=device,
            min_measurement_time_seconds=min_measurement_time_seconds,
            validator=validator,
            cache_clear_policy=cache_clear_policy,
        )
        if not times:
            raise ValueError(
                f"No timing results for methodology: {methodology}",
            )
        return _summarize_statistics(times, return_mode)
