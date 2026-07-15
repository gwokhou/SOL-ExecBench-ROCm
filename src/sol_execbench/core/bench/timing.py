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

from __future__ import annotations

import statistics
from collections.abc import Callable
from typing import Any, Literal, Union

import torch

from sol_execbench.core.bench.io import ShiftingMemoryPoolAllocator


def get_l2_cache_size(device) -> int:
    """Get L2 cache size in bytes for a PyTorch GPU device."""
    props = torch.cuda.get_device_properties(device)
    return int(getattr(props, "L2_cache_size", 128 * 1024 * 1024))


def _summarize_statistics(
    times: list[float],
    return_mode: Literal["mean", "median", "all"],
) -> Union[float, list[float]]:
    """Summarize timing statistics based on return mode."""
    if return_mode == "all":
        return times
    elif return_mode == "mean":
        return statistics.mean(times)
    elif return_mode == "median":
        return statistics.median(times)
    raise ValueError(f"Unknown return_mode: {return_mode}")


def _get_empty_cache_for_benchmark(device) -> torch.Tensor:
    """Create a buffer for clearing GPU L2 cache before benchmark runs."""
    cache_size = max(get_l2_cache_size(device) * 2, 1)
    return torch.empty(int(cache_size), dtype=torch.int8, device=device)


def _clear_cache(cache: torch.Tensor) -> None:
    """Clear the cache buffer by zeroing it."""
    cache.zero_()


def _measurement_budget_reached(
    times_ms: list[float], min_measurement_time_seconds: float | None
) -> bool:
    """Return whether the configured aggregate measurement budget is satisfied."""
    return (
        min_measurement_time_seconds is not None
        and sum(times_ms) >= min_measurement_time_seconds * 1000.0
    )


def clone_args(args: list[Any]) -> list[Any]:
    """Clone tensor arguments to prevent cross-iteration data contamination.

    Returns fresh copies of all tensor arguments so each benchmark iteration
    starts with independent data. Non-tensor arguments are passed through.
    """
    return [arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in args]


def bench_time_with_device_events(
    fn: Callable[..., Any],
    warmup: int = 25,
    rep: int = 100,
    setup: Callable[[], Any] | None = None,
    device: str = "cuda",
    min_measurement_time_seconds: float | None = 0.5,
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
    if min_measurement_time_seconds is not None and min_measurement_time_seconds <= 0:
        raise ValueError("min_measurement_time_seconds must be > 0 or None")
    cache = _get_empty_cache_for_benchmark(device)
    torch.cuda.synchronize()

    if setup is None:
        _fn = fn

        def fn(_):
            return _fn()

        def setup():
            return None

    for _ in range(warmup):
        args = setup()
        _clear_cache(cache)
        fn(args)
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(rep):
        args = setup()
        _clear_cache(cache)
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        fn(args)
        torch.cuda.synchronize()
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))
        if _measurement_budget_reached(times, min_measurement_time_seconds):
            break

    return times


def time_runnable(
    fn: Any,
    inputs: list,
    outputs: list,
    device: str,
    warmup: int = 25,
    rep: int = 100,
    min_measurement_time_seconds: float | None = 0.5,
    return_mode: Literal["mean", "median", "all"] = "median",
    methodology: Literal["events"] = "events",
) -> Union[float, list[float]]:
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
        )
        if not times:
            raise ValueError(f"No timing results for methodology: {methodology}")
        return _summarize_statistics(times, return_mode)
