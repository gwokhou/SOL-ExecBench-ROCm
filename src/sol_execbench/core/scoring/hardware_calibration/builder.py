# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Orchestrate HIP calibration while preserving explicit evidence states."""

from __future__ import annotations

import csv
import hashlib
import io
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable, Protocol

from sol_execbench.core.bench import clock_lock
from sol_execbench.core.scoring.hardware_calibration.environment import (
    GpuEnvironment,
    adapter_for,
)
from sol_execbench.core.scoring.hardware_calibration.hip_probe import (
    HipProbe,
    default_hip_probe,
)
from sol_execbench.core.scoring.hardware_calibration.models import (
    HardwareCalibrationArtifact,
)
from sol_execbench.core.scoring.hardware_calibration.rocprof_compute import (
    ProfilerEnvironment,
    run_rocprof_compute_bench_only,
)


class ClockController(Protocol):
    def lock(self) -> bool: ...

    def unlock(self) -> None: ...


class Rdna4ClockController:
    def lock(self) -> bool:
        return clock_lock.lock_clocks()

    def unlock(self) -> None:
        clock_lock.unlock_clocks()


class ClockLockRequiredError(RuntimeError):
    pass


@dataclass(frozen=True)
class CalibrationRequest:
    environment: GpuEnvironment
    hip_probe: HipProbe | None = None
    require_clock_lock: bool = False
    clock_controller: ClockController | None = None
    profiler_environment: ProfilerEnvironment | None = None
    profiler_run: Callable[..., object] | None = None


def _profile_metadata(
    request: CalibrationRequest, metric_map: dict[str, tuple[tuple[str, str], ...]]
) -> dict[str, object]:
    environment = request.profiler_environment
    if environment is None or environment.state != "measured":
        return {}
    try:
        result = run_rocprof_compute_bench_only(
            environment, run=request.profiler_run or subprocess.run
        )
        raw_output = str(getattr(result, "stdout", "") or "")
    except Exception:
        return {"rocprof_compute_profile_status": "unknown"}
    digest = hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
    try:
        observed = {
            str(row.get("Metric", "")).strip()
            for row in csv.DictReader(io.StringIO(raw_output))
            if str(row.get("Metric", "")).strip()
        }
    except csv.Error:
        observed = set()
    recognised = [
        candidate_key
        for metric in observed
        for candidate_key, _unit in metric_map.get(metric, ())
    ]
    if not recognised:
        return {
            "rocprof_compute_profile_status": "unknown",
            "rocprof_compute_raw_output_sha256": digest,
            "rocprof_compute_recognised_candidate_keys": [],
        }
    return {
        "rocprof_compute_profile_status": "collected",
        "rocprof_compute_raw_output_sha256": digest,
        "rocprof_compute_recognised_candidate_keys": recognised,
    }


def run_calibration(request: CalibrationRequest) -> HardwareCalibrationArtifact:
    """Run declared candidates and return an evidence artifact, never defaults."""
    adapter = adapter_for(request.environment.architecture)
    controller = request.clock_controller
    if controller is None and adapter.supports_clock_lock:
        controller = Rdna4ClockController()
    locked = False
    clock_reason: str | None = None
    if controller is None:
        clock_reason = "clock_lock_adapter_unavailable"
        if request.require_clock_lock:
            raise ClockLockRequiredError(clock_reason)
    else:
        try:
            locked = controller.lock()
        except Exception:
            locked = False
        if not locked:
            clock_reason = "clock_lock_failed"
            if request.require_clock_lock:
                raise ClockLockRequiredError(clock_reason)
    probe = request.hip_probe or default_hip_probe()
    try:
        candidates = tuple(probe.measure(key) for key in adapter.candidates)
    finally:
        if locked and controller is not None:
            controller.unlock()
    metric_map = {
        "Peak FP32": tuple(
            (key.value, "TFLOP/s")
            for key in adapter.candidates
            if key.kind == "compute" and key.operation == "vector"
        )
    }
    metadata: dict[str, object] = {
        "device": request.environment.device,
        "architecture": request.environment.architecture,
        "adapter_family": adapter.family,
        "clock_locked": locked,
    }
    if clock_reason:
        metadata["clock_lock_reason"] = clock_reason
    metadata.update(_profile_metadata(request, metric_map))
    return HardwareCalibrationArtifact(
        generated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        candidates=candidates,
        collection_status="collected",
        validation_status="validated" if locked else "provisional",
        metadata=metadata,
    )
