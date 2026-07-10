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

    def observe_locked(self) -> bool: ...


class Rdna4ClockController:
    def lock(self) -> bool:
        return clock_lock.lock_clocks()

    def unlock(self) -> None:
        clock_lock.unlock_clocks()

    def observe_locked(self) -> bool:
        return clock_lock.verify_clocks()


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
        return {"rocprof_compute_profile_status": "unknown"}
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
    observations: dict[str, bool | None] = {"pre": None, "during": None, "post": None}
    clock_reason: str | None = None
    probe = request.hip_probe or default_hip_probe(
        architecture=request.environment.architecture
    )
    try:
        if controller is None:
            clock_reason = "clock_lock_adapter_unavailable"
            if request.require_clock_lock:
                raise ClockLockRequiredError(clock_reason)
        else:
            observe = getattr(controller, "observe_locked", None)
            if callable(observe):
                try:
                    observations["pre"] = bool(observe())
                except Exception:
                    observations["pre"] = None
            try:
                locked = controller.lock()
            except Exception:
                locked = False
            if not locked:
                clock_reason = "clock_lock_failed"
                if request.require_clock_lock:
                    raise ClockLockRequiredError(clock_reason)
        observe = getattr(controller, "observe_locked", None) if controller else None
        if callable(observe):
            try:
                observations["during"] = bool(observe())
            except Exception:
                observations["during"] = None
        candidates = tuple(probe.measure(key) for key in adapter.candidates)
    finally:
        # Reset is unconditional: a failed/ambiguous lock command can still
        # have changed policy, and diagnostic collection must leave no setting
        # behind.  Reset failure is reflected by absent post observation.
        if controller is not None:
            try:
                controller.unlock()
            except Exception:
                pass
            observe = getattr(controller, "observe_locked", None)
            if callable(observe):
                try:
                    observations["post"] = bool(observe())
                except Exception:
                    observations["post"] = None
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
        "clock_observations": observations,
        "adapter_policy": {
            "requires_clock_lock": adapter.supports_clock_lock,
            "declared_candidate_keys": [key.value for key in adapter.candidates],
        },
        "gpu_uuid": request.environment.uuid,
        "rocm_version": request.environment.rocm_version,
    }
    if clock_reason:
        metadata["clock_lock_reason"] = clock_reason
    metadata.update(_profile_metadata(request, metric_map))
    validated_from_provenance = (
        adapter.supports_clock_lock
        and locked
        and observations["during"] is True
        and observations["post"] is False
        and bool(candidates)
        and all(candidate.state == "measured" for candidate in candidates)
    )
    metadata["validation_provenance"] = {
        "adapter_requires_clock_lock": adapter.supports_clock_lock,
        "clock_locked": locked,
        "clock_observed_during_sampling": observations["during"] is True,
        "clock_reset_observed": observations["post"] is False,
        "all_declared_profiles_measured": all(
            candidate.state == "measured" for candidate in candidates
        ),
    }
    return HardwareCalibrationArtifact(
        generated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        candidates=candidates,
        collection_status="collected",
        validation_status="validated" if validated_from_provenance else "provisional",
        metadata=metadata,
    )
