# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Orchestrate HIP calibration while preserving explicit evidence states."""

from __future__ import annotations

import csv
import hashlib
import io
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable, Protocol

from sol_execbench.core.bench import clock_lock
from sol_execbench.core.platform.arch_capabilities import (
    derive_arch_capability_budget,
)
from sol_execbench.core.platform.isa_validation import (
    IsaCapabilityReport,
    inspect_isa_requirements,
)
from sol_execbench.core.scoring.hardware_calibration.environment import (
    ArchitectureAdapter,
    GpuEnvironment,
    adapter_for,
)
from sol_execbench.core.scoring.hardware_calibration.hip_probe import (
    CalibrationProfileKey,
    HipProbe,
    default_hip_probe,
    matrix_isa_requirements,
)
from sol_execbench.core.scoring.hardware_calibration.models import (
    CALIBRATION_SCHEMA_VERSION,
    CalibrationCandidate,
    HardwareCalibrationArtifact,
)
from sol_execbench.core.scoring.hardware_calibration.rocprof_compute import (
    ProfilerEnvironment,
    run_rocprof_compute_bench_only,
)


class ClockController(Protocol):
    def lock(self) -> bool: ...

    def unlock(self) -> bool | None: ...

    def observe_locked(self) -> bool: ...


class Rdna4ClockController:
    def __init__(self) -> None:
        self._lease: clock_lock.ClockLockLease | None = None

    def lock(self) -> bool:
        self._lease = clock_lock.acquire_clock_lock()
        return self._lease.locked

    def unlock(self) -> bool:
        return self._lease is None or self._lease.release()

    def observe_locked(self) -> bool:
        return clock_lock.verify_clocks()


class ClockLockRequiredError(RuntimeError):
    pass


@dataclass(frozen=True, kw_only=True)
class CalibrationRequest:
    environment: GpuEnvironment
    hip_probe: HipProbe | None = None
    require_clock_lock: bool = False
    clock_controller: ClockController | None = None
    profiler_environment: ProfilerEnvironment | None = None
    profiler_run: Callable[..., object] | None = None
    required_profile_keys: tuple[str, ...] = ()
    requirements_ref: str | None = None
    requirements_sha256: str | None = None
    source_revision: str | None = None
    allow_isa_download: bool = True


@dataclass(frozen=True)
class ClockedMeasurements:
    locked: bool
    observations: dict[str, bool | None]
    clock_reason: str | None
    candidates: tuple[CalibrationCandidate, ...]


def _observe_locked(controller: ClockController) -> bool | None:
    observe = getattr(controller, "observe_locked", None)
    if not callable(observe):
        return None
    try:
        return bool(observe())
    except Exception:
        return None


def _measure_with_clock(
    *,
    controller: ClockController | None,
    probe: HipProbe,
    measurement_keys: tuple[CalibrationProfileKey, ...],
    require_clock_lock: bool,
) -> ClockedMeasurements:
    locked = False
    lock_attempted = False
    observations: dict[str, bool | None] = {"pre": None, "during": None, "post": None}
    clock_reason: str | None = None
    try:
        if controller is None:
            clock_reason = "clock_lock_adapter_unavailable"
            if require_clock_lock:
                raise ClockLockRequiredError(clock_reason)
        else:
            observations["pre"] = _observe_locked(controller)
            if observations["pre"] is True:
                locked = True
            else:
                lock_attempted = True
                try:
                    locked = controller.lock()
                except Exception:
                    locked = False
            if not locked:
                clock_reason = "clock_lock_failed"
                if require_clock_lock:
                    raise ClockLockRequiredError(clock_reason)
        if controller is not None:
            observations["during"] = _observe_locked(controller)
        candidates = tuple(probe.measure(key) for key in measurement_keys)
    finally:
        active_exception = sys.exception()
        cleanup_failed = False
        if controller is not None:
            if lock_attempted:
                try:
                    cleanup_failed = controller.unlock() is False
                except Exception:
                    cleanup_failed = True
            observations["post"] = _observe_locked(controller)
        if cleanup_failed:
            if active_exception is None:
                raise ClockLockRequiredError("clock_unlock_failed")
            active_exception.add_note("clock_unlock_failed")
    return ClockedMeasurements(locked, observations, clock_reason, candidates)


def _resolve_measurement_keys(
    adapter: ArchitectureAdapter, request: CalibrationRequest
) -> tuple[tuple[str, ...], tuple[CalibrationProfileKey, ...]]:
    declared_keys = {key.value for key in adapter.all_candidates}
    required_keys = (
        tuple(sorted(set(request.required_profile_keys)))
        if request.required_profile_keys
        else tuple(sorted(key.value for key in adapter.candidates))
    )
    unknown_required = set(required_keys) - declared_keys
    if unknown_required:
        raise ValueError(
            "calibration requirements contain undeclared profiles: "
            + ", ".join(sorted(unknown_required))
        )
    measurement_keys = tuple(
        key
        for key in adapter.all_candidates
        if key in adapter.candidates or key.value in required_keys
    )
    return required_keys, measurement_keys


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


def _validation_provenance(
    adapter: ArchitectureAdapter,
    measurements: ClockedMeasurements,
    required_keys: tuple[str, ...],
    *,
    injected_probe: bool,
) -> tuple[bool, dict[str, bool]]:
    observations = measurements.observations
    required_profiles_measured = all(
        candidate.state == "measured"
        for candidate in measurements.candidates
        if candidate.key in required_keys
    )
    clock_state_restored = (
        observations["pre"] is not None and observations["post"] == observations["pre"]
    )
    all_required_isa_paths_verified = (
        all(
            candidate.isa_validation is not None
            and candidate.isa_validation.status == "verified"
            for candidate in measurements.candidates
            if candidate.key in required_keys
            and candidate.key.rsplit(".", maxsplit=1)[-1] in {"mfma", "wmma"}
        )
        or injected_probe
    )
    provenance = {
        "adapter_requires_clock_lock": adapter.supports_clock_lock,
        "clock_locked": measurements.locked,
        "clock_observed_during_sampling": observations["during"] is True,
        "clock_state_restored": clock_state_restored,
        "clock_reset_observed": (
            observations["pre"] is False and observations["post"] is False
        ),
        "all_required_profiles_measured": required_profiles_measured,
        "all_required_isa_paths_verified": all_required_isa_paths_verified,
    }
    validated = (
        adapter.supports_clock_lock
        and measurements.locked
        and observations["during"] is True
        and clock_state_restored
        and bool(measurements.candidates)
        and required_profiles_measured
        and all_required_isa_paths_verified
    )
    return validated, provenance


def run_calibration(request: CalibrationRequest) -> HardwareCalibrationArtifact:
    """Run declared candidates and return an evidence artifact, never defaults."""
    adapter, probe, capability_report = _calibration_runtime(request)
    controller = request.clock_controller
    if controller is None and adapter.supports_clock_lock:
        controller = Rdna4ClockController()
    required_keys, measurement_keys = _resolve_measurement_keys(adapter, request)
    measurements = _measure_with_clock(
        controller=controller,
        probe=probe,
        measurement_keys=measurement_keys,
        require_clock_lock=request.require_clock_lock,
    )
    observations = measurements.observations
    candidates = measurements.candidates
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
        "clock_locked": measurements.locked,
        "clock_observations": observations,
        "adapter_policy": {
            "requires_clock_lock": adapter.supports_clock_lock,
            "declared_candidate_keys": [key.value for key in adapter.candidates],
        },
        "gpu_uuid": request.environment.uuid,
        "rocm_version": request.environment.rocm_version,
        "profile_requirements": {
            "architecture": request.environment.architecture,
            "required_profile_keys": list(required_keys),
            "requirements_ref": request.requirements_ref,
            "requirements_sha256": request.requirements_sha256,
        },
        "probe_protocol": {
            "warmup_iterations": 3,
            "timed_samples": 7,
            "selection": "minimum_of_best_three",
        },
        "probe_evidence": {
            key.value: probe.provenance_for(key) for key in measurement_keys
        },
    }
    if capability_report is not None:
        metadata["isa_capability_report"] = capability_report.to_dict()
        metadata["isa_budget_audit"] = _audit_isa_budget(capability_report)
    if request.source_revision is not None:
        metadata["source_revision"] = request.source_revision
    if measurements.clock_reason:
        metadata["clock_lock_reason"] = measurements.clock_reason
    metadata.update(_profile_metadata(request, metric_map))
    validated_from_provenance, provenance = _validation_provenance(
        adapter,
        measurements,
        required_keys,
        injected_probe=request.hip_probe is not None,
    )
    metadata["validation_provenance"] = provenance
    return HardwareCalibrationArtifact(
        generated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        candidates=candidates,
        collection_status="collected",
        validation_status="validated" if validated_from_provenance else "provisional",
        metadata=metadata,
        schema_version=CALIBRATION_SCHEMA_VERSION,
    )


def _calibration_runtime(
    request: CalibrationRequest,
) -> tuple[ArchitectureAdapter, HipProbe, IsaCapabilityReport | None]:
    capability_report: IsaCapabilityReport | None = None
    matrix_unit: str | None = None
    if request.hip_probe is None:
        capability_report = inspect_isa_requirements(
            request.environment.architecture,
            matrix_isa_requirements(),
            allow_download=request.allow_isa_download,
        )
        if len(capability_report.matrix_units) != 1:
            raise RuntimeError(
                "ISA specification does not declare one unambiguous matrix unit"
            )
        matrix_unit = capability_report.matrix_units[0]
    adapter = adapter_for(request.environment.architecture, matrix_unit=matrix_unit)
    probe = request.hip_probe or default_hip_probe(
        architecture=request.environment.architecture,
        allow_isa_download=request.allow_isa_download,
    )
    return adapter, probe, capability_report


def _audit_isa_budget(report: IsaCapabilityReport) -> dict[str, object]:
    """Fail closed when packaged diagnostic budgets contradict the ISA spec."""

    budget = derive_arch_capability_budget(report.architecture)
    if budget is None:
        return {"status": "not_declared", "architecture": report.architecture}
    matrix_units = set(report.matrix_units)
    if budget.matrix_unit not in matrix_units:
        raise RuntimeError(
            "ISA matrix unit contradicts the packaged architecture budget: "
            f"{sorted(matrix_units)} != {budget.matrix_unit}"
        )
    isa_dtypes = {
        dtype
        for instruction in report.supported_instructions
        for dtype in ("bf16", "fp16")
        if dtype.upper() in instruction
    }
    missing_dtypes = isa_dtypes - set(budget.supported_dtypes)
    if missing_dtypes:
        raise RuntimeError(
            "ISA matrix dtype contradicts the packaged architecture budget: "
            + ", ".join(sorted(missing_dtypes))
        )
    return {
        "status": "consistent",
        "architecture": report.architecture,
        "matrix_unit": budget.matrix_unit,
        "matrix_dtypes": sorted(isa_dtypes),
    }
