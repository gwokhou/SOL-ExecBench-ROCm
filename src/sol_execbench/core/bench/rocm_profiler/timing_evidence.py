# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""ROCm profiler timing parsing and live timing collection."""

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Final

from sol_execbench.core.bench.rocm_profiler.calibration import (
    Rocprofv3OverheadCalibration,
)
from sol_execbench.core.bench.rocm_profiler.models import (
    DefaultTimingSelection,
    Rocprofv3TimingEvidence,
    Rocprofv3TimingRow,
)
from sol_execbench.core.bench.rocm_profiler.timing_parsing import (
    parse_rocprofv3_csv,
    summarize_rocprofv3_csv,
)
from sol_execbench.core.bench.timing_policy import (
    TimingBackend,
    TimingPolicy,
    select_timing_policy,
)
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.platform.runtime import resolve_rocm_tool


def select_default_timing(
    policy: TimingPolicy,
    *,
    rocprofv3_available: bool,
) -> DefaultTimingSelection:
    """Resolve whether the default path can use profiler-backed timing."""
    if policy.backend == TimingBackend.ROCPROFV3 and rocprofv3_available:
        return DefaultTimingSelection(
            policy=policy,
            profiler_backed=True,
            fallback_applied=False,
            reason="rocprofv3 is available for the selected timing policy",
        )

    if policy.backend != TimingBackend.ROCPROFV3:
        reason = (
            f"selected policy backend is {policy.backend}, not rocprofv3 "
            "kernel activity timing"
        )
    else:
        reason = "rocprofv3 is unavailable for the selected timing policy"
    fallback_policy = replace(
        select_timing_policy(
            policy.source_type,
            profiler_available=False,
        ),
        reason=reason,
    )
    return DefaultTimingSelection(
        policy=fallback_policy,
        profiler_backed=False,
        fallback_applied=True,
        reason=reason,
    )


def build_compact_timing_evidence(
    *,
    policy: TimingPolicy,
    csv_path: Path,
    tool_version: str,
    gpu_architecture: str,
    warmup_runs: int | None = None,
    iterations: int | None = None,
    min_measurement_time_seconds: float | None = None,
    trial_count: int | None = None,
    clock_locked: bool | None = None,
    profiler_overhead_ms: float | None = None,
) -> tuple[Rocprofv3TimingEvidence, int]:
    """Build summary timing evidence without materializing every CSV row."""
    kernel_rows, duration_ns = summarize_rocprofv3_csv(csv_path)
    parsed_rows = ()
    if kernel_rows > 0:
        parsed_rows = (
            Rocprofv3TimingRow(
                name="compacted_kernel_activity",
                domain="KERNEL_DISPATCH",
                duration_ns=duration_ns,
                raw={"compacted_kernel_activity_rows": str(kernel_rows)},
            ),
        )
    evidence = Rocprofv3TimingEvidence(
        tool_version=tool_version,
        gpu_architecture=gpu_architecture,
        activity_domain=policy.activity_domain,
        aggregation_rule=policy.aggregation_rule,
        backend=policy.backend,
        interpretation=policy.interpretation,
        parsed_rows=parsed_rows,
        warmup_runs=warmup_runs,
        iterations=iterations,
        min_measurement_time_seconds=min_measurement_time_seconds,
        trial_count=trial_count,
        clock_locked=clock_locked,
        fallback_applied=policy.fallback_applied,
        fallback_reason=policy.reason if policy.fallback_applied else None,
        profiler_overhead_ms=profiler_overhead_ms,
    )
    return evidence, kernel_rows


def build_timing_evidence(
    *,
    policy: TimingPolicy,
    csv_content: str,
    tool_version: str,
    gpu_architecture: str,
    warmup_runs: int | None = None,
    iterations: int | None = None,
    min_measurement_time_seconds: float | None = None,
    trial_count: int | None = None,
    clock_locked: bool | None = None,
    profiler_overhead_ms: float | None = None,
) -> Rocprofv3TimingEvidence:
    """Build derived profiler timing evidence from parsed CSV content."""
    return Rocprofv3TimingEvidence(
        tool_version=tool_version,
        gpu_architecture=gpu_architecture,
        activity_domain=policy.activity_domain,
        aggregation_rule=policy.aggregation_rule,
        backend=policy.backend,
        interpretation=policy.interpretation,
        parsed_rows=parse_rocprofv3_csv(csv_content),
        warmup_runs=warmup_runs,
        iterations=iterations,
        min_measurement_time_seconds=min_measurement_time_seconds,
        trial_count=trial_count,
        clock_locked=clock_locked,
        fallback_applied=policy.fallback_applied,
        fallback_reason=policy.reason if policy.fallback_applied else None,
        profiler_overhead_ms=profiler_overhead_ms,
    )


_UNSET_CLOCK: Final = object()


def read_overhead_calibration(
    calibration_path: Path | None,
    *,
    expected_gpu_architecture: str | None = None,
    expected_profiler_executable: str | None = None,
    expected_clock_locked: bool | None | object = _UNSET_CLOCK,
) -> float | None:
    """Read profiler overhead calibration value from a JSON sidecar.

    Returns the ``overhead_ms`` value if the file exists and is valid,
    otherwise None. Never raises — logs warnings on parse errors.
    """
    if calibration_path is None or not calibration_path.exists():
        return None
    try:
        calibration = Rocprofv3OverheadCalibration.model_validate_json(
            calibration_path.read_text(encoding="utf-8"),
        )
        mismatch = _calibration_identity_mismatch(
            calibration,
            expected_gpu_architecture=expected_gpu_architecture,
            expected_profiler_executable=expected_profiler_executable,
            expected_clock_locked=expected_clock_locked,
        )
        if mismatch is not None:
            logging.getLogger(__name__).warning(
                "Ignoring overhead calibration from %s: %s",
                calibration_path,
                mismatch,
            )
            return None
        return calibration.overhead_ms
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        logging.getLogger(__name__).warning(
            "Failed to read overhead calibration from %s: %s",
            calibration_path,
            exc,
        )
        return None


def _calibration_identity_mismatch(
    calibration: Rocprofv3OverheadCalibration,
    *,
    expected_gpu_architecture: str | None,
    expected_profiler_executable: str | None,
    expected_clock_locked: bool | None | object,
) -> str | None:
    if expected_gpu_architecture is not None:
        expected_arch = expected_gpu_architecture.split(":", maxsplit=1)[
            0
        ].lower()
        calibration_arch = calibration.gpu_architecture.split(":", maxsplit=1)[
            0
        ].lower()
        if calibration_arch != expected_arch:
            return f"GPU architecture is {calibration_arch}, expected {expected_arch}"
    if expected_profiler_executable is not None:
        resolved = resolve_rocm_tool(expected_profiler_executable)
        if resolved is None:
            return (
                "current profiler executable could not be resolved: "
                f"{expected_profiler_executable}"
            )
        current_sha256 = sha256_file(resolved)
        if current_sha256 != calibration.profiler_executable_sha256:
            return "profiler executable SHA-256 does not match"
    if (
        expected_clock_locked is not _UNSET_CLOCK
        and calibration.clock_locked != expected_clock_locked
    ):
        return (
            f"clock_locked is {calibration.clock_locked}, "
            f"expected {expected_clock_locked}"
        )
    return None


_read_overhead_calibration = read_overhead_calibration
