# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""CLI orchestration for gfx1200 fusion validation probes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import click

from sol_execbench.cli.protocol import (
    EXIT_RESULT_FAILED,
    EXIT_UNAVAILABLE,
    CliFailure,
    CliResult,
    artifact,
)
from sol_execbench.core.scoring.fusion_validation import (
    FusionValidationArtifact,
    fusion_validation_from_dict,
)
from sol_execbench.core.integrity.checksums import sha256_file
from sol_execbench.core.bench.clock_lock import (
    ClockLockLease,
    acquire_clock_lock,
)
from sol_execbench.core.scoring.hardware_calibration.environment import discover_gpu


Runner = Callable[..., subprocess.CompletedProcess[str]]


@click.group("fusion")
def fusion_cli() -> None:
    """Collect and verify shape-exact fusion-capacity evidence."""


@fusion_cli.command("collect")
@click.option("--device", default=0, show_default=True, type=click.IntRange(min=0))
@click.option("--architecture", required=True)
@click.option(
    "--suite-manifest",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--benchmark-root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--output", required=True, type=click.Path(dir_okay=False, path_type=Path)
)
@click.option("--require-clock-lock", is_flag=True)
def collect_command(
    device: int,
    architecture: str,
    suite_manifest: Path,
    benchmark_root: Path,
    output: Path,
    require_clock_lock: bool,
) -> CliResult:
    """Run the built-in HIP probe (or a manifest override) and retain results."""
    try:
        evidence = collect_fusion_validation(
            device=device,
            architecture=architecture,
            suite_manifest=suite_manifest,
            benchmark_root=benchmark_root,
            output=output,
            require_clock_lock=require_clock_lock,
        )
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        raise CliFailure(
            str(exc), code="environment_unavailable", exit_code=EXIT_UNAVAILABLE
        ) from exc
    failed_capacity = [
        case.evidence_id for case in evidence.cases if case.capacity_status != "passed"
    ]
    failed_performance = [
        case.evidence_id
        for case in evidence.cases
        if case.performance.status != "passed"
    ]
    if failed_capacity:
        raise CliFailure(
            "fusion capacity validation failed",
            code="fusion_capacity_failed",
            exit_code=EXIT_UNAVAILABLE,
            details={"evidence": str(output), "failed_cases": failed_capacity},
        )
    return CliResult(
        data={
            "capacity_status": "passed",
            "performance_status": "passed" if not failed_performance else "failed",
            "failed_performance_cases": failed_performance,
            "sha256": sha256_file(output),
        },
        artifacts=(artifact(output, "fusion_validation"),),
        exit_code=EXIT_RESULT_FAILED if failed_performance else 0,
    )


@fusion_cli.command("verify")
@click.option(
    "--evidence",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--suite-manifest",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def verify_command(evidence: Path, suite_manifest: Path) -> CliResult:
    """Strictly verify schema, suite binding and all capacity decisions."""
    try:
        manifest = _json_object(suite_manifest)
        parsed = fusion_validation_from_dict(_json_object(evidence))
        if parsed.suite_manifest_sha256 != sha256_file(suite_manifest):
            raise ValueError("suite manifest checksum mismatch")
        _verify_suite_coverage(parsed, manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise CliFailure(str(exc), code="invalid_evidence") from exc
    failed_capacity = [
        case.evidence_id for case in parsed.cases if case.capacity_status != "passed"
    ]
    if failed_capacity:
        raise CliFailure(
            "fusion evidence contains failed capacity cases",
            code="fusion_capacity_failed",
            exit_code=EXIT_UNAVAILABLE,
            details={"failed_cases": failed_capacity},
        )
    performance = {case.evidence_id: case.performance.status for case in parsed.cases}
    failed_performance = [
        key for key, value in performance.items() if value != "passed"
    ]
    return CliResult(
        data={
            "capacity_status": "passed",
            "performance_status": "passed" if not failed_performance else "failed",
            "cases": len(parsed.cases),
            "sha256": sha256_file(evidence),
        },
        artifacts=(artifact(evidence, "fusion_validation"),),
        exit_code=EXIT_RESULT_FAILED if failed_performance else 0,
    )


def collect_fusion_validation(
    *,
    device: int,
    architecture: str,
    suite_manifest: Path,
    benchmark_root: Path,
    output: Path,
    require_clock_lock: bool,
    runner: Runner = subprocess.run,
    discover: Callable[[int], object] = discover_gpu,
) -> FusionValidationArtifact:
    """Execute and validate the built-in or manifest-provided probe driver."""
    manifest = _json_object(suite_manifest)
    probe = manifest.get("fusion_probe")
    command = probe.get("command") if isinstance(probe, dict) else None
    if command is None:
        command = _built_in_probe_command()
    if (
        not isinstance(command, (list, tuple))
        or not command
        or not all(isinstance(item, str) and item for item in command)
    ):
        raise ValueError("fusion_probe.command must be a non-empty argv list")
    environment = discover(device)
    live_arch, gpu_uuid, rocm_version = _validated_gpu_identity(
        environment, architecture
    )
    clocks_locked = (
        bool(getattr(environment, "clocks_locked", False))
        or os.environ.get("SOL_EXECBENCH_CLOCKS_LOCKED") == "1"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    clock_lock = ClockLockLease(locked=clocks_locked, acquired=False)
    if require_clock_lock and not clocks_locked:
        clock_lock = acquire_clock_lock()
        clocks_locked = clock_lock.locked
        if not clocks_locked:
            raise RuntimeError("locked GPU clocks are required for fusion collection")
    with clock_lock:
        process_env = _fusion_process_environment(
            device=device,
            live_arch=live_arch,
            gpu_uuid=gpu_uuid,
            rocm_version=rocm_version,
            suite_manifest=suite_manifest,
            benchmark_root=benchmark_root,
            output=output,
            clocks_locked=clocks_locked,
        )
        completed = runner(
            command,
            cwd=benchmark_root,
            env=process_env,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode not in {0, EXIT_RESULT_FAILED}:
            raise RuntimeError(
                f"fusion probe failed with exit {completed.returncode}: {completed.stderr.strip()}"
            )
        parsed = fusion_validation_from_dict(_json_object(output))
        _validate_fusion_artifact(
            parsed=parsed,
            manifest=manifest,
            suite_manifest=suite_manifest,
            live_arch=live_arch,
            gpu_uuid=gpu_uuid,
            rocm_version=rocm_version,
            require_clock_lock=require_clock_lock,
        )
        # Preserve strict parsed normalization rather than accepting non-canonical
        # JSON emitted by the child.
        output.write_text(
            json.dumps(parsed.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return parsed


def _validated_gpu_identity(
    environment: object, architecture: str
) -> tuple[str, str, str]:
    live_arch = str(getattr(environment, "architecture")).lower()
    if live_arch != architecture.lower():
        raise ValueError(
            "architecture assertion does not match runtime discovery: "
            f"{architecture.lower()} != {live_arch}"
        )
    gpu_uuid = getattr(environment, "uuid", None)
    if not gpu_uuid:
        raise RuntimeError("GPU UUID discovery is required for fusion collection")
    rocm_version = getattr(environment, "rocm_version", None)
    if not rocm_version:
        raise RuntimeError("ROCm version discovery is required for fusion collection")
    return live_arch, str(gpu_uuid), str(rocm_version)


def _validate_fusion_artifact(
    *,
    parsed: FusionValidationArtifact,
    manifest: dict[str, Any],
    suite_manifest: Path,
    live_arch: str,
    gpu_uuid: str,
    rocm_version: str,
    require_clock_lock: bool,
) -> None:
    if parsed.suite_manifest_sha256 != sha256_file(suite_manifest):
        raise ValueError("probe output suite manifest checksum mismatch")
    _verify_suite_coverage(parsed, manifest)
    if parsed.architecture != live_arch:
        raise ValueError("probe output architecture mismatch")
    if parsed.gpu_uuid != gpu_uuid:
        raise ValueError("probe output GPU UUID mismatch")
    if parsed.rocm_version != rocm_version:
        raise ValueError("probe output ROCm version mismatch")
    if require_clock_lock and not parsed.clocks_locked:
        raise ValueError("probe output does not attest locked clocks")


def _fusion_process_environment(
    *,
    device: int,
    live_arch: str,
    gpu_uuid: str,
    rocm_version: str,
    suite_manifest: Path,
    benchmark_root: Path,
    output: Path,
    clocks_locked: bool,
) -> dict[str, str]:
    return {
        **os.environ,
        "SOL_EXECBENCH_FUSION_DEVICE": str(device),
        "SOL_EXECBENCH_FUSION_ARCHITECTURE": live_arch,
        "SOL_EXECBENCH_FUSION_GPU_UUID": gpu_uuid,
        "SOL_EXECBENCH_FUSION_ROCM_VERSION": rocm_version,
        "SOL_EXECBENCH_FUSION_SUITE_MANIFEST": str(suite_manifest.resolve()),
        "SOL_EXECBENCH_FUSION_BENCHMARK_ROOT": str(benchmark_root.resolve()),
        "SOL_EXECBENCH_FUSION_OUTPUT": str(output.resolve()),
        "SOL_EXECBENCH_CLOCKS_LOCKED": "1" if clocks_locked else "0",
    }


def _built_in_probe_command() -> tuple[str, ...]:
    """Return the packaged HIP driver argv for standard suite manifests."""
    return (
        sys.executable,
        "-m",
        "sol_execbench.cli.commands.fusion_probe_runner",
    )


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _verify_suite_coverage(
    evidence: FusionValidationArtifact, manifest: dict[str, Any]
) -> None:
    """Require exact coverage of the suite's RMSNorm and tanh fusion slice."""
    workloads = manifest.get("workloads")
    if not isinstance(workloads, list):
        raise ValueError("suite manifest workloads must be a list")
    targets: dict[str, str] = {}
    for row in workloads:
        if not isinstance(row, dict):
            continue
        definition = row.get("definition")
        workload_uuid = row.get("workload_uuid")
        if not isinstance(definition, str) or not isinstance(workload_uuid, str):
            continue
        if "025_rmsnorm_h4096" in definition:
            targets[workload_uuid] = "rmsnorm"
        elif "061_tanh_gated_residual_add_backward" in definition:
            targets[workload_uuid] = "tanh_backward"
    if not targets:
        raise ValueError("suite manifest has no RMSNorm or tanh fusion workloads")
    actual = {case.workload_uuid for case in evidence.cases}
    if actual != set(targets):
        missing = sorted(set(targets) - actual)
        unknown = sorted(actual - set(targets))
        raise ValueError(
            f"fusion evidence shape coverage mismatch; missing={missing}, unknown={unknown}"
        )
    for case in evidence.cases:
        if case.signature.pattern_id != "reduction_epilogue.v1":
            raise ValueError(f"{case.evidence_id} does not prove reduction_epilogue.v1")
        if case.signature.dtype != "fp32":
            raise ValueError(f"{case.evidence_id} does not prove FP32 fusion")
        if not any(shape[-1] == 4096 for shape in case.signature.input_shapes):
            raise ValueError(f"{case.evidence_id} does not prove hidden size 4096")


__all__ = ["collect_fusion_validation", "fusion_cli"]
