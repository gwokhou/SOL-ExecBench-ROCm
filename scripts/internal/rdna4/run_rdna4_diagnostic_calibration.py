#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Collect the frozen gfx1200 diagnostic performance-model parameter package."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from sol_execbench.cli.protocol import (
    EXIT_EXECUTION,
    CliResult,
    artifact,
    response_failure,
    response_success,
)
from sol_execbench.core.bench.clock_lock import acquire_clock_lock
from sol_execbench.core.bench.performance_model.calibration import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    ProbeBatch,
    build_calibration_parameters,
    freeze_probe_configuration,
    parse_probe_metrics,
)
from sol_execbench.core.bench.performance_model.calibration_audit import (
    DiagnosticCalibrationAudit,
)
from sol_execbench.core.bench.performance_model.models import (
    CalibrationIdentity,
    DiagnosticCalibrationProfile,
)
from sol_execbench.core.bench.timing_isolation import (
    verify_clock_state_with_warning,
)
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.evidence.runtime_evidence.collectors import (
    collect_runtime_gpu_telemetry,
)
from sol_execbench.core.evidence.runtime_evidence.models import (
    RuntimeGPUTelemetry,
)
from sol_execbench.core.integrity import (
    sha256_file,
    stable_json_checksum,
)
from sol_execbench.core.integrity.schema_versions import (
    DIAGNOSTIC_CALIBRATION_AUDIT_SCHEMA_VERSION,
)
from sol_execbench.core.platform.amd_smi import parse_gpu_identity
from sol_execbench.core.platform.amdgpu_code_object import extract_code_object
from sol_execbench.core.platform.isa_validation import analyze_isa_disassembly
from sol_execbench.core.platform.runtime import (
    RocmDeviceInfo,
    detect_rocm_device,
    resolve_rocm_tool,
)
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded

PROBE_SOURCE = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "sol_execbench"
    / "data"
    / "hardware_calibration_probes"
    / "diagnostic_microarchitecture.hip"
)
MODES = ("dispatch", "memory", "access", "lds", "reduction", "wmma", "valu")
COMMAND_TIMEOUT_SECONDS = 180.0
COMMAND_NAME = "rdna4 diagnostic calibration"


@dataclass(frozen=True, slots=True)
class _CalibrationContext:
    output: Path
    hipcc: Path
    device: RocmDeviceInfo
    gpu_id: str
    gpu_bdf: str
    compiler_version: str
    tuning_batches: int
    estimation_batches: int


def _compile_probe(hipcc: Path, architecture: str, output: Path) -> None:
    command = [
        str(hipcc),
        str(PROBE_SOURCE),
        "-O3",
        "-std=c++17",
        f"--offload-arch={architecture}",
        "-o",
        str(output),
    ]
    completed = run_in_process_group_bounded(
        command,
        timeout=COMMAND_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"diagnostic probe compilation failed: {completed.stderr}"
        )


def _run_probe_batch(
    binary: Path,
    *,
    phase: str,
    process_batch: int,
    mode: str,
) -> ProbeBatch:
    completed = run_in_process_group_bounded(
        [str(binary), mode],
        timeout=COMMAND_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"diagnostic probe {mode} failed: {completed.stderr}",
        )
    metrics = parse_probe_metrics(completed.stdout)
    if not metrics:
        raise RuntimeError(f"diagnostic probe {mode} emitted no metrics")
    return ProbeBatch(
        phase=phase,
        process_batch=process_batch,
        mode=mode,
        metrics=tuple(metrics),
        clocks_locked=verify_clock_state_with_warning(
            context=f"diagnostic_calibration_{phase}_{mode}",
        ),
    )


def _collect_phase(
    binary: Path,
    *,
    phase: str,
    process_batches: int,
) -> list[ProbeBatch]:
    batches: list[ProbeBatch] = []
    for process_batch in range(process_batches):
        offset = process_batch % len(MODES)
        order = MODES[offset:] + MODES[:offset]
        if process_batch % 2:
            order = tuple(reversed(order))
        for mode in order:
            batches.append(
                _run_probe_batch(
                    binary,
                    phase=phase,
                    process_batch=process_batch,
                    mode=mode,
                ),
            )
    return batches


def _compiler_version(hipcc: Path) -> str:
    completed = run_in_process_group_bounded(
        [str(hipcc), "--version"],
        timeout=30.0,
    )
    if completed.returncode != 0:
        raise RuntimeError("hipcc --version failed")
    lines = [
        line.strip() for line in completed.stdout.splitlines() if line.strip()
    ]
    if not lines:
        raise RuntimeError("hipcc --version returned no version")
    return lines[0]


def _isa_evidence(
    binary: Path,
    architecture: str,
    workspace: Path,
) -> dict[str, object]:
    extracted = extract_code_object(
        binary,
        architecture,
        workspace,
        timeout_seconds=COMMAND_TIMEOUT_SECONDS,
    )
    analysis = analyze_isa_disassembly(
        architecture,
        extracted.disassembly,
        expected_instructions=("V_WMMA_F32_16X16X16_F16",),
    )
    if not analysis.matched_instruction_counts.get(
        "V_WMMA_F32_16X16X16_F16",
    ):
        raise RuntimeError(
            "diagnostic probe did not emit the required WMMA ISA"
        )
    return {
        "code_object_sha256": extracted.sha256,
        "disassembly_sha256": extracted.disassembly_sha256,
        "decoded_instruction_count": analysis.decoded_instruction_count,
        "matched_instruction_counts": dict(analysis.matched_instruction_counts),
        "spec_provenance": analysis.provenance.to_dict(),
    }


def run_calibration(
    *,
    output: Path,
    gpu_id: str,
    tuning_batches: int,
    estimation_batches: int,
) -> Path:
    """Compile, tune, freeze, and collect parameter-estimation evidence."""
    hipcc = resolve_rocm_tool("hipcc")
    if hipcc is None:
        raise RuntimeError("hipcc is unavailable")
    amd_smi = resolve_rocm_tool("amd-smi")
    if amd_smi is None:
        raise RuntimeError("amd-smi is unavailable")
    device = detect_rocm_device()
    if device.gfx_target != "gfx1200":
        raise RuntimeError(
            f"diagnostic calibration requires gfx1200, got {device.gfx_target}"
        )
    observed_gpu_id, gpu_bdf = _gpu_identity(amd_smi, device.index)
    if gpu_id != observed_gpu_id:
        raise RuntimeError(
            f"--gpu-id does not match device {device.index} UUID",
        )
    compiler_version = _compiler_version(hipcc)
    return _run_calibration_workspace(
        _CalibrationContext(
            output=output,
            hipcc=hipcc,
            device=device,
            gpu_id=observed_gpu_id,
            gpu_bdf=gpu_bdf,
            compiler_version=compiler_version,
            tuning_batches=tuning_batches,
            estimation_batches=estimation_batches,
        ),
    )


def _run_calibration_workspace(context: _CalibrationContext) -> Path:
    """Collect evidence and publish calibration within a temporary workspace."""
    with tempfile.TemporaryDirectory(
        prefix="sol_diag_calibration_"
    ) as raw_workspace:
        workspace = Path(raw_workspace)
        binary = workspace / "diagnostic_microarchitecture"
        _compile_probe(context.hipcc, context.device.gfx_target, binary)
        with acquire_clock_lock() as lease:
            if not lease.locked:
                raise RuntimeError("STABLE_PEAK clock lock is required")
            environment = [collect_runtime_gpu_telemetry(phase="pre")]
            tuning = _collect_phase(
                binary,
                phase="tuning",
                process_batches=context.tuning_batches,
            )
            frozen = freeze_probe_configuration(tuning)
            estimation = _collect_phase(
                binary,
                phase="parameter_estimation_after_configuration_freeze",
                process_batches=context.estimation_batches,
            )
            environment.append(collect_runtime_gpu_telemetry(phase="post"))
        isa = _isa_evidence(binary, context.device.gfx_target, workspace)
        audit = DiagnosticCalibrationAudit.model_validate(
            _audit_payload(
                context,
                binary=binary,
                tuning=tuning,
                frozen=frozen,
                estimation=estimation,
                isa=isa,
                environment=environment,
            ),
        )
        audit_path = context.output.with_name(
            f"{context.output.stem}.audit.json"
        )
        atomic_write_json_value(audit_path, audit.model_dump(mode="json"))
        profile = _calibration_profile(
            context,
            audit=audit,
            audit_path=audit_path,
            estimation=estimation,
            frozen=frozen,
        )
        atomic_write_json_value(
            context.output,
            profile.model_dump(mode="json"),
        )
    return context.output


def _calibration_profile(
    context: _CalibrationContext,
    *,
    audit: DiagnosticCalibrationAudit,
    audit_path: Path,
    estimation: Sequence[ProbeBatch],
    frozen: dict[str, str],
) -> DiagnosticCalibrationProfile:
    return DiagnosticCalibrationProfile(
        identity=CalibrationIdentity(
            gpu_architecture="gfx1200",
            gpu_id=context.gpu_id,
            gpu_bdf=context.gpu_bdf,
            rocm_version=context.device.hip_version,
            compiler_version=context.compiler_version,
            clock_mode="locked",
            power_profile="stable_peak",
        ),
        parameters=build_calibration_parameters(estimation, frozen),
        tuning_evidence_sha256=[
            stable_json_checksum(
                [
                    item.model_dump(mode="json")
                    for item in audit.tuning_evidence
                ],
            )
        ],
        parameter_estimation_evidence_sha256=[
            stable_json_checksum(
                [
                    item.model_dump(mode="json")
                    for item in audit.parameter_estimation_evidence
                ],
            ),
            sha256_file(audit_path),
        ],
        probe_evidence_sha256=[
            stable_json_checksum(
                audit.probe_identity.model_dump(mode="json"),
            )
        ],
        configuration_frozen_before_estimation=True,
        bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
    )


def _audit_payload(
    context: _CalibrationContext,
    *,
    binary: Path,
    tuning: Sequence[ProbeBatch],
    frozen: dict[str, str],
    estimation: Sequence[ProbeBatch],
    isa: dict[str, object],
    environment: Sequence[RuntimeGPUTelemetry],
) -> dict[str, object]:
    return {
        "schema_version": DIAGNOSTIC_CALIBRATION_AUDIT_SCHEMA_VERSION,
        "probe_identity": {
            "source_sha256": sha256_file(PROBE_SOURCE),
            "binary_sha256": sha256_file(binary),
            "compiler_sha256": sha256_file(context.hipcc),
            "architecture": context.device.gfx_target,
            "rocm_version": context.device.hip_version,
            "device_name": context.device.name,
            "gpu_id": context.gpu_id,
            "gpu_bdf": context.gpu_bdf,
            "total_memory_bytes": context.device.total_memory_bytes,
            "compiler_version": context.compiler_version,
            "isa": isa,
        },
        "protocol": {
            "design": "two_phase_tuning_then_parameter_estimation",
            "configuration_frozen_before_parameter_estimation": True,
            "tuning_process_batches": len(tuning) // len(MODES),
            "parameter_estimation_process_batches": (
                len(estimation) // len(MODES)
            ),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "clock_mode": "STABLE_PEAK",
        },
        "frozen_configuration": frozen,
        "tuning_evidence": [batch.to_dict() for batch in tuning],
        "parameter_estimation_evidence": [
            batch.to_dict() for batch in estimation
        ],
        "environment": [item.model_dump(mode="json") for item in environment],
    }


def _gpu_identity(amd_smi: Path, device_index: int) -> tuple[str, str]:
    completed = run_in_process_group_bounded(
        [str(amd_smi), "list", "--json"],
        timeout=30.0,
    )
    if completed.returncode != 0:
        raise RuntimeError("amd-smi list --json failed")
    identity = parse_gpu_identity(completed.stdout, device_index)
    if identity.uuid is None or identity.bdf is None:
        raise RuntimeError("amd-smi GPU identity is incomplete")
    return identity.uuid, identity.bdf


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--tuning-batches", type=int, default=3)
    parser.add_argument("--estimation-batches", type=int, default=5)
    arguments = parser.parse_args()
    if arguments.tuning_batches < 1:
        parser.error("--tuning-batches must be positive")
    if arguments.estimation_batches < 5:
        parser.error("--estimation-batches must be at least 5")
    return arguments


def main() -> int:
    """Run the command-line calibration workflow."""
    arguments = _parse_args()
    path = run_calibration(
        output=arguments.output,
        gpu_id=arguments.gpu_id,
        tuning_batches=arguments.tuning_batches,
        estimation_batches=arguments.estimation_batches,
    )
    response = response_success(
        COMMAND_NAME,
        CliResult(
            data={"status": "available", "diagnostic_only": True},
            artifacts=(
                artifact(path, "diagnostic_calibration_json"),
                artifact(
                    path.with_name(f"{path.stem}.audit.json"),
                    "diagnostic_calibration_audit_json",
                ),
            ),
        ),
    )
    print(json.dumps(response, sort_keys=True))
    return 0


def _entrypoint() -> int:
    try:
        return main()
    except Exception as error:  # noqa: BLE001 -- standalone JSON boundary
        print(json.dumps(response_failure(COMMAND_NAME, error), sort_keys=True))
        return EXIT_EXECUTION


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
