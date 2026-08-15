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
    CliExitCode,
    CliResult,
    artifact,
    response_failure,
    response_success,
)
from sol_execbench.core.bench.batch_gpu_qualification import (
    BatchGPUQualificationGate,
    BatchGPUQualificationReceipt,
    BatchGPUQualificationStage,
    LargeBatchGPUTask,
    qualification_artifact,
    qualification_gate_path,
    qualification_parent_stage,
    require_isolated_qualification_root,
    verify_qualification_artifact,
)
from sol_execbench.core.bench.clock_lock import acquire_clock_lock
from sol_execbench.core.bench.performance_model.calibration import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    ProbeBatch,
    build_calibration_parameters,
    build_calibration_surfaces,
    freeze_probe_configuration,
    parse_probe_metrics,
    validate_indexed_read_surface_capacity,
)
from sol_execbench.core.bench.performance_model.calibration_audit import (
    DiagnosticCalibrationAudit,
    calibration_probe_identity_payload,
)
from sol_execbench.core.bench.performance_model.diagnostic_schema_versions import (
    DiagnosticArtifactSchema,
    DiagnosticCalibrationArtifactKind,
)
from sol_execbench.core.bench.performance_model.models import (
    CalibrationIdentity,
    CalibrationSurfaceName,
    DiagnosticCalibrationProfile,
)
from sol_execbench.core.bench.performance_model.vram_policy import (
    MIB,
    DiagnosticVRAMWorkingSetPolicy,
    select_vram_working_set_policy,
)
from sol_execbench.core.bench.timing_isolation import (
    verify_clock_state_with_warning,
)
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_file,
    load_json_value,
)
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
from sol_execbench.core.platform.amd_smi import parse_gpu_identity
from sol_execbench.core.platform.amdgpu_code_object import extract_code_object
from sol_execbench.core.platform.isa_validation import analyze_isa_disassembly
from sol_execbench.core.platform.runtime import (
    RocmDeviceInfo,
    detect_rocm_device,
    detect_rocm_version,
    resolve_rocm_tool,
)
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded
from sol_execbench.core.timestamps import utc_timestamp

PROBE_SOURCE = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "sol_execbench"
    / "data"
    / "hardware_calibration_probes"
    / "diagnostic_microarchitecture.hip"
)
MODES = (
    "dispatch",
    "memory",
    "access",
    "lds",
    "reduction",
    "wmma",
    "valu",
    "indexed_read",
    "atomic_update",
    "fp32_matrix",
    "residency",
    "overlap",
)
COMMAND_TIMEOUT_SECONDS = 180.0
COMMAND_NAME = "rdna4 diagnostic calibration"
_QUALIFICATION_CANARY_MODES = (
    "wmma",
    "memory",
    "indexed_read",
    "atomic_update",
    "overlap",
)
_CAPACITY_GOVERNED_MODES = frozenset({"memory", "indexed_read"})


@dataclass(frozen=True, slots=True, kw_only=True)
class _CalibrationContext:
    output: Path
    hipcc: Path
    device: RocmDeviceInfo
    gpu_id: str
    gpu_bdf: str
    rocm_version: str
    compiler_version: str
    tuning_batches: int
    estimation_batches: int
    vram_policy: DiagnosticVRAMWorkingSetPolicy
    vram_policy_path: Path


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
    vram_policy: DiagnosticVRAMWorkingSetPolicy,
) -> ProbeBatch:
    command = [str(binary), mode]
    if mode in _CAPACITY_GOVERNED_MODES:
        command.append(str(vram_policy.probe_working_set_bytes))
    completed = run_in_process_group_bounded(
        command,
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
    vram_policy: DiagnosticVRAMWorkingSetPolicy,
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
                    vram_policy=vram_policy,
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
    vram_policy_path: Path,
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
    policy = load_json_file(DiagnosticVRAMWorkingSetPolicy, vram_policy_path)
    _verify_vram_policy(policy, device, observed_gpu_id)
    compiler_version = _compiler_version(hipcc)
    rocm_version = detect_rocm_version()
    if rocm_version is None:
        raise RuntimeError("ROCm user-space version is unavailable")
    return _run_calibration_workspace(
        _CalibrationContext(
            output=output,
            hipcc=hipcc,
            device=device,
            gpu_id=observed_gpu_id,
            gpu_bdf=gpu_bdf,
            rocm_version=rocm_version,
            compiler_version=compiler_version,
            tuning_batches=tuning_batches,
            estimation_batches=estimation_batches,
            vram_policy=policy,
            vram_policy_path=vram_policy_path.resolve(),
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
            environment = [
                collect_runtime_gpu_telemetry(
                    phase="pre", device_index=context.device.index
                )
            ]
            tuning = _collect_phase(
                binary,
                phase="tuning",
                process_batches=context.tuning_batches,
                vram_policy=context.vram_policy,
            )
            frozen = freeze_probe_configuration(
                tuning,
                vram_variant=(
                    f"{context.vram_policy.probe_working_set_bytes // MIB}MiB"
                ),
            )
            frozen["vram_policy_sha256"] = sha256_file(context.vram_policy_path)
            estimation = _collect_phase(
                binary,
                phase="parameter_estimation_after_configuration_freeze",
                process_batches=context.estimation_batches,
                vram_policy=context.vram_policy,
            )
            environment.append(
                collect_runtime_gpu_telemetry(
                    phase="post", device_index=context.device.index
                )
            )
        if any(item.pcie_topology is None for item in environment):
            raise RuntimeError("complete PCIe topology evidence is required")
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
    surfaces = build_calibration_surfaces(estimation)
    indexed_read = next(
        surface
        for surface in surfaces
        if surface.name is CalibrationSurfaceName.INDEXED_READ
    )
    validate_indexed_read_surface_capacity(
        indexed_read,
        context.vram_policy.applicability_max_bytes,
    )
    return DiagnosticCalibrationProfile(
        identity=CalibrationIdentity(
            gpu_architecture="gfx1200",
            gpu_id=context.gpu_id,
            gpu_bdf=context.gpu_bdf,
            pcie_topology=audit.probe_identity.pcie_topology,
            rocm_version=context.rocm_version,
            compiler_version=context.compiler_version,
            clock_mode="locked",
            power_profile="stable_peak",
        ),
        parameters=build_calibration_parameters(estimation, frozen),
        surfaces=surfaces,
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
                calibration_probe_identity_payload(audit.probe_identity),
            ),
            sha256_file(context.vram_policy_path),
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
        "schema_version": DiagnosticArtifactSchema.DIAGNOSTIC_CALIBRATION,
        "artifact_kind": DiagnosticCalibrationArtifactKind.AUDIT,
        "probe_identity": {
            "source_sha256": sha256_file(PROBE_SOURCE),
            "binary_sha256": sha256_file(binary),
            "compiler_sha256": sha256_file(context.hipcc),
            "architecture": context.device.gfx_target,
            "rocm_version": context.rocm_version,
            "device_name": context.device.name,
            "gpu_id": context.gpu_id,
            "gpu_bdf": context.gpu_bdf,
            "pcie_topology": environment[0].pcie_topology,
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


def _qualification_root(arguments: argparse.Namespace) -> Path:
    if arguments.qualification_root is None:
        raise ValueError("qualification stages require --qualification-root")
    return require_isolated_qualification_root(
        arguments.qualification_root,
        arguments.output,
    )


def _qualification_subject(hipcc: Path) -> str:
    return stable_json_checksum(
        {
            "probe_source_sha256": sha256_file(PROBE_SOURCE),
            "compiler_sha256": sha256_file(hipcc),
        }
    )


def _qualification_configuration(arguments: argparse.Namespace) -> str:
    policy_path = _required_vram_policy_path(arguments)
    return stable_json_checksum(
        {
            "gpu_id": arguments.gpu_id,
            "tuning_batches": arguments.tuning_batches,
            "estimation_batches": arguments.estimation_batches,
            "vram_policy_sha256": sha256_file(policy_path),
        }
    )


def _qualification_modes(
    stage: BatchGPUQualificationStage,
) -> tuple[str, ...]:
    if stage is BatchGPUQualificationStage.CANARY:
        return _QUALIFICATION_CANARY_MODES
    return MODES


def _static_qualification_receipt(
    arguments: argparse.Namespace,
    hipcc: Path,
) -> BatchGPUQualificationReceipt:
    root = _qualification_root(arguments)
    binary = root / "static" / "diagnostic_microarchitecture"
    binary.parent.mkdir(parents=True, exist_ok=True)
    _compile_probe(hipcc, "gfx1200", binary)
    payload_path = root / "static" / "preflight.json"
    payload = {
        "task": LargeBatchGPUTask.RDNA4_DIAGNOSTIC_CALIBRATION,
        "subject_sha256": _qualification_subject(hipcc),
        "modes": MODES,
        "compile_passed": True,
    }
    atomic_write_json_value(payload_path, payload)
    return BatchGPUQualificationReceipt(
        stage=BatchGPUQualificationStage.STATIC,
        partition="fixed-probes",
        item_ids=MODES,
        input_sha256=stable_json_checksum(payload),
        artifacts=(
            qualification_artifact(root, binary),
            qualification_artifact(root, payload_path),
        ),
    )


def _gpu_qualification_receipt(
    arguments: argparse.Namespace,
    stage: BatchGPUQualificationStage,
    mode: str,
    binary: Path,
    device: RocmDeviceInfo,
    gpu_id: str,
    gpu_bdf: str,
) -> BatchGPUQualificationReceipt:
    root = _qualification_root(arguments)
    path = root / stage.value / mode / "evidence.json"
    input_sha256 = stable_json_checksum(
        {
            "subject_sha256": _qualification_subject(_required_tool("hipcc")),
            "mode": mode,
            "gpu_id": gpu_id,
            "gpu_bdf": gpu_bdf,
        }
    )
    if path.is_file():
        payload = load_json_value(path)
    else:
        batch = _run_probe_batch(
            binary,
            phase=f"qualification_{stage.value}",
            process_batch=0,
            mode=mode,
            vram_policy=_load_vram_policy(arguments),
        )
        payload = {
            "stage": stage,
            "mode": mode,
            "input_sha256": input_sha256,
            "device": _qualification_device(device, gpu_id, gpu_bdf),
            "batch": batch.to_dict(),
            "all_passed": True,
        }
        atomic_write_json_value(path, payload)
    if (
        payload.get("input_sha256") != input_sha256
        or payload.get("device")
        != _qualification_device(device, gpu_id, gpu_bdf)
        or payload.get("all_passed") is not True
    ):
        raise ValueError(f"calibration qualification evidence drift: {path}")
    return BatchGPUQualificationReceipt(
        stage=stage,
        partition=mode,
        item_ids=(mode,),
        input_sha256=input_sha256,
        artifacts=(qualification_artifact(root, path),),
    )


def _required_tool(name: str) -> Path:
    path = resolve_rocm_tool(name)
    if path is None:
        raise RuntimeError(f"{name} is unavailable")
    return path


def _qualification_device(
    device: RocmDeviceInfo,
    gpu_id: str,
    gpu_bdf: str,
) -> dict[str, object]:
    return {
        "name": device.name,
        "gfx_target": device.gfx_target,
        "total_memory_bytes": device.total_memory_bytes,
        "torch_version": device.torch_version,
        "hip_version": device.hip_version,
        "gpu_id": gpu_id,
        "gpu_bdf": gpu_bdf,
    }


def _qualification_hardware(
    arguments: argparse.Namespace,
) -> tuple[RocmDeviceInfo, str, str]:
    device = detect_rocm_device()
    if device.gfx_target != "gfx1200":
        raise RuntimeError(
            f"diagnostic calibration requires gfx1200, got {device.gfx_target}"
        )
    observed_gpu_id, gpu_bdf = _gpu_identity(
        _required_tool("amd-smi"), device.index
    )
    if arguments.gpu_id != observed_gpu_id:
        raise RuntimeError("--gpu-id does not match qualification device UUID")
    _verify_vram_policy(_load_vram_policy(arguments), device, observed_gpu_id)
    return device, observed_gpu_id, gpu_bdf


def _required_vram_policy_path(arguments: argparse.Namespace) -> Path:
    if arguments.vram_policy is None:
        raise ValueError("calibration stage requires --vram-policy")
    return arguments.vram_policy.resolve()


def _load_vram_policy(
    arguments: argparse.Namespace,
) -> DiagnosticVRAMWorkingSetPolicy:
    return load_json_file(
        DiagnosticVRAMWorkingSetPolicy,
        _required_vram_policy_path(arguments),
    )


def _verify_vram_policy(
    policy: DiagnosticVRAMWorkingSetPolicy,
    device: RocmDeviceInfo,
    gpu_id: str,
) -> None:
    expected = select_vram_working_set_policy(
        gpu_architecture=device.gfx_target,
        gpu_id=gpu_id,
        total_memory_bytes=device.total_memory_bytes,
        source_revision=policy.source_revision,
        created_at=policy.created_at,
    )
    if policy != expected:
        raise ValueError("frozen VRAM policy differs from observed hardware")


def _freeze_vram_policy(arguments: argparse.Namespace) -> Path:
    output = arguments.output.resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite VRAM policy: {output}")
    device = detect_rocm_device()
    observed_gpu_id, _gpu_bdf = _gpu_identity(
        _required_tool("amd-smi"), device.index
    )
    if arguments.gpu_id != observed_gpu_id:
        raise RuntimeError("--gpu-id does not match policy device UUID")
    policy = select_vram_working_set_policy(
        gpu_architecture=device.gfx_target,
        gpu_id=observed_gpu_id,
        total_memory_bytes=device.total_memory_bytes,
        source_revision=_source_revision(),
    )
    atomic_write_json_value(output, policy.model_dump(mode="json"))
    return output


def _source_revision() -> str:
    completed = run_in_process_group_bounded(
        ["git", "rev-parse", "HEAD"], timeout=30.0
    )
    revision = completed.stdout.strip()
    if completed.returncode != 0 or len(revision) != 40:
        raise RuntimeError("cannot resolve calibration source revision")
    return revision


def _run_qualification(
    arguments: argparse.Namespace,
    stage: BatchGPUQualificationStage,
) -> BatchGPUQualificationGate:
    root = _qualification_root(arguments)
    gate_path = qualification_gate_path(root, stage)
    if gate_path.is_file():
        return _verify_qualification(arguments, stage)
    hipcc = _required_tool("hipcc")
    parent = qualification_parent_stage(stage)
    parent_hash = None
    if parent is not None:
        _verify_qualification(arguments, parent)
        parent_hash = sha256_file(qualification_gate_path(root, parent))
    if stage is BatchGPUQualificationStage.STATIC:
        receipts = (_static_qualification_receipt(arguments, hipcc),)
    else:
        device, gpu_id, gpu_bdf = _qualification_hardware(arguments)
        binary = root / "static" / "diagnostic_microarchitecture"
        with acquire_clock_lock() as lease:
            if not lease.locked:
                raise RuntimeError("STABLE_PEAK clock lock is required")
            receipts = tuple(
                _gpu_qualification_receipt(
                    arguments,
                    stage,
                    mode,
                    binary,
                    device,
                    gpu_id,
                    gpu_bdf,
                )
                for mode in _qualification_modes(stage)
            )
    gate = BatchGPUQualificationGate(
        task=LargeBatchGPUTask.RDNA4_DIAGNOSTIC_CALIBRATION,
        stage=stage,
        scope_id=arguments.gpu_id,
        subject_sha256=_qualification_subject(hipcc),
        runner_sha256=sha256_file(Path(__file__)),
        configuration_sha256=_qualification_configuration(arguments),
        source_revision=_qualification_subject(hipcc),
        parent_gate_sha256=parent_hash,
        item_ids=tuple(
            item for receipt in receipts for item in receipt.item_ids
        ),
        receipts=receipts,
        created_at=utc_timestamp(),
    )
    atomic_write_json_value(gate_path, gate.model_dump(mode="json"))
    return _verify_qualification(arguments, stage)


def _verify_qualification(
    arguments: argparse.Namespace,
    stage: BatchGPUQualificationStage,
) -> BatchGPUQualificationGate:
    root = _qualification_root(arguments)
    hipcc = _required_tool("hipcc")
    parent = qualification_parent_stage(stage)
    parent_hash = None
    if parent is not None:
        _verify_qualification(arguments, parent)
        parent_hash = sha256_file(qualification_gate_path(root, parent))
    gate = load_json_file(
        BatchGPUQualificationGate, qualification_gate_path(root, stage)
    )
    if not (
        gate.task is LargeBatchGPUTask.RDNA4_DIAGNOSTIC_CALIBRATION
        and gate.stage is stage
        and gate.scope_id == arguments.gpu_id
        and gate.subject_sha256 == _qualification_subject(hipcc)
        and gate.runner_sha256 == sha256_file(Path(__file__))
        and gate.configuration_sha256 == _qualification_configuration(arguments)
        and gate.parent_gate_sha256 == parent_hash
        and gate.item_ids == _qualification_modes(stage)
    ):
        raise ValueError(f"calibration qualification identity drift: {stage}")
    for receipt in gate.receipts:
        for evidence_artifact in receipt.artifacts:
            verify_qualification_artifact(root, evidence_artifact)
    if stage is not BatchGPUQualificationStage.STATIC:
        device, gpu_id, gpu_bdf = _qualification_hardware(arguments)
        for receipt in gate.receipts:
            payload = load_json_value(root / receipt.artifacts[0].path)
            if payload.get("device") != _qualification_device(
                device, gpu_id, gpu_bdf
            ):
                raise ValueError("calibration qualification hardware drift")
    return gate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            *(stage.command for stage in BatchGPUQualificationStage),
            "freeze-policy",
            "run",
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualification-root", type=Path)
    parser.add_argument("--vram-policy", type=Path)
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--tuning-batches", type=int, default=3)
    parser.add_argument("--estimation-batches", type=int, default=5)
    arguments = parser.parse_args()
    if arguments.tuning_batches < 1:
        parser.error("--tuning-batches must be positive")
    if arguments.estimation_batches < 5:
        parser.error("--estimation-batches must be at least 5")
    if arguments.stage != "freeze-policy" and arguments.vram_policy is None:
        parser.error("calibration stage requires --vram-policy")
    if (
        arguments.stage != "freeze-policy"
        and arguments.qualification_root is None
    ):
        parser.error("calibration stage requires --qualification-root")
    return arguments


def main() -> int:
    """Run the command-line calibration workflow."""
    arguments = _parse_args()
    if arguments.stage == "freeze-policy":
        path = _freeze_vram_policy(arguments)
        print(path)
        return 0
    if arguments.stage != "run":
        stage = BatchGPUQualificationStage(
            arguments.stage.removeprefix("qualify-")
        )
        gate = _run_qualification(arguments, stage)
        print(gate.model_dump_json())
        return 0
    _verify_qualification(arguments, BatchGPUQualificationStage.FULL)
    path = run_calibration(
        output=arguments.output,
        gpu_id=arguments.gpu_id,
        tuning_batches=arguments.tuning_batches,
        estimation_batches=arguments.estimation_batches,
        vram_policy_path=_required_vram_policy_path(arguments),
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
        return CliExitCode.EXECUTION


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
