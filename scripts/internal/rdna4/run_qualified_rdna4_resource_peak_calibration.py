# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Qualify, then run the frozen RDNA4 resource-peak v3 producer."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import tempfile
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

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
from sol_execbench.core.bench.gpu_execution import native_gpu_execution
from sol_execbench.core.bench.resource_calibration_timing import (
    CalibrationPhase,
    CalibrationTimings,
)
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_value,
)
from sol_execbench.core.integrity import (
    sha256_bytes,
    sha256_file,
    stable_json_checksum,
)
from sol_execbench.core.process.locks import exclusive_file_lock
from sol_execbench.core.timestamps import utc_timestamp

REPO_ROOT = Path(__file__).resolve().parents[3]
LEGACY_PRODUCER = (
    REPO_ROOT / "scripts/internal/rdna4/run_rdna4_resource_peak_calibration.py"
)
DEFAULT_OUTPUT = REPO_ROOT / "src/solar/audits/rx9060xt_resource_peaks_v3.json"
_CANARY_PROBES = (
    "vector_fp32_fp32.hip",
    "matrix_fp16_fp16_wmma.hip",
    "reduction_fp32_fp32.hip",
    "stream_copy_fp32_fp32.hip",
)


@cache
def _legacy() -> dict[str, Any]:
    """Load the byte-frozen producer without modifying its evidence identity."""
    return runpy.run_path(str(LEGACY_PRODUCER))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            *(stage.command for stage in BatchGPUQualificationStage),
            "run",
        ),
    )
    parser.add_argument("--profile", default="RX_9060_XT")
    parser.add_argument("--gfx", default="gfx1200")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tuning-batches", type=int, default=3)
    parser.add_argument(
        "--measurement-batches",
        "--repeats",
        dest="measurement_batches",
        type=int,
        default=7,
    )
    parser.add_argument("--compile-jobs", type=int, default=1)
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--allow-unlocked", action="store_true")
    parser.add_argument("--qualification-root", type=Path, required=True)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify the existing artifact chain on CPU; no device validation or execution",
    )
    arguments = parser.parse_args(argv)
    if arguments.compile_jobs <= 0:
        parser.error("--compile-jobs must be positive")
    if arguments.tuning_batches <= 0:
        parser.error("--tuning-batches must be positive")
    if arguments.measurement_batches < 5:
        parser.error("--measurement-batches must be at least 5")
    return arguments


def _root(arguments: argparse.Namespace) -> Path:
    return require_isolated_qualification_root(
        arguments.qualification_root,
        arguments.out,
    )


def _probes() -> tuple[Mapping[str, Any], ...]:
    return tuple(_legacy()["PROBES"])


def _probe_names(stage: BatchGPUQualificationStage) -> tuple[str, ...]:
    if stage is BatchGPUQualificationStage.CANARY:
        return _CANARY_PROBES
    return tuple(str(probe["source"]) for probe in _probes())


def _selected_probes(
    stage: BatchGPUQualificationStage,
) -> tuple[Mapping[str, Any], ...]:
    selected = set(_probe_names(stage))
    return tuple(probe for probe in _probes() if probe["source"] in selected)


def _subject(hipcc: str) -> str:
    legacy = _legacy()
    probe_root = Path(legacy["PROBE_DIR"])
    return stable_json_checksum(
        {
            "legacy_producer_sha256": sha256_file(LEGACY_PRODUCER),
            "compiler_sha256": sha256_file(Path(hipcc)),
            "toolchain": _toolchain_identity(hipcc),
            "probes": [
                (
                    probe["source"],
                    sha256_file(probe_root / str(probe["source"])),
                )
                for probe in _probes()
            ],
        }
    )


def _toolchain_identity(hipcc: str) -> dict[str, object]:
    legacy = _legacy()
    configured = os.environ.get("HIP_CLANG_PATH")
    clang = (
        (Path(configured) / "clang++")
        if configured
        else Path(legacy["_required_rocm_tool"]("clang++"))
    )
    return {
        "hipcc_version": legacy["_run"]([hipcc, "--version"]).stdout,
        "clang_path": str(clang),
        "clang_sha256": sha256_file(clang),
        "environment": {
            key: os.environ[key]
            for key in (
                "CPATH",
                "CPLUS_INCLUDE_PATH",
                "LIBRARY_PATH",
                "PATH",
                "LD_LIBRARY_PATH",
                "HIP_PLATFORM",
                "DEVICE_LIB_PATH",
                "HIP_CLANG_PATH",
                "HIP_PATH",
                "ROCM_PATH",
                "HIPCC_COMPILE_FLAGS_APPEND",
                "HIPCC_LINK_FLAGS_APPEND",
            )
            if key in os.environ
        },
    }


def _configuration(arguments: argparse.Namespace) -> str:
    return stable_json_checksum(
        {
            "profile": arguments.profile,
            "gfx": arguments.gfx,
            "tuning_batches": arguments.tuning_batches,
            "measurement_batches": arguments.measurement_batches,
            "allow_unlocked": arguments.allow_unlocked,
        }
    )


def _compiler_defines(probe: Mapping[str, Any]) -> dict[str, int]:
    tuning = probe.get("tuning")
    if tuning is None:
        return {}
    return {tuning.compiler_macro: tuning.candidates[0]}


def _static_receipt(
    arguments: argparse.Namespace,
    hipcc: str,
) -> BatchGPUQualificationReceipt:
    legacy = _legacy()
    root = _root(arguments)
    compiled = legacy["_compile_calibration_binaries"](
        arguments.gfx,
        root / "static",
        hipcc,
        compile_jobs=getattr(arguments, "compile_jobs", 1),
    )
    binaries = tuple(
        path for group in compiled.values() for path in group.values()
    )
    path = root / "static" / "preflight.json"
    item_ids = _probe_names(BatchGPUQualificationStage.STATIC)
    payload = {
        "task": LargeBatchGPUTask.RDNA4_RESOURCE_PEAK_CALIBRATION,
        "subject_sha256": _subject(hipcc),
        "item_ids": item_ids,
        "compile_passed": True,
    }
    atomic_write_json_value(path, payload)
    return BatchGPUQualificationReceipt(
        stage=BatchGPUQualificationStage.STATIC,
        partition="resource-probes",
        item_ids=item_ids,
        input_sha256=stable_json_checksum(payload),
        artifacts=(
            *(qualification_artifact(root, binary) for binary in binaries),
            qualification_artifact(root, path),
        ),
    )


def _binary(arguments: argparse.Namespace, probe: Mapping[str, Any]) -> Path:
    source = Path(_legacy()["PROBE_DIR"]) / str(probe["source"])
    definitions = _compiler_defines(probe)
    label = "-".join(
        f"{name.lower()}-{value}" for name, value in sorted(definitions.items())
    )
    return (
        _root(arguments)
        / "static/binaries"
        / source.stem
        / f"{source.stem}-{label or 'default'}.bin"
    )


def _gpu_receipt(
    arguments: argparse.Namespace,
    stage: BatchGPUQualificationStage,
    probe: Mapping[str, Any],
    subject_sha256: str,
    device: dict[str, Any],
) -> BatchGPUQualificationReceipt:
    root = _root(arguments)
    name = str(probe["source"])
    path = root / stage.value / name / "evidence.json"
    input_sha256 = stable_json_checksum(
        {"subject_sha256": subject_sha256, "probe": name, "device": device}
    )
    if path.is_file():
        payload = load_json_value(path)
    else:
        batch = _legacy()["_run_sample_batch"](
            _binary(arguments, probe), 0, amdsmi=None
        )
        payload = {
            "stage": stage,
            "probe": name,
            "input_sha256": input_sha256,
            "device": device,
            "batch": batch.to_dict(),
            "all_passed": True,
        }
        atomic_write_json_value(path, payload)
    if (
        payload.get("input_sha256") != input_sha256
        or payload.get("device") != device
        or payload.get("all_passed") is not True
    ):
        raise ValueError(f"resource qualification evidence drift: {path}")
    return BatchGPUQualificationReceipt(
        stage=stage,
        partition=name,
        item_ids=(name,),
        input_sha256=input_sha256,
        artifacts=(qualification_artifact(root, path),),
    )


def _require_clock(arguments: argparse.Namespace) -> None:
    clock = _legacy()["_clock_state"]()
    if not clock.get("clock_locked_verified") and not arguments.allow_unlocked:
        raise RuntimeError("resource qualification requires locked clocks")


@dataclass(frozen=True)
class QualificationIdentity:
    """Invocation-local CPU identity, recomputed on every CLI call."""

    hipcc: str
    subject: str
    runner: str
    configuration: str
    revision: str


def _identity(arguments: argparse.Namespace) -> QualificationIdentity:
    hipcc = _legacy()["_required_rocm_tool"]("hipcc")
    return QualificationIdentity(
        hipcc,
        _subject(hipcc),
        sha256_file(Path(__file__)),
        _configuration(arguments),
        _legacy()["_git_revision"](),
    )


def _check_device(
    arguments: argparse.Namespace, gates: list[BatchGPUQualificationGate]
) -> dict[str, Any]:
    _require_clock(arguments)
    device = _legacy()["_device_identity"]()
    if device["gfx_target"] != arguments.gfx:
        raise RuntimeError(
            f"visible GPU is {device['gfx_target']}, expected {arguments.gfx}"
        )
    root = _root(arguments)
    for gate in gates:
        if gate.stage is BatchGPUQualificationStage.STATIC:
            continue
        for receipt in gate.receipts:
            payload = load_json_value(root / receipt.artifacts[0].path)
            if payload.get("device") != device:
                raise ValueError("resource qualification device identity drift")
    return device


def _run_qualification(
    arguments: argparse.Namespace,
    stage: BatchGPUQualificationStage,
) -> BatchGPUQualificationGate:
    # Competing static publishers fail before touching binaries; no GPU lock.
    lock = (
        exclusive_file_lock(_root(arguments) / ".static-compile.lock")
        if stage is BatchGPUQualificationStage.STATIC
        else nullcontext()
    )
    with lock:
        return _run_qualification_locked(arguments, stage)


def _run_qualification_locked(
    arguments: argparse.Namespace,
    stage: BatchGPUQualificationStage,
) -> BatchGPUQualificationGate:
    timings = CalibrationTimings()
    try:
        with timings.measure(CalibrationPhase.PREPARE):
            identity = _identity(arguments)
            root = _root(arguments)
            gate_path = qualification_gate_path(root, stage)
            gates: list[BatchGPUQualificationGate] = []
            gate_hashes: dict[BatchGPUQualificationStage, str] = {}
            existing = gate_path.is_file()
            receipts: tuple[BatchGPUQualificationReceipt, ...] = ()
            parent_hash = None
            predecessor = (
                stage if existing else qualification_parent_stage(stage)
            )
            if predecessor is not None:
                _verify_qualification(
                    arguments, predecessor, identity, gates, gate_hashes
                )
                parent_hash = gate_hashes[predecessor]
            if stage is BatchGPUQualificationStage.STATIC and not existing:
                receipts = (_static_receipt(arguments, identity.hipcc),)
        if stage is not BatchGPUQualificationStage.STATIC:
            with (
                timings.measure(CalibrationPhase.GPU_LIFECYCLE),
                native_gpu_execution(
                    manage_clocks=not arguments.allow_unlocked
                ),
            ):
                device = _check_device(arguments, gates)
                if not existing:
                    receipts = tuple(
                        _gpu_receipt(
                            arguments, stage, probe, identity.subject, device
                        )
                        for probe in _selected_probes(stage)
                    )
        with timings.measure(CalibrationPhase.FINALIZE):
            if existing:
                return gates[-1]
            return _write_gate(
                arguments, stage, identity, receipts, parent_hash
            )
    finally:
        timings.report()


def _write_gate(
    arguments: argparse.Namespace,
    stage: BatchGPUQualificationStage,
    identity: QualificationIdentity,
    receipts: tuple[BatchGPUQualificationReceipt, ...],
    parent_hash: str | None,
) -> BatchGPUQualificationGate:
    root = _root(arguments)
    parent = qualification_parent_stage(stage)
    if (
        parent is not None
        and sha256_file(qualification_gate_path(root, parent)) != parent_hash
    ):
        raise ValueError(
            "resource qualification parent changed during execution"
        )
    gate = BatchGPUQualificationGate(
        task=LargeBatchGPUTask.RDNA4_RESOURCE_PEAK_CALIBRATION,
        stage=stage,
        scope_id=f"{arguments.profile}:{arguments.gfx}",
        subject_sha256=identity.subject,
        runner_sha256=identity.runner,
        configuration_sha256=identity.configuration,
        source_revision=identity.revision,
        parent_gate_sha256=parent_hash,
        item_ids=tuple(
            item for receipt in receipts for item in receipt.item_ids
        ),
        receipts=receipts,
        created_at=utc_timestamp(),
    )
    atomic_write_json_value(
        qualification_gate_path(root, stage), gate.model_dump(mode="json")
    )
    return gate


def _verify_qualification(
    arguments: argparse.Namespace,
    stage: BatchGPUQualificationStage,
    identity: QualificationIdentity | None = None,
    gates: list[BatchGPUQualificationGate] | None = None,
    gate_hashes: dict[BatchGPUQualificationStage, str] | None = None,
) -> BatchGPUQualificationGate:
    root = _root(arguments)
    identity = identity or _identity(arguments)
    gate_hashes = {} if gate_hashes is None else gate_hashes
    parent = qualification_parent_stage(stage)
    parent_hash = None
    if parent is not None:
        _verify_qualification(arguments, parent, identity, gates, gate_hashes)
        parent_hash = gate_hashes[parent]
    raw = qualification_gate_path(root, stage).read_bytes()
    gate = BatchGPUQualificationGate.model_validate_json(raw)
    gate_hashes[stage] = sha256_bytes(raw)
    if not (
        gate.task is LargeBatchGPUTask.RDNA4_RESOURCE_PEAK_CALIBRATION
        and gate.stage is stage
        and gate.scope_id == f"{arguments.profile}:{arguments.gfx}"
        and gate.subject_sha256 == identity.subject
        and gate.runner_sha256 == identity.runner
        and gate.configuration_sha256 == identity.configuration
        and gate.source_revision == identity.revision
        and gate.parent_gate_sha256 == parent_hash
        and gate.item_ids == _probe_names(stage)
    ):
        raise ValueError(f"resource qualification identity drift: {stage}")
    for receipt in gate.receipts:
        for artifact in receipt.artifacts:
            verify_qualification_artifact(root, artifact)
    if gates is not None:
        gates.append(gate)
    return gate


def _run_calibration(arguments: argparse.Namespace) -> int:
    if arguments.workdir is not None:
        return _run_prepared_calibration(arguments, arguments.workdir)
    with tempfile.TemporaryDirectory(
        prefix="solar-resource-peak-calibration-"
    ) as temporary:
        return _run_prepared_calibration(arguments, Path(temporary))


def _prepare_qualified_calibration(
    arguments: argparse.Namespace,
    gates: list[BatchGPUQualificationGate],
    legacy: dict[str, Any],
) -> Any:
    """Consume static binaries; never compile inside a host-managed GPU job."""
    root = _root(arguments)
    static = gates[0]
    expected = {
        (root / artifact.path).resolve(): artifact.sha256
        for receipt in static.receipts
        for artifact in receipt.artifacts
    }
    binaries = legacy["_calibration_binary_paths"](root / "static")
    paths = [path for group in binaries.values() for path in group.values()]
    if any(path.resolve() not in expected for path in paths):
        raise ValueError("static qualification is missing a calibration binary")
    prepared = legacy["CalibrationPreparation"](
        legacy["_required_rocm_tool"]("hipcc"),
        binaries,
        {path: expected[path.resolve()] for path in paths},
        legacy["_isa_spec_evidence"](arguments.gfx),
    )
    prepared.verify()
    return prepared


def _run_prepared_calibration(
    arguments: argparse.Namespace, workdir: Path
) -> int:
    timings = CalibrationTimings()
    try:
        with timings.measure(CalibrationPhase.PREPARE):
            gates: list[BatchGPUQualificationGate] = []
            _verify_qualification(
                arguments, BatchGPUQualificationStage.FULL, gates=gates
            )
            legacy = _legacy()
            preparation = _prepare_qualified_calibration(
                arguments, gates, legacy
            )
        with (
            timings.measure(CalibrationPhase.GPU_LIFECYCLE),
            native_gpu_execution(manage_clocks=not arguments.allow_unlocked),
        ):
            _check_device(arguments, gates)
            return int(legacy["_calibrate"](arguments, workdir, preparation))
    finally:
        timings.report()


def main(argv: list[str] | None = None) -> int:
    """Validate CPU inputs before acquiring the native GPU lifecycle."""
    arguments = _parse_args(argv)
    stage = (
        BatchGPUQualificationStage.FULL
        if arguments.stage == "run"
        else BatchGPUQualificationStage(
            arguments.stage.removeprefix("qualify-")
        )
    )
    if arguments.check_only:
        gate = _verify_qualification(arguments, stage)
        print(
            json.dumps(
                {
                    "stage": stage.value,
                    "artifact_chain_verified": True,
                    "device_readback_performed": False,
                    "items": len(gate.item_ids),
                }
            )
        )
        return 0
    if arguments.stage == "run":
        return _run_calibration(arguments)
    gate = _run_qualification(arguments, stage)
    print(gate.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
