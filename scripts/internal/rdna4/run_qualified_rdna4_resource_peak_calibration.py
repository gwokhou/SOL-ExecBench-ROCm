# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Qualify, then run the frozen RDNA4 resource-peak v3 producer."""

from __future__ import annotations

import argparse
import tempfile
from functools import cache
from pathlib import Path
from typing import cast

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
from sol_execbench.core.bench.frozen_resource_peak_producer import (
    FrozenResourcePeakProducer,
    ResourcePeakProbe,
    ResourcePeakTuning,
    load_frozen_resource_peak_producer,
)
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_file,
    load_json_value,
)
from sol_execbench.core.integrity import sha256_file, stable_json_checksum
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
def _producer() -> FrozenResourcePeakProducer:
    """Return the typed adapter for the byte-frozen historical producer."""
    return load_frozen_resource_peak_producer(LEGACY_PRODUCER)


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
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--allow-unlocked", action="store_true")
    parser.add_argument("--qualification-root", type=Path, required=True)
    arguments = parser.parse_args(argv)
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


def _probes() -> tuple[ResourcePeakProbe, ...]:
    return _producer().probes


def _probe_names(stage: BatchGPUQualificationStage) -> tuple[str, ...]:
    if stage is BatchGPUQualificationStage.CANARY:
        return _CANARY_PROBES
    return tuple(str(probe["source"]) for probe in _probes())


def _selected_probes(
    stage: BatchGPUQualificationStage,
) -> tuple[ResourcePeakProbe, ...]:
    selected = set(_probe_names(stage))
    return tuple(probe for probe in _probes() if probe["source"] in selected)


def _subject(hipcc: str) -> str:
    producer = _producer()
    probe_root = producer.probe_dir
    return stable_json_checksum(
        {
            "legacy_producer_sha256": sha256_file(LEGACY_PRODUCER),
            "compiler_sha256": sha256_file(Path(hipcc)),
            "probes": [
                (
                    probe["source"],
                    sha256_file(probe_root / str(probe["source"])),
                )
                for probe in _probes()
            ],
        }
    )


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


def _compiler_defines(probe: ResourcePeakProbe) -> dict[str, int]:
    tuning = probe.get("tuning")
    if tuning is None:
        return {}
    tuning = cast(ResourcePeakTuning, tuning)
    return {tuning.compiler_macro: tuning.candidates[0]}


def _static_receipt(
    arguments: argparse.Namespace,
    hipcc: str,
) -> BatchGPUQualificationReceipt:
    producer = _producer()
    root = _root(arguments)
    binaries = tuple(
        producer.compile_probe(
            producer.probe_dir / str(probe["source"]),
            root / "static",
            hipcc,
            arguments.gfx,
            compiler_defines=_compiler_defines(probe),
        )
        for probe in _probes()
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


def _binary(arguments: argparse.Namespace, probe: ResourcePeakProbe) -> Path:
    source = _producer().probe_dir / str(probe["source"])
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
    probe: ResourcePeakProbe,
    hipcc: str,
    device: dict[str, object],
) -> BatchGPUQualificationReceipt:
    root = _root(arguments)
    name = str(probe["source"])
    path = root / stage.value / name / "evidence.json"
    input_sha256 = stable_json_checksum(
        {"subject_sha256": _subject(hipcc), "probe": name, "device": device}
    )
    if path.is_file():
        payload = load_json_value(path)
    else:
        batch = _producer().run_sample_batch(_binary(arguments, probe), 0)
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
    clock = _producer().clock_state()
    if not clock.get("clock_locked_verified") and not arguments.allow_unlocked:
        raise RuntimeError("resource qualification requires locked clocks")


def _run_qualification(
    arguments: argparse.Namespace,
    stage: BatchGPUQualificationStage,
) -> BatchGPUQualificationGate:
    root = _root(arguments)
    gate_path = qualification_gate_path(root, stage)
    if gate_path.is_file():
        return _verify_qualification(arguments, stage)
    producer = _producer()
    hipcc = producer.required_rocm_tool("hipcc")
    parent = qualification_parent_stage(stage)
    parent_hash = None
    if parent is not None:
        _verify_qualification(arguments, parent)
        parent_hash = sha256_file(qualification_gate_path(root, parent))
    if stage is BatchGPUQualificationStage.STATIC:
        receipts = (_static_receipt(arguments, hipcc),)
    else:
        _require_clock(arguments)
        device = producer.device_identity()
        if device["gfx_target"] != arguments.gfx:
            raise RuntimeError(
                f"visible GPU is {device['gfx_target']}, expected {arguments.gfx}"
            )
        receipts = tuple(
            _gpu_receipt(arguments, stage, probe, hipcc, device)
            for probe in _selected_probes(stage)
        )
    gate = BatchGPUQualificationGate(
        task=LargeBatchGPUTask.RDNA4_RESOURCE_PEAK_CALIBRATION,
        stage=stage,
        scope_id=f"{arguments.profile}:{arguments.gfx}",
        subject_sha256=_subject(hipcc),
        runner_sha256=sha256_file(Path(__file__)),
        configuration_sha256=_configuration(arguments),
        source_revision=producer.git_revision(),
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
    root = _root(arguments)
    producer = _producer()
    hipcc = producer.required_rocm_tool("hipcc")
    parent = qualification_parent_stage(stage)
    parent_hash = None
    if parent is not None:
        _verify_qualification(arguments, parent)
        parent_hash = sha256_file(qualification_gate_path(root, parent))
    gate = load_json_file(
        BatchGPUQualificationGate,
        qualification_gate_path(root, stage),
    )
    if not (
        gate.task is LargeBatchGPUTask.RDNA4_RESOURCE_PEAK_CALIBRATION
        and gate.stage is stage
        and gate.scope_id == f"{arguments.profile}:{arguments.gfx}"
        and gate.subject_sha256 == _subject(hipcc)
        and gate.runner_sha256 == sha256_file(Path(__file__))
        and gate.configuration_sha256 == _configuration(arguments)
        and gate.source_revision == producer.git_revision()
        and gate.parent_gate_sha256 == parent_hash
        and gate.item_ids == _probe_names(stage)
    ):
        raise ValueError(f"resource qualification identity drift: {stage}")
    for receipt in gate.receipts:
        for artifact in receipt.artifacts:
            verify_qualification_artifact(root, artifact)
    if stage is not BatchGPUQualificationStage.STATIC:
        _require_clock(arguments)
        device = producer.device_identity()
        for receipt in gate.receipts:
            payload = load_json_value(root / receipt.artifacts[0].path)
            if payload.get("device") != device:
                raise ValueError("resource qualification device identity drift")
    return gate


def _run_calibration(arguments: argparse.Namespace) -> int:
    _verify_qualification(arguments, BatchGPUQualificationStage.FULL)
    producer = _producer()
    if arguments.workdir is not None:
        return producer.calibrate(arguments, arguments.workdir)
    with tempfile.TemporaryDirectory(
        prefix="solar-resource-peak-calibration-"
    ) as temporary:
        return producer.calibrate(arguments, Path(temporary))


def main(argv: list[str] | None = None) -> int:
    """Run one uniform qualification stage or the gated calibration."""
    arguments = _parse_args(argv)
    if arguments.stage == "run":
        return _run_calibration(arguments)
    stage = BatchGPUQualificationStage(arguments.stage.removeprefix("qualify-"))
    gate = _run_qualification(arguments, stage)
    print(gate.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
