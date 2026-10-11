# SPDX-License-Identifier: Apache-2.0
"""Content-bound qualification for the fixed development upgrade experiment."""

import argparse
from collections.abc import Callable
from pathlib import Path

from sol_execbench.core.bench.batch_gpu_qualification import (
    BatchGPUQualificationGate,
    BatchGPUQualificationReceipt,
    BatchGPUQualificationStage,
    LargeBatchGPUTask,
    qualification_artifact,
    qualification_gate_path,
    qualification_parent_stage,
    require_isolated_qualification_root,
    select_risk_first_axis_extrema,
    verify_qualification_artifact,
)
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_value,
)
from sol_execbench.core.data.solution_instance import Solution
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.integrity import (
    sha256_file,
    stable_json_checksum,
    verify_artifact_file,
)
from sol_execbench.core.timestamps import utc_timestamp

WORKER = Path(__file__).with_name("case_worker.py")
STACKS = ("new", "old")


def validate_inputs(root: Path) -> dict:
    """Require the entire frozen denominator and all current wire schemas."""
    manifest = load_json_value(root / "manifest.json")
    if (
        len(manifest["items"]) != 24
        or len({i["id"] for i in manifest["items"]}) != 24
    ):
        raise ValueError("expected exactly 24 unique frozen cases")
    if (
        stable_json_checksum(manifest["files"])
        != manifest["denominator_sha256"]
    ):
        raise ValueError("denominator identity drift")
    for relative, digest in manifest["files"].items():
        verify_artifact_file(
            root,
            relative,
            expected_sha256=digest,
            expected_size_bytes=(root / relative).stat().st_size,
        )
    for item in manifest["items"]:
        for field, model in (
            ("definition", Definition),
            ("workload", Workload),
            ("candidate", Solution),
        ):
            if item[field] not in manifest["files"]:
                raise ValueError("unbound case file")
            model.model_validate(load_json_value(root / item[field]))
        if item["candidate_sha256"] != manifest["files"][item["candidate"]]:
            raise ValueError("candidate identity drift")
        if item["tensors_sha256"] != manifest["files"].get(item["tensors"]):
            raise ValueError("frozen tensor identity drift")
    return manifest


def selected_items(
    manifest: dict, stage: BatchGPUQualificationStage
) -> list[dict]:
    """Choose every family-axis extreme, then run highest storage risk first."""
    items = manifest["items"]
    if stage is BatchGPUQualificationStage.CANARY:
        items = [
            item
            for family in sorted({i["family"] for i in items})
            for item in select_risk_first_axis_extrema(
                [i for i in items if i["family"] == family],
                item_id=lambda i: i["id"],
                axes=lambda i: i["axes"],
            )
        ]
    return sorted(
        items, key=lambda i: (-i["minimum_tensor_storage_bytes"], i["id"])
    )


def identities(root: Path, plan: Path) -> dict:
    """Bind exact plan, frozen bytes and all local acquisition implementation."""
    sources = [WORKER, Path(__file__), Path(__file__).with_name("acquire.py")]
    return {
        "subject_sha256": sha256_file(root / "manifest.json"),
        "configuration_sha256": sha256_file(plan),
        "runner_sha256": stable_json_checksum(
            {p.name: sha256_file(p) for p in sources}
        ),
    }


def require_gate(
    root: Path,
    plan: Path,
    qualification: Path,
    stage: BatchGPUQualificationStage,
) -> BatchGPUQualificationGate:
    """Reverify the complete predecessor chain and every retained result."""
    manifest = validate_inputs(root)
    gate_path = qualification_gate_path(qualification, stage)
    gate = BatchGPUQualificationGate.model_validate(load_json_value(gate_path))
    protocol = load_json_value(plan)
    expected_ids = tuple(
        f"{stack}:{item['id']}"
        for stack in STACKS
        for item in selected_items(manifest, stage)
    )
    if (
        gate.task is not LargeBatchGPUTask.STACK_UPGRADE_SCREENING
        or gate.stage is not stage
        or gate.item_ids != expected_ids
    ):
        raise ValueError("qualification task/stage/denominator mismatch")
    if (
        gate.scope_id != protocol["scope_id"]
        or gate.source_revision != protocol["source_revision"]
    ):
        raise ValueError("qualification source or scope mismatch")
    if any(
        getattr(gate, key) != value
        for key, value in identities(root, plan).items()
    ):
        raise ValueError("qualification input or runner identity drift")
    _verify_receipts(gate, qualification, manifest, protocol)
    parent = qualification_parent_stage(stage)
    if parent is not None:
        require_gate(root, plan, qualification, parent)
        if gate.parent_gate_sha256 != sha256_file(
            qualification_gate_path(qualification, parent)
        ):
            raise ValueError("qualification predecessor drift")
    return gate


def _verify_receipts(
    gate: BatchGPUQualificationGate, root: Path, manifest: dict, protocol: dict
) -> None:
    observed = []
    for receipt in gate.receipts:
        if receipt.input_sha256 != gate.subject_sha256:
            raise ValueError("qualification receipt input drift")
        for artifact in receipt.artifacts:
            path = verify_qualification_artifact(root, artifact)
            result = load_json_value(path)
            if gate.stage is BatchGPUQualificationStage.STATIC:
                if result != manifest:
                    raise ValueError("static qualification denominator drift")
            else:
                if (
                    result["correct"] is not True
                    or result["timed"] is not False
                ):
                    raise ValueError("invalid qualification observation")
                if result["image"] != protocol["images"][result["stack"]]:
                    raise ValueError("qualification image drift")
                observed.append(f"{result['stack']}:{result['id']}")
    if (
        gate.stage is not BatchGPUQualificationStage.STATIC
        and tuple(observed) != gate.item_ids
    ):
        raise ValueError(
            "qualification observations do not cover selected inputs"
        )


def _gpu_artifacts(
    protocol: dict,
    root: Path,
    qualification: Path,
    stage_root: Path,
    items: list[dict],
    execute: Callable,
) -> tuple:
    artifacts = []
    for stack in STACKS:
        for index, item in enumerate(items):
            output = stage_root / f"{stack}-{index}.json"
            result = execute(protocol, root, item, stack, output, timed=False)
            if result["correct"] is not True:
                raise ValueError("incorrect qualification output")
            if stack == "old":
                new_result = load_json_value(stage_root / f"new-{index}.json")
                if result["input_sha256"] != new_result["input_sha256"]:
                    raise ValueError("qualification old/new inputs differ")
            artifacts.append(qualification_artifact(qualification, output))
    return tuple(artifacts)


def qualify(
    root: Path,
    plan: Path,
    qualification: Path,
    stage: BatchGPUQualificationStage,
    execute: Callable,
) -> BatchGPUQualificationGate:
    """Issue a gate only after every selected stack/case succeeds."""
    manifest = validate_inputs(root)
    protocol = load_json_value(plan)
    require_isolated_qualification_root(
        qualification, Path(protocol["measurements_root"])
    )
    parent = qualification_parent_stage(stage)
    if parent is not None:
        require_gate(root, plan, qualification, parent)
    path = qualification_gate_path(qualification, stage)
    if path.exists():
        return require_gate(root, plan, qualification, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = identities(root, plan)
    items = selected_items(manifest, stage)
    artifacts = []
    if stage is BatchGPUQualificationStage.STATIC:
        static = path.parent / "inputs.json"
        atomic_write_json_value(static, manifest)
        artifacts.append(qualification_artifact(qualification, static))
    else:
        artifacts.extend(
            _gpu_artifacts(
                protocol, root, qualification, path.parent, items, execute
            )
        )
    ids = tuple(f"{stack}:{item['id']}" for stack in STACKS for item in items)
    gate = BatchGPUQualificationGate(
        task=LargeBatchGPUTask.STACK_UPGRADE_SCREENING,
        stage=stage,
        scope_id=protocol["scope_id"],
        source_revision=protocol["source_revision"],
        **identity,
        parent_gate_sha256=sha256_file(
            qualification_gate_path(qualification, parent)
        )
        if parent
        else None,
        item_ids=ids,
        created_at=utc_timestamp(),
        receipts=(
            BatchGPUQualificationReceipt(
                stage=stage,
                partition="both_stacks",
                item_ids=ids,
                input_sha256=identity["subject_sha256"],
                artifacts=tuple(artifacts),
            ),
        ),
    )
    if identity != identities(root, plan):
        raise ValueError("qualification sources changed while running")
    _verify_receipts(gate, qualification, manifest, protocol)
    atomic_write_json_value(path, gate.model_dump(mode="json"))
    return gate


def main() -> None:
    """Use the common mandatory stage names for a development-only experiment."""
    from scripts.internal.upgrade.acquire import execute

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=[stage.command for stage in BatchGPUQualificationStage]
    )
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--qualification-root", type=Path, required=True)
    args = parser.parse_args()
    qualify(
        args.inputs.resolve(),
        args.plan.resolve(),
        args.qualification_root.resolve(),
        BatchGPUQualificationStage(args.stage.removeprefix("qualify-")),
        execute,
    )


if __name__ == "__main__":
    main()
