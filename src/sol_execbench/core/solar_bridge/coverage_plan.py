"""Read-only C0 campaign planning and exact resume identity construction."""

from dataclasses import asdict
from pathlib import Path

from sol_execbench.core.dataset.corpus import load_target_descriptor
from sol_execbench.core.dataset.target_view import (
    MaterializedTargetView,
    load_materialized_target_view,
    require_llm_core_scope,
)
from sol_execbench.core.integrity import sha256_file, stable_json_checksum
from sol_execbench.core.integrity.source_tree import source_tree_manifest
from sol_execbench.core.solar_bridge.coverage_models import (
    COVERAGE_PATHS,
    COVERAGE_SOURCE_SHA256,
    COVERAGE_TARGET_SHA256,
    SLOT_ORDER,
    CoverageOptions,
    CoverageRow,
    CoverageSpec,
    TargetCoverageReport,
)
from sol_execbench.core.solar_bridge.resource_policy import (
    formal_mapper_thread_count,
)
from sol_execbench.core.solar_bridge.workload_context import (
    workload_verification_policy,
)
from solar.verification.execution import semantic_runtime_identity


def load_coverage_target(
    source: Path, view: Path, descriptor: Path
) -> MaterializedTargetView:
    """Require the frozen corpus and target declaration before any worker entry."""
    if (
        sha256_file(source) != COVERAGE_SOURCE_SHA256
        or sha256_file(descriptor) != COVERAGE_TARGET_SHA256
    ):
        raise ValueError("M0-C0 frozen manifest or target declaration drift")
    target = load_materialized_target_view(source, view)
    require_llm_core_scope(target)
    if target.view.target != load_target_descriptor(descriptor):
        raise ValueError("M0-C0 target view differs from frozen descriptor")
    return target


def coverage_rows(
    target: MaterializedTargetView, device: str
) -> list[CoverageRow]:
    """Enumerate every row without loading reference code or allocating inputs."""
    entries = {item.semantic_id: item for item in target.source.entries}
    records = sorted(
        target.view.workloads,
        key=lambda r: (
            r.slot_id != "smoke",
            r.semantic_id,
            SLOT_ORDER.index(r.slot_id),
        ),
    )
    rows = []
    for record in records:
        for ir_path in COVERAGE_PATHS:
            policy = workload_verification_policy(
                target.definitions[record.semantic_id],
                target.workloads[record.uuid],
                device,
            )
            rows.append(
                CoverageRow(
                    key=f"{CoverageSpec.M0}/{target.view.workload_view_digest}/{record.semantic_id}/{record.uuid}/{ir_path}",
                    semantic_id=record.semantic_id,
                    workload_uuid=record.uuid,
                    ir_path=ir_path,
                    slot=record.slot_id,
                    role=record.role,
                    family=entries[record.semantic_id].operation_family,
                    policy=asdict(policy),
                )
            )
    return rows


def new_coverage_report(
    target: MaterializedTargetView,
    options: CoverageOptions,
    mapper_identity: dict[str, object],
) -> TargetCoverageReport:
    """Bind source, runtime, generated metadata, policy, mapper and budgets."""
    rows = coverage_rows(target, options.device)
    identity = {
        "spec": CoverageSpec.M0,
        "source": source_tree_manifest(Path(__file__).resolve().parents[3]),
        "runtime": semantic_runtime_identity(),
        "target_view": target.view.model_dump(mode="json"),
        "view_file_sha256": sha256_file(target.path),
        "options": options.model_dump(mode="json"),
        "mapper": mapper_identity,
        "mapper_threads": formal_mapper_thread_count(),
        "rows": [row.model_dump(mode="json") for row in rows],
    }
    return TargetCoverageReport(
        identity=identity,
        identity_sha256=stable_json_checksum(identity),
        decisions=[d.model_dump(mode="json") for d in target.view.decisions],
        rows=rows,
    )
