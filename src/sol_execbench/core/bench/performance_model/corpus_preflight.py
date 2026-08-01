"""CPU-only structural preflight for the preregistered gfx1200 corpus."""

from __future__ import annotations

import argparse
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from sol_execbench.core.data.base_model import (
    CurrentSchemaModel,
    NonEmptyString,
    StrictArtifactModel,
)
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.json_utils import load_jsonl_file
from sol_execbench.core.data.solution_instance import Solution
from sol_execbench.core.data.workload import TopKRoutingCheck, Workload
from sol_execbench.core.integrity import sha256_file

DESIGN_SCHEMA = "rdna4_diagnostic_corpus_design.v1"
DESIGN_KIND = "adjacent_shape_stratified_three_way_rotation"
CASES_PER_BATCH = 20
PHASES_PER_FAMILY = 3
UNIVERSE_CASES_PER_FAMILY = CASES_PER_BATCH * PHASES_PER_FAMILY

_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    allow_inf_nan=False,
)


class DiagnosticFamily(StrEnum):
    """Closed family vocabulary for the diagnostic validation corpus."""

    ELEMENTWISE = "elementwise"
    TRANSPOSE = "transpose"
    REDUCTION_NORM = "reduction_norm"
    MATMUL = "matmul"
    SOFTMAX = "softmax"
    CROSS_ENTROPY = "cross_entropy"
    INDEXED_READ = "indexed_read"
    INDEXED_UPDATE = "indexed_update"
    COMPOSITE_GRAPH = "composite_graph"
    TRANSFORMER_BLOCK = "transformer_block"
    CONCURRENT_GRAPH = "concurrent_graph"


class DiagnosticPhase(StrEnum):
    """Closed preregistered collection-phase vocabulary."""

    POINT_FIT = "point_fit"
    CONFORMAL = "conformal"
    HELD_OUT = "held_out"


class DiagnosticCorpusRole(StrEnum):
    """Authority partition assigned to a preregistered case."""

    DEVELOPMENT = "development"
    HELD_OUT = "held_out"


class DiagnosticCollectionStatus(StrEnum):
    """Recoverable status states emitted by collection runners."""

    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"
    ALREADY_COLLECTED = "already_collected"


FAMILIES = tuple(DiagnosticFamily)
PHASES = tuple(DiagnosticPhase)
PHASE_ROLES = {
    DiagnosticPhase.POINT_FIT: DiagnosticCorpusRole.DEVELOPMENT,
    DiagnosticPhase.CONFORMAL: DiagnosticCorpusRole.DEVELOPMENT,
    DiagnosticPhase.HELD_OUT: DiagnosticCorpusRole.HELD_OUT,
}
EXPECTED_CASES = len(DiagnosticFamily) * len(DiagnosticPhase) * CASES_PER_BATCH
TERMINAL_STATUSES = frozenset(
    {
        DiagnosticCollectionStatus.DONE,
        DiagnosticCollectionStatus.ALREADY_COLLECTED,
    }
)


class DiagnosticCorpusCase(StrictArtifactModel):
    """One frozen case identity in the preregistered design."""

    model_config = _CONFIG

    axes: dict[NonEmptyString, int] = Field(min_length=1)
    case_id: NonEmptyString
    family: DiagnosticFamily
    global_index: int = Field(ge=0)
    phase: DiagnosticPhase
    role: DiagnosticCorpusRole
    workload_uuid: NonEmptyString

    @model_validator(mode="after")
    def role_matches_phase(self) -> DiagnosticCorpusCase:
        """Reject development/held-out authority drift."""
        if self.role is not PHASE_ROLES[self.phase]:
            raise ValueError("diagnostic corpus phase/role mismatch")
        return self


class DiagnosticPhaseCaseCounts(StrictArtifactModel):
    """Exact per-phase case counts for each diagnostic family."""

    model_config = _CONFIG

    point_fit: Literal[20]
    conformal: Literal[20]
    held_out: Literal[20]


class DiagnosticCorpusDesign(CurrentSchemaModel):
    """Frozen 11-family, three-phase diagnostic corpus design."""

    model_config = _CONFIG
    current_schema_version = DESIGN_SCHEMA

    schema_version: Literal["rdna4_diagnostic_corpus_design.v1"]
    design: Literal["adjacent_shape_stratified_three_way_rotation"]
    cases_per_family: DiagnosticPhaseCaseCounts
    universe_cases_per_family: Literal[60]
    universe_start: int = Field(ge=0)
    configuration_frozen_before_collection: Literal[True]
    cases: list[DiagnosticCorpusCase] = Field(
        min_length=EXPECTED_CASES,
        max_length=EXPECTED_CASES,
    )

    @model_validator(mode="after")
    def cases_form_exact_partition(self) -> DiagnosticCorpusDesign:
        """Require unique identities and exactly 33 batches of 20."""
        for field in ("case_id", "workload_uuid"):
            values = [getattr(case, field) for case in self.cases]
            repeated = _duplicates(values)
            if repeated:
                raise ValueError(
                    f"diagnostic corpus repeats {field}: {sorted(repeated)}"
                )
        counts = Counter((case.family, case.phase) for case in self.cases)
        expected = {
            (family, phase)
            for family in DiagnosticFamily
            for phase in DiagnosticPhase
        }
        if set(counts) != expected or any(
            count != CASES_PER_BATCH for count in counts.values()
        ):
            raise ValueError(
                "diagnostic corpus is not partitioned into 33x20 batches"
            )
        for family in DiagnosticFamily:
            indexes = [
                case.global_index
                for case in self.cases
                if case.family is family
            ]
            if len(indexes) != len(set(indexes)):
                raise ValueError(
                    f"diagnostic corpus repeats {family} global indexes"
                )
        return self


class DiagnosticCollectionRecord(StrictArtifactModel):
    """One append-only collection status record."""

    model_config = _CONFIG

    axes: dict[NonEmptyString, int] = Field(min_length=1)
    case_id: NonEmptyString
    collect_seconds: float | None = Field(default=None, ge=0)
    error: str | None = Field(default=None, max_length=1024)
    family: DiagnosticFamily
    global_index: int = Field(ge=0)
    index: int = Field(ge=0)
    phase: DiagnosticPhase
    solar_seconds: float | None = Field(default=None, ge=0)
    status: DiagnosticCollectionStatus
    workload_uuid: NonEmptyString


class CollectionBatch(StrictArtifactModel):
    """One deterministic family/phase collection batch."""

    model_config = _CONFIG

    family: DiagnosticFamily
    phase: DiagnosticPhase
    role: DiagnosticCorpusRole
    case_ids: list[NonEmptyString] = Field(
        min_length=CASES_PER_BATCH,
        max_length=CASES_PER_BATCH,
    )


class ResumeSummary(StrictArtifactModel):
    """Validated resume state derived from persisted evidence."""

    model_config = _CONFIG

    records: int = Field(ge=0)
    resolved: int = Field(ge=0)
    pending: int = Field(ge=0)
    evidence_sha256: dict[str, str]


class PreflightSummary(StrictArtifactModel):
    """Machine-readable result of the complete CPU preflight."""

    model_config = _CONFIG

    ok: Literal[True] = True
    design_sha256: str
    cases: int = Field(ge=0)
    families: int = Field(ge=0)
    batches: list[CollectionBatch]
    resume: ResumeSummary | None = None


def _duplicates(values: list[str]) -> set[str]:
    counts = Counter(values)
    return {value for value, count in counts.items() if count > 1}


def validate_design(payload: object) -> list[DiagnosticCorpusCase]:
    """Validate a design payload and return its typed frozen cases."""
    return DiagnosticCorpusDesign.model_validate(payload).cases


def build_batches(
    cases: list[DiagnosticCorpusCase],
) -> list[CollectionBatch]:
    """Return the deterministic 33-batch collection plan."""
    batches: list[CollectionBatch] = []
    for family in DiagnosticFamily:
        for phase in DiagnosticPhase:
            selected = sorted(
                (
                    case
                    for case in cases
                    if case.family is family and case.phase is phase
                ),
                key=lambda case: case.global_index,
            )
            batches.append(
                CollectionBatch(
                    family=family,
                    phase=phase,
                    role=PHASE_ROLES[phase],
                    case_ids=[case.case_id for case in selected],
                )
            )
    return batches


def _checked_outputs(workload: Workload) -> set[str]:
    outputs: set[str] = set()
    for check in workload.checks:
        if isinstance(check, TopKRoutingCheck):
            outputs.update((check.ids_output, check.weights_output))
        else:
            outputs.add(check.output)
    return outputs


def _validate_workload(
    definition: Definition,
    workload: Workload,
    family: DiagnosticFamily,
) -> None:
    if set(workload.inputs) != set(definition.inputs):
        raise ValueError(f"{family} workload input inventory mismatch")
    if _checked_outputs(workload) != set(definition.outputs):
        raise ValueError(f"{family} workload output inventory mismatch")
    if set(workload.axes) != set(definition.axes):
        raise ValueError(f"{family} workload axis inventory mismatch")


def _validate_problem(
    problem_dir: Path,
    family: DiagnosticFamily,
    family_cases: list[DiagnosticCorpusCase],
) -> None:
    definition = Definition.model_validate_json(
        (problem_dir / "definition.json").read_text(encoding="utf-8")
    )
    solution = Solution.model_validate_json(
        (problem_dir / "solution.json").read_text(encoding="utf-8")
    )
    workloads = load_jsonl_file(Workload, problem_dir / "workload.jsonl")
    if len(workloads) != UNIVERSE_CASES_PER_FAMILY:
        raise ValueError(f"{family} must contain 60 workloads")
    for workload in workloads:
        _validate_workload(definition, workload, family)
    by_uuid = {workload.uuid: workload for workload in workloads}
    if len(by_uuid) != len(workloads):
        raise ValueError(f"{family} repeats workload UUIDs")
    if solution.definition != definition.name:
        raise ValueError(f"{family} solution targets another definition")
    for case in family_cases:
        workload = by_uuid.get(case.workload_uuid)
        if workload is None or workload.axes != case.axes:
            raise ValueError(f"{case.case_id} does not match authored workload")


def validate_problems(
    root: Path,
    cases: list[DiagnosticCorpusCase],
) -> None:
    """Validate typed problem artifacts and their design correspondence."""
    problems = root / "problems"
    observed = {path.name for path in problems.iterdir() if path.is_dir()}
    if observed != {family.value for family in DiagnosticFamily}:
        raise ValueError("diagnostic corpus problem-family inventory mismatch")
    for family in DiagnosticFamily:
        selected = [case for case in cases if case.family is family]
        _validate_problem(problems / family.value, family, selected)


def _load_status_records(path: Path) -> list[DiagnosticCollectionRecord]:
    return [
        DiagnosticCollectionRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _record_matches_case(
    record: DiagnosticCollectionRecord,
    case: DiagnosticCorpusCase,
) -> bool:
    return (
        record.family is case.family
        and record.phase is case.phase
        and record.global_index == case.global_index
        and record.workload_uuid == case.workload_uuid
        and record.axes == case.axes
    )


def validate_status_log(
    status_path: Path,
    case_root: Path,
    cases: list[DiagnosticCorpusCase],
) -> ResumeSummary:
    """Validate resumable status identity against persisted evidence."""
    known = {case.case_id: case for case in cases}
    records = _load_status_records(status_path)
    latest: dict[str, DiagnosticCollectionRecord] = {}
    hashes: dict[str, str] = {}
    for record in records:
        case = known.get(record.case_id)
        if case is None:
            raise ValueError(
                f"status log references unknown case {record.case_id!r}"
            )
        if not _record_matches_case(record, case):
            raise ValueError(f"status identity mismatch for {record.case_id}")
        latest[record.case_id] = record
    for case_id, record in latest.items():
        if record.status not in TERMINAL_STATUSES:
            continue
        case = known[case_id]
        evidence = (
            case_root
            / case.phase.value
            / case.family.value
            / case_id
            / "trace.jsonl.performance-evidence.json"
        )
        if not evidence.is_file():
            raise ValueError(f"terminal status lacks evidence: {case_id}")
        hashes[case_id] = sha256_file(evidence)
    return ResumeSummary(
        records=len(records),
        resolved=len(hashes),
        pending=len(cases) - len(hashes),
        evidence_sha256=dict(sorted(hashes.items())),
    )


def preflight(
    corpus_root: Path,
    *,
    status_log: Path | None = None,
    case_root: Path | None = None,
) -> PreflightSummary:
    """Run every CPU-only corpus and optional resume check."""
    design_path = corpus_root / "design.json"
    design = DiagnosticCorpusDesign.model_validate_json(
        design_path.read_text(encoding="utf-8")
    )
    validate_problems(corpus_root, design.cases)
    resume: ResumeSummary | None = None
    if status_log is not None:
        if case_root is None:
            raise ValueError("--case-root is required with --status-log")
        resume = validate_status_log(status_log, case_root, design.cases)
    return PreflightSummary(
        design_sha256=sha256_file(design_path),
        cases=len(design.cases),
        families=len(DiagnosticFamily),
        batches=build_batches(design.cases),
        resume=resume,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--status-log", type=Path)
    parser.add_argument("--case-root", type=Path)
    return parser.parse_args()


def main() -> int:
    """Run the preflight and print its machine-readable result."""
    arguments = _parse_args()
    result = preflight(
        arguments.corpus_root,
        status_log=arguments.status_log,
        case_root=arguments.case_root,
    )
    print(result.model_dump_json(indent=2))
    return 0
