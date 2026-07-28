# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""AKA-derived corpus: manifest validation, materialization, and audit.

Replaces the NVIDIA-parquet corpus engine. Benchmark problems are authored
artifacts derived from AMD AgentKernelArena (AKA) tasks under the methodology in
``docs/internal/aka-sol-task-source-research.md`` §8 and the SOL-ExecBench paper
(arXiv 2603.19173) §3. Each problem is committed under
``problems/AMD_AKA/<suite>/<name>/`` as ``definition.json`` + ``workload.jsonl``
(+ optional ``reference.py``), in the backward-compatible schema consumed by
``sol_execbench.cli.evaluation.problem_io`` and the rest of the harness.

The manifest pins the AKA source revision and records per-problem checksums and
provenance. ``materialize`` mirrors the authored problems into a local tree and
``audit`` verifies them. Reference implementations are AKA's own correctness
oracles (``module_fn`` for torch2hip/torch2flydsl; the test-file torch oracle for
instruction2triton), so correctness is inherited from AKA's per-task contract;
``scripts/aka_equivalence_check.py`` re-validates each against the AKA original.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from sol_execbench.core.bench.reference_protocol import (
    MAX_REFERENCE_TENSOR_STORAGE_BYTES,
)
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.definition_models import DType
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.dataset.aka_compatibility import (
    DEFAULT_PROBE_TIMEOUT_SECONDS,
    AKACorpusSelection,
    AKAExecutionTarget,
    AKAMaterializationTarget,
    Probe,
    load_execution_targets,
    select_corpus_for_target,
    static_reference_storage,
)
from sol_execbench.core.dataset.aka_contract import (
    AKA_MANIFEST_SCHEMA_VERSION,
    AKA_OFFICIAL_BASELINE_ID,
    AKA_REQUIRED_RELEASE_EVIDENCE,
    AKAArtifactRole,
    AKACorpusRole,
    AKAFusionDepth,
    AKAOfficialScoringStatus,
    AKAOperation,
    AKAPassKind,
    AKAReleasePolicy,
    AKASourceFamily,
    AKASuite,
)
from sol_execbench.core.dataset.aka_tolerance import (
    validate_calibration_binding,
)
from sol_execbench.core.integrity import (
    sha256_file,
    validate_relative_artifact_path,
    validate_sha256,
)
from sol_execbench.core.platform.runtime import derive_cache_clear_policy

AKA_REPOSITORY = "https://github.com/AMD-AGI/AgentKernelArena"
AKA_REVISION = "869228138e07e773b61dd7fc1d8cdc0435c7b405"
AKA_LICENSE = "Apache-2.0"
AKA_PROVENANCE_CLASS = "ecosystem_grounded"

FORMAL_ARCHITECTURE = "solar:RX_9060_XT"
FORMAL_GFX_TARGET = "gfx1200"
FORMAL_ARCHITECTURE_SHA256 = (
    "abe2510a07805017c74f214c49990db365ccf59f1622ac1ba58434bf1351e71b"
)

# Corpus-size bounds. The initial seed landed at 15 problems; the friendliness
# expansion (docs/internal/aka-expansion-friendliness.md) grows it to 37 across
# the three handling categories. The upper bound leaves room for growth
# while still bounding the manifest validator's check.
SEED_SET_MIN_PROBLEMS = 15
SEED_SET_MAX_PROBLEMS = 48


@dataclass(frozen=True)
class AKAArtifactBinding:
    """One content-addressed file within an AKA task."""

    role: AKAArtifactRole
    path: str
    sha256: str


@dataclass(frozen=True)
class AKACorpusEntry:
    """One AKA-derived problem selected into the seed set."""

    slot: str
    task_path: str
    problem_name: str
    operation: AKAOperation
    dtype: DType
    pass_kind: AKAPassKind
    fusion_depth: AKAFusionDepth
    source_family: AKASourceFamily
    suite: AKASuite
    role: AKACorpusRole = AKACorpusRole.SCORED
    exclusion_reason_code: str = ""
    workload_uuids: tuple[str, ...] = ()
    aka_artifacts: tuple[AKAArtifactBinding, ...] = ()
    golden: dict[str, Any] = field(default_factory=dict)

    @property
    def relative_problem_dir(self) -> Path:
        """Return the corpus-relative problem directory."""
        return Path(self.suite) / self.problem_name


@dataclass(frozen=True)
class AKACorpusManifest:
    """Validated AKA-derived corpus manifest."""

    path: Path
    source: dict[str, Any]
    execution_targets: dict[str, AKAExecutionTarget]
    formal_analysis: dict[str, Any]
    entries: tuple[AKACorpusEntry, ...]
    materialized_problem_sha256: dict[str, dict[str, str]]
    formal_coverage_requirements: dict[str, Any]
    official_scoring: dict[str, Any]
    tolerance_calibration: dict[str, Any] = field(default_factory=dict)

    @property
    def authored_root(self) -> Path:
        """Return the root directory containing authored problems."""
        return self.path.parent

    @classmethod
    def load(cls, path: str | Path) -> AKACorpusManifest:
        """Load and validate a corpus manifest."""
        manifest_path = Path(path).resolve()
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        _validate_manifest_header(data)
        entries = tuple(_load_entry(item) for item in data.get("entries") or [])
        coverage = dict(data.get("formal_coverage_requirements") or {})
        _validate_entries(entries, coverage)
        problem_records = data.get("materialized_problems") or []
        _validate_canonical_problem_inventory(entries, problem_records)
        materialized_problem_sha256 = {
            str(item["path"]): {
                "definition_sha256": str(item["definition_sha256"]),
                "workload_sha256": str(item["workload_sha256"]),
            }
            for item in problem_records
        }
        _validate_authored_problems(
            manifest_path.parent,
            entries,
            materialized_problem_sha256,
        )
        calibration = dict(data.get("tolerance_calibration") or {})
        validate_calibration_binding(
            authored_root=manifest_path.parent,
            binding=calibration,
            entries=entries,
            source_revision=str(
                (data.get("source") or {}).get("revision") or "",
            ),
            formal_gfx_target=str(
                (data.get("formal_analysis") or {}).get("formal_gfx_target")
                or "",
            ),
        )
        return cls(
            manifest_path,
            dict(data.get("source") or {}),
            load_execution_targets(data.get("execution_targets") or {}),
            dict(data.get("formal_analysis") or {}),
            entries,
            materialized_problem_sha256,
            coverage,
            dict(data.get("official_scoring") or {}),
            calibration,
        )

    def materialize(
        self,
        output_root: str | Path,
        *,
        target: AKAMaterializationTarget,
        probe_timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
        probe: Probe | None = None,
    ) -> Path:
        """Select executable workloads and mirror them atomically."""
        execution_target = self.execution_targets.get(target.device.gfx_target)
        if execution_target is None:
            raise ValueError(
                f"unsupported AKA execution target: {target.device.gfx_target}",
            )
        selection = select_corpus_for_target(
            authored_root=self.authored_root,
            entries=self.entries,
            execution_target=execution_target,
            target=target,
            probe_timeout_seconds=probe_timeout_seconds,
            probe=probe,
        )
        output = Path(output_root).resolve()
        if output.exists():
            raise FileExistsError(
                f"materialization output already exists: {output}",
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent),
        )
        try:
            records = _mirror_selection(self.authored_root, staging, selection)
            _write_materialization_manifest(
                self,
                staging,
                records,
                selection,
                target,
                probe_timeout_seconds,
            )
            staging.replace(output)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return output

    def audit(self, output_root: str | Path) -> dict[str, Any]:
        """Validate local files against manifest checksums and the schema."""
        output = Path(output_root).resolve()
        record_path = output / "materialization-manifest.yaml"
        record = yaml.safe_load(record_path.read_text(encoding="utf-8")) or {}
        if int(record.get("schema_version", 0)) != 2:
            raise ValueError(
                "AKA materialization record must use schema_version 2",
            )
        if record.get("aka_manifest_sha256") != sha256_file(self.path):
            raise ValueError("corpus manifest identity changed")
        if record.get("source", {}).get("revision") != self.source.get(
            "revision",
        ):
            raise ValueError(
                "materialization record pins a different AKA revision",
            )
        target = _audit_materialization_target(record.get("target"), self)
        decisions = _audit_decision_partition(
            self.authored_root,
            self.entries,
            record.get("workload_decisions"),
        )
        problems = record.get("problems") or []
        _audit_selected_inventory(self.entries, problems, decisions)
        for item in problems:
            _audit_materialized_problem(
                self.authored_root,
                output,
                item,
                self.materialized_problem_sha256[str(item["path"])],
            )
        _audit_no_unrecorded_problem_files(output, problems)
        selected_paths = {str(item["path"]) for item in problems}
        selected_entries = tuple(
            entry
            for entry in self.entries
            if entry.relative_problem_dir.as_posix() in selected_paths
        )
        workload_count = sum(
            len(item.get("workload_uuids") or ()) for item in problems
        )
        coverage = _coverage_report(self, selected_entries, workload_count)
        if record.get("coverage") != coverage:
            raise ValueError("materialized coverage report is inconsistent")
        return {
            "status": "valid",
            "problems": len(problems),
            "workloads": workload_count,
            "excluded_workloads": sum(
                not bool(item.get("included")) for item in decisions
            ),
            "scored": sum(
                entry.role == AKACorpusRole.SCORED for entry in selected_entries
            ),
            "compatibility_sentinels": sum(
                entry.role == AKACorpusRole.COMPATIBILITY_SENTINEL
                for entry in selected_entries
            ),
            "target_incompatible": sum(
                entry.role == AKACorpusRole.TARGET_INCOMPATIBLE
                for entry in self.entries
            ),
            "gfx_target": target,
            "coverage": coverage,
            "source_repository": self.source.get("repository"),
            "source_revision": self.source.get("revision"),
        }

    def audit_aka_provenance(self, aka_root: str | Path) -> dict[str, Any]:
        """Bind the generated problems to the pinned AKA commit.

        Verifies the local AKA clone is checked out at the pinned revision and
        that every corpus entry's content-addressed config, semantic reference,
        and correctness runner match the files at that commit. This makes the
        manifest-recorded binding between the generated problems and the
        original AKA source tree verifiable end-to-end.
        """
        root = Path(aka_root).resolve()
        head_file = root / ".aka-head"
        if not head_file.is_file():
            raise ValueError(
                "AKA clone missing .aka-head; run scripts/fetch_aka_source.sh first",
            )
        head = head_file.read_text().strip()
        if head != self.source.get("revision"):
            raise ValueError(
                f"AKA clone at {head[:12]} but corpus pins "
                f"{str(self.source.get('revision'))[:12]}",
            )
        checked = 0
        for entry in self.entries:
            task_root = root / entry.task_path
            for artifact in entry.aka_artifacts:
                path = task_root / artifact.path
                if not path.is_file() or sha256_file(path) != artifact.sha256:
                    raise ValueError(
                        f"AKA {artifact.role} mismatch for {entry.task_path} "
                        f"at {head[:12]}",
                    )
                checked += 1
        return {
            "status": "bound",
            "revision": head,
            "entries_verified": len(self.entries),
            "checksums_verified": checked,
        }


def _validate_manifest_header(data: Mapping[str, Any]) -> None:
    if int(data.get("schema_version", 0)) != AKA_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"AKA corpus manifest must use schema_version {AKA_MANIFEST_SCHEMA_VERSION}",
        )
    source = data.get("source") or {}
    if source.get("repository") != AKA_REPOSITORY:
        raise ValueError(
            "corpus must derive from the AMD AgentKernelArena repository",
        )
    if source.get("revision") != AKA_REVISION:
        raise ValueError("corpus AKA revision is not the pinned revision")
    if source.get("license") != AKA_LICENSE:
        raise ValueError("corpus source license must be Apache-2.0")
    if source.get("provenance_class") != AKA_PROVENANCE_CLASS:
        raise ValueError("corpus provenance class must be ecosystem_grounded")
    load_execution_targets(data.get("execution_targets") or {})
    formal = data.get("formal_analysis") or {}
    if formal.get("architecture_profile") != FORMAL_ARCHITECTURE:
        raise ValueError(
            "formal corpus must reference the packaged RX 9060 XT profile",
        )
    if formal.get("formal_gfx_target") != FORMAL_GFX_TARGET:
        raise ValueError("formal corpus must target gfx1200")
    if formal.get("architecture_profile_sha256") != FORMAL_ARCHITECTURE_SHA256:
        raise ValueError("formal architecture profile identity changed")
    scoring = data.get("official_scoring") or {}
    try:
        status = AKAOfficialScoringStatus(str(scoring.get("status")))
    except ValueError:
        raise ValueError(
            "official scoring availability must be explicit",
        ) from None
    if status is AKAOfficialScoringStatus.AVAILABLE:
        try:
            policy = AKAReleasePolicy(str(scoring.get("release_policy")))
        except ValueError:
            raise ValueError(
                "official scoring release policy is invalid",
            ) from None
        if policy is not AKAReleasePolicy.CONTENT_ADDRESSED_PUBLISHER_V1:
            raise ValueError("official scoring release policy is unsupported")
        if scoring.get("baseline_id") != AKA_OFFICIAL_BASELINE_ID:
            raise ValueError(
                "official scoring baseline identity is not canonical",
            )
        required = tuple(
            str(item) for item in scoring.get("required_evidence") or ()
        )
        expected = tuple(AKA_REQUIRED_RELEASE_EVIDENCE)
        if required != expected:
            raise ValueError("official scoring evidence denominator is invalid")


def _load_entry(data: Mapping[str, Any]) -> AKACorpusEntry:
    artifacts = tuple(
        AKAArtifactBinding(
            role=AKAArtifactRole(str(item["role"])),
            path=str(item["path"]),
            sha256=str(item["sha256"]),
        )
        for item in (data.get("aka_artifacts") or ())
    )
    return AKACorpusEntry(
        slot=str(data["slot"]),
        task_path=str(data["task_path"]),
        problem_name=str(data["problem_name"]),
        operation=AKAOperation(str(data["operation"])),
        dtype=DType(str(data["dtype"])),
        pass_kind=AKAPassKind(str(data["pass_kind"])),
        fusion_depth=AKAFusionDepth(str(data["fusion_depth"])),
        source_family=AKASourceFamily(str(data["source_family"])),
        suite=AKASuite(str(data["suite"])),
        role=AKACorpusRole(str(data.get("role", AKACorpusRole.SCORED))),
        exclusion_reason_code=str(data.get("exclusion_reason_code", "")),
        workload_uuids=tuple(
            str(u) for u in (data.get("workload_uuids") or ())
        ),
        aka_artifacts=artifacts,
        golden=dict(data.get("golden") or {}),
    )


def _validate_entries(
    entries: tuple[AKACorpusEntry, ...],
    coverage: Mapping[str, Any],
) -> None:
    if not (SEED_SET_MIN_PROBLEMS <= len(entries) <= SEED_SET_MAX_PROBLEMS):
        raise ValueError(
            f"AKA seed set must contain {SEED_SET_MIN_PROBLEMS}.."
            f"{SEED_SET_MAX_PROBLEMS} problems, got {len(entries)}",
        )
    names = [entry.problem_name for entry in entries]
    if len(set(names)) != len(names):
        raise ValueError("AKA corpus problem names must be unique")
    if len({entry.slot for entry in entries}) != len(entries):
        raise ValueError("AKA corpus slots must be unique")
    if len({entry.task_path for entry in entries}) != len(entries):
        raise ValueError("AKA corpus task paths must be unique")
    if not all(entry.task_path.startswith("tasks/") for entry in entries):
        raise ValueError("AKA corpus entries must reference tasks/ paths")
    for entry in entries:
        validate_relative_artifact_path(entry.task_path, "AKA task path")
        _validate_artifact_bindings(entry)
    unknown_roles = sorted(
        {str(entry.role) for entry in entries} - set(AKACorpusRole),
    )
    if unknown_roles:
        raise ValueError(
            f"AKA corpus entries use unknown roles: {unknown_roles}",
        )
    for entry in entries:
        if entry.role == AKACorpusRole.TARGET_INCOMPATIBLE:
            if not entry.exclusion_reason_code:
                raise ValueError(
                    "target-incompatible corpus entries require an exclusion reason",
                )
        elif entry.exclusion_reason_code:
            raise ValueError(
                "only target-incompatible corpus entries may declare an exclusion reason",
            )
    sentinels = [
        entry
        for entry in entries
        if entry.role == AKACorpusRole.COMPATIBILITY_SENTINEL
    ]
    # The DType enum names OCP FP8 as "float8_e4m3fn" / "float8_e5m2", so accept
    # Accept either the "fp8" or "float8" prefix for the FP8 sentinel.
    if sentinels and not all(
        entry.dtype.startswith(("fp8", "float8")) for entry in sentinels
    ):
        raise ValueError("compatibility sentinels must be FP8 AKA tasks")
    if sum(entry.role == AKACorpusRole.SCORED for entry in entries) == 0:
        raise ValueError("AKA corpus must contain at least one scored entry")
    _validate_coverage_truth(entries, coverage)


def _validate_artifact_bindings(entry: AKACorpusEntry) -> None:
    roles = [artifact.role for artifact in entry.aka_artifacts]
    required_roles = set(AKAArtifactRole)
    if set(roles) != required_roles or len(roles) != len(required_roles):
        raise ValueError(
            f"AKA entry {entry.problem_name} must bind exactly one artifact for "
            f"each role: {sorted(required_roles)}",
        )
    for artifact in entry.aka_artifacts:
        validate_relative_artifact_path(artifact.path, "AKA artifact path")
        validate_sha256(
            artifact.sha256,
            f"AKA artifact checksum for {entry.problem_name}/{artifact.role}",
        )


def _validate_coverage_truth(
    entries: tuple[AKACorpusEntry, ...],
    coverage: Mapping[str, Any],
) -> None:
    """The declared coverage axes must truthfully aggregate the entries."""
    axes = coverage.get("axes") or {}
    tag_keys = {
        "operation": "operation",
        "dtype": "dtype",
        "pass_kind": "pass_kind",
        "fusion_depth": "fusion_depth",
        "source_family": "source_family",
        "suite": "suite",
    }
    for axis_name, entry_attr in tag_keys.items():
        declared = axes.get(axis_name) or {}
        if not declared:
            continue
        actual: dict[str, int] = {}
        for entry in entries:
            value = _manifest_value(getattr(entry, entry_attr))
            actual[value] = actual.get(value, 0) + 1
        if actual != {k: int(v) for k, v in declared.items()}:
            raise ValueError(
                f"coverage axis {axis_name!r} does not match entries: "
                f"declared={declared}, actual={actual}",
            )
    for combo in coverage.get("combinations") or []:
        min_count = int(combo.get("min_count", 0))
        if min_count <= 0:
            continue
        matched = sum(
            1 for entry in entries if _entry_matches_combo(entry, combo)
        )
        if matched < min_count:
            raise ValueError(
                f"coverage combination unmet ({matched}<{min_count}): {combo}",
            )


def _entry_matches_combo(
    entry: AKACorpusEntry,
    combo: Mapping[str, Any],
) -> bool:
    mapping = {
        "operation": entry.operation,
        "dtype": entry.dtype,
        "pass_kind": entry.pass_kind,
        "pass": entry.pass_kind,
        "fusion_depth": entry.fusion_depth,
        "source_family": entry.source_family,
        "suite": entry.suite,
    }
    return all(
        str(mapping.get(k)) == str(v)
        for k, v in combo.items()
        if k != "min_count"
    )


def _manifest_value(value: object) -> str:
    return str(value)


def _canonical_workload_lines(path: Path) -> tuple[list[str], dict[str, str]]:
    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines() if line
    ]
    by_uuid: dict[str, str] = {}
    for line in lines:
        uuid = str(json.loads(line).get("uuid") or "")
        if not uuid or uuid in by_uuid:
            raise ValueError(
                f"canonical workload UUID is missing or duplicated: {path}",
            )
        by_uuid[uuid] = line
    return lines, by_uuid


def _mirror_selection(
    authored_root: Path,
    staging: Path,
    selection: AKACorpusSelection,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for problem in selection.problems:
        entry = problem.entry
        src = authored_root / entry.relative_problem_dir
        dst = staging / entry.relative_problem_dir
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / "definition.json", dst / "definition.json")
        _, workload_lines = _canonical_workload_lines(src / "workload.jsonl")
        selected_uuids = [workload.uuid for workload in problem.workloads]
        selected_payload = (
            "\n".join(workload_lines[uuid] for uuid in selected_uuids) + "\n"
        )
        (dst / "workload.jsonl").write_text(selected_payload, encoding="utf-8")
        reference_py = src / "reference.py"
        if reference_py.is_file():
            shutil.copy2(reference_py, dst / "reference.py")
        definition_path = dst / "definition.json"
        workload_path = dst / "workload.jsonl"
        records.append(
            {
                "path": entry.relative_problem_dir.as_posix(),
                "task_path": entry.task_path,
                "definition_sha256": sha256_file(definition_path),
                "source_workload_sha256": sha256_file(src / "workload.jsonl"),
                "workload_sha256": sha256_file(workload_path),
                "workload_uuids": selected_uuids,
            },
        )
    return records


def _write_materialization_manifest(
    manifest: AKACorpusManifest,
    staging: Path,
    records: list[dict[str, Any]],
    selection: AKACorpusSelection,
    target: AKAMaterializationTarget,
    probe_timeout_seconds: float,
) -> None:
    payload = {
        "schema_version": 2,
        "source": {
            "repository": manifest.source.get("repository"),
            "revision": manifest.source.get("revision"),
            "license": manifest.source.get("license"),
            "provenance_class": manifest.source.get("provenance_class"),
        },
        "aka_manifest_sha256": sha256_file(manifest.path),
        "target": target.to_dict(),
        "selection_policy": {
            "static_filter": "manifest_supported_tensor_dtypes",
            "live_probe": "trusted_reference_and_harness_minimum",
            "probe_timeout_seconds": probe_timeout_seconds,
            "unknown_targets": "fail_closed",
        },
        "problems": records,
        "workload_decisions": [
            decision.to_dict() for decision in selection.decisions
        ],
        "coverage": _selection_coverage(manifest, selection),
    }
    (staging / "materialization-manifest.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
    )


def _validate_canonical_problem_inventory(
    entries: tuple[AKACorpusEntry, ...],
    problems: Any,
) -> None:
    """Require the local record to name every corpus problem exactly once."""
    if not isinstance(problems, list) or any(
        not isinstance(item, Mapping) for item in problems
    ):
        raise ValueError(
            "materialization problem inventory must be a list of objects",
        )
    expected = {entry.relative_problem_dir.as_posix() for entry in entries}
    recorded = [str(item.get("path", "")) for item in problems]
    if len(recorded) != len(set(recorded)):
        raise ValueError(
            "materialization problem inventory contains duplicate paths",
        )
    if set(recorded) != expected:
        missing = sorted(expected - set(recorded))
        unexpected = sorted(set(recorded) - expected)
        raise ValueError(
            "materialization problem inventory mismatch: "
            f"missing={missing}, unexpected={unexpected}",
        )


def _validate_authored_problems(
    authored_root: Path,
    entries: tuple[AKACorpusEntry, ...],
    sha256_record: Mapping[str, Mapping[str, str]],
) -> None:
    """Authored problems must exist, match manifest checksums, and validate."""
    for entry in entries:
        root = authored_root / entry.relative_problem_dir
        definition_path = root / "definition.json"
        workload_path = root / "workload.jsonl"
        if not definition_path.is_file() or not workload_path.is_file():
            raise ValueError(
                f"authored AKA problem missing on disk: {entry.relative_problem_dir}",
            )
        relative = entry.relative_problem_dir.as_posix()
        expected = sha256_record.get(relative, {})
        if expected.get("definition_sha256") and (
            sha256_file(definition_path) != expected["definition_sha256"]
        ):
            raise ValueError(
                f"authored definition checksum mismatch: {entry.relative_problem_dir}",
            )
        if expected.get("workload_sha256") and (
            sha256_file(workload_path) != expected["workload_sha256"]
        ):
            raise ValueError(
                f"authored workload checksum mismatch: {entry.relative_problem_dir}",
            )
        definition = Definition.model_validate_json(definition_path.read_text())
        workloads = [
            Workload.model_validate_json(line)
            for line in workload_path.read_text().splitlines()
            if line.strip()
        ]
        payload_incompatibility = tuple(
            static_reference_storage(definition, workload).reference_case_bytes
            > MAX_REFERENCE_TENSOR_STORAGE_BYTES
            for workload in workloads
        )
        if any(payload_incompatibility) and entry.role == AKACorpusRole.SCORED:
            raise ValueError(
                "scored AKA problem exceeds the static trusted-reference IPC "
                f"limit: {entry.relative_problem_dir}",
            )
        if (
            entry.role == AKACorpusRole.TARGET_INCOMPATIBLE
            and entry.exclusion_reason_code == "reference_ipc_payload_limit"
            and (
                not payload_incompatibility or not all(payload_incompatibility)
            )
        ):
            raise ValueError(
                "every workload in a target-incompatible AKA problem must exceed "
                "the static trusted-reference IPC limit: "
                f"{entry.relative_problem_dir}",
            )
        for uuid in entry.workload_uuids:
            _verify_workload_uuid_present(workload_path, uuid)


def _verify_workload_uuid_present(workload_path: Path, uuid: str) -> None:
    uuids = {
        str(json.loads(line).get("uuid"))
        for line in workload_path.read_text().splitlines()
        if line.strip()
    }
    if uuid not in uuids:
        raise ValueError(f"selected workload uuid missing on disk: {uuid}")


def _selection_coverage(
    manifest: AKACorpusManifest,
    selection: AKACorpusSelection,
) -> dict[str, Any]:
    selected_entries = tuple(problem.entry for problem in selection.problems)
    return _coverage_report(
        manifest,
        selected_entries,
        sum(len(problem.workloads) for problem in selection.problems),
    )


def _coverage_report(
    manifest: AKACorpusManifest,
    selected_entries: tuple[AKACorpusEntry, ...],
    workload_count: int,
) -> dict[str, Any]:
    axes: dict[str, dict[str, int]] = {}
    for axis in (
        "operation",
        "dtype",
        "pass_kind",
        "fusion_depth",
        "source_family",
        "suite",
    ):
        counts: dict[str, int] = {}
        for entry in selected_entries:
            value = str(getattr(entry, axis))
            counts[value] = counts.get(value, 0) + 1
        axes[axis] = counts
    gaps: list[dict[str, Any]] = []
    for combo in (
        manifest.formal_coverage_requirements.get("combinations") or []
    ):
        minimum = int(combo.get("min_count", 0))
        matched = sum(
            _entry_matches_combo(entry, combo) for entry in selected_entries
        )
        if matched < minimum:
            gaps.append({"requirement": dict(combo), "matched": matched})
    return {
        "problem_count": len(selected_entries),
        "workload_count": workload_count,
        "axes": axes,
        "formal_coverage_gaps": gaps,
    }


def _canonical_workload_inventory(
    authored_root: Path,
    entries: tuple[AKACorpusEntry, ...],
) -> dict[tuple[str, str], str]:
    inventory: dict[tuple[str, str], str] = {}
    for entry in entries:
        relative = entry.relative_problem_dir.as_posix()
        _, lines = _canonical_workload_lines(
            authored_root / entry.relative_problem_dir / "workload.jsonl",
        )
        for uuid, line in lines.items():
            inventory[(relative, uuid)] = line
    return inventory


def _audit_materialization_target(raw: Any, manifest: AKACorpusManifest) -> str:
    if not isinstance(raw, Mapping):
        raise ValueError("materialization target evidence must be an object")
    gfx_target = str(raw.get("gfx_target") or "")
    if gfx_target not in manifest.execution_targets:
        raise ValueError(f"unsupported materialized gfx target: {gfx_target}")
    cache = raw.get("cache_clear") or {}
    if not isinstance(cache, Mapping):
        raise ValueError("cache-clear evidence must be an object")
    detected = raw.get("l2_cache_bytes")
    detected_l2 = (
        detected if isinstance(detected, int) and detected > 0 else None
    )
    expected = derive_cache_clear_policy(detected_l2)
    if cache.get("detected_l2_bytes") != expected.detected_l2_bytes:
        raise ValueError("cache-clear L2 evidence is inconsistent")
    if cache.get("clear_buffer_bytes") != expected.clear_buffer_bytes:
        raise ValueError("cache-clear buffer does not follow the 2x-L2 policy")
    if cache.get("source") != expected.source:
        raise ValueError("cache-clear policy source is inconsistent")
    if cache.get("fallback_reason") != expected.fallback_reason:
        raise ValueError("cache-clear fallback evidence is inconsistent")
    return gfx_target


def _audit_decision_partition(
    authored_root: Path,
    entries: tuple[AKACorpusEntry, ...],
    raw: Any,
) -> list[Mapping[str, Any]]:
    if not isinstance(raw, list) or any(
        not isinstance(item, Mapping) for item in raw
    ):
        raise ValueError(
            "workload decision inventory must be a list of objects",
        )
    expected = set(_canonical_workload_inventory(authored_root, entries))
    observed = [
        (str(item.get("path") or ""), str(item.get("workload_uuid") or ""))
        for item in raw
    ]
    if len(observed) != len(set(observed)):
        raise ValueError(
            "workload decision inventory contains duplicate workloads",
        )
    if set(observed) != expected:
        raise ValueError(
            "workload decisions do not partition the canonical corpus",
        )
    for item in raw:
        if not isinstance(item.get("included"), bool):
            raise ValueError(
                "workload decisions must record a boolean included value",
            )
        if not item.get("stage") or not item.get("reason_code"):
            raise ValueError("workload decisions require stage and reason_code")
    return raw


def _audit_selected_inventory(
    entries: tuple[AKACorpusEntry, ...],
    problems: Any,
    decisions: list[Mapping[str, Any]],
) -> None:
    if not isinstance(problems, list) or any(
        not isinstance(item, Mapping) for item in problems
    ):
        raise ValueError(
            "materialization problem inventory must be a list of objects",
        )
    canonical_paths = {
        entry.relative_problem_dir.as_posix() for entry in entries
    }
    paths = [str(item.get("path") or "") for item in problems]
    if len(paths) != len(set(paths)) or not set(paths) <= canonical_paths:
        raise ValueError(
            "materialized problem inventory is duplicated or unknown",
        )
    included: dict[str, list[str]] = {}
    for decision in decisions:
        if decision["included"]:
            included.setdefault(str(decision["path"]), []).append(
                str(decision["workload_uuid"]),
            )
    if set(paths) != set(included):
        raise ValueError(
            "materialized problems do not match included decisions",
        )
    for item in problems:
        path = str(item["path"])
        if list(item.get("workload_uuids") or ()) != included[path]:
            raise ValueError(
                f"materialized workload inventory mismatch: {path}",
            )


def _audit_materialized_problem(
    authored_root: Path,
    output: Path,
    item: Mapping[str, Any],
    expected: Mapping[str, str],
) -> None:
    root = output / str(item["path"])
    definition_path = root / "definition.json"
    workload_path = root / "workload.jsonl"
    if item.get("definition_sha256") != expected["definition_sha256"]:
        raise ValueError(f"definition record mismatch: {item['path']}")
    canonical_workload = authored_root / str(item["path"]) / "workload.jsonl"
    if item.get("source_workload_sha256") != expected["workload_sha256"]:
        raise ValueError(f"source workload record mismatch: {item['path']}")
    if sha256_file(canonical_workload) != expected["workload_sha256"]:
        raise ValueError(f"source workload identity mismatch: {item['path']}")
    if sha256_file(definition_path) != expected["definition_sha256"]:
        raise ValueError(f"definition identity mismatch: {item['path']}")
    if sha256_file(workload_path) != item.get("workload_sha256"):
        raise ValueError(f"workload identity mismatch: {item['path']}")
    _, source_lines = _canonical_workload_lines(canonical_workload)
    expected_payload = (
        "\n".join(source_lines[str(uuid)] for uuid in item["workload_uuids"])
        + "\n"
    )
    if workload_path.read_text(encoding="utf-8") != expected_payload:
        raise ValueError(f"selected workload payload mismatch: {item['path']}")
    Definition.model_validate_json(definition_path.read_text())
    for line in workload_path.read_text().splitlines():
        if line.strip():
            Workload.model_validate_json(line)


def _audit_no_unrecorded_problem_files(
    output: Path,
    problems: list[Mapping[str, Any]],
) -> None:
    expected = {str(item["path"]) for item in problems}
    observed = {
        path.parent.relative_to(output).as_posix()
        for path in output.glob("*/*/definition.json")
    }
    workload_paths = {
        path.parent.relative_to(output).as_posix()
        for path in output.glob("*/*/workload.jsonl")
    }
    if observed != expected or workload_paths != expected:
        raise ValueError(
            "materialization contains unrecorded or missing problem files",
        )


__all__ = [
    "AKA_LICENSE",
    "AKA_PROVENANCE_CLASS",
    "AKA_REPOSITORY",
    "AKA_REVISION",
    "FORMAL_ARCHITECTURE",
    "FORMAL_ARCHITECTURE_SHA256",
    "FORMAL_GFX_TARGET",
    "SEED_SET_MAX_PROBLEMS",
    "SEED_SET_MIN_PROBLEMS",
    "AKACorpusEntry",
    "AKACorpusManifest",
]
