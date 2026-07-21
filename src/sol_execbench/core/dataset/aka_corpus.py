# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""AKA-derived corpus: manifest validation, materialization, and audit.

Replaces the NVIDIA-parquet corpus engine. Benchmark problems are authored
artifacts derived from AMD AgentKernelArena (AKA) tasks under the methodology in
``docs/internal/aka-sol-task-source-research.md`` §8 and the SOL-ExecBench paper
(arXiv 2603.19173) §3. Each problem is committed under
``problems/RX_9060_XT/<suite>/<name>/`` as ``definition.json`` + ``workload.jsonl``
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.integrity import sha256_file

AKA_REPOSITORY = "https://github.com/AMD-AGI/AgentKernelArena"
AKA_REVISION = "869228138e07e773b61dd7fc1d8cdc0435c7b405"
AKA_LICENSE = "Apache-2.0"
AKA_PROVENANCE_CLASS = "ecosystem_grounded"

FORMAL_ARCHITECTURE = "solar:RX_9060_XT"
FORMAL_GFX_TARGET = "gfx1200"
FORMAL_ARCHITECTURE_SHA256 = (
    "944aa6f9383a565bd4e636b068ee077e7415f38f15517b69cb78b6ea32c9a8dd"
)

# Seed-set bounds (replace the NVIDIA 15-workload / 14-scored + 1-sentinel pin).
SEED_SET_MIN_PROBLEMS = 15
SEED_SET_MAX_PROBLEMS = 25

_AUTHORED_PROBLEM_FILES = ("definition.json", "workload.jsonl")


@dataclass(frozen=True)
class AkaCorpusEntry:
    """One AKA-derived problem selected into the seed set."""

    slot: str
    task_path: str
    problem_name: str
    operation: str
    dtype: str
    pass_kind: str
    fusion_depth: str
    source_family: str
    suite: str
    role: str = "scored"
    workload_uuids: tuple[str, ...] = ()
    aka_config_sha256: str = ""
    aka_source_sha256: str = ""
    aka_runner_sha256: str = ""
    golden: dict[str, Any] = field(default_factory=dict)

    @property
    def relative_problem_dir(self) -> Path:
        return Path(self.suite) / self.problem_name


@dataclass(frozen=True)
class AkaCorpusManifest:
    """Validated AKA-derived corpus manifest."""

    path: Path
    source: dict[str, Any]
    target: dict[str, Any]
    entries: tuple[AkaCorpusEntry, ...]
    materialized_problem_sha256: dict[str, dict[str, str]]
    formal_coverage_requirements: dict[str, Any]
    official_scoring: dict[str, Any]

    @property
    def authored_root(self) -> Path:
        return self.path.parent

    @classmethod
    def load(cls, path: str | Path) -> "AkaCorpusManifest":
        manifest_path = Path(path).resolve()
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        _validate_manifest_header(data)
        entries = tuple(_load_entry(item) for item in data.get("entries") or [])
        coverage = dict(data.get("formal_coverage_requirements") or {})
        _validate_entries(entries, coverage)
        problem_records = data.get("materialized_problems") or []
        _validate_problem_inventory(entries, problem_records)
        materialized_problem_sha256 = {
            str(item["path"]): {
                "definition_sha256": str(item["definition_sha256"]),
                "workload_sha256": str(item["workload_sha256"]),
            }
            for item in problem_records
        }
        _validate_authored_problems(
            manifest_path.parent, entries, materialized_problem_sha256
        )
        return cls(
            manifest_path,
            dict(data.get("source") or {}),
            dict(data.get("target") or {}),
            entries,
            materialized_problem_sha256,
            coverage,
            dict(data.get("official_scoring") or {}),
        )

    def materialize(self, output_root: str | Path) -> Path:
        """Mirror authored problems into an untracked local tree atomically."""
        output = Path(output_root).resolve()
        if output.exists():
            raise FileExistsError(f"materialization output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
        try:
            records = _mirror_entries(self.authored_root, staging, self.entries)
            _write_materialization_manifest(self, staging, records)
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
        if record.get("aka_manifest_sha256") != sha256_file(self.path):
            raise ValueError("corpus manifest identity changed")
        if record.get("source", {}).get("revision") != self.source.get("revision"):
            raise ValueError("materialization record pins a different AKA revision")
        problems = record.get("problems") or []
        _validate_problem_inventory(self.entries, problems)
        for item in problems:
            _audit_problem(
                output,
                item,
                self.materialized_problem_sha256[str(item["path"])],
            )
        _audit_selected_workloads(self.entries, output)
        return {
            "status": "valid",
            "problems": len(problems),
            "scored": sum(entry.role == "scored" for entry in self.entries),
            "compatibility_sentinels": sum(
                entry.role == "compatibility_sentinel" for entry in self.entries
            ),
            "source_repository": self.source.get("repository"),
            "source_revision": self.source.get("revision"),
        }

    def audit_aka_provenance(self, aka_root: str | Path) -> dict[str, Any]:
        """Bind the generated problems to the pinned AKA commit.

        Verifies the local AKA clone is checked out at the pinned revision and
        that every corpus entry's per-task file checksums (config / source /
        runner) match the files at that commit. This makes the
        manifest-recorded binding between the generated problems and the
        original AKA source tree verifiable end-to-end.
        """
        root = Path(aka_root).resolve()
        head_file = root / ".aka-head"
        if not head_file.is_file():
            raise ValueError(
                "AKA clone missing .aka-head; run scripts/fetch_aka_source.sh first"
            )
        head = head_file.read_text().strip()
        if head != self.source.get("revision"):
            raise ValueError(
                f"AKA clone at {head[:12]} but corpus pins "
                f"{str(self.source.get('revision'))[:12]}"
            )
        checked = 0
        for entry in self.entries:
            for attr, relative in (
                ("aka_config_sha256", "config.yaml"),
                ("aka_runner_sha256", "eval_tools/correctness_check.py"),
            ):
                expected = getattr(entry, attr)
                if not expected:
                    continue
                path = root / entry.task_path / relative
                if not path.is_file() or sha256_file(path) != expected:
                    raise ValueError(
                        f"AKA {attr} mismatch for {entry.task_path} at {head[:12]}"
                    )
                checked += 1
            source_sha = entry.aka_source_sha256
            if source_sha:
                func_dir = root / entry.task_path / "pytorch_code_functional"
                files = sorted(func_dir.glob("*.py")) if func_dir.is_dir() else []
                if len(files) != 1 or sha256_file(files[0]) != source_sha:
                    raise ValueError(
                        f"AKA source mismatch for {entry.task_path} at {head[:12]}"
                    )
                checked += 1
        return {
            "status": "bound",
            "revision": head,
            "entries_verified": len(self.entries),
            "checksums_verified": checked,
        }


def _validate_manifest_header(data: Mapping[str, Any]) -> None:
    if int(data.get("schema_version", 0)) != 3:
        raise ValueError("AKA corpus manifest must use schema_version 3")
    source = data.get("source") or {}
    if source.get("repository") != AKA_REPOSITORY:
        raise ValueError("corpus must derive from the AMD AgentKernelArena repository")
    if source.get("revision") != AKA_REVISION:
        raise ValueError("corpus AKA revision is not the pinned revision")
    if source.get("license") != AKA_LICENSE:
        raise ValueError("corpus source license must be Apache-2.0")
    if source.get("provenance_class") != AKA_PROVENANCE_CLASS:
        raise ValueError("corpus provenance class must be ecosystem_grounded")
    target = data.get("target") or {}
    if target.get("architecture_profile") != FORMAL_ARCHITECTURE:
        raise ValueError("formal corpus must reference the packaged RX 9060 XT profile")
    if target.get("formal_gfx_target") != FORMAL_GFX_TARGET:
        raise ValueError("formal corpus must target gfx1200")
    if target.get("architecture_profile_sha256") != FORMAL_ARCHITECTURE_SHA256:
        raise ValueError("formal architecture profile identity changed")
    scoring = data.get("official_scoring") or {}
    if scoring.get("status") not in {"available", "unavailable"}:
        raise ValueError("official scoring availability must be explicit")


def _load_entry(data: Mapping[str, Any]) -> AkaCorpusEntry:
    return AkaCorpusEntry(
        slot=str(data["slot"]),
        task_path=str(data["task_path"]),
        problem_name=str(data["problem_name"]),
        operation=str(data["operation"]),
        dtype=str(data["dtype"]),
        pass_kind=str(data["pass_kind"]),
        fusion_depth=str(data["fusion_depth"]),
        source_family=str(data["source_family"]),
        suite=str(data["suite"]),
        role=str(data.get("role", "scored")),
        workload_uuids=tuple(str(u) for u in (data.get("workload_uuids") or ())),
        aka_config_sha256=str(data.get("aka_config_sha256", "")),
        aka_source_sha256=str(data.get("aka_source_sha256", "")),
        aka_runner_sha256=str(data.get("aka_runner_sha256", "")),
        golden=dict(data.get("golden") or {}),
    )


def _validate_entries(
    entries: tuple[AkaCorpusEntry, ...], coverage: Mapping[str, Any]
) -> None:
    if not (SEED_SET_MIN_PROBLEMS <= len(entries) <= SEED_SET_MAX_PROBLEMS):
        raise ValueError(
            f"AKA seed set must contain {SEED_SET_MIN_PROBLEMS}.."
            f"{SEED_SET_MAX_PROBLEMS} problems, got {len(entries)}"
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
    sentinels = [entry for entry in entries if entry.role == "compatibility_sentinel"]
    if sentinels and not all(entry.dtype.startswith("fp8") for entry in sentinels):
        raise ValueError("compatibility sentinels must be FP8 AKA tasks")
    if sum(entry.role == "scored" for entry in entries) == 0:
        raise ValueError("AKA corpus must contain at least one scored entry")
    _validate_coverage_truth(entries, coverage)


def _validate_coverage_truth(
    entries: tuple[AkaCorpusEntry, ...], coverage: Mapping[str, Any]
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
            value = getattr(entry, entry_attr)
            actual[value] = actual.get(value, 0) + 1
        if actual != {k: int(v) for k, v in declared.items()}:
            raise ValueError(
                f"coverage axis {axis_name!r} does not match entries: "
                f"declared={declared}, actual={actual}"
            )
    for combo in coverage.get("combinations") or []:
        min_count = int(combo.get("min_count", 0))
        if min_count <= 0:
            continue
        matched = sum(1 for entry in entries if _entry_matches_combo(entry, combo))
        if matched < min_count:
            raise ValueError(
                f"coverage combination unmet ({matched}<{min_count}): {combo}"
            )


def _entry_matches_combo(entry: AkaCorpusEntry, combo: Mapping[str, Any]) -> bool:
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
        str(mapping.get(k)) == str(v) for k, v in combo.items() if k != "min_count"
    )


def _mirror_entries(
    authored_root: Path, staging: Path, entries: tuple[AkaCorpusEntry, ...]
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for entry in entries:
        src = authored_root / entry.relative_problem_dir
        dst = staging / entry.relative_problem_dir
        dst.mkdir(parents=True, exist_ok=True)
        for name in _AUTHORED_PROBLEM_FILES:
            shutil.copy2(src / name, dst / name)
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
                "workload_sha256": sha256_file(workload_path),
            }
        )
    return records


def _write_materialization_manifest(
    manifest: AkaCorpusManifest, staging: Path, records: list[dict[str, str]]
) -> None:
    payload = {
        "schema_version": 1,
        "source": {
            "repository": manifest.source.get("repository"),
            "revision": manifest.source.get("revision"),
            "license": manifest.source.get("license"),
            "provenance_class": manifest.source.get("provenance_class"),
        },
        "aka_manifest_sha256": sha256_file(manifest.path),
        "problems": records,
    }
    (staging / "materialization-manifest.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False)
    )


def _validate_problem_inventory(
    entries: tuple[AkaCorpusEntry, ...], problems: Any
) -> None:
    """Require the local record to name every corpus problem exactly once."""
    if not isinstance(problems, list) or any(
        not isinstance(item, Mapping) for item in problems
    ):
        raise ValueError("materialization problem inventory must be a list of objects")
    expected = {entry.relative_problem_dir.as_posix() for entry in entries}
    recorded = [str(item.get("path", "")) for item in problems]
    if len(recorded) != len(set(recorded)):
        raise ValueError("materialization problem inventory contains duplicate paths")
    if set(recorded) != expected:
        missing = sorted(expected - set(recorded))
        unexpected = sorted(set(recorded) - expected)
        raise ValueError(
            "materialization problem inventory mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _validate_authored_problems(
    authored_root: Path,
    entries: tuple[AkaCorpusEntry, ...],
    sha256_record: Mapping[str, Mapping[str, str]],
) -> None:
    """Authored problems must exist, match manifest checksums, and validate."""
    for entry in entries:
        root = authored_root / entry.relative_problem_dir
        definition_path = root / "definition.json"
        workload_path = root / "workload.jsonl"
        if not definition_path.is_file() or not workload_path.is_file():
            raise ValueError(
                f"authored AKA problem missing on disk: {entry.relative_problem_dir}"
            )
        relative = entry.relative_problem_dir.as_posix()
        expected = sha256_record.get(relative, {})
        if expected.get("definition_sha256") and (
            sha256_file(definition_path) != expected["definition_sha256"]
        ):
            raise ValueError(
                f"authored definition checksum mismatch: {entry.relative_problem_dir}"
            )
        if expected.get("workload_sha256") and (
            sha256_file(workload_path) != expected["workload_sha256"]
        ):
            raise ValueError(
                f"authored workload checksum mismatch: {entry.relative_problem_dir}"
            )
        Definition.model_validate_json(definition_path.read_text())
        for line in workload_path.read_text().splitlines():
            if line.strip():
                Workload.model_validate_json(line)
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


def _audit_problem(
    output: Path, item: Mapping[str, Any], expected: Mapping[str, str]
) -> None:
    root = output / str(item["path"])
    definition_path = root / "definition.json"
    workload_path = root / "workload.jsonl"
    if item.get("definition_sha256") != expected["definition_sha256"]:
        raise ValueError(f"definition record mismatch: {item['path']}")
    if item.get("workload_sha256") != expected["workload_sha256"]:
        raise ValueError(f"workload record mismatch: {item['path']}")
    if sha256_file(definition_path) != expected["definition_sha256"]:
        raise ValueError(f"definition identity mismatch: {item['path']}")
    if sha256_file(workload_path) != expected["workload_sha256"]:
        raise ValueError(f"workload identity mismatch: {item['path']}")
    Definition.model_validate_json(definition_path.read_text())
    for line in workload_path.read_text().splitlines():
        if line.strip():
            Workload.model_validate_json(line)


def _audit_selected_workloads(
    entries: tuple[AkaCorpusEntry, ...], output: Path
) -> None:
    for entry in entries:
        path = output / entry.relative_problem_dir / "workload.jsonl"
        workloads = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        for uuid in entry.workload_uuids:
            matches = [item for item in workloads if item.get("uuid") == uuid]
            if len(matches) != 1:
                raise ValueError(f"selected workload missing: {uuid}")


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
    "AkaCorpusEntry",
    "AkaCorpusManifest",
]
