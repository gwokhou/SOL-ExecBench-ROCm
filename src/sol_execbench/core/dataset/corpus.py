# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Pinned public corpus selection and local materialization."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.integrity import sha256_file, stable_json_checksum

OFFICIAL_DATASET_ID = "nvidia/SOL-ExecBench"
OFFICIAL_DATASET_REVISION = "63699402f003496acc3af4eb534a5304a8ac1ea9"
FORMAL_ARCHITECTURE = "solar:RX_9060_XT"
FORMAL_GFX_TARGET = "gfx1200"
FORMAL_ARCHITECTURE_SHA256 = (
    "944aa6f9383a565bd4e636b068ee077e7415f38f15517b69cb78b6ea32c9a8dd"
)
OFFICIAL_CORPUS_MANIFEST_SHA256 = (
    "a89d07c1c0eb0e74275d17b1ff9e09058f97826190b2ebbbb53bab4b706248e6"
)


@dataclass(frozen=True)
class CorpusEntry:
    """One exact workload selected from the pinned official dataset."""

    slot: str
    config: str
    problem: str
    workload_uuid: str
    official_row_sha256: str
    official_workload_sha256: str
    operation: str | None = None
    role: str = "scored"

    @property
    def relative_problem_dir(self) -> Path:
        return Path(self.config) / self.problem


@dataclass(frozen=True)
class CorpusManifest:
    """Validated root-level corpus manifest."""

    path: Path
    parquet_sha256: dict[str, str]
    entries: tuple[CorpusEntry, ...]
    materialized_problem_sha256: dict[str, dict[str, str]]
    official_scoring: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "CorpusManifest":
        manifest_path = Path(path).resolve()
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        _validate_manifest_header(data)
        source = data["source"]
        parquet = {
            str(name): str(digest)
            for name, digest in (source.get("parquet_sha256") or {}).items()
        }
        entries = tuple(_load_entry(item) for item in data.get("entries") or [])
        _validate_entries(entries, parquet)
        problem_records = data.get("materialized_problems") or []
        _validate_problem_inventory(entries, problem_records)
        materialized_problem_sha256 = {
            str(item["path"]): {
                "definition_sha256": str(item["definition_sha256"]),
                "workload_sha256": str(item["workload_sha256"]),
            }
            for item in problem_records
        }
        if sha256_file(manifest_path) != OFFICIAL_CORPUS_MANIFEST_SHA256:
            raise ValueError("public corpus manifest identity changed")
        return cls(
            manifest_path,
            parquet,
            entries,
            materialized_problem_sha256,
            dict(data.get("official_scoring") or {}),
        )

    def materialize(self, source_root: str | Path, output_root: str | Path) -> Path:
        """Select exact rows into an untracked local problem tree atomically."""
        source = Path(source_root).resolve()
        output = Path(output_root).resolve()
        if output.exists():
            raise FileExistsError(f"materialization output already exists: {output}")
        _verify_parquet_files(source, self.parquet_sha256)
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
        try:
            records = _materialize_entries(self, source, staging)
            _write_materialization_manifest(self, staging, records)
            staging.replace(output)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return output

    def audit(self, output_root: str | Path) -> dict[str, Any]:
        """Validate local files against both public and materialized identities."""
        output = Path(output_root).resolve()
        record_path = output / "materialization-manifest.yaml"
        record = yaml.safe_load(record_path.read_text(encoding="utf-8")) or {}
        if record.get("corpus_manifest_sha256") != sha256_file(self.path):
            raise ValueError("public corpus manifest identity changed")
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
            "workloads": len(self.entries),
            "scored": sum(entry.role == "scored" for entry in self.entries),
            "compatibility_sentinels": sum(
                entry.role == "compatibility_sentinel" for entry in self.entries
            ),
        }


def _validate_manifest_header(data: Mapping[str, Any]) -> None:
    if int(data.get("schema_version", 0)) != 2:
        raise ValueError("corpus manifest must use schema_version 2")
    source = data.get("source") or {}
    if source.get("dataset_id") != OFFICIAL_DATASET_ID:
        raise ValueError("corpus must use the official NVIDIA dataset")
    if source.get("revision") != OFFICIAL_DATASET_REVISION:
        raise ValueError("corpus dataset revision is not the reviewed revision")
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


def _load_entry(data: Mapping[str, Any]) -> CorpusEntry:
    return CorpusEntry(
        slot=str(data["slot"]),
        config=str(data["config"]),
        problem=str(data["problem"]),
        workload_uuid=str(data["workload_uuid"]),
        official_row_sha256=str(data["official_row_sha256"]),
        official_workload_sha256=str(data["official_workload_sha256"]),
        operation=str(data["operation"]),
        role=str(data.get("role", "scored")),
    )


def _validate_entries(
    entries: tuple[CorpusEntry, ...], parquet: Mapping[str, str]
) -> None:
    if len(entries) != 15:
        raise ValueError("RX 9060 XT corpus must contain exactly 15 workloads")
    if len({entry.slot for entry in entries}) != len(entries):
        raise ValueError("corpus slots must be unique")
    identities = {
        (entry.config, entry.problem, entry.workload_uuid) for entry in entries
    }
    if len(identities) != len(entries):
        raise ValueError("corpus workload identities must be unique")
    if any(entry.config not in parquet for entry in entries):
        raise ValueError("corpus entry references an unpinned parquet file")
    if any(not entry.operation for entry in entries):
        raise ValueError("formal corpus entries must declare an operation")
    roles = [entry.role for entry in entries]
    if roles.count("scored") != 14 or roles.count("compatibility_sentinel") != 1:
        raise ValueError("corpus must contain 14 scored entries and one sentinel")


def _verify_parquet_files(source: Path, expected: Mapping[str, str]) -> None:
    for config, digest in expected.items():
        path = source / "data" / f"{config}.parquet"
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"official parquet identity mismatch: {config}")


def _materialize_entries(
    manifest: CorpusManifest, source: Path, staging: Path
) -> list[dict[str, str]]:
    import pandas as pd

    frames: dict[str, Any] = {}
    selected: dict[Path, list[dict[str, Any]]] = {}
    definitions: dict[Path, dict[str, Any]] = {}
    for entry in manifest.entries:
        if entry.config not in frames:
            frames[entry.config] = pd.read_parquet(
                source / "data" / f"{entry.config}.parquet"
            )
        row = _select_official_row(frames[entry.config], entry)
        workload = _select_official_workload(row, entry)
        relative = entry.relative_problem_dir
        definitions[relative] = _definition_from_row(row, op_type=entry.operation)
        selected.setdefault(relative, []).append(workload)
    return [
        _write_problem(staging, relative, definitions[relative], workloads)
        for relative, workloads in selected.items()
    ]


def _select_official_row(frame: Any, entry: CorpusEntry) -> dict[str, Any]:
    matches = frame[frame["name"] == entry.problem]
    if len(matches) != 1:
        raise ValueError(f"official problem missing or duplicated: {entry.problem}")
    row = matches.iloc[0].to_dict()
    normalized = {key: (None if value is None else value) for key, value in row.items()}
    if stable_json_checksum(normalized) != entry.official_row_sha256:
        raise ValueError(f"official row drifted: {entry.problem}")
    return row


def _select_official_workload(
    row: Mapping[str, Any], entry: CorpusEntry
) -> dict[str, Any]:
    workloads = json.loads(str(row["workloads"]))
    matches = [
        workload
        for workload in workloads
        if str(workload.get("uuid")) == entry.workload_uuid
    ]
    if len(matches) != 1:
        raise ValueError(f"official workload missing: {entry.workload_uuid}")
    raw_workload = matches[0]
    if stable_json_checksum(raw_workload) != entry.official_workload_sha256:
        raise ValueError(f"official workload drifted: {entry.workload_uuid}")
    workload = dict(raw_workload)
    tolerance = workload.get("tolerance")
    if isinstance(tolerance, dict) and "required_match_ratio" in tolerance:
        normalized_tolerance = dict(tolerance)
        normalized_tolerance["required_matched_ratio"] = normalized_tolerance.pop(
            "required_match_ratio"
        )
        workload["tolerance"] = normalized_tolerance
    Workload.model_validate(workload)
    return workload


def _definition_from_row(
    row: Mapping[str, Any], *, op_type: str | None = None
) -> dict[str, Any]:
    resolved_op_type = str(row.get("op_type") or op_type or "").strip()
    if not resolved_op_type:
        raise ValueError("official problem definition is missing op_type")
    definition: dict[str, Any] = {
        "name": str(row["name"]),
        "op_type": resolved_op_type,
        "description": str(row.get("description") or ""),
        "axes": json.loads(str(row["axes"])),
        "inputs": json.loads(str(row["inputs"])),
        "outputs": json.loads(str(row["outputs"])),
        "reference": str(row["reference"]),
    }
    custom = row.get("custom_inputs_entrypoint")
    if isinstance(custom, str) and custom:
        definition["custom_inputs_entrypoint"] = custom
    hf_id = row.get("hf_id")
    if isinstance(hf_id, str) and hf_id:
        definition["hf_id"] = hf_id
    Definition.model_validate(definition)
    return definition


def _write_problem(
    staging: Path,
    relative: Path,
    definition: Mapping[str, Any],
    workloads: list[dict[str, Any]],
) -> dict[str, str]:
    root = staging / relative
    root.mkdir(parents=True, exist_ok=True)
    definition_path = root / "definition.json"
    workload_path = root / "workload.jsonl"
    definition_path.write_text(json.dumps(definition, indent=2) + "\n")
    workload_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in workloads)
    )
    return {
        "path": relative.as_posix(),
        "definition_sha256": sha256_file(definition_path),
        "workload_sha256": sha256_file(workload_path),
    }


def _write_materialization_manifest(
    manifest: CorpusManifest, staging: Path, records: list[dict[str, str]]
) -> None:
    payload = {
        "schema_version": 1,
        "dataset_id": OFFICIAL_DATASET_ID,
        "dataset_revision": OFFICIAL_DATASET_REVISION,
        "corpus_manifest_sha256": sha256_file(manifest.path),
        "problems": records,
    }
    (staging / "materialization-manifest.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False)
    )


def _validate_problem_inventory(
    entries: tuple[CorpusEntry, ...], problems: Any
) -> None:
    """Require the local record to name every immutable corpus problem exactly once."""
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
        Workload.model_validate_json(line)


def _audit_selected_workloads(entries: tuple[CorpusEntry, ...], output: Path) -> None:
    for entry in entries:
        path = output / entry.relative_problem_dir / "workload.jsonl"
        workloads = [json.loads(line) for line in path.read_text().splitlines()]
        matches = [
            item for item in workloads if item.get("uuid") == entry.workload_uuid
        ]
        if len(matches) != 1:
            raise ValueError(f"selected workload missing: {entry.workload_uuid}")
        if stable_json_checksum(matches[0]) != entry.official_workload_sha256:
            raise ValueError(
                f"selected workload identity mismatch: {entry.workload_uuid}"
            )


__all__ = [
    "CorpusEntry",
    "CorpusManifest",
    "FORMAL_GFX_TARGET",
    "FORMAL_ARCHITECTURE_SHA256",
    "OFFICIAL_CORPUS_MANIFEST_SHA256",
    "OFFICIAL_DATASET_ID",
    "OFFICIAL_DATASET_REVISION",
]
