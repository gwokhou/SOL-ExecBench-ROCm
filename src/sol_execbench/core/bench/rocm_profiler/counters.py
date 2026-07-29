# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Bounded rocprofv3 counter discovery, job generation, and CSV alignment."""

from __future__ import annotations

import csv
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import ConfigDict, Field

from sol_execbench.core.bench.performance_model.models import (
    DispatchEvidence,
    EvidenceReference,
    ResourceFootprint,
)
from sol_execbench.core.data.base_model import BaseModelWithDocstrings
from sol_execbench.core.integrity import sha256_file

MAX_COUNTER_CSV_BYTES = 128 * 1024 * 1024
ROCPROFV3_AVAIL_EXECUTABLE = "rocprofv3-avail"
COUNTER_MANIFEST_SCHEMA_VERSION = "sol_execbench.rocprofv3_counter_manifest.v1"
_FIELD_TOKEN = re.compile(r"[^a-z0-9]+")
_COUNTER_TOKEN = re.compile(r"[^A-Za-z0-9]+")
_NUMBER = re.compile(
    r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*([A-Za-z%]*)\s*$",
)
_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)


class CounterChoice(BaseModelWithDocstrings):
    """Ordered counter alternatives for one normalized model metric."""

    model_config = _MODEL_CONFIG

    metric: str
    alternatives: list[str] = Field(min_length=1)
    required: bool = True


class CounterGroup(BaseModelWithDocstrings):
    """One hardware-compatible rocprofv3 counter pass."""

    model_config = _MODEL_CONFIG

    name: str
    counters: list[CounterChoice] = Field(min_length=1)


class CounterManifest(BaseModelWithDocstrings):
    """Versioned per-architecture counter selection policy."""

    model_config = _MODEL_CONFIG

    schema_version: str
    architecture: str
    rocm_compatibility: str
    groups: list[CounterGroup] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class CounterPassCSV:
    """One pass-indexed counter CSV artifact."""

    pass_index: int
    path: Path


@dataclass(slots=True)
class _MutableDispatch:
    workload_uuid: str
    candidate_sha256: str
    dispatch_id: str
    correlation_id: str | None
    kernel_symbol: str
    grid: tuple[int, int, int]
    workgroup: tuple[int, int, int]
    iteration_ordinal: int
    pass_index: int
    counters: dict[str, float]
    runtime_footprint: ResourceFootprint
    start_timestamp_ns: int | None
    end_timestamp_ns: int | None
    source: EvidenceReference


def load_counter_manifest(path: str | Path) -> CounterManifest:
    """Load a strict versioned counter manifest."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    manifest = CounterManifest.model_validate(payload)
    if manifest.schema_version != COUNTER_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported_counter_manifest_schema")
    return manifest


def parse_available_counters(output: str) -> frozenset[str]:
    """Parse both ``list --pmc`` and ``info --pmc`` text formats."""
    names = {
        match.group(1)
        for match in re.finditer(
            r"(?im)^\s*Counter_Name\s*:\s*([A-Za-z0-9_]+)\s*$",
            output,
        )
    }
    for match in re.finditer(r"(?im)^\s*PMC\s*:\s*(.*)$", output):
        names.update(re.findall(r"\b[A-Za-z][A-Za-z0-9_]+\b", match.group(1)))
    if names:
        return frozenset(names)
    return frozenset(
        token
        for token in re.findall(r"\b[A-Za-z][A-Za-z0-9_]+\b", output)
        if token.upper() == token and "_" in token
    )


def parse_available_architectures(output: str) -> frozenset[str]:
    """Return the exact GPU architectures named by ``rocprofv3-avail``."""
    return frozenset(
        match.group(1).lower()
        for match in re.finditer(
            r"(?im)^\s*Name\s*:\s*(gfx[0-9a-z]+)\s*$",
            output,
        )
    )


def select_counter_groups(
    manifest: CounterManifest,
    available: Iterable[str],
) -> tuple[list[list[str]], list[str]]:
    """Select supported alternatives in fail-safe single-counter passes."""
    supported = set(available)
    groups: list[list[str]] = []
    missing: list[str] = []
    for group in manifest.groups:
        for choice in group.counters:
            counter = next(
                (name for name in choice.alternatives if name in supported),
                None,
            )
            if counter is not None:
                # gfx12 counter-block compatibility varies by driver and
                # firmware. A singleton pass is always auditable and avoids
                # rocprofiler aborting an otherwise valid collection.
                groups.append([counter])
            elif choice.required:
                missing.append(f"{group.name}:{choice.metric}")
    return groups, missing


def write_counter_job(
    path: str | Path,
    groups: Sequence[Sequence[str]],
    *,
    output_directory: str,
) -> None:
    """Atomically write a controlled multi-pass rocprofv3 YAML job."""
    if not groups or any(not group for group in groups):
        raise ValueError("counter groups must not be empty")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "jobs": [
            {
                "pmc": list(group),
                "output_directory": output_directory,
                "output_format": ["csv", "rocpd"],
                "truncate_kernels": False,
            }
            for group in groups
        ],
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def build_rocprofv3_counter_command(
    application_command: Sequence[str],
    *,
    input_path: str | Path,
    executable: str = "rocprofv3",
) -> list[str]:
    """Build the explicit controlled counter-collection command."""
    if not application_command:
        raise ValueError("application_command must not be empty")
    return [executable, "--input", str(input_path), "--", *application_command]


def parse_and_align_counter_passes(
    passes: Sequence[CounterPassCSV],
    *,
    workload_uuid: str,
    candidate_sha256: str,
    required_counters: Iterable[str] = (),
) -> list[DispatchEvidence]:
    """Parse and fail-closed align dispatch evidence across profiler passes."""
    if not passes:
        return []
    parsed = [
        _parse_counter_pass(
            counter_pass,
            workload_uuid=workload_uuid,
            candidate_sha256=candidate_sha256,
        )
        for counter_pass in passes
    ]
    return _align_passes(parsed, set(required_counters))


def counter_names_in_csv(path: str | Path) -> frozenset[str]:
    """Return normalized counter names from one bounded counter CSV."""
    source = Path(path)
    if source.stat().st_size > MAX_COUNTER_CSV_BYTES:
        raise ValueError("counter_csv_too_large")
    names: set[str] = set()
    with source.open(newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            row = {_normalize_field(key): value for key, value in raw.items()}
            names.add(
                normalize_counter_name(
                    _required_field(row, "countername", "counter"),
                ),
            )
    if not names:
        raise ValueError("counter_csv_contains_no_counters")
    return frozenset(names)


def counter_pass_index(path: str | Path) -> int | None:
    """Return the pass/PMC index encoded in an artifact path."""
    for part in Path(path).parts:
        match = re.fullmatch(r"(?:pass|pmc)[_-](\d+)", part.lower())
        if match:
            return int(match.group(1))
    return None


def normalize_counter_name(value: str) -> str:
    """Normalize a rocprof counter name to the model's canonical spelling."""
    return _COUNTER_TOKEN.sub("_", value).strip("_").upper()


def _parse_counter_pass(
    counter_pass: CounterPassCSV,
    *,
    workload_uuid: str,
    candidate_sha256: str,
) -> list[_MutableDispatch]:
    path = counter_pass.path
    if path.stat().st_size > MAX_COUNTER_CSV_BYTES:
        raise ValueError("counter_csv_too_large")
    source = EvidenceReference(
        kind="rocprofv3_counter_csv",
        path=str(path),
        sha256=sha256_file(path),
    )
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    order: list[str] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            row = {_normalize_field(key): value for key, value in raw.items()}
            dispatch_id = _field(row, "dispatchid", "dispatch")
            if dispatch_id is None:
                raise ValueError("counter_csv_missing_dispatch_id")
            if dispatch_id not in grouped:
                order.append(dispatch_id)
            grouped[dispatch_id].append(row)
    occurrence: Counter[tuple[str, tuple[int, int, int], tuple[int, int, int]]]
    occurrence = Counter()
    result: list[_MutableDispatch] = []
    for dispatch_id in order:
        dispatch = _dispatch_from_rows(
            grouped[dispatch_id],
            workload_uuid=workload_uuid,
            candidate_sha256=candidate_sha256,
            pass_index=counter_pass.pass_index,
            source=source,
        )
        base = (dispatch.kernel_symbol, dispatch.grid, dispatch.workgroup)
        dispatch.iteration_ordinal = occurrence[base]
        occurrence[base] += 1
        result.append(dispatch)
    return result


def _dispatch_from_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    workload_uuid: str,
    candidate_sha256: str,
    pass_index: int,
    source: EvidenceReference,
) -> _MutableDispatch:
    first = rows[0]
    kernel = _required_field(first, "kernelname", "name")
    grid = _dimensions(first, "gridsize", "grid")
    workgroup = _dimensions(first, "workgroupsize", "workgroup")
    counters: dict[str, float] = {}
    for row in rows:
        name = _required_field(row, "countername", "counter")
        normalized_name = normalize_counter_name(name)
        value = _counter_value(
            _required_field(row, "countervalue", "value"),
            counter_name=normalized_name,
        )
        if normalized_name in counters and counters[normalized_name] != value:
            raise ValueError("counter_value_conflict_within_pass")
        counters[normalized_name] = value
    return _MutableDispatch(
        workload_uuid=workload_uuid,
        candidate_sha256=candidate_sha256,
        dispatch_id=_required_field(first, "dispatchid", "dispatch"),
        correlation_id=_field(first, "correlationid", "correlation"),
        kernel_symbol=kernel,
        grid=grid,
        workgroup=workgroup,
        iteration_ordinal=0,
        pass_index=pass_index,
        counters=counters,
        runtime_footprint=ResourceFootprint(
            vgpr_count=_optional_int(_field(first, "vgprcount")),
            sgpr_count=_optional_int(_field(first, "sgprcount")),
            lds_bytes=_optional_int(_field(first, "ldsblocksize", "ldssize")),
            scratch_bytes=_optional_int(_field(first, "scratchsize")),
        ),
        start_timestamp_ns=_optional_int(
            _field(first, "starttimestamp", "starttimestampns"),
        ),
        end_timestamp_ns=_optional_int(
            _field(first, "endtimestamp", "endtimestampns"),
        ),
        source=source,
    )


def _align_passes(
    parsed: Sequence[Sequence[_MutableDispatch]],
    required: set[str],
) -> list[DispatchEvidence]:
    maps = [{_alignment_key(item): item for item in items} for items in parsed]
    all_keys = set().union(*(mapping.keys() for mapping in maps))
    aligned: list[DispatchEvidence] = []
    for key in sorted(all_keys):
        matches = [mapping.get(key) for mapping in maps]
        concrete = [match for match in matches if match is not None]
        mismatch = len(concrete) != len(maps)
        counter_conflict = _has_counter_conflict(concrete)
        footprint_conflict = _has_footprint_conflict(concrete)
        first = concrete[0]
        counters = _merged_counters(concrete)
        missing = sorted(
            {normalize_counter_name(name) for name in required}
            - counters.keys(),
        )
        reasons: list[str] = []
        if mismatch:
            reasons.append("cross_pass_alignment_mismatch")
        if counter_conflict:
            reasons.append("cross_pass_counter_conflict")
        if footprint_conflict:
            reasons.append("cross_pass_footprint_conflict")
        if missing:
            reasons.append("missing_required_counters")
        aligned.append(
            DispatchEvidence(
                workload_uuid=first.workload_uuid,
                candidate_sha256=first.candidate_sha256,
                dispatch_id=first.dispatch_id,
                correlation_id=first.correlation_id,
                kernel_symbol=first.kernel_symbol,
                grid=first.grid,
                workgroup=first.workgroup,
                iteration_ordinal=first.iteration_ordinal,
                counter_passes=sorted(item.pass_index for item in concrete),
                counters=counters,
                runtime_footprint=first.runtime_footprint,
                start_timestamp_ns=first.start_timestamp_ns,
                end_timestamp_ns=first.end_timestamp_ns,
                valid=not reasons,
                reason_codes=reasons,
                sources=[item.source for item in concrete],
            ),
        )
    return aligned


def _alignment_key(
    dispatch: _MutableDispatch,
) -> tuple[str, tuple[int, int, int], tuple[int, int, int], int]:
    return (
        dispatch.kernel_symbol,
        dispatch.grid,
        dispatch.workgroup,
        dispatch.iteration_ordinal,
    )


def _has_counter_conflict(dispatches: Sequence[_MutableDispatch]) -> bool:
    values: dict[str, float] = {}
    for dispatch in dispatches:
        for name, value in dispatch.counters.items():
            if name in values and values[name] != value:
                return True
            values[name] = value
    return False


def _has_footprint_conflict(
    dispatches: Sequence[_MutableDispatch],
) -> bool:
    return len({dispatch.runtime_footprint for dispatch in dispatches}) > 1


def _merged_counters(
    dispatches: Sequence[_MutableDispatch],
) -> dict[str, float]:
    merged: dict[str, float] = {}
    for dispatch in dispatches:
        merged.update(dispatch.counters)
    return dict(sorted(merged.items()))


def _normalize_field(value: str | None) -> str:
    return _FIELD_TOKEN.sub("", value.lower()) if value else ""


def _field(row: Mapping[str, str], *names: str) -> str | None:
    for name in names:
        value = row.get(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _required_field(row: Mapping[str, str], *names: str) -> str:
    value = _field(row, *names)
    if value is None:
        raise ValueError(f"counter_csv_missing_{names[0]}")
    return value


def _dimensions(
    row: Mapping[str, str],
    combined_name: str,
    prefix: str,
) -> tuple[int, int, int]:
    combined = _field(row, combined_name)
    if combined is not None:
        values = [int(value) for value in re.findall(r"\d+", combined)]
        if not values:
            raise ValueError(f"invalid_{combined_name}")
        padded = values + [1, 1]
        return padded[0], padded[1], padded[2]
    values = [
        _optional_int(_field(row, f"{prefix}size{axis}", f"{prefix}{axis}"))
        for axis in ("x", "y", "z")
    ]
    return (
        values[0] if values[0] is not None else 1,
        values[1] if values[1] is not None else 1,
        values[2] if values[2] is not None else 1,
    )


def _counter_value(value: str, *, counter_name: str) -> float:
    match = _NUMBER.fullmatch(value.replace(",", ""))
    if match is None:
        raise ValueError("invalid_counter_value")
    number = float(match.group(1))
    suffix = match.group(2).lower()
    scale = {
        "": 1.0,
        "%": 1.0,
        "b": 1.0,
        "kb": 1_000.0,
        "mb": 1_000_000.0,
        "gb": 1_000_000_000.0,
        "kib": 1_024.0,
        "mib": 1_048_576.0,
        "gib": 1_073_741_824.0,
    }.get(suffix)
    if scale is None:
        raise ValueError("unsupported_counter_unit")
    normalized = number * scale
    if not suffix and counter_name in {"FETCH_SIZE", "WRITE_SIZE"}:
        return normalized * 1024.0
    return normalized


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    return int(value)


__all__ = [
    "COUNTER_MANIFEST_SCHEMA_VERSION",
    "MAX_COUNTER_CSV_BYTES",
    "ROCPROFV3_AVAIL_EXECUTABLE",
    "CounterGroup",
    "CounterManifest",
    "CounterPassCSV",
    "build_rocprofv3_counter_command",
    "counter_names_in_csv",
    "counter_pass_index",
    "load_counter_manifest",
    "normalize_counter_name",
    "parse_and_align_counter_passes",
    "parse_available_architectures",
    "parse_available_counters",
    "select_counter_groups",
    "write_counter_job",
]
