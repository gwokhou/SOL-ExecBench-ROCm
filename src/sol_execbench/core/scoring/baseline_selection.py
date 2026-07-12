# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Deterministic selection evidence for a release baseline portfolio."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from statistics import median
from typing import Any, Iterable


BASELINE_SELECTION_SCHEMA_VERSION = "sol_execbench.baseline_selection.v1"
_TIE_TOLERANCE_REL = 0.02


def _sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class BaselineCandidate:
    """Three independent correct measurements of one candidate implementation."""

    definition: str
    workload_uuid: str
    candidate: str
    solution_sha256: str
    backend: str
    backend_version: str
    build_id: str
    dependencies: tuple[str, ...]
    timings_ms: tuple[float, ...]
    correctness_passed: bool

    def __post_init__(self) -> None:
        for name in (
            "definition",
            "workload_uuid",
            "candidate",
            "backend",
            "backend_version",
            "build_id",
        ):
            if (
                not isinstance(getattr(self, name), str)
                or not getattr(self, name).strip()
            ):
                raise ValueError(f"{name} must be non-empty")
        _sha256(self.solution_sha256, "solution_sha256")
        if len(self.timings_ms) != 3 or any(
            not math.isfinite(value) or value <= 0.0 for value in self.timings_ms
        ):
            raise ValueError("timings_ms must contain exactly three positive values")
        if any(not value.strip() for value in self.dependencies):
            raise ValueError("dependencies must contain non-empty strings")
        if not self.correctness_passed:
            raise ValueError("only correctness-passing candidates may be selected")

    @property
    def key(self) -> tuple[str, str]:
        return self.definition, self.workload_uuid

    @property
    def median_ms(self) -> float:
        return float(median(self.timings_ms))

    @property
    def spread_rel(self) -> float:
        return (max(self.timings_ms) - min(self.timings_ms)) / self.median_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "definition": self.definition,
            "workload_uuid": self.workload_uuid,
            "candidate": self.candidate,
            "solution_sha256": self.solution_sha256,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "build_id": self.build_id,
            "dependencies": list(self.dependencies),
            "timings_ms": list(self.timings_ms),
            "median_ms": self.median_ms,
            "spread_rel": self.spread_rel,
            "correctness_passed": self.correctness_passed,
        }


@dataclass(frozen=True)
class BaselineSelection:
    """Winner record for one definition/workload shape signature."""

    winner: BaselineCandidate
    candidates: tuple[BaselineCandidate, ...]
    tie_tolerance_rel: float = _TIE_TOLERANCE_REL

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("selection requires candidates")
        if any(candidate.key != self.winner.key for candidate in self.candidates):
            raise ValueError("selection candidates must share a workload key")
        if self.winner not in self.candidates:
            raise ValueError("winner must be one of selection candidates")
        if not math.isfinite(self.tie_tolerance_rel) or self.tie_tolerance_rel < 0.0:
            raise ValueError("tie_tolerance_rel must be finite and non-negative")
        if self.winner != choose_baseline_candidate(
            self.candidates, self.tie_tolerance_rel
        ):
            raise ValueError("winner does not follow deterministic selection policy")

    def to_dict(self) -> dict[str, Any]:
        return {
            "definition": self.winner.definition,
            "workload_uuid": self.winner.workload_uuid,
            "winner": self.winner.candidate,
            "tie_tolerance_rel": self.tie_tolerance_rel,
            "candidates": [
                candidate.to_dict()
                for candidate in sorted(
                    self.candidates, key=lambda item: item.candidate
                )
            ],
        }


def choose_baseline_candidate(
    candidates: Iterable[BaselineCandidate],
    tie_tolerance_rel: float = _TIE_TOLERANCE_REL,
) -> BaselineCandidate:
    """Choose minimum median, then spread, dependency count, and stable name."""
    entries = tuple(candidates)
    if not entries:
        raise ValueError("cannot choose from an empty candidate set")
    fastest = min(entry.median_ms for entry in entries)
    tied = tuple(
        entry
        for entry in entries
        if entry.median_ms <= fastest * (1.0 + tie_tolerance_rel)
    )
    return min(
        tied,
        key=lambda entry: (
            entry.spread_rel,
            len(entry.dependencies),
            entry.candidate,
        ),
    )


def build_baseline_selection_manifest(
    *,
    scope: str,
    candidates: Iterable[BaselineCandidate],
    required_workload_keys: Iterable[tuple[str, str]],
) -> BaselineSelectionManifest:
    """Select one reproducible baseline candidate for every frozen workload.

    The input is deliberately a flat stream of candidate measurements: this
    makes it possible to retain every losing candidate in the result while
    rejecting a portfolio that silently omits or adds a workload.
    """
    grouped: dict[tuple[str, str], list[BaselineCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.key, []).append(candidate)
    required = tuple(required_workload_keys)
    required_set = set(required)
    if len(required) != len(required_set):
        raise ValueError("required workload keys must be unique")
    missing = required_set - set(grouped)
    unexpected = set(grouped) - required_set
    if missing or unexpected:
        details = []
        if missing:
            details.append(
                "missing=" + ", ".join(f"{a}:{b}" for a, b in sorted(missing))
            )
        if unexpected:
            details.append(
                "unexpected=" + ", ".join(f"{a}:{b}" for a, b in sorted(unexpected))
            )
        raise ValueError(
            "baseline candidates do not match frozen suite: " + "; ".join(details)
        )
    selections = tuple(
        BaselineSelection(
            winner=choose_baseline_candidate(grouped[key]),
            candidates=tuple(grouped[key]),
        )
        for key in sorted(required_set)
    )
    return BaselineSelectionManifest(scope=scope, selections=selections)


@dataclass(frozen=True)
class BaselineSelectionManifest:
    """Immutable selected baseline portfolio for a fixed suite."""

    scope: str
    selections: tuple[BaselineSelection, ...]
    payload_sha256: str | None = None
    schema_version: str = BASELINE_SELECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != BASELINE_SELECTION_SCHEMA_VERSION:
            raise ValueError("unsupported baseline selection schema")
        if not self.scope.strip() or not self.selections:
            raise ValueError("scope and selections must be non-empty")
        keys = [selection.winner.key for selection in self.selections]
        if len(keys) != len(set(keys)):
            raise ValueError("baseline selection manifest has duplicate workloads")
        expected = hashlib.sha256(
            json.dumps(
                self._payload_dict(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        if self.payload_sha256 is not None and self.payload_sha256 != expected:
            raise ValueError("baseline selection manifest checksum does not match")
        object.__setattr__(self, "payload_sha256", expected)

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope,
            "selections": [
                selection.to_dict()
                for selection in sorted(
                    self.selections, key=lambda item: item.winner.key
                )
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_dict(), "payload_sha256": self.payload_sha256}


def baseline_selection_manifest_from_dict(
    payload: dict[str, Any],
) -> BaselineSelectionManifest:
    """Strictly parse a persisted baseline-selection manifest."""
    expected = {"schema_version", "scope", "selections", "payload_sha256"}
    if set(payload) != expected or not isinstance(payload["selections"], list):
        raise ValueError("baseline selection manifest has invalid fields")
    selections = tuple(_selection_from_dict(raw) for raw in payload["selections"])
    return BaselineSelectionManifest(
        schema_version=str(payload["schema_version"]),
        scope=str(payload["scope"]),
        selections=selections,
        payload_sha256=str(payload["payload_sha256"]),
    )


def baseline_candidates_from_dict(payload: object) -> tuple[BaselineCandidate, ...]:
    """Strictly parse the flat candidate-record input accepted by the CLI."""
    if not isinstance(payload, list):
        raise ValueError("baseline candidates must be a JSON list")
    return tuple(_candidate_from_dict(item) for item in payload)


def _selection_from_dict(raw: Any) -> BaselineSelection:
    expected = {
        "definition",
        "workload_uuid",
        "winner",
        "tie_tolerance_rel",
        "candidates",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("baseline selection has invalid fields")
    candidates = tuple(
        _candidate_from_dict(candidate) for candidate in raw["candidates"]
    )
    winner = next(
        (candidate for candidate in candidates if candidate.candidate == raw["winner"]),
        None,
    )
    if winner is None:
        raise ValueError("baseline selection winner is not a candidate")
    if winner.key != (raw["definition"], raw["workload_uuid"]):
        raise ValueError("baseline selection key does not match its winner")
    return BaselineSelection(
        winner=winner,
        candidates=candidates,
        tie_tolerance_rel=float(raw["tie_tolerance_rel"]),
    )


def _candidate_from_dict(raw: Any) -> BaselineCandidate:
    expected = {
        "definition",
        "workload_uuid",
        "candidate",
        "solution_sha256",
        "backend",
        "backend_version",
        "build_id",
        "dependencies",
        "timings_ms",
        "median_ms",
        "spread_rel",
        "correctness_passed",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("baseline candidate has invalid fields")
    candidate = BaselineCandidate(
        definition=str(raw["definition"]),
        workload_uuid=str(raw["workload_uuid"]),
        candidate=str(raw["candidate"]),
        solution_sha256=str(raw["solution_sha256"]),
        backend=str(raw["backend"]),
        backend_version=str(raw["backend_version"]),
        build_id=str(raw["build_id"]),
        dependencies=tuple(str(value) for value in raw["dependencies"]),
        timings_ms=tuple(float(value) for value in raw["timings_ms"]),
        correctness_passed=bool(raw["correctness_passed"]),
    )
    if not math.isclose(float(raw["median_ms"]), candidate.median_ms):
        raise ValueError("baseline candidate median does not match timings")
    if not math.isclose(float(raw["spread_rel"]), candidate.spread_rel):
        raise ValueError("baseline candidate spread does not match timings")
    return candidate
