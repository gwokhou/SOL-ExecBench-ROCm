# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Canonical full-dataset denominator and static bound-coverage evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.platform.arch_capabilities import (
    load_packaged_arch_capability_budget,
)
from sol_execbench.core.scoring.amd_bound_estimate.estimates import (
    estimate_bound_work,
    resolve_architecture_profile_paths,
)
from sol_execbench.core.scoring.amd_bound_graph.builder import build_static_bound_graph
from sol_execbench.core.scoring.amd_sol.fusion import build_fusion_groups
from sol_execbench.core.scoring.hardware_profile_requirements import (
    hardware_profile_requirements_from_dict,
    requirements_from_estimates,
)
from sol_execbench.core.scoring.hardware_calibration.environment import adapter_for


FULL_SUITE_SCHEMA_VERSION = "sol_execbench.canonical_suite.v1"
FULL_SUITE_COVERAGE_SCHEMA_VERSION = "sol_execbench.full_suite_coverage.v1"
FULL_SUITE_SCOPE = "gfx1200:sol-execbench:235-problems:3957-workloads"
FULL_SUITE_PROBLEM_COUNT = 235
FULL_SUITE_WORKLOAD_COUNT = 3957
OFFICIAL_AGGREGATION_POLICY = "fixed_suite_denominator_zero_for_blocked"
DERIVED_AGGREGATION_POLICY = "available_scored_workloads_mean"
_CONFIDENCE_RANK = {"supported": 0, "inexact": 1, "unsupported": 2}


def _canonical_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _json_lines(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} workload must be an object")
        rows.append((line_number - 1, payload))
    return rows


def _problem_paths(benchmark_root: Path) -> list[Path]:
    return sorted(
        path.parent
        for path in Path(benchmark_root).rglob("definition.json")
        if path.with_name("workload.jsonl").is_file()
    )


def _source_tree_digest(problem_paths: Iterable[Path], benchmark_root: Path) -> str:
    digest = hashlib.sha256()
    for problem_path in problem_paths:
        for filename in ("definition.json", "workload.jsonl"):
            path = problem_path / filename
            relative = path.relative_to(benchmark_root).as_posix().encode()
            content = path.read_bytes()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
    return digest.hexdigest()


def build_full_suite_manifest(
    benchmark_root: Path,
    *,
    architecture: str = "gfx1200",
    expected_problem_count: int | None = FULL_SUITE_PROBLEM_COUNT,
    expected_workload_count: int | None = FULL_SUITE_WORKLOAD_COUNT,
) -> dict[str, Any]:
    """Build the UUID-pinned full denominator from a benchmark checkout."""
    root = Path(benchmark_root)
    problems = _problem_paths(root)
    workloads: list[dict[str, Any]] = []
    definitions: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for problem_path in problems:
        relative = problem_path.relative_to(root)
        if len(relative.parts) < 2:
            raise ValueError(
                f"problem path must include category and problem id: {relative}"
            )
        category = relative.parts[0]
        problem_id = relative.as_posix()
        definition = Definition.model_validate_json(
            (problem_path / "definition.json").read_text(encoding="utf-8")
        )
        rows = _json_lines(problem_path / "workload.jsonl")
        definitions.append(
            {
                "category": category,
                "problem_id": problem_id,
                "definition": definition.name,
                "workload_count": len(rows),
            }
        )
        for row_index, raw_workload in rows:
            workload = Workload.model_validate(raw_workload)
            key = (definition.name, str(workload.uuid))
            if key in seen:
                raise ValueError(f"duplicate definition/workload key: {key!r}")
            seen.add(key)
            workloads.append(
                {
                    "definition": definition.name,
                    "workload_uuid": str(workload.uuid),
                    "category": category,
                    "problem_id": problem_id,
                    "row_index": row_index,
                }
            )
    if (
        expected_problem_count is not None
        and len(definitions) != expected_problem_count
    ):
        raise ValueError(
            f"full suite must contain {expected_problem_count} problems; got {len(definitions)}"
        )
    if (
        expected_workload_count is not None
        and len(workloads) != expected_workload_count
    ):
        raise ValueError(
            f"full suite must contain {expected_workload_count} workloads; got {len(workloads)}"
        )
    payload: dict[str, Any] = {
        "schema_version": FULL_SUITE_SCHEMA_VERSION,
        "architecture": architecture,
        "scope": (
            FULL_SUITE_SCOPE
            if expected_problem_count == FULL_SUITE_PROBLEM_COUNT
            and expected_workload_count == FULL_SUITE_WORKLOAD_COUNT
            else f"{architecture}:sol-execbench:{len(definitions)}-problems:{len(workloads)}-workloads"
        ),
        "source_tree_sha256": _source_tree_digest(problems, root),
        "problem_denominator": len(definitions),
        "workload_denominator": len(workloads),
        "official_aggregation_policy": OFFICIAL_AGGREGATION_POLICY,
        "derived_aggregation_policy": DERIVED_AGGREGATION_POLICY,
        "definitions": definitions,
        "workloads": workloads,
    }
    payload["payload_sha256"] = _canonical_digest(payload)
    return payload


def validate_full_suite_manifest(payload: dict[str, Any]) -> None:
    """Strictly validate denominator counts, keys, policies, and checksum."""
    required = {
        "schema_version",
        "architecture",
        "scope",
        "source_tree_sha256",
        "problem_denominator",
        "workload_denominator",
        "official_aggregation_policy",
        "derived_aggregation_policy",
        "definitions",
        "workloads",
        "payload_sha256",
    }
    if set(payload) != required:
        raise ValueError("canonical suite manifest has invalid fields")
    if payload["schema_version"] != FULL_SUITE_SCHEMA_VERSION:
        raise ValueError("unsupported canonical suite manifest schema")
    expected = _canonical_digest(
        {key: value for key, value in payload.items() if key != "payload_sha256"}
    )
    if payload["payload_sha256"] != expected:
        raise ValueError("canonical suite manifest checksum mismatch")
    definitions = payload["definitions"]
    workloads = payload["workloads"]
    if not isinstance(definitions, list) or not isinstance(workloads, list):
        raise ValueError("canonical suite definitions and workloads must be lists")
    if payload["problem_denominator"] != len(definitions):
        raise ValueError("problem denominator does not match definitions")
    if payload["workload_denominator"] != len(workloads):
        raise ValueError("workload denominator does not match workloads")
    keys = [(row["definition"], row["workload_uuid"]) for row in workloads]
    if len(keys) != len(set(keys)):
        raise ValueError("canonical suite contains duplicate workload keys")
    if payload["official_aggregation_policy"] != OFFICIAL_AGGREGATION_POLICY:
        raise ValueError("canonical suite official aggregation policy is invalid")
    if payload["derived_aggregation_policy"] != DERIVED_AGGREGATION_POLICY:
        raise ValueError("canonical suite derived aggregation policy is invalid")


def build_full_suite_coverage(
    benchmark_root: Path,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build workload, family, fusion, and exact hardware-profile coverage."""
    validate_full_suite_manifest(manifest)
    root = Path(benchmark_root)
    budget = load_packaged_arch_capability_budget(str(manifest["architecture"]))
    declared_profile_keys = {
        key.value for key in adapter_for(str(manifest["architecture"])).all_candidates
    }
    operator_counts: Counter[tuple[str, str]] = Counter()
    operator_name_counts: Counter[tuple[str, str, str]] = Counter()
    fusion_counts: Counter[tuple[str, str]] = Counter()
    workload_counts: Counter[str] = Counter()
    blocker_counts: Counter[str] = Counter()
    profile_workloads: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)
    all_estimates = []
    authority_eligible_estimates = []
    workload_rows: list[dict[str, Any]] = []

    entries = {
        (row["definition"], row["workload_uuid"]): row for row in manifest["workloads"]
    }
    for definition_row in manifest["definitions"]:
        problem_path = root / definition_row["problem_id"]
        definition = Definition.model_validate_json(
            (problem_path / "definition.json").read_text(encoding="utf-8")
        )
        for _, raw_workload in _json_lines(problem_path / "workload.jsonl"):
            workload = Workload.model_validate(raw_workload)
            key = (definition.name, str(workload.uuid))
            if key not in entries:
                raise ValueError(
                    f"workload is missing from canonical manifest: {key!r}"
                )
            graph = build_static_bound_graph(definition, workload)
            estimates = resolve_architecture_profile_paths(
                estimate_bound_work(graph), str(manifest["architecture"])
            )
            groups = build_fusion_groups(graph, estimates, capability_budget=budget)
            all_estimates.extend(estimates)
            for estimate in estimates:
                confidence = estimate.confidence.value
                operator_counts[(estimate.op_family.value, confidence)] += 1
                operator_name_counts[
                    (estimate.op_family.value, estimate.op_name, confidence)
                ] += 1
                for profile in _estimate_profile_keys(estimate):
                    profile_workloads[profile].add(key)
            for group in groups:
                fusion_counts[(group.pattern_id, group.confidence.value)] += 1
            worst = max(
                (
                    *(estimate.confidence.value for estimate in estimates),
                    *(group.confidence.value for group in groups),
                ),
                key=_CONFIDENCE_RANK.__getitem__,
                default="unsupported",
            )
            blockers = set()
            if any(item.confidence.value == "unsupported" for item in estimates):
                blockers.add("unsupported_operator_estimate")
            if any(item.confidence.value == "inexact" for item in estimates):
                blockers.add("inexact_operator_estimate")
            if any(group.confidence.value != "supported" for group in groups):
                blockers.add("fusion_group_not_supported")
            unavailable_profiles = sorted(
                {
                    profile
                    for estimate in estimates
                    for profile in _estimate_profile_keys(estimate)
                    if profile not in declared_profile_keys
                }
            )
            if unavailable_profiles:
                blockers.add("hardware_profile_probe_unavailable")
            for blocker in blockers:
                blocker_counts[blocker] += 1
            if not blockers:
                authority_eligible_estimates.extend(estimates)
            workload_counts[worst] += 1
            workload_rows.append(
                {
                    **entries[key],
                    "node_count": len(estimates),
                    "fusion_group_count": len(groups),
                    "worst_confidence": worst,
                    "blocker_codes": sorted(blockers),
                    "inexact_operator_names": sorted(
                        {
                            f"{estimate.op_family.value}:{estimate.op_name}"
                            for estimate in estimates
                            if estimate.confidence.value == "inexact"
                        }
                    ),
                    "unsupported_operator_names": sorted(
                        {
                            f"{estimate.op_family.value}:{estimate.op_name}"
                            for estimate in estimates
                            if estimate.confidence.value == "unsupported"
                        }
                    ),
                    "inexact_fusion_patterns": sorted(
                        {
                            group.pattern_id
                            for group in groups
                            if group.confidence.value != "supported"
                        }
                    ),
                    "unavailable_hardware_profiles": unavailable_profiles,
                }
            )

    requirements = requirements_from_estimates(
        architecture=str(manifest["architecture"]),
        estimates=authority_eligible_estimates,
        scope=str(manifest["scope"]),
    ).to_dict()
    coverage: dict[str, Any] = {
        "schema_version": FULL_SUITE_COVERAGE_SCHEMA_VERSION,
        "architecture": manifest["architecture"],
        "scope": manifest["scope"],
        "suite_manifest_sha256": manifest["payload_sha256"],
        "summary": {
            "problem_count": manifest["problem_denominator"],
            "workload_count": len(workload_rows),
            "node_count": len(all_estimates),
            "authority_eligible_workload_count": sum(
                not row["blocker_codes"] for row in workload_rows
            ),
            "workloads_by_worst_confidence": dict(sorted(workload_counts.items())),
            "workloads_by_blocker": dict(sorted(blocker_counts.items())),
        },
        "operator_family_coverage": _nested_counts(operator_counts),
        "operator_name_coverage": _operator_name_counts(operator_name_counts),
        "fusion_pattern_coverage": _nested_counts(fusion_counts),
        "hardware_profile_coverage": [
            {"profile_key": key, "workload_count": len(value)}
            for key, value in sorted(profile_workloads.items())
        ],
        "workloads": sorted(
            workload_rows, key=lambda row: (row["definition"], row["workload_uuid"])
        ),
    }
    coverage["payload_sha256"] = _canonical_digest(coverage)
    return coverage, requirements


def validate_full_suite_coverage(
    coverage: dict[str, Any],
    requirements: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    """Verify that static coverage exactly accounts for the frozen denominator."""
    validate_full_suite_manifest(manifest)
    required = {
        "schema_version",
        "architecture",
        "scope",
        "suite_manifest_sha256",
        "summary",
        "operator_family_coverage",
        "operator_name_coverage",
        "fusion_pattern_coverage",
        "hardware_profile_coverage",
        "workloads",
        "payload_sha256",
    }
    if set(coverage) != required:
        raise ValueError("full suite coverage has invalid fields")
    if coverage["schema_version"] != FULL_SUITE_COVERAGE_SCHEMA_VERSION:
        raise ValueError("unsupported full suite coverage schema")
    expected = _canonical_digest(
        {key: value for key, value in coverage.items() if key != "payload_sha256"}
    )
    if coverage["payload_sha256"] != expected:
        raise ValueError("full suite coverage checksum mismatch")
    if coverage["suite_manifest_sha256"] != manifest["payload_sha256"]:
        raise ValueError("full suite coverage manifest checksum mismatch")
    rows = coverage["workloads"]
    if not isinstance(rows, list):
        raise ValueError("full suite coverage workloads must be a list")
    expected_keys = {
        (row["definition"], row["workload_uuid"]) for row in manifest["workloads"]
    }
    actual_keys = {(row["definition"], row["workload_uuid"]) for row in rows}
    if actual_keys != expected_keys or len(rows) != len(actual_keys):
        raise ValueError("full suite coverage does not exactly match denominator")
    summary = coverage["summary"]
    if summary["problem_count"] != manifest["problem_denominator"]:
        raise ValueError("full suite coverage problem count mismatch")
    if summary["workload_count"] != manifest["workload_denominator"]:
        raise ValueError("full suite coverage workload count mismatch")
    confidence_total = sum(summary["workloads_by_worst_confidence"].values())
    if confidence_total != len(rows):
        raise ValueError("full suite coverage confidence counts do not close")
    eligible = sum(not row["blocker_codes"] for row in rows)
    if summary["authority_eligible_workload_count"] != eligible:
        raise ValueError("full suite coverage authority-eligible count mismatch")
    parsed_requirements = hardware_profile_requirements_from_dict(requirements)
    if parsed_requirements.architecture != manifest["architecture"]:
        raise ValueError("hardware requirements architecture mismatch")
    if parsed_requirements.scope != manifest["scope"]:
        raise ValueError("hardware requirements scope mismatch")


def _estimate_profile_keys(estimate: Any) -> tuple[str, ...]:
    keys = []
    if estimate.flops > 0.0 and all(
        (
            estimate.compute_operation,
            estimate.input_dtype,
            estimate.output_dtype,
            estimate.compute_path,
        )
    ):
        keys.append(
            f"compute.{estimate.compute_operation}.{estimate.input_dtype}."
            f"{estimate.output_dtype}.{estimate.compute_path}"
        )
    if estimate.total_bytes > 0.0 and all(
        (
            estimate.memory_access,
            estimate.input_dtype,
            estimate.output_dtype,
            estimate.memory_path,
        )
    ):
        keys.append(
            f"memory.{estimate.memory_access}.{estimate.input_dtype}."
            f"{estimate.output_dtype}.{estimate.memory_path}"
        )
    return tuple(keys)


def _nested_counts(counts: Counter[tuple[str, str]]) -> list[dict[str, Any]]:
    names = sorted({name for name, _ in counts})
    return [
        {
            "name": name,
            "supported": counts[(name, "supported")],
            "inexact": counts[(name, "inexact")],
            "unsupported": counts[(name, "unsupported")],
            "total": sum(counts[(name, value)] for value in _CONFIDENCE_RANK),
        }
        for name in names
    ]


def _operator_name_counts(
    counts: Counter[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    keys = sorted({(family, name) for family, name, _ in counts})
    return [
        {
            "family": family,
            "op_name": name,
            "supported": counts[(family, name, "supported")],
            "inexact": counts[(family, name, "inexact")],
            "unsupported": counts[(family, name, "unsupported")],
            "total": sum(counts[(family, name, value)] for value in _CONFIDENCE_RANK),
        }
        for family, name in keys
    ]


__all__ = [
    "DERIVED_AGGREGATION_POLICY",
    "FULL_SUITE_COVERAGE_SCHEMA_VERSION",
    "FULL_SUITE_PROBLEM_COUNT",
    "FULL_SUITE_SCHEMA_VERSION",
    "FULL_SUITE_SCOPE",
    "FULL_SUITE_WORKLOAD_COUNT",
    "OFFICIAL_AGGREGATION_POLICY",
    "build_full_suite_coverage",
    "build_full_suite_manifest",
    "validate_full_suite_coverage",
    "validate_full_suite_manifest",
]
