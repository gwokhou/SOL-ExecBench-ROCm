#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Gate AMD SOL publication on held-out external measurement contradictions.

The JSONL input is deliberately provider-neutral.  Each row contains a bound
identity plus the complete ``PerformanceProviderResult`` payload emitted by an
external compiler/search provider (for example TorchInductor, Origami or
rocMLIR).  Measurements remain diagnostics: this script can only reject a
floor that a faster held-out implementation disproves; it never upgrades one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from sol_execbench.core.scoring.amd_sol import (
    PerformanceProviderResult,
    amd_sol_bound_from_dict,
)
from sol_execbench.core.integrity.checksums import sha256_file


def _sha256_payload(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_bound_artifacts(path: Path) -> dict[tuple[str, str], object]:
    artifacts: dict[tuple[str, str], object] = {}
    for artifact_path in sorted(path.rglob("*.amd-sol.json")):
        artifact = amd_sol_bound_from_dict(
            json.loads(artifact_path.read_text(encoding="utf-8"))
        )
        key = (artifact.definition, str(artifact.workload_uuid))
        if key in artifacts:
            raise ValueError(f"duplicate bound artifact for {key}")
        artifacts[key] = artifact
    if not artifacts:
        raise ValueError(f"no AMD SOL artifacts found below {path}")
    return artifacts


def _measurement_rows(
    path: Path, *, require_profiler_trace: bool = False
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        required = {"definition", "workload_uuid", "provider_result"}
        if set(row) != required:
            raise ValueError(
                f"{path}:{line_number} must contain exactly {sorted(required)}"
            )
        if not isinstance(row["definition"], str) or not isinstance(
            row["workload_uuid"], str
        ):
            raise ValueError(f"{path}:{line_number} has invalid workload identity")
        if not isinstance(row["provider_result"], dict):
            raise ValueError(f"{path}:{line_number} provider_result must be an object")
        result = PerformanceProviderResult(**row["provider_result"])
        if result.measured_latency_ms is None:
            raise ValueError(
                f"{path}:{line_number} must provide measured_latency_ms for held-out gating"
            )
        _validate_provider_raw_evidence(
            result,
            source_path=path,
            require_profiler_trace=require_profiler_trace,
        )
        rows.append(
            {
                "definition": row["definition"],
                "workload_uuid": row["workload_uuid"],
                "provider_result": result,
            }
        )
    if not rows:
        raise ValueError(f"{path} contains no held-out measurements")
    return rows


def _validate_provider_raw_evidence(
    result: PerformanceProviderResult,
    *,
    source_path: Path,
    require_profiler_trace: bool,
) -> None:
    """Require a local, checksummed provider record before publication gating."""
    assert result.raw_evidence_ref is not None
    assert result.raw_evidence_sha256 is not None
    raw_path = Path(result.raw_evidence_ref)
    if not raw_path.is_absolute():
        raw_path = source_path.parent / raw_path
    if not raw_path.is_file() or sha256_file(raw_path) != result.raw_evidence_sha256:
        raise ValueError("provider raw evidence is missing or checksum-mismatched")
    if not require_profiler_trace:
        return
    try:
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("provider raw evidence is not valid JSON") from exc
    trace = raw.get("profiler_trace") if isinstance(raw, dict) else None
    if not isinstance(trace, dict) or set(trace) != {"ref", "sha256"}:
        raise ValueError("provider raw evidence lacks profiler trace")
    reference, checksum = trace["ref"], trace["sha256"]
    if not isinstance(reference, str) or not isinstance(checksum, str):
        raise ValueError("provider profiler trace reference is invalid")
    trace_path = Path(reference)
    if not trace_path.is_absolute():
        trace_path = raw_path.parent / trace_path
    if not trace_path.is_file() or sha256_file(trace_path) != checksum:
        raise ValueError("provider profiler trace is missing or checksum-mismatched")


def _strata_rows(path: Path, *, stratum_kind: str) -> dict[tuple[str, str], str]:
    """Load the independently declared calibration/held-out family assignment."""
    assignments: dict[tuple[str, str], str] = {}
    required = {"definition", "workload_uuid", "split", "stratum_kind", "stratum"}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError(
                f"{path}:{line_number} must contain exactly {sorted(required)}"
            )
        if row["split"] not in {"calibration", "heldout"}:
            raise ValueError(f"{path}:{line_number} has invalid split")
        if row["stratum_kind"] != stratum_kind:
            raise ValueError(
                f"{path}:{line_number} stratum_kind must be {stratum_kind!r}"
            )
        if not all(
            isinstance(row[key], str) and row[key].strip()
            for key in ("definition", "workload_uuid", "stratum")
        ):
            raise ValueError(f"{path}:{line_number} has invalid identity or stratum")
        key = (row["definition"], row["workload_uuid"])
        if key in assignments:
            raise ValueError(
                f"{path}:{line_number} duplicates stratum identity {key!r}"
            )
        assignments[key] = f"{row['split']}:{row['stratum']}"
    if not assignments:
        raise ValueError(f"{path} contains no stratum assignments")
    return assignments


def _ratio_summary(ratios: list[float]) -> dict[str, float | int]:
    if not ratios:
        return {"count": 0}
    ordered = sorted(ratios)

    def percentile(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p10": percentile(0.10),
        "median": percentile(0.50),
        "p90": percentile(0.90),
        "max": ordered[-1],
    }


def _assess_measurements(
    artifacts: dict[tuple[str, str], object],
    rows: list[dict[str, Any]],
) -> tuple[
    int, list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]
]:
    """Return matched count, contradictions, missing artifacts, and ratios."""
    contradictions: list[dict[str, object]] = []
    unmatched: list[dict[str, object]] = []
    ratios: list[dict[str, object]] = []
    matched = 0
    for row in rows:
        key = (row["definition"], row["workload_uuid"])
        artifact = artifacts.get(key)
        result = row["provider_result"]
        assert isinstance(result, PerformanceProviderResult)
        if artifact is None:
            unmatched.append(
                {
                    "definition": key[0],
                    "workload_uuid": key[1],
                    "provider_name": result.provider_name,
                    "provider_revision": result.provider_revision,
                    "raw_evidence_ref": result.raw_evidence_ref,
                    "raw_evidence_sha256": result.raw_evidence_sha256,
                    "reason": "missing_v5_bound_artifact",
                }
            )
            continue
        if result.target_architecture != artifact.hardware_model.architecture:
            raise ValueError(f"provider architecture does not match {key}")
        matched += 1
        assert result.measured_latency_ms is not None
        ratio = artifact.t_sol_floor_ms / result.measured_latency_ms
        ratios.append(
            {
                "definition": artifact.definition,
                "workload_uuid": str(artifact.workload_uuid),
                "ratio": ratio,
            }
        )
        if ratio > 1.0:
            contradictions.append(
                {
                    "definition": artifact.definition,
                    "workload_uuid": str(artifact.workload_uuid),
                    "t_sol_floor_ms": artifact.t_sol_floor_ms,
                    "fastest_known_ms": result.measured_latency_ms,
                    "ratio": ratio,
                    "provider_name": result.provider_name,
                    "provider_revision": result.provider_revision,
                    "raw_evidence_ref": result.raw_evidence_ref,
                    "raw_evidence_sha256": result.raw_evidence_sha256,
                }
            )
    return matched, contradictions, unmatched, ratios


def _validate_stratified_split(
    *,
    held_out_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    assignments: dict[tuple[str, str], str],
    stratum_kind: str,
) -> dict[str, object]:
    held_out_keys = {(row["definition"], row["workload_uuid"]) for row in held_out_rows}
    calibration_keys = {
        (row["definition"], row["workload_uuid"]) for row in calibration_rows
    }
    overlap = held_out_keys & calibration_keys
    if overlap:
        raise ValueError("calibration and held-out measurements overlap by workload")
    missing = (held_out_keys | calibration_keys) - set(assignments)
    if missing:
        raise ValueError("measurement lacks a calibration/held-out stratum assignment")
    held_out_strata = {assignments[key] for key in held_out_keys}
    calibration_strata = {assignments[key] for key in calibration_keys}
    if any(not assignment.startswith("heldout:") for assignment in held_out_strata):
        raise ValueError("held-out measurement has a non-heldout stratum assignment")
    if any(
        not assignment.startswith("calibration:") for assignment in calibration_strata
    ):
        raise ValueError(
            "calibration measurement has a non-calibration stratum assignment"
        )
    held_out_groups = {
        assignment.removeprefix("heldout:") for assignment in held_out_strata
    }
    calibration_groups = {
        assignment.removeprefix("calibration:") for assignment in calibration_strata
    }
    shared_groups = held_out_groups & calibration_groups
    if shared_groups:
        raise ValueError(
            f"{stratum_kind} strata leak across calibration and held-out splits"
        )
    return {
        "status": "validated",
        "stratum_kind": stratum_kind,
        "calibration_measurement_count": len(calibration_rows),
        "held_out_measurement_count": len(held_out_rows),
        "calibration_strata": sorted(calibration_groups),
        "held_out_strata": sorted(held_out_groups),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bounds-dir", required=True, type=Path)
    parser.add_argument("--held-out-jsonl", required=True, type=Path)
    parser.add_argument("--calibration-jsonl", type=Path)
    parser.add_argument("--strata-jsonl", type=Path)
    parser.add_argument("--stratum-kind", choices=("problem_family", "shape_family"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--require-profiler-trace",
        action="store_true",
        help="Require each held-out provider measurement to bind a profiler trace.",
    )
    parser.add_argument("--fail-on-contradiction", action="store_true")
    args = parser.parse_args()

    artifacts = _load_bound_artifacts(args.bounds_dir)
    if any(value is not None for value in (args.calibration_jsonl, args.strata_jsonl)):
        if not all(
            value is not None
            for value in (args.calibration_jsonl, args.strata_jsonl, args.stratum_kind)
        ):
            raise ValueError(
                "calibration-jsonl, strata-jsonl, and stratum-kind must be supplied together"
            )
    held_out_rows = _measurement_rows(
        args.held_out_jsonl, require_profiler_trace=args.require_profiler_trace
    )
    matched, contradictions, unmatched, held_out_ratios = _assess_measurements(
        artifacts, held_out_rows
    )
    calibration_payload: dict[str, object] | None = None
    stratification: dict[str, object] = {"status": "not_requested"}
    calibration_contradictions: list[dict[str, object]] = []
    calibration_unmatched: list[dict[str, object]] = []
    if args.calibration_jsonl is not None:
        calibration_rows = _measurement_rows(
            args.calibration_jsonl, require_profiler_trace=args.require_profiler_trace
        )
        (
            calibration_matched,
            calibration_contradictions,
            calibration_unmatched,
            calibration_ratios,
        ) = _assess_measurements(artifacts, calibration_rows)
        assert args.strata_jsonl is not None and args.stratum_kind is not None
        stratification = _validate_stratified_split(
            held_out_rows=held_out_rows,
            calibration_rows=calibration_rows,
            assignments=_strata_rows(args.strata_jsonl, stratum_kind=args.stratum_kind),
            stratum_kind=args.stratum_kind,
        )
        calibration_payload = {
            "matched_measurements": calibration_matched,
            "unmatched_measurement_count": len(calibration_unmatched),
            "contradiction_count": len(calibration_contradictions),
            "floor_to_fastest_known_ratio": _ratio_summary(
                [float(row["ratio"]) for row in calibration_ratios]
            ),
        }
    payload: dict[str, object] = {
        "schema_version": "sol_execbench.amd_sol_heldout_contradictions.v1",
        "bounds_dir": str(args.bounds_dir),
        "held_out_jsonl": str(args.held_out_jsonl),
        "matched_measurements": matched,
        "unmatched_measurement_count": len(unmatched),
        "contradiction_count": len(contradictions),
        "floor_to_fastest_known_ratio": _ratio_summary(
            [float(row["ratio"]) for row in held_out_ratios]
        ),
        "stratification": stratification,
        "calibration": calibration_payload,
        "publication_eligible": (
            not contradictions
            and not unmatched
            and not calibration_contradictions
            and not calibration_unmatched
        ),
        "contradictions": sorted(
            contradictions,
            key=lambda item: (-float(item["ratio"]), str(item["workload_uuid"])),
        ),
        "unmatched_measurements": sorted(
            unmatched,
            key=lambda item: (str(item["definition"]), str(item["workload_uuid"])),
        ),
    }
    payload["checksum_sha256"] = _sha256_payload(payload)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({"contradictions": len(contradictions), "output": str(args.output)})
    )
    rejected = (
        contradictions
        or unmatched
        or calibration_contradictions
        or calibration_unmatched
    )
    return 1 if rejected and args.fail_on_contradiction else 0


if __name__ == "__main__":
    raise SystemExit(main())
