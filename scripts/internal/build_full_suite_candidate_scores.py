#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Derive AMD-native scores from a full-suite candidate trace and frozen bounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sol_execbench.core.scoring.amd_score import (
    AmdNativeScore,
    score_amd_native_trace_workload,
)
from sol_execbench.core.scoring.amd_score.report_inputs import parse_score_report_trace
from sol_execbench.core.scoring.amd_score.reports import write_amd_score_report
from sol_execbench.core.scoring.amd_sol import amd_sol_bound_from_dict
from sol_execbench.core.scoring.baseline_artifact import load_scoring_baseline_artifact


def _trace_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(payload, list) or not all(
        isinstance(row, dict) for row in payload
    ):
        raise ValueError(f"{path} must contain a JSON trace list or JSONL")
    return payload


def _authority_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("workloads") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("authority input must contain workloads")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("authority workload must be an object")
        definition = row.get("definition")
        uuid = row.get("workload_uuid")
        if not isinstance(definition, str) or not isinstance(uuid, str):
            raise ValueError("authority workload has invalid identity")
        result[(definition, uuid)] = row
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument(
        "--timing-evidence",
        type=Path,
        help="Timing evidence cited by scored rows; defaults to --trace.",
    )
    parser.add_argument("--authority-input", required=True, type=Path)
    parser.add_argument("--closure-dir", required=True, type=Path)
    parser.add_argument("--scoring-baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    baseline = load_scoring_baseline_artifact(args.scoring_baseline)
    authority = _authority_rows(args.authority_input)
    artifacts: dict[Path, Any] = {}
    scores: list[AmdNativeScore] = []
    for record in _trace_records(args.trace):
        trace = parse_score_report_trace(record)
        key = (trace.definition, trace.workload.uuid)
        row = authority.get(key)
        if row is None:
            raise ValueError(
                f"candidate trace workload is outside authority input: {key!r}"
            )
        if row.get("official_blockers"):
            continue
        bound_ref = row.get("bound_ref")
        bound_sha = row.get("bound_sha256")
        model_ref = row.get("hardware_model_ref")
        if not all(
            isinstance(value, str) and value
            for value in (bound_ref, bound_sha, model_ref)
        ):
            raise ValueError(
                f"candidate trace has no authoritative bound/model: {key!r}"
            )
        bound_path = args.closure_dir / bound_ref
        artifact = artifacts.get(bound_path)
        if artifact is None:
            artifact = amd_sol_bound_from_dict(
                json.loads(bound_path.read_text(encoding="utf-8"))
            )
            artifacts[bound_path] = artifact
        scores.append(
            score_amd_native_trace_workload(
                trace,
                artifact,
                trace_ref=str(args.trace),
                timing_evidence_ref=str(args.timing_evidence or args.trace),
                sol_bound_ref=bound_ref,
                baseline_ref=f"{baseline.source}#{key[0]}:{key[1]}",
                baseline_artifact=baseline,
                hardware_model_ref=model_ref,
            )
        )
    if len({(score.definition, score.workload_uuid) for score in scores}) != len(
        scores
    ):
        raise ValueError("candidate trace contains duplicate workload identities")
    write_amd_score_report(
        args.output,
        scores,
        problem_count=len({score.definition for score in scores}),
        baseline_entry_count=len(baseline.entries),
    )
    print(
        json.dumps({"scores": len(scores), "output": str(args.output)}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
