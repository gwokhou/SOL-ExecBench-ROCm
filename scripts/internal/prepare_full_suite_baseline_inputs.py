#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Canonicalize a full-suite selected baseline run into release inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from sol_execbench.core.integrity.checksums import sha256_file


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _eligible_keys(coverage: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (str(row["definition"]), str(row["workload_uuid"]))
        for row in coverage["workloads"]
        if not row["blocker_codes"]
    }


def _authority_key_sets(
    authority: dict[str, Any],
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    workloads = authority.get("workloads")
    if not isinstance(workloads, list):
        raise ValueError("authority input must contain a workloads list")
    eligible: set[tuple[str, str]] = set()
    blocked: set[tuple[str, str]] = set()
    all_keys: set[tuple[str, str]] = set()
    for row in workloads:
        if not isinstance(row, dict):
            raise ValueError("authority input workloads must contain objects")
        definition = row.get("definition")
        workload_uuid = row.get("workload_uuid")
        if not isinstance(definition, str) or not isinstance(workload_uuid, str):
            raise ValueError("authority workloads require definition and workload_uuid")
        key = (definition, workload_uuid)
        if key in all_keys:
            raise ValueError(f"duplicate authority workload: {key!r}")
        all_keys.add(key)
        blockers = row.get("official_blockers")
        if not isinstance(blockers, list) or not all(
            isinstance(code, str) for code in blockers
        ):
            raise ValueError(
                "authority workload official_blockers must be a string list"
            )
        if blockers:
            blocked.add(key)
            continue
        eligible.add(key)
    return eligible, blocked


def _trace_records(run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("traces.json")):
        text = path.read_text(encoding="utf-8")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = [json.loads(line) for line in text.splitlines() if line.strip()]
        if not isinstance(payload, list) or not all(
            isinstance(row, dict) for row in payload
        ):
            raise ValueError(f"{path} must contain a trace record list or JSONL")
        records.extend(payload)
    return records


def _key(record: dict[str, Any]) -> tuple[str, str]:
    workload = record.get("workload")
    definition = record.get("definition")
    uuid = workload.get("uuid") if isinstance(workload, dict) else None
    if not isinstance(definition, str) or not isinstance(uuid, str):
        raise ValueError("trace record is missing definition/workload UUID")
    return definition, uuid


def _validate_trace(record: dict[str, Any]) -> dict[str, Any]:
    evaluation = record.get("evaluation")
    if not isinstance(evaluation, dict) or evaluation.get("status") != "PASSED":
        raise ValueError(f"baseline trace {_key(record)!r} did not pass")
    performance = evaluation.get("performance")
    latency = performance.get("latency_ms") if isinstance(performance, dict) else None
    if not isinstance(latency, (int, float)) or latency <= 0.0:
        raise ValueError(f"baseline trace {_key(record)!r} has no positive latency")
    if "Clocks locked: yes" not in str(evaluation.get("log", "")):
        raise ValueError(f"baseline trace {_key(record)!r} lacks clock-lock evidence")
    environment = evaluation.get("environment")
    if not isinstance(environment, dict):
        raise ValueError(f"baseline trace {_key(record)!r} lacks environment evidence")
    return environment


def _solution_portfolio(run_dir: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    definitions = sorted({str(record["definition"]) for record in records})
    entries: list[dict[str, str]] = []
    for definition in definitions:
        candidates = sorted(
            path
            for path in run_dir.rglob("solution.json")
            if path.parent.name == definition
        )
        if len(candidates) != 1:
            raise ValueError(
                f"expected one emitted solution for {definition}; got {len(candidates)}"
            )
        path = candidates[0]
        entries.append(
            {
                "definition": definition,
                "relative_path": path.relative_to(run_dir).as_posix(),
                "sha256": sha256_file(path),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": "sol_execbench.baseline_solution_portfolio.v1",
        "solution_kind": "pytorch_rocm_reference_portfolio",
        "entries": entries,
    }
    payload["payload_sha256"] = _canonical_digest(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("coverage", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--authority-input",
        type=Path,
        help=(
            "Authority input whose workloads with empty official_blockers define "
            "the expected trace denominator. Defaults to statically eligible coverage."
        ),
    )
    parser.add_argument(
        "--solution-run-dir",
        type=Path,
        help=(
            "Directory containing the emitted per-definition solution.json files. "
            "Defaults to run_dir; useful when an independent rerun only retains traces."
        ),
    )
    args = parser.parse_args()
    coverage = _json(args.coverage)
    if not isinstance(coverage, dict):
        raise ValueError("coverage must be an object")
    if args.authority_input is None:
        expected = _eligible_keys(coverage)
        explicitly_blocked: set[tuple[str, str]] = set()
    else:
        authority = _json(args.authority_input)
        if not isinstance(authority, dict):
            raise ValueError("authority input must be an object")
        expected, explicitly_blocked = _authority_key_sets(authority)
    records = _trace_records(args.run_dir)
    by_key = {_key(record): record for record in records}
    if len(by_key) != len(records):
        raise ValueError("baseline run contains duplicate trace workloads")
    missing = sorted(expected - set(by_key))
    unknown_extra = sorted(set(by_key) - expected - explicitly_blocked)
    if missing or unknown_extra:
        raise ValueError(
            "baseline trace coverage mismatch: "
            f"missing={missing[:3]}, extra={unknown_extra[:3]}"
        )
    selected_records = [by_key[key] for key in sorted(expected)]
    environments = {
        _canonical_digest(_validate_trace(record)): _validate_trace(record)
        for record in selected_records
    }
    if len(environments) != 1:
        raise ValueError("baseline trace environment fingerprint is not uniform")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "baseline-trace.jsonl"
    trace_path.write_text(
        "".join(
            json.dumps(by_key[key], sort_keys=True, separators=(",", ":")) + "\n"
            for key in sorted(expected)
        ),
        encoding="utf-8",
    )
    portfolio = _solution_portfolio(
        args.solution_run_dir or args.run_dir, selected_records
    )
    portfolio_path = output_dir / "baseline-solution-portfolio.json"
    portfolio_path.write_text(
        json.dumps(portfolio, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    environment = next(iter(environments.values()))
    inputs = {
        "trace_ref": trace_path.name,
        "trace_sha256": sha256_file(trace_path),
        "solution_ref": portfolio_path.name,
        "solution_sha256": portfolio["payload_sha256"],
        "environment_fingerprint": _canonical_digest(environment),
        "clock_policy": "rocm_smi_stable_peak_lock_with_post_reset",
        "timing_policy": "latency_ms",
        "compiler_build_id": "hipcc-7.1.52802-26aae437f6",
        "eligible_workloads": len(expected),
    }
    (output_dir / "baseline-inputs.json").write_text(
        json.dumps(inputs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(inputs, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
