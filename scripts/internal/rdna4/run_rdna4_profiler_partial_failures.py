# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Classify partial RDNA4 profiler-backed timing failures."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sol_execbench.core.dataset.inventory import build_dataset_inventory
from sol_execbench.core.dataset.profiler_timing_coverage import (
    ProfilerTimingCoverageReport,
    build_profiler_timing_coverage_report,
)
from sol_execbench.core.dataset.readiness import classify_rocm_readiness
from sol_execbench.core.text_utils import markdown_table_cell as _markdown_cell

DEFAULT_DATASET_ROOT = Path("data/SOL-ExecBench/benchmark")
DEFAULT_OUTPUT_DIR = Path("out/rdna4-profiler-partial-failure-classification")
DEFAULT_TIMING_EVIDENCE_DIRS = (
    Path("out/rdna4-profiler-sharded-closure-l1026-20260608/timing"),
    Path("out/rdna4-profiler-workload-aggregate-20260608-v2/timing"),
    Path("out/rdna4-profiler-backed-timing-full-20260608/timing"),
    Path("out/rdna4-timing-evidence/timing"),
)
DEFAULT_EXPECTED_PROBLEM_DENOMINATOR = 235
CLAIM_BOUNDARY = (
    "Classification only. Partial profiler-backed targets do not count as "
    "full profiler-backed timing until every expected workload passes with "
    "usable rocprofv3 kernel activity evidence."
)
OOM_LOG_MARKERS = (
    "HIP out of memory",
    "out of memory",
    "Tried to allocate",
)
TIMEOUT_LOG_MARKERS = (
    "TimeoutExpired",
    "timed out",
    "TIMEOUT",
)


def run_classification(
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    timing_evidence_dirs: Sequence[Path] = DEFAULT_TIMING_EVIDENCE_DIRS,
    expected_problem_denominator: int | None = DEFAULT_EXPECTED_PROBLEM_DENOMINATOR,
) -> int:
    """Build and write partial profiler failure classification artifacts."""
    inventory = build_dataset_inventory(dataset_root)
    readiness = classify_rocm_readiness(inventory, dataset_root=dataset_root)
    coverage = build_profiler_timing_coverage_report(
        readiness,
        dataset_root=dataset_root,
        timing_evidence_dirs=tuple(timing_evidence_dirs),
        expected_problem_denominator=expected_problem_denominator,
    )
    report = build_partial_failure_classification(
        coverage,
        timing_evidence_dirs=timing_evidence_dirs,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "partial-failure-classification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "partial-failure-classification.md").write_text(
        render_partial_failure_classification_markdown(report),
        encoding="utf-8",
    )
    _write_decision_lists(output_dir, report)
    return 0


def build_partial_failure_classification(
    coverage: ProfilerTimingCoverageReport,
    *,
    timing_evidence_dirs: Sequence[Path] = (),
) -> dict[str, Any]:
    """Return a deterministic ledger for partial profiler-backed targets."""
    rows: list[dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    for problem in coverage.problems:
        if problem.status not in {
            "partial_profiler_backed",
            "reference_oom_blocked",
        }:
            continue
        evidence = problem.evidence
        raw_payload = _load_optional_json(Path(evidence.path)) if evidence else None
        source_workloads = _source_workloads(raw_payload)
        trace_counts = _trace_counts(evidence, source_workloads)
        failure_details = failure_trace_details(raw_payload, source_workloads)
        attempted_workload_count = (
            len(source_workloads)
            if source_workloads
            else (evidence.profiled_workload_count if evidence is not None else None)
        )
        failure_mode = classify_failure_mode(
            trace_counts=trace_counts,
            profiled_workload_count=attempted_workload_count,
            expected_workload_count=(
                evidence.expected_workload_count if evidence is not None else None
            ),
            kernel_activity_rows=(
                evidence.kernel_activity_rows if evidence is not None else 0
            ),
        )
        blocker = blocker_class(
            trace_counts=trace_counts,
            failure_details=failure_details,
        )
        decision = closure_decision(failure_mode, blocker_class=blocker)
        row = {
            "problem_id": problem.problem_id,
            "category": problem.category,
            "problem_path": problem.problem_path,
            "workload_count": problem.workload_count,
            "evidence_path": evidence.path if evidence is not None else None,
            "kernel_activity_rows": (
                evidence.kernel_activity_rows if evidence is not None else 0
            ),
            "profiled_workload_count": (
                evidence.profiled_workload_count if evidence is not None else None
            ),
            "attempted_workload_count": attempted_workload_count,
            "expected_workload_count": (
                evidence.expected_workload_count if evidence is not None else None
            ),
            "trace_status_counts": trace_counts,
            "failure_status_counts": _failure_status_counts(trace_counts),
            "failure_details": failure_details,
            "failure_mode": failure_mode,
            "blocker_class": blocker,
            "closure_decision": decision,
            "failure_reason": _failure_reason(evidence),
        }
        rows.append(row)
        failure_counts[failure_mode] += 1
        decision_counts[row["closure_decision"]] += 1
    blocker_counts = Counter(row["blocker_class"] for row in rows)
    rows.sort(key=lambda item: (item["closure_decision"], item["problem_id"]))
    return {
        "schema_version": "sol_execbench.rdna4_partial_profiler_failures.v1",
        "claim_boundary": CLAIM_BOUNDARY,
        "coverage_summary": {
            "problem_denominator": coverage.totals.problem_denominator,
            "expected_problem_denominator": coverage.expected_problem_denominator,
            "profiler_backed_problems": coverage.totals.profiler_backed_problems,
            "partial_profiler_backed_problems": (
                coverage.totals.partial_profiler_backed_problems
            ),
            "profiler_blocked_problems": coverage.totals.profiler_blocked_problems,
            "fallback_timing_problems": coverage.totals.fallback_timing_problems,
            "readiness_blocked_problems": (coverage.totals.readiness_blocked_problems),
        },
        "partial_target_count": len(rows),
        "failure_mode_counts": dict(sorted(failure_counts.items())),
        "blocker_class_counts": dict(sorted(blocker_counts.items())),
        "closure_decision_counts": dict(sorted(decision_counts.items())),
        "timing_evidence_dirs": [
            Path(path).as_posix() for path in timing_evidence_dirs
        ],
        "targets": rows,
    }


def classify_failure_mode(
    *,
    trace_counts: dict[str, int],
    profiled_workload_count: int | None,
    expected_workload_count: int | None,
    kernel_activity_rows: int,
) -> str:
    """Classify one partial target from normalized status counters."""
    expected = expected_workload_count or 0
    profiled = profiled_workload_count or 0
    if expected > 0 and profiled < expected:
        return "incomplete_workload_coverage"
    if kernel_activity_rows <= 0:
        return "profiler_evidence_gap"
    failures = _failure_status_counts(trace_counts)
    failure_statuses = set(failures)
    if failure_statuses == {"TIMEOUT"}:
        return "timeout_only"
    if "TIMEOUT" in failure_statuses:
        return "mixed_timeout_failure"
    if failure_statuses == {"INVALID_REFERENCE"}:
        return "invalid_reference_only"
    if failure_statuses == {"RUNTIME_ERROR"}:
        return "runtime_error_only"
    if {"INVALID_REFERENCE", "RUNTIME_ERROR"} & failure_statuses:
        return "mixed_correctness_runtime"
    if "PROFILER_BLOCKED" in failure_statuses:
        return "profiler_evidence_gap"
    if failure_statuses:
        return "other_workload_failure"
    return "profiler_evidence_gap"


def blocker_class(
    *,
    trace_counts: dict[str, int],
    failure_details: Sequence[dict[str, Any]],
) -> str:
    """Classify the actionable blocker behind a partial profiler target."""
    failure_statuses = _failure_status_counts(trace_counts)
    if any(detail.get("timeout_detected") is True for detail in failure_details) or (
        "TIMEOUT" in failure_statuses
    ):
        return "timeout"
    if any(detail.get("oom_detected") is True for detail in failure_details):
        if "PROFILER_BLOCKED" in _failure_status_counts(trace_counts):
            if any(detail.get("phase") == "correctness" for detail in failure_details):
                return "profiler_closure_oom_blocked"
            return "memory_oom_with_profiler_gap"
        oom_phases = {
            detail.get("phase")
            for detail in failure_details
            if detail.get("oom_detected") is True
        }
        if "correctness" in oom_phases:
            return "profiler_closure_oom_blocked"
        if "gen_inputs" in oom_phases:
            return "gen_inputs_oom_blocked"
        if "user_function" in oom_phases:
            return "user_solution_oom"
        return "reference_oom_blocked"
    if "PROFILER_BLOCKED" in _failure_status_counts(trace_counts):
        return "profiler_evidence_gap"
    return "non_oom_workload_failure"


def closure_decision(failure_mode: str, *, blocker_class: str | None = None) -> str:
    """Map failure mode to a closure decision."""
    if blocker_class in {
        "reference_oom_blocked",
        "gen_inputs_oom_blocked",
        "user_solution_oom",
        "profiler_closure_oom_blocked",
        "memory_oom_with_profiler_gap",
        "reference_oom_with_profiler_gap",
    }:
        return "blocked_on_reference_oom"
    if blocker_class == "timeout":
        return "blocked_on_timeout"
    if failure_mode == "invalid_reference_only":
        return "blocked_on_correctness"
    if failure_mode == "runtime_error_only":
        return "blocked_on_runtime"
    if failure_mode in {"mixed_correctness_runtime", "other_workload_failure"}:
        return "blocked_on_mixed_failures"
    if failure_mode == "incomplete_workload_coverage":
        return "complete_missing_workload_slices"
    return "inspect_profiler_evidence_gap"


def render_partial_failure_classification_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown classification report."""
    lines = [
        "# RDNA4 Partial Profiler Failure Classification",
        "",
        f"- Classified targets: `{report['partial_target_count']}`",
        "",
        "## Closure Decisions",
        "",
        "| Decision | Problems |",
        "| --- | ---: |",
    ]
    for decision, count in report["closure_decision_counts"].items():
        lines.append(f"| {decision} | {count} |")
    lines.extend(
        [
            "",
            "## Failure Modes",
            "",
            "| Failure Mode | Problems |",
            "| --- | ---: |",
        ]
    )
    for mode, count in report["failure_mode_counts"].items():
        lines.append(f"| {mode} | {count} |")
    lines.extend(
        [
            "",
            "## Blocker Classes",
            "",
            "| Blocker Class | Problems |",
            "| --- | ---: |",
        ]
    )
    for blocker, count in report.get("blocker_class_counts", {}).items():
        lines.append(f"| {blocker} | {count} |")
    lines.extend(
        [
            "",
            "## Targets",
            "",
            "| Problem | Failure Mode | Blocker Class | Decision | Status Counts |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for target in report["targets"]:
        lines.append(
            "| "
            f"{target['problem_id']} | "
            f"{target['failure_mode']} | "
            f"{target.get('blocker_class', '')} | "
            f"{target['closure_decision']} | "
            f"{_markdown_cell(json.dumps(target['trace_status_counts'], sort_keys=True))} |"
        )
    lines.extend(["", "## Claim Boundary", "", report["claim_boundary"], ""])
    return "\n".join(lines)


def _write_decision_lists(output_dir: Path, report: dict[str, Any]) -> None:
    for existing in output_dir.glob("*.txt"):
        existing.unlink()
    by_decision: dict[str, list[str]] = defaultdict(list)
    for target in report["targets"]:
        by_decision[target["closure_decision"]].append(target["problem_id"])
    for decision, problem_ids in by_decision.items():
        (output_dir / f"{decision}.txt").write_text(
            "".join(f"{problem_id}\n" for problem_id in sorted(problem_ids)),
            encoding="utf-8",
        )


def _failure_status_counts(trace_counts: dict[str, int]) -> dict[str, int]:
    return {
        status: count
        for status, count in sorted(trace_counts.items())
        if status != "PASSED" and count > 0
    }


def failure_trace_details(
    payload: dict[str, Any] | None,
    source_workloads: Sequence[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Extract non-PASSED workload failure details from timing sidecars."""
    if source_workloads:
        return _source_workload_failure_details(source_workloads)
    return _payload_failure_details(payload)


def _source_workload_failure_details(
    source_workloads: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for workload in source_workloads:
        replacement_path = workload.get("replacement_path")
        workload_index = _int_or_none(workload.get("workload_index"))
        slice_payload = (
            _load_optional_json(Path(replacement_path))
            if isinstance(replacement_path, str)
            else None
        )
        slice_details = _payload_failure_details(
            slice_payload,
            default_workload_index=workload_index,
        )
        if slice_details:
            details.extend(slice_details)
            continue
        status = workload.get("status")
        if status == "profiler_blocked":
            details.append(
                {
                    "workload_index": workload_index,
                    "workload_uuid": workload.get("workload_uuid"),
                    "status": "PROFILER_BLOCKED",
                    "phase": "profiler",
                    "oom_detected": False,
                    "timeout_detected": False,
                    "log_head": str(workload.get("failure_reason") or ""),
                }
            )
    return details


def _payload_failure_details(
    payload: dict[str, Any] | None,
    *,
    default_workload_index: int | None = None,
) -> list[dict[str, Any]]:
    stdout = payload.get("stdout") if isinstance(payload, dict) else None
    details: list[dict[str, Any]] = []
    if isinstance(stdout, str):
        for line_index, line in enumerate(stdout.splitlines()):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            evaluation = record.get("evaluation")
            if not isinstance(evaluation, dict):
                continue
            status = evaluation.get("status")
            if status == "PASSED" or not isinstance(status, str):
                continue
            workload = record.get("workload")
            log = str(evaluation.get("log") or "")
            details.append(
                {
                    "workload_index": (
                        default_workload_index
                        if default_workload_index is not None
                        else line_index
                    ),
                    "workload_uuid": (
                        workload.get("uuid") if isinstance(workload, dict) else None
                    ),
                    "axes": workload.get("axes")
                    if isinstance(workload, dict)
                    else None,
                    "status": status,
                    "phase": _failure_phase(log),
                    "oom_detected": _is_oom_log(log),
                    "timeout_detected": _is_timeout_log(log) or status == "TIMEOUT",
                    "log_head": log.splitlines()[0][:240] if log else "",
                }
            )
    stderr = payload.get("stderr") if isinstance(payload, dict) else None
    if isinstance(stderr, str) and _is_oom_log(stderr):
        details.append(
            {
                "workload_index": (
                    default_workload_index
                    if default_workload_index is not None
                    else len(details)
                ),
                "workload_uuid": None,
                "axes": None,
                "status": "PROFILER_BLOCKED",
                "phase": _failure_phase(stderr),
                "oom_detected": True,
                "timeout_detected": False,
                "log_head": _stderr_log_head(stderr),
            }
        )
    return details


def _failure_phase(log: str) -> str:
    if log.startswith("gen_inputs failed:"):
        return "gen_inputs"
    if log.startswith("Reference run() failed:"):
        return "reference"
    if log.startswith("User function failed:"):
        return "user_function"
    if "compute_error_stats" in log or "correctness.py" in log:
        return "correctness"
    return "unknown"


def _is_oom_log(log: str) -> bool:
    return any(marker in log for marker in OOM_LOG_MARKERS)


def _is_timeout_log(log: str) -> bool:
    return any(marker in log for marker in TIMEOUT_LOG_MARKERS)


def _stderr_log_head(stderr: str) -> str:
    for line in stderr.splitlines():
        if _is_oom_log(line):
            return line[:240]
    return stderr.splitlines()[0][:240] if stderr.splitlines() else ""


def _trace_counts(
    evidence: Any, source_workloads: list[dict[str, Any]]
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    if evidence is not None:
        counts.update(evidence.trace_status_counts)
    for workload in source_workloads:
        status = workload.get("status")
        if status == "profiler_blocked":
            counts["PROFILER_BLOCKED"] += 1
    return dict(sorted(counts.items()))


def _source_workloads(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    evidence = payload.get("evidence") if isinstance(payload, dict) else None
    if not isinstance(evidence, dict):
        return []
    source = evidence.get("source_workloads")
    if not isinstance(source, list):
        return []
    return [item for item in source if isinstance(item, dict)]


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _failure_reason(evidence: Any) -> str:
    if evidence is None:
        return ""
    if evidence.replacement_failure_reason:
        return evidence.replacement_failure_reason
    if evidence.fallback_reason:
        return evidence.fallback_reason
    return ""


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--timing-evidence-dir",
        action="append",
        type=Path,
        default=None,
        help="Problem-level timing sidecar root; may be provided more than once.",
    )
    parser.add_argument(
        "--expected-problem-denominator",
        type=int,
        default=DEFAULT_EXPECTED_PROBLEM_DENOMINATOR,
    )
    parser.add_argument(
        "--no-expected-problem-denominator",
        action="store_true",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    expected = (
        None
        if args.no_expected_problem_denominator
        else args.expected_problem_denominator
    )
    timing_dirs = (
        tuple(args.timing_evidence_dir)
        if args.timing_evidence_dir
        else DEFAULT_TIMING_EVIDENCE_DIRS
    )
    try:
        return run_classification(
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            timing_evidence_dirs=timing_dirs,
            expected_problem_denominator=expected,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
