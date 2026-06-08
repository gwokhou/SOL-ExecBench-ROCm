# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Run a bounded RDNA4 profiler-backed timing smoke check."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sol_execbench.core.bench.rocm_profiler import (
    ROCPROFV3_EXECUTABLE,
    ProfilerRunner,
    collect_source_timing_evidence,
)
from sol_execbench.core.dataset.runner import build_cli_command

DEFAULT_PROBLEM_DIR = Path("examples/triton/rmsnorm")
DEFAULT_OUTPUT_DIR = Path("out/rdna4-profiler-backed-timing-smoke")
DEFAULT_OUTPUT_FILE = "rdna4-triton-rmsnorm-timing"
CLAIM_BOUNDARY = (
    "Bounded RDNA4 profiler-backed timing smoke evidence only; not full paper "
    "validation, score authority, leaderboard readiness, or broader hardware "
    "validation."
)


def run_smoke(
    *,
    problem_dir: Path = DEFAULT_PROBLEM_DIR,
    solution_path: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    workload_limit: int = 1,
    timeout: int = 600,
    tool_version: str = "rocprofv3",
    gpu_architecture: str = "gfx1200",
    clock_locked: bool = True,
    allow_fallback: bool = False,
    rocprofv3_available: bool | None = None,
    runner: ProfilerRunner | None = None,
) -> int:
    """Run the smoke check and return a process-style status code."""
    if workload_limit <= 0:
        raise ValueError("workload_limit must be positive")

    solution_path = solution_path or problem_dir / "solution_triton.json"
    definition_path = problem_dir / "definition.json"
    workload_path = problem_dir / "workload.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    limited_workload = output_dir / "workload.smoke.jsonl"
    _write_limited_workload(workload_path, limited_workload, workload_limit)

    solution = _load_json(solution_path)
    languages = _solution_languages(solution)
    command = build_cli_command(
        definition_path=definition_path,
        workload_path=limited_workload,
        solution_path=solution_path,
        timeout=timeout,
    )
    available = (
        shutil.which(ROCPROFV3_EXECUTABLE) is not None
        if rocprofv3_available is None
        else rocprofv3_available
    )

    timing_dir = output_dir / "rocprofv3"
    result = collect_source_timing_evidence(
        application_command=command,
        languages=languages,
        output_directory=timing_dir,
        output_file=DEFAULT_OUTPUT_FILE,
        tool_version=tool_version,
        gpu_architecture=gpu_architecture,
        rocprofv3_available=available,
        runner=runner,
        trial_count=1,
        clock_locked=clock_locked,
    )
    timing_payload = result.to_dict()
    timing_path = output_dir / "timing.json"
    timing_path.write_text(json.dumps(timing_payload, indent=2, sort_keys=True) + "\n")

    summary = _build_summary(
        command=command,
        problem_dir=problem_dir,
        solution_path=solution_path,
        limited_workload=limited_workload,
        workload_limit=workload_limit,
        languages=languages,
        rocprofv3_available=available,
        timing_path=timing_path,
        timing_payload=timing_payload,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "summary.md").write_text(_render_summary_markdown(summary))
    return 0 if summary["profiler_collected"] or allow_fallback else 1


def _write_limited_workload(source: Path, destination: Path, limit: int) -> None:
    lines = [line for line in source.read_text(encoding="utf-8").splitlines() if line]
    if not lines:
        raise ValueError(f"workload file has no records: {source}")
    destination.write_text("\n".join(lines[:limit]) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _solution_languages(solution: dict[str, Any]) -> tuple[str, ...]:
    languages = solution.get("spec", {}).get("languages", ())
    if isinstance(languages, str):
        return (languages,)
    if isinstance(languages, Sequence):
        return tuple(str(language) for language in languages)
    return ()


def _build_summary(
    *,
    command: Sequence[str],
    problem_dir: Path,
    solution_path: Path,
    limited_workload: Path,
    workload_limit: int,
    languages: Sequence[str],
    rocprofv3_available: bool,
    timing_path: Path,
    timing_payload: dict[str, Any],
) -> dict[str, Any]:
    selection = timing_payload.get("selection") or {}
    policy = (selection.get("policy") or {}) if isinstance(selection, dict) else {}
    evidence = timing_payload.get("evidence") or {}
    return {
        "schema_version": "sol_execbench.rdna4_profiler_timing_smoke.v1",
        "status": "profiler_backed"
        if timing_payload.get("profiler_collected") is True
        else "fallback",
        "profiler_collected": timing_payload.get("profiler_collected") is True,
        "problem_dir": str(problem_dir),
        "solution_path": str(solution_path),
        "limited_workload": str(limited_workload),
        "workload_limit": workload_limit,
        "languages": list(languages),
        "rocprofv3_available": rocprofv3_available,
        "command": list(command),
        "timing_path": str(timing_path),
        "csv_path": timing_payload.get("csv_path"),
        "policy_backend": policy.get("backend"),
        "activity_domain": policy.get("activity_domain"),
        "fallback_reason": selection.get("reason")
        if isinstance(selection, dict)
        else None,
        "kernel_duration_ms": evidence.get("kernel_duration_ms")
        if isinstance(evidence, dict)
        else None,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# RDNA4 Profiler-Backed Timing Smoke",
        "",
        f"- Status: `{summary['status']}`",
        f"- Profiler collected: `{str(summary['profiler_collected']).lower()}`",
        f"- Policy backend: `{summary['policy_backend']}`",
        f"- Activity domain: `{summary['activity_domain']}`",
        f"- Workload limit: `{summary['workload_limit']}`",
        f"- Timing path: `{summary['timing_path']}`",
        f"- CSV path: `{summary['csv_path']}`",
        "",
        "## Claim Boundary",
        "",
        summary["claim_boundary"],
        "",
    ]
    if summary.get("fallback_reason"):
        lines.extend(["## Fallback Reason", "", str(summary["fallback_reason"]), ""])
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-dir", type=Path, default=DEFAULT_PROBLEM_DIR)
    parser.add_argument("--solution", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workload-limit", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--timing-tool-version", default="rocprofv3")
    parser.add_argument("--gpu-architecture", default="gfx1200")
    parser.add_argument(
        "--clock-locked", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Exit successfully even when profiler-backed timing was not collected.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_smoke(
            problem_dir=args.problem_dir,
            solution_path=args.solution,
            output_dir=args.output_dir,
            workload_limit=args.workload_limit,
            timeout=args.timeout,
            tool_version=args.timing_tool_version,
            gpu_architecture=args.gpu_architecture,
            clock_locked=args.clock_locked,
            allow_fallback=args.allow_fallback,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
