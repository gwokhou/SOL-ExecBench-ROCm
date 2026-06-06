#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Run bounded engineering-prerelease validation and write review artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

SCHEMA_VERSION = "sol_execbench.release_candidate_validation.v1"
DEFAULT_OUTPUT_DIR = Path("out/release_candidate_validation")
DEFAULT_LOG_TAIL_CHARS = 4000

DEFAULT_CPU_COMMAND = [
    "uv",
    "run",
    "pytest",
    "tests/sol_execbench/test_contract.py",
    "tests/sol_execbench/test_trust_summary.py",
    "tests/sol_execbench/test_trust_summary_script.py",
    "tests/sol_execbench/test_consistency_script.py",
    "tests/sol_execbench/test_claim_upgrade_script.py",
    "tests/sol_execbench/test_run_dataset_execution_closure.py",
    "-q",
]
DEFAULT_DOCTOR_COMMAND = ["uv", "run", "sol-execbench", "doctor", "--json"]
DEFAULT_ROCM_PYTEST_COMMAND = [
    "uv",
    "run",
    "pytest",
    "-m",
    "requires_rocm",
    "-q",
    "-rs",
]
DEFAULT_DOCKER_COMMAND = ["./scripts/run_docker.sh", "--dry-run"]

TOKEN_PATTERN = re.compile(
    r"(?ix)"
    r"("
    r"(?:[A-Z0-9_]*?(?:TOKEN|SECRET|PASSWORD|PASSWD|API[_-]?KEY|CREDENTIAL)[A-Z0-9_-]*?)"
    r"|authorization"
    r")"
    r"(\s*:\s*bearer\s+|\s*[:=]\s*)"
    r"([^\s'\"]+)"
)
_TOKEN_PREFIX_OVERLAP_CHARS = 512
_TOKEN_VALUE_DELIMITERS = set(" \t\r\n'\"")


@dataclass(frozen=True)
class ValidationResult:
    name: str
    command: list[str]
    status: str
    classification: str
    next_action: str
    duration_s: float
    returncode: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    artifact_paths: list[str] = field(default_factory=list)
    evidence: dict[str, object] = field(default_factory=dict)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[ValidationResult] = []
    if args.skip_cpu:
        results.append(
            _skipped_result(
                name="cpu_safe_validation",
                command=_command_from_args(args.cpu_command, DEFAULT_CPU_COMMAND),
                classification="diagnostic-only",
                next_action="Run without --skip-cpu before publishing a release candidate.",
            )
        )
    else:
        results.append(
            _run_check(
                name="cpu_safe_validation",
                command=_command_from_args(args.cpu_command, DEFAULT_CPU_COMMAND),
                failure_classification="blocking",
                failure_next_action="Fix the failing CPU-safe validation command before publishing.",
                log_tail_chars=args.log_tail_chars,
            )
        )

    if args.include_rocm_smoke:
        doctor_command = _command_from_args(
            args.rocm_doctor_command, DEFAULT_DOCTOR_COMMAND
        )
        rocm_pytest_command = _command_from_args(
            args.rocm_pytest_command,
            DEFAULT_ROCM_PYTEST_COMMAND,
        )
        doctor_result = _run_check(
            name="rocm_doctor",
            command=doctor_command,
            failure_classification="deferred",
            failure_next_action="Review ROCm runtime availability; this is optional smoke evidence for prerelease.",
            evidence=_clock_policy_evidence(),
            log_tail_chars=args.log_tail_chars,
        )
        results.append(doctor_result)
        results.append(
            _run_check(
                name="rocm_pytest_smoke",
                command=rocm_pytest_command,
                failure_classification=_optional_smoke_classification,
                failure_next_action="Run on a ROCm-capable host or keep this prerelease evidence marked deferred.",
                evidence=_clock_policy_evidence(),
                log_tail_chars=args.log_tail_chars,
            )
        )

    if args.include_docker_smoke:
        results.append(
            _run_check(
                name="docker_smoke",
                command=_command_from_args(args.docker_command, DEFAULT_DOCKER_COMMAND),
                failure_classification="deferred",
                failure_next_action="Review Docker/ROCm wrapper availability; this is optional container user-space evidence.",
                evidence={
                    "authority_boundary": (
                        "Docker smoke evidence is not native-host validation, "
                        "paper parity, score authority, or leaderboard authority."
                    )
                },
                log_tail_chars=args.log_tail_chars,
            )
        )

    if args.include_dataset_slice:
        if args.dataset_dir is None:
            raise SystemExit("--include-dataset-slice requires --dataset-dir")
        if args.dataset_limit is None or args.dataset_limit <= 0:
            raise SystemExit(
                "--include-dataset-slice requires positive --dataset-limit"
            )
        results.extend(_dataset_slice_results(args, output_dir, args.log_tail_chars))

    payload = _build_payload(results)
    json_path = output_dir / "release_candidate_validation.json"
    markdown_path = output_dir / "release_candidate_validation.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    return 1 if payload["overall_status"] == "blocking" else 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bounded SOL ExecBench ROCm engineering-prerelease validation.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--cpu-command", nargs="+", default=None)
    parser.add_argument("--include-rocm-smoke", action="store_true")
    parser.add_argument("--rocm-doctor-command", nargs="+", default=None)
    parser.add_argument("--rocm-pytest-command", nargs="+", default=None)
    parser.add_argument("--include-docker-smoke", action="store_true")
    parser.add_argument("--docker-command", nargs="+", default=None)
    parser.add_argument("--include-dataset-slice", action="store_true")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--dataset-limit", type=int, default=None)
    parser.add_argument("--dataset-command", nargs=argparse.REMAINDER, default=None)
    parser.add_argument("--trust-summary-command", nargs="+", default=None)
    parser.add_argument("--log-tail-chars", type=int, default=DEFAULT_LOG_TAIL_CHARS)
    return parser.parse_args(argv)


def _command_from_args(value: list[str] | None, default: list[str]) -> list[str]:
    return list(value) if value else list(default)


def _dataset_slice_results(
    args: argparse.Namespace,
    output_dir: Path,
    log_tail_chars: int,
) -> list[ValidationResult]:
    closure_path = output_dir / "execution_closure.json"
    trust_json = output_dir / "trust_summary.json"
    trust_md = output_dir / "trust_summary.md"
    default_dataset_command = [
        "uv",
        "run",
        "scripts/run_dataset.py",
        str(args.dataset_dir),
        "--limit",
        str(args.dataset_limit),
        "--rerun",
        "--execution-closure",
        str(closure_path),
    ]
    dataset_command = _command_from_args(args.dataset_command, default_dataset_command)
    if args.dataset_command:
        _validate_dataset_command_override(dataset_command, args.dataset_limit)
    trust_command = _command_from_args(
        args.trust_summary_command,
        [
            "uv",
            "run",
            "scripts/report_trust_summary.py",
            "--execution-closure",
            str(closure_path),
            "--json-out",
            str(trust_json),
            "--markdown-out",
            str(trust_md),
        ],
    )
    dataset_result = _run_check(
        name="bounded_dataset_slice",
        command=dataset_command,
        failure_classification="deferred",
        failure_next_action="Review dataset assets/readiness; keep this prerelease evidence marked deferred until resolved.",
        artifact_paths=[str(closure_path)],
        evidence={
            "dataset_dir": str(args.dataset_dir),
            "dataset_limit": args.dataset_limit,
            "paper_scale_boundary": "This bounded slice is not full 235-problem paper validation.",
        },
        log_tail_chars=log_tail_chars,
    )
    if dataset_result.status == "passed" and closure_path.exists():
        trust_result = _run_check(
            name="trust_summary",
            command=trust_command,
            failure_classification="diagnostic-only",
            failure_next_action="Generate trust summary after required input sidecars exist.",
            artifact_paths=[str(trust_json), str(trust_md)],
            log_tail_chars=log_tail_chars,
        )
    else:
        trust_result = _skipped_result(
            name="trust_summary",
            command=trust_command,
            classification="diagnostic-only",
            next_action="Generate trust summary after execution_closure.json exists.",
        )
    return [dataset_result, trust_result]


def _validate_dataset_command_override(command: list[str], dataset_limit: int) -> None:
    if not _command_has_option_value(command, "--limit", str(dataset_limit)):
        raise SystemExit("--dataset-command must include the requested bounded --limit")
    if "--rerun" not in command:
        raise SystemExit("--dataset-command must include --rerun")
    if "--execution-closure" not in command:
        raise SystemExit("--dataset-command must write --execution-closure")


def _command_has_option_value(command: list[str], option: str, value: str) -> bool:
    for index, part in enumerate(command):
        if part == option and index + 1 < len(command) and command[index + 1] == value:
            return True
        if part == f"{option}={value}":
            return True
    return False


def _run_check(
    *,
    name: str,
    command: list[str],
    failure_classification: str | Callable[["_CommandOutput"], str],
    failure_next_action: str,
    artifact_paths: list[str] | None = None,
    evidence: dict[str, object] | None = None,
    log_tail_chars: int = DEFAULT_LOG_TAIL_CHARS,
) -> ValidationResult:
    started = time.monotonic()
    stdout_path = _temporary_stream_path(name, "stdout")
    stderr_path = _temporary_stream_path(name, "stderr")
    try:
        try:
            completed = _run_command_to_files(command, stdout_path, stderr_path)
        except FileNotFoundError as exc:
            duration_s = time.monotonic() - started
            return ValidationResult(
                name=name,
                command=command,
                status="unavailable",
                classification="deferred",
                next_action=f"Install or expose required command before collecting this evidence: {exc.filename}",
                duration_s=round(duration_s, 3),
                returncode=None,
                stderr_tail=_tail(str(exc), log_tail_chars),
                artifact_paths=artifact_paths or [],
                evidence=evidence or {},
            )

        duration_s = time.monotonic() - started
        stdout_tail = _tail_file(stdout_path, log_tail_chars)
        stderr_tail = _tail_file(stderr_path, log_tail_chars)
        if completed.returncode == 0:
            return ValidationResult(
                name=name,
                command=command,
                status="passed",
                classification="diagnostic-only",
                next_action="Review recorded artifacts before publishing.",
                duration_s=round(duration_s, 3),
                returncode=completed.returncode,
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail,
                artifact_paths=artifact_paths or [],
                evidence=evidence or {},
            )
        classification = (
            failure_classification(
                _CommandOutput(command, completed.returncode, stdout_path, stderr_path)
            )
            if callable(failure_classification)
            else failure_classification
        )
        return ValidationResult(
            name=name,
            command=command,
            status="failed",
            classification=classification,
            next_action=failure_next_action,
            duration_s=round(duration_s, 3),
            returncode=completed.returncode,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            artifact_paths=artifact_paths or [],
            evidence=evidence or {},
        )
    finally:
        stdout_path.unlink(missing_ok=True)
        stderr_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class _CommandOutput:
    args: list[str]
    returncode: int
    stdout_path: Path
    stderr_path: Path


def _optional_smoke_classification(completed: _CommandOutput) -> str:
    if _output_contains_any(
        completed.stdout_path,
        completed.stderr_path,
        ("skipped", "requires_rocm", "no rocm"),
    ):
        return "deferred"
    return "diagnostic-only"


def _skipped_result(
    *,
    name: str,
    command: list[str],
    classification: str,
    next_action: str,
) -> ValidationResult:
    return ValidationResult(
        name=name,
        command=command,
        status="skipped",
        classification=classification,
        next_action=next_action,
        duration_s=0.0,
    )


def _clock_policy_evidence() -> dict[str, object]:
    keys = (
        "SOL_EXECBENCH_CLOCKS_LOCKED",
        "SOL_EXECBENCH_GPU_CLK_MHZ",
        "SOL_EXECBENCH_DRAM_CLK_MHZ",
    )
    return {
        "clock_policy": {
            key: os.environ.get(key) for key in keys if os.environ.get(key)
        },
        "authority_boundary": (
            "ROCm smoke evidence is diagnostic prerelease evidence, not full "
            "hardware validation, correctness authority, timing authority, "
            "score authority, paper parity, or leaderboard authority."
        ),
    }


def _tail(value: str, limit: int = DEFAULT_LOG_TAIL_CHARS) -> str:
    if limit <= 0:
        return ""
    redacted = TOKEN_PATTERN.sub(
        lambda match: f"{match.group(1)}{match.group(2)}<redacted>", value
    )
    return redacted[-limit:]


def _append_tail(tail: str, text: str, limit: int) -> str:
    if not text:
        return tail
    return (tail + text)[-limit:]


def _redacted_tail_file(path: Path, limit: int) -> str:
    tail = ""
    pending = ""
    in_secret = False
    chunk_size = max(limit, 8192)

    with path.open("rb") as handle:
        for raw_chunk in iter(lambda: handle.read(chunk_size), b""):
            text = raw_chunk.decode(errors="replace")
            if in_secret:
                delimiter_index = next(
                    (
                        index
                        for index, char in enumerate(text)
                        if char in _TOKEN_VALUE_DELIMITERS
                    ),
                    None,
                )
                if delimiter_index is None:
                    continue
                tail = _append_tail(tail, text[delimiter_index], limit)
                text = text[delimiter_index + 1 :]
                in_secret = False

            pending += text
            while pending:
                match = TOKEN_PATTERN.search(pending)
                if match is None:
                    if len(pending) > _TOKEN_PREFIX_OVERLAP_CHARS:
                        emit = pending[:-_TOKEN_PREFIX_OVERLAP_CHARS]
                        tail = _append_tail(tail, emit, limit)
                        pending = pending[-_TOKEN_PREFIX_OVERLAP_CHARS:]
                    break

                tail = _append_tail(tail, pending[: match.start()], limit)
                tail = _append_tail(
                    tail,
                    f"{match.group(1)}{match.group(2)}<redacted>",
                    limit,
                )
                if match.end() == len(pending):
                    pending = ""
                    in_secret = True
                    break
                pending = pending[match.end() :]

    if not in_secret:
        tail = _append_tail(tail, _tail(pending, max(limit, 8192)), limit)
    return tail


def _temporary_stream_path(name: str, stream_name: str) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    with tempfile.NamedTemporaryFile(
        prefix=f"sol_execbench_{safe_name}_{stream_name}_",
        suffix=".log",
        delete=False,
    ) as handle:
        return Path(handle.name)


def _run_command_to_files(
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
) -> subprocess.CompletedProcess[str]:
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_handle,
        stderr_path.open(
            "w",
            encoding="utf-8",
        ) as stderr_handle,
    ):
        completed = subprocess.run(
            command,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            check=False,
        )
    # Some tests monkeypatch subprocess.run and return captured text while
    # ignoring stdout/stderr file handles. Mirror that text into the stream
    # files so production and test paths exercise the same tail logic.
    if completed.stdout:
        stdout_path.write_text(completed.stdout, encoding="utf-8")
    if completed.stderr:
        stderr_path.write_text(completed.stderr, encoding="utf-8")
    return completed


def _tail_file(path: Path, limit: int = DEFAULT_LOG_TAIL_CHARS) -> str:
    if limit <= 0:
        return ""
    try:
        return _redacted_tail_file(path, limit)
    except OSError:
        return ""


def _output_contains_any(
    stdout_path: Path,
    stderr_path: Path,
    needles: tuple[str, ...],
) -> bool:
    for path in (stdout_path, stderr_path):
        try:
            with path.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    lowered = line.lower()
                    if any(needle in lowered for needle in needles):
                        return True
        except OSError:
            continue
    return False


def _build_payload(results: list[ValidationResult]) -> dict[str, object]:
    result_payload = [asdict(result) for result in results]
    blocking = [result for result in results if result.classification == "blocking"]
    failed_blocking = [result for result in blocking if result.status not in {"passed"}]
    if failed_blocking:
        overall_status = "blocking"
    elif any(
        result.status in {"failed", "unavailable", "skipped"} for result in results
    ):
        overall_status = "review_needed"
    else:
        overall_status = "passed"
    return {
        "schema_version": SCHEMA_VERSION,
        "overall_status": overall_status,
        "summary": {
            "total": len(results),
            "passed": sum(result.status == "passed" for result in results),
            "failed": sum(result.status == "failed" for result in results),
            "skipped": sum(result.status == "skipped" for result in results),
            "unavailable": sum(result.status == "unavailable" for result in results),
            "blocking": sum(result.classification == "blocking" for result in results),
            "deferred": sum(result.classification == "deferred" for result in results),
            "diagnostic_only": sum(
                result.classification == "diagnostic-only" for result in results
            ),
        },
        "claim_boundary": {
            "engineering_prerelease_only": True,
            "full_235_problem_validation": False,
            "upstream_solar_parity": False,
            "leaderboard_ready": False,
            "hard_sandbox": False,
            "mi300x_cdna3_full_suite_validated": False,
            "cdna4_validated": False,
        },
        "results": result_payload,
    }


def _render_markdown(payload: dict[str, object]) -> str:
    summary = payload["summary"]
    assert isinstance(summary, dict)
    lines = [
        "# Release Candidate Validation",
        "",
        f"**Schema:** `{payload['schema_version']}`",
        f"**Overall status:** `{payload['overall_status']}`",
        "",
        "## Summary",
        "",
        "| Metric | Count |",
        "|--------|-------|",
    ]
    for key in (
        "total",
        "passed",
        "failed",
        "skipped",
        "unavailable",
        "blocking",
        "deferred",
        "diagnostic_only",
    ):
        lines.append(f"| `{key}` | {summary[key]} |")
    lines.extend(["", "## Results", ""])
    for result in payload["results"]:
        assert isinstance(result, dict)
        command = " ".join(str(part) for part in result["command"])
        lines.extend(
            [
                f"### {result['name']}",
                "",
                f"- Status: `{result['status']}`",
                f"- Classification: `{result['classification']}`",
                f"- Next action: {result['next_action']}",
                f"- Command: `{command}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Claim Boundary",
            "",
            "This artifact is engineering prerelease evidence only. It is not full "
            "235-problem paper validation, upstream SOLAR parity, hosted "
            "leaderboard readiness, hard sandbox evidence, CDNA4 validation, or "
            "full MI300X validation under CDNA3.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
