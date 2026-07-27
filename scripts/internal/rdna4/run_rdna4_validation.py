#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Run or verify a content-addressed local gfx1200 validation bundle."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.platform.environment import (
    build_environment_diagnostics,
)
from sol_execbench.core.platform.rdna4_validation import (
    Rdna4EnvironmentIdentity,
    build_validation_manifest,
    validate_environment_payload,
    verify_validation_directory,
)
from sol_execbench.core.process.subprocesses import (
    run_in_process_group_bounded,
    run_in_process_group_to_files,
)
from sol_execbench.core.timestamps import utc_timestamp

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TIMEOUT_SECONDS = 3600.0


def _git_state() -> tuple[str, bool]:
    revision = run_in_process_group_bounded(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        timeout=10,
    )
    status = run_in_process_group_bounded(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        timeout=10,
    )
    if revision.returncode != 0 or status.returncode != 0:
        raise RuntimeError("could not resolve the validation source state")
    return revision.stdout.strip(), bool(status.stdout.strip())


def _prepare_output(path: Path) -> Path:
    output = path.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"validation output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _collect_environment(output: Path) -> Rdna4EnvironmentIdentity:
    diagnostics = build_environment_diagnostics().model_dump(mode="json")
    atomic_write_json_value(output / "environment-doctor.json", diagnostics)
    return validate_environment_payload(diagnostics)


def _attestation() -> dict[str, object]:
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return {"kind": "local_unsigned", "trusted_execution": False}
    return {
        "kind": "github_actions_self_hosted",
        "trusted_execution": False,
        "repository": os.environ.get("GITHUB_REPOSITORY"),
        "workflow_ref": os.environ.get("GITHUB_WORKFLOW_REF"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
    }


def _run_tests(output: Path, timeout_seconds: float) -> int:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-n",
        "0",
        "--strict-markers",
        "-m",
        "requires_rdna4",
        f"--junitxml={output / 'pytest-rdna4.xml'}",
        "tests/",
    ]
    try:
        completed = run_in_process_group_to_files(
            command,
            output / "pytest.stdout.txt",
            output / "pytest.stderr.txt",
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return 124
    return completed.returncode


def _run(output_path: Path, timeout_seconds: float) -> int:
    source_revision, source_dirty = _git_state()
    output = _prepare_output(output_path)
    environment = _collect_environment(output)
    pytest_returncode = _run_tests(output, timeout_seconds)
    artifacts = [
        output / "environment-doctor.json",
        output / "pytest-rdna4.xml",
        output / "pytest.stdout.txt",
        output / "pytest.stderr.txt",
    ]
    manifest = build_validation_manifest(
        directory=output,
        source_revision=source_revision,
        source_dirty=source_dirty,
        generated_at=utc_timestamp(),
        environment=environment,
        pytest_returncode=pytest_returncode,
        artifact_paths=artifacts,
        attestation=_attestation(),
    )
    atomic_write_json_value(output / "manifest.json", manifest)
    verify_validation_directory(output)
    print(output / "manifest.json")
    return 0 if manifest["status"] == "passed" else 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--output-dir",
        type=Path,
        metavar="BUNDLE_DIR",
        help="Create a new validation bundle in this directory.",
    )
    mode.add_argument(
        "--verify",
        type=Path,
        metavar="BUNDLE_DIR",
        help="Verify a validation bundle directory containing manifest.json.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="RDNA4 pytest timeout in seconds.",
    )
    parser.add_argument("--expected-source-revision")
    parser.add_argument("--require-release-eligible", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run or verify the local RDNA4 validation bundle."""
    args = _parse_args(argv)
    if args.verify is not None:
        verify_validation_directory(
            args.verify.resolve(),
            expected_source_revision=args.expected_source_revision,
            require_release_eligible=args.require_release_eligible,
        )
        print(args.verify.resolve() / "manifest.json")
        return 0
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --verify is used")
    return _run(args.output_dir, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
