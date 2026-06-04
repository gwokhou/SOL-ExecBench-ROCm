#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Guard against redistributing restricted dataset artifacts."""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_PATH = REPO_ROOT / "provenance.toml"


@dataclass(frozen=True)
class DatasetRedistributionFinding:
    path: str
    source_id: str
    source_name: str
    redistribution_class: str
    mode: str
    message: str


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    policy = load_dataset_policy(args.provenance)

    findings: list[DatasetRedistributionFinding] = []
    if args.staged:
        findings.extend(check_paths(_staged_paths(args.repo_root), policy, mode="repository"))
    for release_root in args.release_root:
        findings.extend(check_release_root(release_root, policy))
    if args.path:
        findings.extend(check_paths(args.path, policy, mode=args.mode))

    if args.json:
        import json

        payload = {
            "schema_version": "sol_execbench.dataset_redistribution_check.v1",
            "overall_status": "blocking" if findings else "passed",
            "findings": [finding.__dict__ for finding in findings],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for finding in findings:
            print(
                f"{finding.mode}: {finding.path}: {finding.message} "
                f"({finding.source_id}, {finding.redistribution_class})",
                file=sys.stderr,
            )

    return 1 if findings else 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail if restricted dataset content is staged or included in release bundles.",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--provenance", type=Path, default=PROVENANCE_PATH)
    parser.add_argument("--staged", action="store_true", help="Check git staged paths.")
    parser.add_argument(
        "--release-root",
        type=Path,
        action="append",
        default=[],
        help="Check every file under a release or prerelease bundle root.",
    )
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Check an explicit repository-relative path. May be repeated.",
    )
    parser.add_argument(
        "--mode",
        choices=("repository", "release"),
        default="repository",
        help="Redistribution mode for --path checks.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def load_dataset_policy(path: Path = PROVENANCE_PATH) -> dict[str, object]:
    provenance = tomllib.loads(path.read_text(encoding="utf-8"))
    policy = provenance.get("dataset_policy")
    if not isinstance(policy, dict):
        raise ValueError("provenance.toml must define [dataset_policy]")
    sources = policy.get("sources")
    if not isinstance(sources, list):
        raise ValueError("provenance.toml [dataset_policy] must define sources")
    return policy


def check_release_root(
    release_root: Path,
    policy: dict[str, object],
) -> list[DatasetRedistributionFinding]:
    if not release_root.exists():
        return []
    paths = [
        path.relative_to(release_root).as_posix()
        for path in release_root.rglob("*")
        if path.is_file()
    ]
    return check_paths(paths, policy, mode="release")


def check_paths(
    paths: Iterable[str],
    policy: dict[str, object],
    *,
    mode: str,
) -> list[DatasetRedistributionFinding]:
    findings: list[DatasetRedistributionFinding] = []
    for raw_path in paths:
        normalized = _normalize_path(raw_path)
        for source in _sources(policy):
            if not _matches_source(normalized, source):
                continue
            if _is_allowed(source, mode=mode):
                continue
            findings.append(
                DatasetRedistributionFinding(
                    path=normalized,
                    source_id=str(source["id"]),
                    source_name=str(source.get("name", source["id"])),
                    redistribution_class=str(source.get("redistribution_class", "")),
                    mode=mode,
                    message=(
                        f"{source.get('name', source['id'])} is not allowed for "
                        f"{mode} redistribution by this project"
                    ),
                )
            )
            break
    return findings


def _sources(policy: dict[str, object]) -> list[dict[str, object]]:
    return [source for source in policy["sources"] if isinstance(source, dict)]


def _matches_source(path: str, source: dict[str, object]) -> bool:
    globs = source.get("path_globs", [])
    if not isinstance(globs, list):
        return False
    return any(
        isinstance(pattern, str) and fnmatch.fnmatchcase(path, pattern)
        for pattern in globs
    )


def _is_allowed(source: dict[str, object], *, mode: str) -> bool:
    if mode == "repository":
        return source.get("repository_redistribution") is True
    if mode == "release":
        return source.get("release_bundle_redistribution") is True
    raise ValueError(f"unknown redistribution mode: {mode}")


def _staged_paths(repo_root: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "git staged path query failed")
    return [line for line in completed.stdout.splitlines() if line.strip()]


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


if __name__ == "__main__":
    raise SystemExit(main())
