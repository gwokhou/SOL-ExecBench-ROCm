#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Build the canonical 235-problem suite, static coverage, and profile requirements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sol_execbench.core.scoring.full_suite import (
    build_full_suite_coverage,
    build_full_suite_manifest,
    validate_full_suite_coverage,
)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    manifest = build_full_suite_manifest(args.benchmark_root)
    _write(args.output_dir / "canonical-suite.json", manifest)
    if not args.manifest_only:
        coverage, requirements = build_full_suite_coverage(
            args.benchmark_root, manifest
        )
        validate_full_suite_coverage(coverage, requirements, manifest)
        _write(args.output_dir / "static-coverage.json", coverage)
        _write(args.output_dir / "hardware-profile-requirements.json", requirements)
    print(
        json.dumps(
            {
                "problem_count": manifest["problem_denominator"],
                "workload_count": manifest["workload_denominator"],
                "scope": manifest["scope"],
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
