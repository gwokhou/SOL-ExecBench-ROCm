#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Build the canonical 235-problem suite, authority coverage, and profile requirements."""

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
    parser.add_argument(
        "--analysis-timeout-seconds",
        type=int,
        default=60,
        help="maximum authority graph/estimate/fusion analysis time per workload",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="worker processes; 0 chooses a CPU- and memory-bounded value",
    )
    parser.add_argument(
        "--worker-tasks-per-child",
        type=int,
        default=64,
        help=(
            "workload tasks before planned worker recycling; the RSS supervisor "
            "still immediately reaps a worker that exceeds its memory limit"
        ),
    )
    parser.add_argument(
        "--analysis-log-file",
        type=Path,
        help="full asynchronous torch.export diagnostics; defaults under output_dir",
    )
    args = parser.parse_args()

    manifest = build_full_suite_manifest(args.benchmark_root)
    _write(args.output_dir / "canonical-suite.json", manifest)
    if not args.manifest_only:
        coverage, requirements = build_full_suite_coverage(
            args.benchmark_root,
            manifest,
            analysis_timeout_seconds=args.analysis_timeout_seconds,
            workers=args.workers,
            worker_tasks_per_child=args.worker_tasks_per_child,
            analysis_log_path=(
                args.analysis_log_file
                if args.analysis_log_file is not None
                else args.output_dir / "authority-analysis.log"
            ),
        )
        validate_full_suite_coverage(coverage, requirements, manifest)
        _write(args.output_dir / "authority-coverage.json", coverage)
        _write(args.output_dir / "hardware-profile-requirements.json", requirements)
    print(
        json.dumps(
            {
                "problem_count": manifest["problem_denominator"],
                "workload_count": manifest["workload_denominator"],
                "scope": manifest["scope"],
                "output_dir": str(args.output_dir),
                "analysis_log_file": str(
                    args.analysis_log_file
                    if args.analysis_log_file is not None
                    else args.output_dir / "authority-analysis.log"
                )
                if not args.manifest_only
                else None,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
