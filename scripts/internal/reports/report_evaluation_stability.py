#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Generate evaluation stability JSON and Markdown reports from timing evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from sol_execbench.core.reports.evaluation_stability import (
    build_evaluation_stability_report,
    load_json,
    write_evaluation_stability_reports,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate SOL ExecBench evaluation stability diagnostics.",
    )
    parser.add_argument("--timing-evidence", type=Path, action="append", default=[])
    parser.add_argument(
        "--json-out", "--json-output", dest="json_out", type=Path, required=True
    )
    parser.add_argument(
        "--markdown-out",
        "--markdown-output",
        dest="markdown_out",
        type=Path,
        required=True,
    )
    parser.add_argument("--created-at", default=None)
    parser.add_argument("--noise-cv-threshold", type=float, default=0.10)
    parser.add_argument("--min-samples", type=int, default=3)
    args = parser.parse_args(argv)

    report = build_evaluation_stability_report(
        timing_evidence=[load_json(path) for path in args.timing_evidence],
        source_paths=list(args.timing_evidence),
        created_at=args.created_at,
        noise_cv_threshold=args.noise_cv_threshold,
        min_samples=args.min_samples,
    )
    write_evaluation_stability_reports(
        report,
        json_path=args.json_out,
        markdown_path=args.markdown_out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
