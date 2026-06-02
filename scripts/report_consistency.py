#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Generate cross-report consistency JSON and Markdown reports from sidecars."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from sol_execbench.core.consistency import (
    build_consistency_report,
    load_json,
    write_consistency_reports,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate SOL ExecBench cross-report consistency diagnostics.",
    )
    parser.add_argument("--execution-closure", type=Path, default=None)
    parser.add_argument("--paper-denominator", type=Path, default=None)
    parser.add_argument("--matrix-report", type=Path, default=None)
    parser.add_argument("--runtime-evidence", type=Path, default=None)
    parser.add_argument("--static-evidence", type=Path, default=None)
    parser.add_argument("--amd-score-report", type=Path, default=None)
    parser.add_argument("--amd-sol-report", type=Path, default=None)
    parser.add_argument("--solar-derivation", type=Path, default=None)
    parser.add_argument("--amd-bound-sanity", type=Path, default=None)
    parser.add_argument("--json-out", "--json-output", dest="json_out", type=Path, required=True)
    parser.add_argument(
        "--markdown-out",
        "--markdown-output",
        dest="markdown_out",
        type=Path,
        required=True,
    )
    parser.add_argument("--created-at", default=None)
    args = parser.parse_args(argv)

    report = build_consistency_report(
        execution_closure=_optional_json(args.execution_closure),
        paper_denominator=_optional_json(args.paper_denominator),
        matrix_report=_optional_json(args.matrix_report),
        runtime_evidence=_optional_json(args.runtime_evidence),
        static_evidence=_optional_json(args.static_evidence),
        amd_score_report=_optional_json(args.amd_score_report),
        amd_sol_report=_optional_json(args.amd_sol_report),
        solar_derivation=_optional_json(args.solar_derivation),
        amd_bound_sanity=_optional_json(args.amd_bound_sanity),
        source_paths={
            "execution_closure": args.execution_closure,
            "paper_denominator": args.paper_denominator,
            "matrix_report": args.matrix_report,
            "runtime_evidence": args.runtime_evidence,
            "static_evidence": args.static_evidence,
            "amd_score_report": args.amd_score_report,
            "amd_sol_report": args.amd_sol_report,
            "solar_derivation": args.solar_derivation,
            "amd_bound_sanity": args.amd_bound_sanity,
        },
        created_at=args.created_at,
    )
    write_consistency_reports(
        report,
        json_path=args.json_out,
        markdown_path=args.markdown_out,
    )
    return 0


def _optional_json(path: Path | None) -> dict | None:
    return load_json(path) if path else None


if __name__ == "__main__":
    raise SystemExit(main())
