#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Generate paper denominator JSON and Markdown reports from sidecars."""

from __future__ import annotations

import argparse
from pathlib import Path

from sol_execbench.core.dataset.paper_denominator import (
    build_paper_denominator_report,
    load_json,
    write_paper_denominator_reports,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SOL ExecBench paper denominator reports.",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--ready-subset", type=Path, default=None)
    parser.add_argument("--execution-closure", type=Path, required=True)
    parser.add_argument("--amd-score-report", type=Path, default=None)
    parser.add_argument("--amd-sol-artifact", type=Path, action="append", default=[])
    parser.add_argument("--solar-artifact", type=Path, action="append", default=[])
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--created-at", default=None)
    args = parser.parse_args()

    manifest = load_json(args.manifest) if args.manifest else None
    inventory = load_json(args.inventory)
    readiness = load_json(args.readiness)
    ready_subset = load_json(args.ready_subset) if args.ready_subset else None
    execution_closure = load_json(args.execution_closure)
    amd_score_report = (
        load_json(args.amd_score_report) if args.amd_score_report else None
    )
    report = build_paper_denominator_report(
        manifest=manifest,
        inventory=inventory,
        readiness=readiness,
        ready_subset=ready_subset,
        execution_closure=execution_closure,
        amd_score_report=amd_score_report,
        amd_sol_artifacts=args.amd_sol_artifact,
        solar_artifacts=args.solar_artifact,
        source_paths={
            "manifest": args.manifest,
            "inventory": args.inventory,
            "readiness": args.readiness,
            "ready_subset": args.ready_subset,
            "execution_closure": args.execution_closure,
            "amd_score_report": args.amd_score_report,
        },
        created_at=args.created_at,
    )
    write_paper_denominator_reports(
        report,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )


if __name__ == "__main__":
    main()
