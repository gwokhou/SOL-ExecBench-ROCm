#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate trust summary JSON and Markdown reports from evidence sidecars."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from sol_execbench.core.trust_summary import (
    build_trust_summary_report,
    load_json,
    write_trust_summary_reports,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate SOL ExecBench trust summary diagnostics.",
    )
    parser.add_argument("--consistency-report", type=Path, default=None)
    parser.add_argument("--evaluation-stability", type=Path, default=None)
    parser.add_argument("--claim-upgrade", type=Path, default=None)
    parser.add_argument("--execution-closure", type=Path, default=None)
    parser.add_argument("--paper-denominator", type=Path, default=None)
    parser.add_argument("--matrix-report", type=Path, default=None)
    parser.add_argument("--amd-score-report", type=Path, default=None)
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

    report = build_trust_summary_report(
        consistency_report=_optional_json(args.consistency_report),
        evaluation_stability=_optional_json(args.evaluation_stability),
        claim_upgrade=_optional_json(args.claim_upgrade),
        execution_closure=_optional_json(args.execution_closure),
        paper_denominator=_optional_json(args.paper_denominator),
        matrix_report=_optional_json(args.matrix_report),
        amd_score_report=_optional_json(args.amd_score_report),
        amd_bound_sanity=_optional_json(args.amd_bound_sanity),
        source_paths={
            "consistency_report": args.consistency_report,
            "evaluation_stability": args.evaluation_stability,
            "claim_upgrade": args.claim_upgrade,
            "execution_closure": args.execution_closure,
            "paper_denominator": args.paper_denominator,
            "matrix_report": args.matrix_report,
            "amd_score_report": args.amd_score_report,
            "amd_bound_sanity": args.amd_bound_sanity,
        },
        created_at=args.created_at,
    )
    write_trust_summary_reports(
        report,
        json_path=args.json_out,
        markdown_path=args.markdown_out,
    )
    return 0


def _optional_json(path: Path | None) -> dict | None:
    return load_json(path) if path else None


if __name__ == "__main__":
    raise SystemExit(main())
