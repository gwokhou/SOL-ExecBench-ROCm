#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Diff two diagnostic ROCm Compatibility Matrix reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sol_execbench.core.matrix_diff import (
    diff_matrix_reports,
    load_matrix_report,
    matrix_report_diff_to_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diff two diagnostic ROCm Compatibility Matrix reports.",
    )
    parser.add_argument("old_report", type=Path)
    parser.add_argument("new_report", type=Path)
    parser.add_argument("--old-label", default="old")
    parser.add_argument("--new-label", default="new")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    return parser


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    diff = diff_matrix_reports(
        load_matrix_report(args.old_report),
        load_matrix_report(args.new_report),
        old_label=args.old_label,
        new_label=args.new_label,
    )
    if args.json_out is not None:
        _write_text(
            args.json_out,
            json.dumps(diff.to_dict(), sort_keys=True, indent=2) + "\n",
        )
    if args.markdown_out is not None:
        _write_text(args.markdown_out, matrix_report_diff_to_markdown(diff))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
