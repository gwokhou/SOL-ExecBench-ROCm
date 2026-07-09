#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Export diagnostic ROCm Compatibility Matrix JSON Schemas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sol_execbench.core.platform.compatibility import (
    export_matrix_entry_json_schema,
    export_matrix_json_schemas,
    export_rocm_compatibility_matrix_report_json_schema,
)


SCHEMA_FILENAMES = {
    "matrix_entry": "matrix-entry.schema.json",
    "rocm_compatibility_matrix_report": "rocm-compatibility-matrix-report.schema.json",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export diagnostic ROCm Compatibility Matrix JSON Schemas.",
    )
    parser.add_argument(
        "--model",
        choices=("matrix-entry", "report", "all"),
        required=True,
        help="Matrix schema model to export.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file for a single schema export.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for --model all.",
    )
    return parser


def _write_schema(path: Path, schema: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(schema, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.model == "all":
        if args.output is not None:
            parser.error("--model all writes --output-dir, not --output")
        if args.output_dir is None:
            parser.error("--model all requires --output-dir")
        schemas = export_matrix_json_schemas()
        for schema_name in sorted(schemas):
            _write_schema(
                args.output_dir / SCHEMA_FILENAMES[schema_name], schemas[schema_name]
            )
        return 0

    if args.output_dir is not None:
        parser.error("--output-dir is only valid with --model all")
    if args.output is None:
        parser.error("--model matrix-entry/report requires --output")

    if args.model == "matrix-entry":
        schema = export_matrix_entry_json_schema()
    else:
        schema = export_rocm_compatibility_matrix_report_json_schema()
    _write_schema(args.output, schema)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
