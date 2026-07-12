#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Generate AMD bound sanity JSON and Markdown reports from sidecars."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from sol_execbench.core.scoring.amd_bound_sanity.builder import (
    build_amd_bound_sanity_report,
)
from sol_execbench.core.scoring.amd_bound_sanity.io import (
    load_json,
    write_amd_bound_sanity_reports,
)
from sol_execbench.core.scoring.amd_bound_sanity.inputs import SanityInputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SOL ExecBench AMD bound sanity diagnostic reports.",
    )
    parser.add_argument("--trace", type=Path, action="append", default=[])
    parser.add_argument("--execution-closure", type=Path, required=True)
    parser.add_argument("--amd-sol-artifact", type=Path, action="append", default=[])
    parser.add_argument("--solar-artifact", type=Path, action="append", default=[])
    parser.add_argument("--amd-score-report", type=Path, default=None)
    parser.add_argument("--compatibility-matrix", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--created-at", default=None)
    args = parser.parse_args()

    execution_closure = load_json(args.execution_closure)
    amd_score_report = (
        load_json(args.amd_score_report) if args.amd_score_report else None
    )
    compatibility_matrix = (
        load_json(args.compatibility_matrix) if args.compatibility_matrix else None
    )

    report = build_amd_bound_sanity_report(
        SanityInputs(
            trace_refs=_load_trace_refs(args.trace),
            execution_closure=execution_closure,
            amd_sol_artifacts=[_load_artifact(path) for path in args.amd_sol_artifact],
            solar_artifacts=[_load_artifact(path) for path in args.solar_artifact],
            amd_score_report=amd_score_report,
            compatibility_matrix=compatibility_matrix,
            source_paths={
                "execution_closure": args.execution_closure,
                "amd_score_report": args.amd_score_report,
                "compatibility_matrix": args.compatibility_matrix,
            },
            created_at=args.created_at,
        )
    )
    write_amd_bound_sanity_reports(
        report,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )


def _load_trace_refs(paths: list[Path]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for path in paths:
        payload = load_json(path)
        if isinstance(payload, list):
            refs.extend(_ref_with_path(item, fallback_path=path) for item in payload)
        else:
            refs.append(_ref_with_path(payload, fallback_path=path))
    return refs


def _load_artifact(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object artifact at {path}")
    payload = dict(payload)
    payload.setdefault("path", str(path))
    return payload


def _ref_with_path(payload: object, *, fallback_path: Path) -> dict[str, Any]:
    if isinstance(payload, dict):
        ref = dict(payload)
        ref.setdefault("path", str(fallback_path))
        return ref
    return {"path": str(fallback_path), "ref": str(payload)}


if __name__ == "__main__":
    main()
