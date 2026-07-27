#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Validate every executable AKA workload/output against its pinned source oracle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from sol_execbench.core.dataset.aka_corpus import AkaCorpusManifest
from sol_execbench.core.dataset.aka_equivalence import (
    CrosscheckStatus,
    check_problem_equivalence,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "problems" / "AMD_AKA" / "manifest.yaml",
    )
    parser.add_argument(
        "--aka-root",
        type=Path,
        default=REPO_ROOT / "data" / "AgentKernelArena",
    )
    parser.add_argument(
        "--problems-root",
        type=Path,
        default=REPO_ROOT / "problems" / "AMD_AKA",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto' selects a ROCm GPU when available, otherwise CPU",
    )
    parser.add_argument(
        "--max-workloads",
        type=int,
        help="diagnostic limit; omit to validate every workload",
    )
    return parser.parse_args()


def _device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def main() -> int:
    """Run semantic-equivalence checks for selected AKA corpus entries."""
    args = _parse_args()
    if not args.aka_root.is_dir():
        raise FileNotFoundError(
            f"pinned AKA clone is required: {args.aka_root}",
        )
    if args.max_workloads is not None and args.max_workloads <= 0:
        raise ValueError("--max-workloads must be positive")
    manifest = AkaCorpusManifest.load(args.manifest)
    manifest.audit_aka_provenance(args.aka_root)
    reports = [
        check_problem_equivalence(
            entry,
            args.problems_root / entry.relative_problem_dir,
            args.aka_root,
            device=_device(args.device),
            max_workloads=args.max_workloads,
        )
        for entry in manifest.entries
    ]
    for report in reports:
        status = "PASS" if report.passed else "FAIL"
        print(
            f"[{status}] {report.problem_name}: {report.workloads_checked} workloads, "
            f"{report.outputs_checked} outputs, cross-check={report.crosscheck}; "
            f"{report.detail}",
        )
    passed = sum(report.passed for report in reports)
    crossed = sum(
        report.crosscheck is CrosscheckStatus.PASSED for report in reports
    )
    not_applicable = sum(
        report.crosscheck is CrosscheckStatus.NOT_APPLICABLE
        for report in reports
    )
    print(
        f"{passed}/{len(reports)} problems passed; "
        f"{crossed} source-equivalent, {not_applicable} explicitly not applicable",
    )
    return 0 if passed == len(reports) else 1


if __name__ == "__main__":
    sys.exit(main())
