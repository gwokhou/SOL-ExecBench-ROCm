#!/usr/bin/env python3
"""Reject removed paths from current user and architecture docs."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURRENT_DOCS = (
    ROOT / "README.md",
    *sorted((ROOT / "docs" / "user").glob("*.md")),
    ROOT / "docs" / "internal" / "architecture_navigation.md",
    ROOT / "docs" / "internal" / "coupling_governance.md",
)
RETIRED_REFERENCES = (
    "scripts/run_dataset.py",
    "scripts/run_derived_isolated.py",
    "scripts/download_solexecbench.py",
    "sol-execbench dataset migrate",
    "sol-execbench baseline compare",
    "sol-execbench baseline export",
    "sol_execbench.core.data.contract",
    "sol_execbench.core.scoring.suite_score",
    "sol_execbench.core.scoring.confidence",
    "sol_execbench.sol_score",
    "solar.extraction",
    "examples/hip_cpp/",
    "examples/hipblas/",
    "examples/pytorch/",
    "examples/triton/",
    "examples/miopen/softmax/",
    "examples/ck/gemm/",
    "examples/rocwmma/gemm/",
    "tests/sol_execbench/test_cdna3_hardware_marker.py",
)


def main() -> int:
    failures: list[str] = []
    for path in CURRENT_DOCS:
        text = path.read_text(encoding="utf-8")
        for reference in RETIRED_REFERENCES:
            if reference in text:
                failures.append(
                    f"{path.relative_to(ROOT)} references retired path {reference!r}"
                )
    if failures:
        print("\n".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
