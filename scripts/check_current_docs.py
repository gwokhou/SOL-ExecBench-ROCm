#!/usr/bin/env python3
"""Reject retired references and broken local links in current documentation."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURRENT_DOCS = (
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "SECURITY.md",
    *sorted((ROOT / "docs").rglob("*.md")),
    ROOT / "scripts" / "internal" / "README.md",
)
RETIRED_REFERENCES = (
    "docs/internal/solar-three-stage-readiness.md",
    "docs/user/research_preview.md",
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
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\((?P<target>[^)\s]+)")
EXTERNAL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def _broken_local_links(path: Path, text: str) -> list[str]:
    failures: list[str] = []
    for match in MARKDOWN_LINK.finditer(text):
        target = match.group("target")
        if target.startswith("#") or EXTERNAL_SCHEME.match(target):
            continue
        local_target = target.partition("#")[0]
        if local_target and not (path.parent / local_target).exists():
            failures.append(
                f"{path.relative_to(ROOT)} has broken local link {target!r}"
            )
    return failures


def main() -> int:
    failures: list[str] = []
    for path in CURRENT_DOCS:
        text = path.read_text(encoding="utf-8")
        for reference in RETIRED_REFERENCES:
            if reference in text:
                failures.append(
                    f"{path.relative_to(ROOT)} references retired path {reference!r}"
                )
        failures.extend(_broken_local_links(path, text))
    if failures:
        print("\n".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
