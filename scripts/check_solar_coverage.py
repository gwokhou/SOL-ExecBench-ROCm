"""Enforce separate line and branch coverage floors for one source package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sol_execbench.core.dataset.schema_versions import (
    COVERAGE_POLICY_SCHEMA_VERSION,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "scripts" / "solar_coverage_policy.json"


def _percentage(covered: int, total: int) -> float:
    return 100.0 if total == 0 else 100.0 * covered / total


def _summary_percentages(summary: dict[str, Any]) -> tuple[float, float]:
    return (
        _percentage(
            int(summary["covered_lines"]),
            int(summary["num_statements"]),
        ),
        _percentage(
            int(summary.get("covered_branches", 0)),
            int(summary.get("num_branches", 0)),
        ),
    )


def _resolve_file(
    files: dict[str, Any],
    expected: str,
) -> dict[str, Any] | None:
    matches = [
        value for name, value in files.items() if name.endswith(expected)
    ]
    return matches[0] if len(matches) == 1 else None


def _combined_percentages(
    entries: list[dict[str, Any]],
) -> tuple[float, float]:
    summaries = [entry["summary"] for entry in entries]
    totals = {
        key: sum(int(summary.get(key, 0)) for summary in summaries)
        for key in (
            "covered_lines",
            "num_statements",
            "covered_branches",
            "num_branches",
        )
    }
    return _summary_percentages(totals)


def check_report(report: dict[str, Any], policy: dict[str, Any]) -> list[str]:
    """Return human-readable policy violations for one coverage JSON report."""
    if policy.get("schema_version") != COVERAGE_POLICY_SCHEMA_VERSION:
        return [
            (
                "coverage policy must use current "
                f"schema_version={COVERAGE_POLICY_SCHEMA_VERSION}"
            ),
        ]
    files = report.get("files")
    if not isinstance(files, dict):
        return ["coverage report has no files mapping"]
    failures: list[str] = []
    for path, floors in policy["files"].items():
        entry = _resolve_file(files, path)
        if entry is None:
            failures.append(f"missing or ambiguous coverage entry: {path}")
            continue
        line, branch = _summary_percentages(entry["summary"])
        if line < float(floors["line"]):
            failures.append(f"{path}: line {line:.2f}% < {floors['line']:.2f}%")
        if branch < float(floors["branch"]):
            failures.append(
                f"{path}: branch {branch:.2f}% < {floors['branch']:.2f}%",
            )

    for name, group in policy.get("groups", {}).items():
        entries: list[dict[str, Any]] = []
        for path in group["paths"]:
            entry = _resolve_file(files, path)
            if entry is None:
                failures.append(
                    f"{name}: missing or ambiguous coverage entry: {path}",
                )
            else:
                entries.append(entry)
        if len(entries) != len(group["paths"]):
            continue
        line, branch = _combined_percentages(entries)
        if line < float(group["line"]):
            failures.append(
                f"{name}: line {line:.2f}% < {group['line']:.2f}%",
            )
        if branch < float(group["branch"]):
            failures.append(
                f"{name}: branch {branch:.2f}% < {group['branch']:.2f}%",
            )

    package = str(policy.get("package", "solar"))
    source_prefix = f"src/{package}/"
    vendor_prefix = f"src/{package}/_vendor/"
    package_summaries = [
        entry["summary"]
        for name, entry in files.items()
        if source_prefix in name and vendor_prefix not in name
    ]
    if not package_summaries:
        failures.append(f"coverage report has no project-owned {package} files")
        return failures
    line, branch = _combined_percentages(
        [{"summary": summary} for summary in package_summaries],
    )
    global_floors = policy["global"]
    if line < float(global_floors["line"]):
        failures.append(
            f"{package} total: line {line:.2f}% < {global_floors['line']:.2f}%",
        )
    if branch < float(global_floors["branch"]):
        failures.append(
            f"{package} total: branch {branch:.2f}% < {global_floors['branch']:.2f}%",
        )
    return failures


def main() -> int:
    """Check SOLAR coverage data against repository policy."""
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    failures = check_report(report, policy)
    if failures:
        print("Coverage policy failed:\n- " + "\n- ".join(failures))
        return 1
    print("Coverage policy passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
