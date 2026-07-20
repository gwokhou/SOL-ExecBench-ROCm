"""Enforce separate line and branch coverage floors for Solar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "scripts" / "solar_coverage_policy.json"


def _percentage(covered: int, total: int) -> float:
    return 100.0 if total == 0 else 100.0 * covered / total


def _summary_percentages(summary: dict[str, Any]) -> tuple[float, float]:
    return (
        _percentage(int(summary["covered_lines"]), int(summary["num_statements"])),
        _percentage(
            int(summary.get("covered_branches", 0)),
            int(summary.get("num_branches", 0)),
        ),
    )


def _resolve_file(files: dict[str, Any], expected: str) -> dict[str, Any] | None:
    matches = [value for name, value in files.items() if name.endswith(expected)]
    return matches[0] if len(matches) == 1 else None


def check_report(report: dict[str, Any], policy: dict[str, Any]) -> list[str]:
    """Return human-readable policy violations for one coverage JSON report."""
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
            failures.append(f"{path}: branch {branch:.2f}% < {floors['branch']:.2f}%")

    solar = [
        entry["summary"]
        for name, entry in files.items()
        if "src/solar/" in name and "src/solar/_vendor/" not in name
    ]
    if not solar:
        failures.append("coverage report has no project-owned Solar files")
        return failures
    totals = {
        key: sum(int(summary.get(key, 0)) for summary in solar)
        for key in (
            "covered_lines",
            "num_statements",
            "covered_branches",
            "num_branches",
        )
    }
    line, branch = _summary_percentages(totals)
    global_floors = policy["global"]
    if line < float(global_floors["line"]):
        failures.append(f"Solar total: line {line:.2f}% < {global_floors['line']:.2f}%")
    if branch < float(global_floors["branch"]):
        failures.append(
            f"Solar total: branch {branch:.2f}% < {global_floors['branch']:.2f}%"
        )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    failures = check_report(report, policy)
    if failures:
        print("Solar coverage policy failed:\n- " + "\n- ".join(failures))
        return 1
    print("Solar coverage policy passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
