from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts/check_solar_coverage.py"
)
SPEC = spec_from_file_location("check_coverage_policy", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
coverage_policy = module_from_spec(SPEC)
SPEC.loader.exec_module(coverage_policy)


def _entry(
    *,
    covered_lines: int = 8,
    statements: int = 10,
    covered_branches: int = 6,
    branches: int = 8,
) -> dict[str, object]:
    return {
        "summary": {
            "covered_lines": covered_lines,
            "num_statements": statements,
            "covered_branches": covered_branches,
            "num_branches": branches,
        },
    }


def _policy() -> dict[str, object]:
    return {
        "package": "example",
        "global": {"line": 70.0, "branch": 55.0},
        "files": {
            "src/example/critical.py": {"line": 80.0, "branch": 65.0},
        },
    }


def test_accepts_package_and_critical_file_above_floors() -> None:
    report = {
        "files": {
            "src/example/critical.py": _entry(),
            "src/example/other.py": _entry(),
            "src/unrelated/module.py": _entry(
                covered_lines=0,
                covered_branches=0,
            ),
        },
    }

    assert coverage_policy.check_report(report, _policy()) == []


def test_reports_missing_critical_file_and_package() -> None:
    failures = coverage_policy.check_report({"files": {}}, _policy())

    assert failures == [
        "missing or ambiguous coverage entry: src/example/critical.py",
        "coverage report has no project-owned example files",
    ]


def test_enforces_file_line_and_branch_floors_independently() -> None:
    report = {
        "files": {
            "src/example/critical.py": _entry(
                covered_lines=7,
                covered_branches=5,
            ),
        },
    }

    assert coverage_policy.check_report(report, _policy()) == [
        "src/example/critical.py: line 70.00% < 80.00%",
        "src/example/critical.py: branch 62.50% < 65.00%",
    ]


def test_enforces_group_floors_over_refactored_modules() -> None:
    policy = _policy()
    policy["groups"] = {
        "refactored subsystem": {
            "paths": [
                "src/example/weak.py",
                "src/example/strong.py",
            ],
            "line": 80.0,
            "branch": 65.0,
        },
    }
    report = {
        "files": {
            "src/example/critical.py": _entry(),
            "src/example/weak.py": _entry(
                covered_lines=6,
                covered_branches=4,
            ),
            "src/example/strong.py": _entry(
                covered_lines=10,
                covered_branches=8,
            ),
        },
    }

    assert coverage_policy.check_report(report, policy) == []


def test_reports_missing_group_entries() -> None:
    policy = _policy()
    policy["groups"] = {
        "refactored subsystem": {
            "paths": [
                "src/example/weak.py",
                "src/example/missing.py",
            ],
            "line": 80.0,
            "branch": 65.0,
        },
    }
    report = {
        "files": {
            "src/example/critical.py": _entry(),
            "src/example/weak.py": _entry(),
        },
    }

    assert coverage_policy.check_report(report, policy) == [
        (
            "refactored subsystem: missing or ambiguous coverage entry: "
            "src/example/missing.py"
        ),
    ]


def test_reports_group_line_and_branch_floors() -> None:
    policy = _policy()
    policy["groups"] = {
        "refactored subsystem": {
            "paths": [
                "src/example/weak.py",
                "src/example/strong.py",
            ],
            "line": 80.0,
            "branch": 65.0,
        },
    }
    report = {
        "files": {
            "src/example/critical.py": _entry(),
            "src/example/weak.py": _entry(
                covered_lines=6,
                covered_branches=4,
            ),
            "src/example/strong.py": _entry(),
        },
    }

    assert coverage_policy.check_report(report, policy) == [
        "refactored subsystem: line 70.00% < 80.00%",
        "refactored subsystem: branch 62.50% < 65.00%",
    ]


def test_enforces_package_totals_without_counting_vendor_or_other_packages() -> (
    None
):
    report = {
        "files": {
            "src/example/critical.py": _entry(),
            "src/example/weak.py": _entry(
                covered_lines=0,
                covered_branches=0,
            ),
            "src/example/_vendor/copied.py": _entry(),
            "src/other/strong.py": _entry(),
        },
    }

    assert coverage_policy.check_report(report, _policy()) == [
        "example total: line 40.00% < 70.00%",
        "example total: branch 37.50% < 55.00%",
    ]
