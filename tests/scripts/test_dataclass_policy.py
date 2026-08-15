"""Contract tests for the first-party dataclass policy."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts" / "check_dataclass_policy.py"
SPEC = importlib.util.spec_from_file_location("check_dataclass_policy", SCRIPT)
assert SPEC and SPEC.loader
policy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(policy)


def test_repository_dataclasses_follow_policy() -> None:
    assert (
        policy.dataclass_violations(
            policy.repository_python_files(Path(__file__).parents[2])
        )
        == []
    )


def test_policy_reports_each_missing_option(tmp_path: Path) -> None:
    source = tmp_path / "models.py"
    source.write_text(
        "from dataclasses import dataclass\n"
        "@dataclass(frozen=True)\n"
        "class Legacy:\n"
        "    value: str\n",
        encoding="utf-8",
    )
    assert policy.dataclass_violations([source]) == [
        f"{source}:3: Legacy requires slots=True, kw_only=True"
    ]
