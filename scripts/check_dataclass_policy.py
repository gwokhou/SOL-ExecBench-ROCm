#!/usr/bin/env python3
"""Enforce keyword-only slotted first-party dataclasses."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIRST_PARTY_ROOTS = ("src", "tests", "scripts")
EXCLUDED_PARTS = frozenset({"_vendor", "vendor", "generated", ".venv"})


def dataclass_violations(paths: Iterable[Path]) -> list[str]:
    """Return stable findings for non-slotted or positional dataclasses."""
    violations: list[str] = []
    for path in sorted(paths):
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            decorator = next(
                (
                    item
                    for item in node.decorator_list
                    if _decorator_name(item) == "dataclass"
                ),
                None,
            )
            if decorator is None:
                continue
            options = _decorator_options(decorator)
            missing = [
                option
                for option in ("slots", "kw_only")
                if options.get(option) is not True
            ]
            if missing:
                violations.append(
                    f"{path}:{node.lineno}: {node.name} requires "
                    + ", ".join(f"{option}=True" for option in missing)
                )
    return violations


def _decorator_name(decorator: ast.expr) -> str | None:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


def _decorator_options(decorator: ast.expr) -> dict[str, object]:
    if not isinstance(decorator, ast.Call):
        return {}
    return {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in decorator.keywords
        if keyword.arg in {"slots", "kw_only"}
    }


def repository_python_files(repo_root: Path) -> Iterable[Path]:
    """Yield policy-scoped first-party Python sources."""
    for root_name in FIRST_PARTY_ROOTS:
        yield from (repo_root / root_name).rglob("*.py")


def main(argv: Sequence[str] | None = None) -> int:
    """Check the repository or explicit Python paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    paths = args.paths or list(repository_python_files(args.repo_root))
    violations = dataclass_violations(paths)
    for violation in violations:
        print(violation)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
