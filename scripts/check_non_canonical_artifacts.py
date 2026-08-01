#!/usr/bin/env python3
"""Fail when git-ignored curated artifacts reference non-current schema contracts.

`check_schema_versions.py` audits only tracked and non-ignored files and
deliberately excludes ``data/``, ``out/``, and ``dist/`` from traversal. This
companion gate closes that blind spot: it walks the *curated* top-level
children of ``data/`` (skipping generated and vendored roots) and fails on any
unsupported schema identifier unless the file lives under a ``NON_CANONICAL.md``
marker directory.

Rule: mark it or fix it. A directory that contains ``NON_CANONICAL.md`` exempts
its entire subtree (including the marker itself). Anything else in scope must
reference only current schema identifiers from the shared registries, or no
schema identifiers at all.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from sol_execbench.core.integrity.schema_versions import (
    CURRENT_SCHEMA_VERSIONS,
)
from solar.schema_versions import CURRENT_STRING_SCHEMA_VERSIONS

ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOT = ROOT / "data"
# Generated / vendored data roots are out of scope: outputs/ holds transient
# diagnostic artifacts, AgentKernelArena/ is a vendored corpus clone, and
# fusion-sample/ is a sample problem.
EXEMPT_DATA_ROOTS = frozenset(
    {
        "AgentKernelArena",
        "fusion-sample",
        "outputs",
    }
)
MARKER = "NON_CANONICAL.md"
SCHEMA_ID_RE = re.compile(
    r"(?:sol_execbench|solar)(?:\.[a-z0-9_]+)+\.v\d+",
)
# Content-based, not line-anchored: curated data files may be inline JSON
# (``{"schema_version": 6}``) or YAML-style; both must be caught.
NUMERIC_SCHEMA_FIELD_RE = re.compile(
    r'(?:"schema_version"|schema_version)[ \t]*:[ \t]*(\d+)\b',
)
CURRENT_SCHEMA_IDENTIFIERS = (
    CURRENT_SCHEMA_VERSIONS | CURRENT_STRING_SCHEMA_VERSIONS
)
TEXT_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".h",
        ".hip",
        ".json",
        ".md",
        ".py",
        ".toml",
        ".yaml",
        ".yml",
    }
)


def scan_paths(
    scan_root: Path,
    exempt_roots: Iterable[str] = EXEMPT_DATA_ROOTS,
) -> Iterable[Path]:
    """Yield repo-relative text files under *scan_root*, skipping exempt roots.

    Top-level files directly under *scan_root* are in scope. Exempt roots are
    never traversed, so large generated/vendored trees stay untouched.
    """
    exempt = set(exempt_roots)
    for path in sorted(scan_root.glob("*")):
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            yield path.relative_to(scan_root)
    for root in sorted(scan_root.iterdir()):
        if not root.is_dir() or root.name in exempt:
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in TEXT_SUFFIXES:
                yield path.relative_to(scan_root)


def is_marked(rel_path: Path, scan_root: Path) -> bool:
    """Return True when *rel_path* is the marker or sits under a marked directory.

    A directory is marked when it contains ``NON_CANONICAL.md``. The marker
    file itself is always exempt so the gate can run on the marker text.
    """
    if rel_path.name == MARKER:
        return True
    for parent in rel_path.parents:
        if (scan_root / parent / MARKER).exists():
            return True
    return False


def audit_text(path: Path, content: str) -> list[str]:
    """Return findings for one in-scope file's schema contracts."""
    findings: list[str] = []
    for schema_id in SCHEMA_ID_RE.findall(content):
        if schema_id not in CURRENT_SCHEMA_IDENTIFIERS:
            findings.append(
                f"{path}: unsupported schema identifier {schema_id}",
            )
    for version in NUMERIC_SCHEMA_FIELD_RE.findall(content):
        findings.append(
            f"{path}: unregistered numeric schema artifact "
            f"(schema_version {version})",
        )
    return findings


def collect_findings(
    scan_root: Path,
    exempt_roots: Iterable[str] = EXEMPT_DATA_ROOTS,
) -> list[str]:
    """Audit in-scope files and return unsupported-contract findings."""
    findings: list[str] = []
    for rel_path in scan_paths(scan_root, exempt_roots):
        if is_marked(rel_path, scan_root):
            continue
        try:
            content = (scan_root / rel_path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        findings.extend(audit_text(rel_path, content))
    return findings


def main() -> int:
    """Run the git-ignored curated-artifact schema audit."""
    findings = sorted(collect_findings(SCAN_ROOT))
    if findings:
        print("\n".join(findings))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
