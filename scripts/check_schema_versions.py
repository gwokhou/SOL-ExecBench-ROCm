#!/usr/bin/env python3
"""Fail when first-party files contain non-current schema contracts."""

from __future__ import annotations

import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from sol_execbench.core.integrity.schema_versions import CURRENT_SCHEMA_VERSIONS

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_ID_RE = re.compile(r"sol_execbench(?:\.[a-z0-9_]+)+\.v\d+")
VERSION_SUFFIX_RE = re.compile(r"\.v\d+$")
UPSTREAM_TOLERANCE_FIELD = "required_" + "match_ratio"
UPSTREAM_FIELD_ALLOWLIST: set[Path] = set()
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    ".uv-cache",
    "__pycache__",
    "data",
    "dist",
    "out",
}
EXCLUDED_PREFIXES = {
    Path("src/sol_execbench/_vendor"),
    Path("src/solar/_vendor"),
}
TEXT_SUFFIXES = {
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


def audit_text(path: Path, content: str) -> tuple[list[str], dict[str, set[str]]]:
    """Return findings and schema IDs grouped by family for one file."""
    findings: list[str] = []
    families: dict[str, set[str]] = defaultdict(set)
    for schema_id in SCHEMA_ID_RE.findall(content):
        family = VERSION_SUFFIX_RE.sub("", schema_id)
        families[family].add(schema_id)
        if schema_id not in CURRENT_SCHEMA_VERSIONS:
            findings.append(f"{path}: unsupported schema identifier {schema_id}")
    if UPSTREAM_TOLERANCE_FIELD in content and path not in UPSTREAM_FIELD_ALLOWLIST:
        findings.append(f"{path}: upstream tolerance name escaped the import boundary")
    return findings, families


def audit_paths(paths: Iterable[Path]) -> list[str]:
    """Audit readable first-party files and enforce one version per family."""
    findings: list[str] = []
    families: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        try:
            content = (ROOT / path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        path_findings, path_families = audit_text(path, content)
        findings.extend(path_findings)
        for family, versions in path_families.items():
            families[family].update(versions)
    for family, versions in sorted(families.items()):
        if len(versions) > 1:
            findings.append(f"{family}: multiple schema versions: {sorted(versions)}")
    return findings


def first_party_paths() -> tuple[Path, ...]:
    """List tracked and new non-ignored files without traversing generated data."""
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        timeout=10,
    )
    paths = (Path(item) for item in result.stdout.decode().split("\0") if item)
    return tuple(
        path
        for path in paths
        if path.suffix in TEXT_SUFFIXES
        and not EXCLUDED_PARTS.intersection(path.parts)
        and not any(path.is_relative_to(prefix) for prefix in EXCLUDED_PREFIXES)
    )


def main() -> int:
    """Run the repository schema audit."""
    retired_roots = (".planning", ".superpowers", "docs/examples", "docs/releases")
    findings = [
        f"retired archive root still exists: {path}"
        for path in retired_roots
        if (ROOT / path).exists()
    ]
    findings.extend(audit_paths(first_party_paths()))
    if findings:
        print("\n".join(findings))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
