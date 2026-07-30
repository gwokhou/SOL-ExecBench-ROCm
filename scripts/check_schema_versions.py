#!/usr/bin/env python3
"""Fail when first-party files contain non-current schema contracts."""

from __future__ import annotations

import ast
import re
import subprocess
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from sol_execbench.core.integrity.schema_versions import (
    CURRENT_NUMERIC_SCHEMA_VERSIONS as EXECBENCH_NUMERIC_SCHEMA_VERSIONS,
    CURRENT_SCHEMA_VERSIONS,
)
from solar.schema_versions import (
    CURRENT_NUMERIC_SCHEMA_VERSIONS as SOLAR_NUMERIC_SCHEMA_VERSIONS,
    CURRENT_STRING_SCHEMA_VERSIONS,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_ID_RE = re.compile(
    r"(?:sol_execbench|solar)(?:\.[a-z0-9_]+)+\.v\d+",
)
VERSION_SUFFIX_RE = re.compile(r"\.v\d+$")
CURRENT_SCHEMA_IDENTIFIERS = (
    CURRENT_SCHEMA_VERSIONS | CURRENT_STRING_SCHEMA_VERSIONS
)
CURRENT_NUMERIC_SCHEMA_VERSIONS = (
    EXECBENCH_NUMERIC_SCHEMA_VERSIONS | SOLAR_NUMERIC_SCHEMA_VERSIONS
)
NUMERIC_SCHEMA_FIELD_RE = re.compile(
    r'(?m)^[ \t]*(?:"schema_version"|schema_version)[ \t]*:[ \t]*(\d+)\b',
)
NUMERIC_SCHEMA_FILES = {
    Path("problems/AMD_AKA/manifest.yaml"): "aka_corpus_manifest",
    Path("problems/AMD_AKA/tolerance-calibration.json"): (
        "aka_tolerance_calibration"
    ),
    Path("scripts/sol_execbench_coverage_policy.json"): "coverage_policy",
    Path("scripts/solar_coverage_policy.json"): "coverage_policy",
}
NUMERIC_SCHEMA_PROSE_PATTERNS = {
    "aka_corpus_manifest": (
        re.compile(r"\bmanifest schema v(\d+)\b", re.IGNORECASE),
        re.compile(r"\bschema v(\d+) (?:RX 9060 XT )?corpus\b", re.IGNORECASE),
    ),
    "extended_einsum_ir": (
        re.compile(r"\bExtended schema v(\d+)\b", re.IGNORECASE),
    ),
    "operator_graph": (
        re.compile(r"\boperator[- ]graph schema v(\d+)\b", re.IGNORECASE),
    ),
}
UPSTREAM_TOLERANCE_FIELD = "required_" + "match_ratio"
UPSTREAM_FIELD_ALLOWLIST: set[Path] = set()
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    ".uv-cache",
    "__pycache__",
}
EXCLUDED_PREFIXES = {
    Path("data"),
    Path("dist"),
    Path("out"),
    Path("src/sol_execbench/_vendor"),
    Path("src/solar/_vendor"),
}
RETIRED_SCHEMA_PATHS = {
    Path("scripts/internal/migrate_workload_checks.py"),
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


def _positive_int(node: ast.AST | None) -> ast.Constant | None:
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and node.value > 0
    ):
        return node
    return None


def _mentions_schema_version(node: ast.AST) -> bool:
    return any(
        (isinstance(item, ast.Name) and "schema_version" in item.id.lower())
        or (isinstance(item, ast.Constant) and item.value == "schema_version")
        for item in ast.walk(node)
    )


class _NumericSchemaLiteralVisitor(ast.NodeVisitor):
    """Collect raw positive numeric schema versions from production Python."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.findings: list[str] = []

    def _record(self, node: ast.Constant) -> None:
        self.findings.append(
            f"{self.path}:{node.lineno}: raw numeric schema version "
            "must use the current family constant",
        )

    def visit_Dict(self, node: ast.Dict) -> None:
        for key, value in zip(node.keys, node.values, strict=True):
            if (
                isinstance(key, ast.Constant)
                and key.value == "schema_version"
                and (literal := _positive_int(value)) is not None
            ):
                self._record(literal)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        literal = _positive_int(node.value)
        if literal is not None and any(
            _mentions_schema_version(target) for target in node.targets
        ):
            self._record(literal)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        literal = _positive_int(node.value)
        if literal is not None and _mentions_schema_version(node.target):
            self._record(literal)
        self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword) -> None:
        literal = _positive_int(node.value)
        if (
            literal is not None
            and node.arg is not None
            and "schema_version" in node.arg.lower()
        ):
            self._record(literal)
        self.generic_visit(node)

    def _visit_arguments(self, arguments: ast.arguments) -> None:
        positional = [*arguments.posonlyargs, *arguments.args]
        defaults = [
            *(None for _ in range(len(positional) - len(arguments.defaults))),
            *arguments.defaults,
        ]
        pairs = [
            *zip(positional, defaults, strict=True),
            *zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True),
        ]
        for argument, default in pairs:
            if (
                "schema_version" in argument.arg.lower()
                and (literal := _positive_int(default)) is not None
            ):
                self._record(literal)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_arguments(node.args)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_arguments(node.args)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        operands = [node.left, *node.comparators]
        if any(_mentions_schema_version(item) for item in operands):
            for operand in operands:
                if (literal := _positive_int(operand)) is not None:
                    self._record(literal)
        self.generic_visit(node)


def _python_numeric_schema_findings(path: Path, content: str) -> list[str]:
    """Reject raw positive numeric schema versions outside their registries."""
    if path.suffix != ".py" or path.parts[0] not in {"src", "scripts"}:
        return []
    if path in {
        Path("src/sol_execbench/core/integrity/schema_versions.py"),
        Path("src/solar/schema_versions.py"),
    }:
        return []
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []
    visitor = _NumericSchemaLiteralVisitor(path)
    visitor.visit(tree)
    return visitor.findings


def _numeric_schema_findings(path: Path, content: str) -> list[str]:
    """Validate numeric artifacts and versioned schema prose by family."""
    findings = _python_numeric_schema_findings(path, content)
    observed = [
        int(value) for value in NUMERIC_SCHEMA_FIELD_RE.findall(content)
    ]
    family = NUMERIC_SCHEMA_FILES.get(path)
    if family is not None:
        expected = CURRENT_NUMERIC_SCHEMA_VERSIONS[family]
        if observed != [expected]:
            findings.append(
                f"{path}: {family} must contain only schema version {expected}",
            )
    elif observed and path.suffix in {".json", ".yaml", ".yml"}:
        findings.append(
            f"{path}: unregistered numeric schema artifact",
        )
    for prose_family, patterns in NUMERIC_SCHEMA_PROSE_PATTERNS.items():
        expected = CURRENT_NUMERIC_SCHEMA_VERSIONS[prose_family]
        for pattern in patterns:
            for match in pattern.finditer(content):
                observed = int(match.group(1))
                if observed != expected:
                    findings.append(
                        f"{path}: unsupported {prose_family} schema version "
                        f"{observed}; current is {expected}",
                    )
    return findings


def audit_text(
    path: Path,
    content: str,
) -> tuple[list[str], dict[str, set[str]]]:
    """Return findings and schema IDs grouped by family for one file."""
    findings: list[str] = []
    families: dict[str, set[str]] = defaultdict(set)
    findings.extend(_numeric_schema_findings(path, content))
    for schema_id in SCHEMA_ID_RE.findall(content):
        family = VERSION_SUFFIX_RE.sub("", schema_id)
        families[family].add(schema_id)
        if schema_id not in CURRENT_SCHEMA_IDENTIFIERS:
            findings.append(
                f"{path}: unsupported schema identifier {schema_id}",
            )
    if (
        UPSTREAM_TOLERANCE_FIELD in content
        and path not in UPSTREAM_FIELD_ALLOWLIST
    ):
        findings.append(
            f"{path}: upstream tolerance name escaped the import boundary",
        )
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
            findings.append(
                f"{family}: multiple schema versions: {sorted(versions)}",
            )
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
    retired_roots = (
        ".planning",
        ".superpowers",
        "docs/examples",
        "docs/releases",
    )
    findings = [
        f"retired archive root still exists: {path}"
        for path in retired_roots
        if (ROOT / path).exists()
    ]
    findings.extend(
        f"retired schema path still exists: {path}"
        for path in sorted(RETIRED_SCHEMA_PATHS)
        if (ROOT / path).exists()
    )
    findings.extend(audit_paths(first_party_paths()))
    if findings:
        print("\n".join(findings))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
