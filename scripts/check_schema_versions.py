#!/usr/bin/env python3
"""Fail when first-party files contain non-current schema contracts."""

from __future__ import annotations

import ast
import re
import subprocess
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from types import MappingProxyType, ModuleType

from sol_execbench.core.dataset import (
    schema_versions as dataset_schema_versions,
)
from sol_execbench.core.integrity.artifact_registry import (
    ARTIFACT_SCHEMA_MEMBERS,
    ARTIFACT_SCHEMA_REGISTRIES,
    CURRENT_NUMERIC_ARTIFACT_SCHEMAS,
    CURRENT_STRING_ARTIFACT_SCHEMAS,
)
from sol_execbench.core.integrity.protocol_versions import (
    CURRENT_WIRE_PROTOCOLS,
    WireProtocol,
)
from solar import schema_versions as solar_schema_versions
from solar.schema_versions import (
    CURRENT_NUMERIC_SCHEMA_VERSIONS as SOLAR_NUMERIC_SCHEMA_VERSIONS,
    CURRENT_STRING_SCHEMA_VERSIONS,
    SchemaVersion as SolarSchemaVersion,
)

ROOT = Path(__file__).resolve().parents[1]
VERSION_SUFFIX_RE = re.compile(r"\.v\d+$")
CURRENT_SCHEMA_IDENTIFIERS = (
    CURRENT_STRING_ARTIFACT_SCHEMAS | CURRENT_STRING_SCHEMA_VERSIONS
)
CURRENT_VERSIONED_WIRE_IDENTIFIERS = (
    CURRENT_SCHEMA_IDENTIFIERS | CURRENT_WIRE_PROTOCOLS
)
NON_NAMESPACED_SCHEMA_FAMILIES = frozenset(
    VERSION_SUFFIX_RE.sub("", schema_id)
    for schema_id in CURRENT_SCHEMA_IDENTIFIERS
    if not schema_id.startswith(("sol_execbench.", "solar."))
)
VERSIONED_WIRE_PREFIXES = (
    r"(?:sol_execbench|solar)(?:\.[a-z0-9_]+)+",
    *map(re.escape, sorted(NON_NAMESPACED_SCHEMA_FAMILIES)),
)
VERSIONED_WIRE_ID_RE = re.compile(
    rf"(?:{'|'.join(VERSIONED_WIRE_PREFIXES)})\.v\d+"
)
CURRENT_NUMERIC_SCHEMA_VERSIONS = MappingProxyType(
    {
        **CURRENT_NUMERIC_ARTIFACT_SCHEMAS,
        **SOLAR_NUMERIC_SCHEMA_VERSIONS,
    }
)
READ_ONLY_MAPPING_TYPE = type(MappingProxyType({}))
EXECBENCH_SCHEMA_REGISTRIES = frozenset(
    {
        Path(
            "src/sol_execbench/core/bench/performance_model/diagnostic_schema_versions.py"
        ),
        Path(
            "src/sol_execbench/core/bench/performance_model/lifecycle/schema_versions.py"
        ),
        Path(
            "src/sol_execbench/core/bench/performance_model/schema_versions.py"
        ),
        Path("src/sol_execbench/core/bench/rocm_profiler/schema_versions.py"),
        Path("src/sol_execbench/core/control_plane_schema_versions.py"),
        Path("src/sol_execbench/core/data/schema_versions.py"),
        Path("src/sol_execbench/core/dataset/schema_versions.py"),
        Path("src/sol_execbench/core/platform/schema_versions.py"),
        Path("src/sol_execbench/core/scoring/schema_versions.py"),
        Path("src/sol_execbench/tools/amd_isa/schema_versions.py"),
    }
)
ARTIFACT_AGGREGATE_REGISTRY = Path(
    "src/sol_execbench/core/integrity/artifact_registry.py"
)
ARTIFACT_AGGREGATE_IMPORT = "sol_execbench.core.integrity.artifact_registry"
ARTIFACT_AGGREGATE_IMPORT_ALLOWLIST = frozenset(
    {
        Path("scripts/check_non_canonical_artifacts.py"),
        Path("scripts/check_schema_versions.py"),
    }
)
SOLAR_SCHEMA_REGISTRY = Path("src/solar/schema_versions.py")
PROTOCOL_REGISTRY = Path(
    "src/sol_execbench/core/integrity/protocol_versions.py"
)
ARTIFACT_SCHEMA_CLASS_NAMES = frozenset(
    registry.__name__ for registry in ARTIFACT_SCHEMA_REGISTRIES
)
ARTIFACT_SCHEMA_OWNER_BY_VALUE = {
    member.value: type(member).__name__ for member in ARTIFACT_SCHEMA_MEMBERS
}
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
            "must use the matching family constant",
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


class _SchemaAccessPolicyVisitor(ast.NodeVisitor):
    """Reject schema lookup and coercion patterns that weaken exact contracts."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.findings: list[str] = []

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "SCHEMA_VERSIONS"
        ):
            self.findings.append(
                f"{self.path}:{node.lineno}: import the named schema-version "
                "enum member instead of indexing SCHEMA_VERSIONS",
            )
        self.generic_visit(node)

    def _reject_enum_relay(
        self,
        targets: Iterable[ast.AST],
        value: ast.AST | None,
    ) -> None:
        if not (
            isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.value.id
            in ARTIFACT_SCHEMA_CLASS_NAMES | {"SolarSchemaVersion"}
        ):
            return
        for target in targets:
            if isinstance(target, ast.Name) and target.id.endswith(
                "_SCHEMA_VERSION"
            ):
                self.findings.append(
                    f"{self.path}:{target.lineno}: use {value.value.id}."
                    f"{value.attr} directly instead of relaying it through "
                    f"{target.id}",
                )

    def visit_Assign(self, node: ast.Assign) -> None:
        self._reject_enum_relay(node.targets, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._reject_enum_relay((node.target,), node.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in {"int", "str"}
            and any(_mentions_schema_version(item) for item in node.args)
        ):
            self.findings.append(
                f"{self.path}:{node.lineno}: schema_version readers must not "
                f"coerce with {node.func.id}()",
            )
        self.generic_visit(node)


class _ArtifactAggregateImportVisitor(ast.NodeVisitor):
    """Keep the cross-domain aggregate out of production dependencies."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.findings: list[str] = []

    def _record(self, node: ast.Import | ast.ImportFrom) -> None:
        self.findings.append(
            f"{self.path}:{node.lineno}: artifact_registry is audit-only; "
            "import the owning domain schema enum directly",
        )

    def visit_Import(self, node: ast.Import) -> None:
        if any(
            alias.name == ARTIFACT_AGGREGATE_IMPORT
            or alias.name.startswith(f"{ARTIFACT_AGGREGATE_IMPORT}.")
            for alias in node.names
        ):
            self._record(node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == ARTIFACT_AGGREGATE_IMPORT:
            self._record(node)
        self.generic_visit(node)


def _python_numeric_schema_findings(path: Path, content: str) -> list[str]:
    """Reject raw positive numeric schema versions outside their registries."""
    if path.suffix != ".py" or path.parts[0] not in {"src", "scripts"}:
        return []
    if path in {
        *EXECBENCH_SCHEMA_REGISTRIES,
        SOLAR_SCHEMA_REGISTRY,
    }:
        return []
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []
    visitor = _NumericSchemaLiteralVisitor(path)
    visitor.visit(tree)
    access_visitor = _SchemaAccessPolicyVisitor(path)
    access_visitor.visit(tree)
    aggregate_import_visitor = _ArtifactAggregateImportVisitor(path)
    if path not in ARTIFACT_AGGREGATE_IMPORT_ALLOWLIST:
        aggregate_import_visitor.visit(tree)
    return [
        *visitor.findings,
        *access_visitor.findings,
        *aggregate_import_visitor.findings,
    ]


def registry_findings() -> list[str]:
    """Require immutable registries derived exactly from their enums."""
    registries = {
        "sol_execbench CURRENT_NUMERIC_ARTIFACT_SCHEMAS": (
            CURRENT_NUMERIC_ARTIFACT_SCHEMAS
        ),
        "solar CURRENT_NUMERIC_SCHEMA_VERSIONS": SOLAR_NUMERIC_SCHEMA_VERSIONS,
    }
    findings = [
        f"{name} must be a read-only mapping"
        for name, registry in registries.items()
        if not isinstance(registry, READ_ONLY_MAPPING_TYPE)
    ]
    expected_members = tuple(
        member for registry in ARTIFACT_SCHEMA_REGISTRIES for member in registry
    )
    expected_numeric = _numeric_schema_constants(dataset_schema_versions)
    expected_solar_numeric = _numeric_schema_constants(solar_schema_versions)
    if expected_members != ARTIFACT_SCHEMA_MEMBERS:
        findings.append("ARTIFACT_SCHEMA_MEMBERS must derive from domain enums")
    if dict(CURRENT_NUMERIC_ARTIFACT_SCHEMAS) != expected_numeric:
        findings.append(
            "CURRENT_NUMERIC_ARTIFACT_SCHEMAS must contain every "
            "family constant",
        )
    if dict(SOLAR_NUMERIC_SCHEMA_VERSIONS) != expected_solar_numeric:
        findings.append(
            "solar CURRENT_NUMERIC_SCHEMA_VERSIONS must contain every family "
            "constant",
        )
    expected_string_values = frozenset(
        member.value for member in ARTIFACT_SCHEMA_MEMBERS
    )
    if expected_string_values != CURRENT_STRING_ARTIFACT_SCHEMAS:
        findings.append(
            "CURRENT_STRING_ARTIFACT_SCHEMAS must derive from domain enums"
        )
    if len(CURRENT_STRING_ARTIFACT_SCHEMAS) != len(ARTIFACT_SCHEMA_MEMBERS):
        findings.append("domain artifact schema values must be unique")
    if (
        frozenset(protocol.value for protocol in WireProtocol)
        != CURRENT_WIRE_PROTOCOLS
    ):
        findings.append("CURRENT_WIRE_PROTOCOLS must derive from WireProtocol")
    if len(CURRENT_WIRE_PROTOCOLS) != len(WireProtocol):
        findings.append("WireProtocol values must be unique")
    if CURRENT_SCHEMA_IDENTIFIERS & CURRENT_WIRE_PROTOCOLS:
        findings.append("schema and protocol registries must be disjoint")
    if (
        frozenset(version.value for version in SolarSchemaVersion)
        != CURRENT_STRING_SCHEMA_VERSIONS
    ):
        findings.append(
            "solar CURRENT_STRING_SCHEMA_VERSIONS must derive from SchemaVersion"
        )
    if len(CURRENT_STRING_SCHEMA_VERSIONS) != len(SolarSchemaVersion):
        findings.append("solar SchemaVersion values must be unique")
    return findings


def _numeric_schema_constants(module: ModuleType) -> dict[str, int]:
    suffix = "_SCHEMA_VERSION"
    return {
        name.removesuffix(suffix).lower(): value
        for name, value in vars(module).items()
        if name.endswith(suffix)
        and isinstance(value, int)
        and not isinstance(value, bool)
    }


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
    """Return findings and versioned wire IDs grouped by family."""
    findings: list[str] = []
    families: dict[str, set[str]] = defaultdict(set)
    findings.extend(_numeric_schema_findings(path, content))
    for identifier in VERSIONED_WIRE_ID_RE.findall(content):
        family = VERSION_SUFFIX_RE.sub("", identifier)
        families[family].add(identifier)
        if identifier not in CURRENT_VERSIONED_WIRE_IDENTIFIERS:
            findings.append(
                f"{path}: unsupported versioned wire identifier {identifier}",
            )
        elif (
            path.suffix == ".py"
            and path.parts[0] in {"src", "scripts"}
            and (
                (
                    identifier in CURRENT_STRING_ARTIFACT_SCHEMAS
                    and path not in EXECBENCH_SCHEMA_REGISTRIES
                )
                or (
                    identifier in CURRENT_STRING_SCHEMA_VERSIONS
                    and path != SOLAR_SCHEMA_REGISTRY
                )
                or (
                    identifier in CURRENT_WIRE_PROTOCOLS
                    and path != PROTOCOL_REGISTRY
                )
            )
        ):
            owner = (
                "WireProtocol"
                if identifier in CURRENT_WIRE_PROTOCOLS
                else ARTIFACT_SCHEMA_OWNER_BY_VALUE.get(
                    identifier,
                    "SolarSchemaVersion",
                )
            )
            findings.append(
                f"{path}: versioned wire identifier {identifier} must be "
                f"referenced through {owner}",
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
                f"{family}: multiple versioned wire identifiers: "
                f"{sorted(versions)}",
            )
    return findings


def registry_usage_findings(paths: Iterable[Path]) -> list[str]:
    """Reject canonical registry entries with no non-test owner or artifact."""
    registry_paths = {
        *EXECBENCH_SCHEMA_REGISTRIES,
        ARTIFACT_AGGREGATE_REGISTRY,
        SOLAR_SCHEMA_REGISTRY,
        PROTOCOL_REGISTRY,
    }
    contents: list[str] = []
    for path in paths:
        if path in registry_paths or path.parts[0] == "tests":
            continue
        try:
            contents.append((ROOT / path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue
    corpus = "\n".join(contents)
    findings = [
        "unused artifact schema registration: "
        f"{type(version).__name__}.{version.name}"
        for version in ARTIFACT_SCHEMA_MEMBERS
        if f"{type(version).__name__}.{version.name}" not in corpus
        and version.value not in corpus
    ]
    findings.extend(
        f"unused wire protocol registration: WireProtocol.{protocol.name}"
        for protocol in WireProtocol
        if f"WireProtocol.{protocol.name}" not in corpus
        and protocol.value not in corpus
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
    paths = first_party_paths()
    findings.extend(registry_findings())
    findings.extend(registry_usage_findings(paths))
    findings.extend(audit_paths(paths))
    if findings:
        print("\n".join(findings))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
