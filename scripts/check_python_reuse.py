#!/usr/bin/env python3
"""Reject known duplicate constants and hand-written memoization patterns."""

from __future__ import annotations

import ast
import subprocess
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path("scripts/check_python_reuse.py")
SOURCE_PREFIXES = (
    Path("src/sol_execbench"),
    Path("src/solar"),
    Path("scripts"),
)
EXCLUDED_PREFIXES = (
    Path("src/sol_execbench/_vendor"),
    Path("src/solar/_vendor"),
)
CANONICAL_LITERAL_OWNERS = {
    "capacity_constrained_tile_aware_v1": Path("src/solar/contracts.py"),
    "rocprofv3-counters": Path(
        "src/sol_execbench/cli/evaluation/profile_mode.py",
    ),
}
CANONICAL_ENUM_OWNERS = {
    ("low", "medium", "high"): (
        Path("src/sol_execbench/core/bench/diagnostic_sidecar.py"),
        "DiagnosticConfidence",
    ),
}
ENUM_MODELED_CONSTANT_NAMES = frozenset(
    {
        "COUNTER_REASON_ARTIFACT_INCOMPLETE",
        "COUNTER_REASON_AVAIL_FAILED",
        "COUNTER_REASON_COLLECTED",
        "COUNTER_REASON_PMC_CHECK_FAILED",
        "COUNTER_REASON_UNSUPPORTED",
        "DECISION_AUTO",
        "DECISION_NONE",
        "EXIT_EXECUTION",
        "EXIT_INPUT",
        "EXIT_RESULT_FAILED",
        "EXIT_SUCCESS",
        "EXIT_UNAVAILABLE",
        "GEN_INPUTS_DEVICE_MISMATCH",
        "GEN_INPUTS_ERROR",
        "GEN_INPUTS_OOM_BLOCKED",
        "GEN_INPUTS_SCHEMA_MISMATCH",
        "GEN_INPUTS_TIMEOUT",
        "PROFILE_NONE",
        "PROFILE_ROCPROFV3",
        "PROFILE_ROCPROFV3_COUNTERS",
        "ROCPROF_REASON_ARTIFACTS_REGISTERED",
        "ROCPROF_REASON_COMMAND_FAILED",
        "ROCPROF_REASON_COMMAND_TIMEOUT",
        "ROCPROF_REASON_DIAGNOSTIC_LOG_REGISTERED",
        "ROCPROF_REASON_NO_REGISTERED_ARTIFACTS",
        "ROCPROF_REASON_PARTIAL_ARTIFACT_COVERAGE",
        "ROCPROF_REASON_UNAVAILABLE",
        "STATIC_EVIDENCE_AUTO",
        "STATIC_EVIDENCE_NONE",
    },
)
CHECKSUM_MODULES = {
    Path("src/sol_execbench/core/integrity/checksums.py"),
    Path("src/solar/artifacts/checksums.py"),
}


def _subscript_container_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        return node.value.id
    return None


def _called_assignment_containers(nodes: Iterable[ast.stmt]) -> set[str]:
    containers: set[str] = set()
    for statement in nodes:
        for node in ast.walk(statement):
            assignments: list[tuple[ast.AST, ast.AST | None]] = []
            if isinstance(node, ast.Assign):
                assignments.extend(
                    (target, node.value) for target in node.targets
                )
            elif isinstance(node, ast.AnnAssign):
                assignments.append((node.target, node.value))
            for target, value in assignments:
                name = _subscript_container_name(target)
                if name is not None and isinstance(value, ast.Call):
                    containers.add(name)
    return containers


class _ReuseVisitor(ast.NodeVisitor):
    """Collect repository reuse-policy violations from one Python module."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.findings: list[str] = []

    def visit_Constant(self, node: ast.Constant) -> None:
        if (
            isinstance(node.value, str)
            and (owner := CANONICAL_LITERAL_OWNERS.get(node.value)) is not None
            and self.path not in {owner, SELF}
        ):
            self.findings.append(
                f"{self.path}:{node.lineno}: import canonical literal "
                f"{node.value!r} from {owner}",
            )

    def visit_Module(self, node: ast.Module) -> None:
        local_enum_names = {
            statement.name
            for statement in node.body
            if isinstance(statement, ast.ClassDef)
            and any(
                isinstance(base, ast.Name)
                and base.id in {"Enum", "IntEnum", "StrEnum"}
                for base in statement.bases
            )
        }
        for statement in node.body:
            assignments: list[tuple[ast.Name, ast.AST | None]] = []
            if isinstance(statement, ast.Assign):
                assignments.extend(
                    (target, statement.value)
                    for target in statement.targets
                    if isinstance(target, ast.Name)
                )
            elif isinstance(statement, ast.AnnAssign) and isinstance(
                statement.target,
                ast.Name,
            ):
                assignments.append((statement.target, statement.value))
            for target, value in assignments:
                if target.id in ENUM_MODELED_CONSTANT_NAMES:
                    self.findings.append(
                        f"{self.path}:{target.lineno}: model {target.id} "
                        "as a member of its canonical enum",
                    )
                if (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id == "str"
                    and len(value.args) == 1
                    and isinstance(value.args[0], ast.Attribute)
                    and isinstance(value.args[0].value, ast.Name)
                    and value.args[0].value.id in local_enum_names
                ):
                    self.findings.append(
                        f"{self.path}:{target.lineno}: do not duplicate an "
                        "enum member as a module constant; derive boundary "
                        "collections from the enum",
                    )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self.path not in CHECKSUM_MODULES and isinstance(
            node.func,
            ast.Attribute,
        ):
            direct_file_digest = (
                node.func.attr == "file_digest"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == "sha256"
            )
            read_bytes_digest = node.func.attr == "sha256" and any(
                isinstance(argument, ast.Call)
                and isinstance(argument.func, ast.Attribute)
                and argument.func.attr == "read_bytes"
                for argument in node.args
            )
            if direct_file_digest or read_bytes_digest:
                self.findings.append(
                    f"{self.path}:{node.lineno}: use the package-owned "
                    "sha256_file helper",
                )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if any(
            isinstance(base, ast.Name) and base.id == "StrEnum"
            for base in node.bases
        ):
            values = tuple(
                statement.value.value
                for statement in node.body
                if isinstance(statement, (ast.Assign, ast.AnnAssign))
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            )
            owner = CANONICAL_ENUM_OWNERS.get(values)
            if owner is not None and (self.path, node.name) != owner:
                owner_path, owner_name = owner
                self.findings.append(
                    f"{self.path}:{node.lineno}: reuse canonical enum "
                    f"{owner_name} from {owner_path}",
                )
        self.generic_visit(node)

    def _manual_memoization_container(
        self,
        statement: ast.stmt,
        following: ast.stmt | None,
    ) -> str | None:
        if not isinstance(statement, ast.If) or not isinstance(
            following,
            ast.Return,
        ):
            return None
        test = statement.test
        if (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.NotIn)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Name)
        ):
            container = test.comparators[0].id
            returned = _subscript_container_name(following.value)
            if (
                returned == container
                and container in _called_assignment_containers(statement.body)
            ):
                return container
        return None

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        following = [*node.body[1:], None]
        for statement, next_statement in zip(node.body, following, strict=True):
            container = self._manual_memoization_container(
                statement,
                next_statement,
            )
            if container is not None:
                self.findings.append(
                    f"{self.path}:{statement.lineno}: replace hand-written "
                    f"{container} memoization with functools.cache",
                )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)


def audit_source(path: Path, content: str) -> list[str]:
    """Return reuse-policy findings for one Python source file."""
    try:
        tree = ast.parse(content, filename=str(path))
    except SyntaxError:
        return []
    visitor = _ReuseVisitor(path)
    visitor.visit(tree)
    return visitor.findings


def first_party_python_paths() -> tuple[Path, ...]:
    """Return tracked and new first-party Python files in governed roots."""
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
        if path.suffix == ".py"
        and any(path.is_relative_to(prefix) for prefix in SOURCE_PREFIXES)
        and not any(path.is_relative_to(prefix) for prefix in EXCLUDED_PREFIXES)
    )


def audit_paths(paths: Iterable[Path]) -> list[str]:
    """Audit readable paths against constant and memoization ownership."""
    findings: list[str] = []
    for path in paths:
        try:
            content = (ROOT / path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        findings.extend(audit_source(path, content))
    return findings


def main() -> int:
    """Run repository Python reuse guardrails."""
    findings = audit_paths(first_party_python_paths())
    if findings:
        print("\n".join(findings))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
