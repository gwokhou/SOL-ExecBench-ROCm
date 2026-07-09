# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reward hack defenses for SOL ExecBench evaluation.

Provides detection functions for four common reward-hacking patterns.
The identity of torch.cuda.Event.elapsed_time is captured at module load
time — before any user code is imported — so patching after the fact is
detected.
"""

from __future__ import annotations

import ast
from typing import Any, Iterable

import torch

from sol_execbench.core.bench.reward_hack.models import (
    _CACHE_METHODS,
    _PRECISION_DOWNGRADE_RULE,
    _PRECISION_ATTRS,
    _PRECISION_DTYPE_NAMES,
    _PROCESS_ATTRS,
    _PROCESS_PREFIXES,
    _RISKY_DECODE_CALLS,
    _RISKY_FILE_CALLS,
    _RISKY_IMPORT_ROOTS,
    _RISKY_METHODS,
    _RULE_BY_NAME,
    _STATIC_RULES,
    _SourceRule,
    SourceReview,
    SourceReviewIssue,
)


def review_solution_sources(
    solution: Any,
    *,
    output_dtypes: dict[str, torch.dtype] | None = None,
) -> SourceReview:
    """Statically review solution sources for reward-hack patterns.

    The review is intentionally conservative for execution-blocking exploit
    families while remaining structured and inspectable for tests and tooling.
    """
    issues: list[SourceReviewIssue] = []
    float32_contract = bool(output_dtypes) and all(
        dtype == torch.float32 for dtype in output_dtypes.values()
    )
    for source in getattr(solution, "sources", []):
        path = str(getattr(source, "path", ""))
        content = str(getattr(source, "content", ""))
        if path.endswith(".py"):
            issues.extend(_match_python_source(path, content, float32_contract))
        else:
            for rule in _STATIC_RULES:
                issues.extend(_match_rule(path, content, rule))
            if float32_contract:
                issues.extend(_match_rule(path, content, _PRECISION_DOWNGRADE_RULE))
    return SourceReview(issues=tuple(issues))


def _match_python_source(
    path: str,
    content: str,
    float32_contract: bool,
) -> Iterable[SourceReviewIssue]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        issues: list[SourceReviewIssue] = []
        for rule in _STATIC_RULES:
            issues.extend(_match_rule(path, content, rule))
        if float32_contract:
            issues.extend(_match_rule(path, content, _PRECISION_DOWNGRADE_RULE))
        return tuple(issues)

    visitor = _PythonSourceReviewVisitor(path, content, float32_contract)
    visitor.visit(tree)
    return tuple(visitor.issues)


class _PythonSourceReviewVisitor(ast.NodeVisitor):
    def __init__(self, path: str, content: str, float32_contract: bool) -> None:
        self.path = path
        self.content = content
        self.float32_contract = float32_contract
        self.issues: list[SourceReviewIssue] = []
        self.aliases: dict[str, str] = {}

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", maxsplit=1)[0]
            local_name = alias.asname or root
            self.aliases[local_name] = alias.name
            if root in _RISKY_IMPORT_ROOTS:
                self._add("unauthorized_file_or_loader", node, alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        root = module.split(".", maxsplit=1)[0]
        if root in _RISKY_IMPORT_ROOTS:
            self._add("unauthorized_file_or_loader", node, module)
        for alias in node.names:
            if alias.name == "*":
                continue
            local_name = alias.asname or alias.name
            imported_name = f"{module}.{alias.name}" if module else alias.name
            self.aliases[local_name] = imported_name
            if module == "os" and _is_process_attr(alias.name):
                self._add("unauthorized_file_or_loader", node, imported_name)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if _assignment_creates_semantic_cache(node.targets, node.value):
            self._add("semantic_output_cache", node, self._node_source(node))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None and _assignment_creates_semantic_cache(
            [node.target], node.value
        ):
            self._add("semantic_output_cache", node, self._node_source(node))
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        if any("cache" in name.lower() for name in node.names):
            self._add("semantic_output_cache", node, ", ".join(node.names))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = self._resolved_name(node.func)
        if self._is_hidden_stream_call(name):
            self._add("hidden_async_stream", node, name)
        if self._is_cache_call(name):
            self._add("semantic_output_cache", node, name)
        if self._is_unauthorized_call(node, name):
            self._add(
                "unauthorized_file_or_loader", node, name or self._node_source(node)
            )
        if self.float32_contract and self._is_precision_downgrade_call(node, name):
            self._add("precision_downgrade", node, name or self._node_source(node))
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_decorators(node.decorator_list)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_decorators(node.decorator_list)
        self.generic_visit(node)

    def _is_hidden_stream_call(self, name: str) -> bool:
        return name in {
            "torch.cuda.Stream",
            "torch.cuda.stream",
            "torch.cuda.ExternalStream",
        } or name.endswith(".wait_stream")

    def _is_cache_call(self, name: str) -> bool:
        return name in {
            "functools.lru_cache",
            "lru_cache",
            "hashlib.md5",
            "hashlib.sha1",
            "hashlib.sha256",
        } or name.endswith(tuple(f".{method}" for method in _CACHE_METHODS))

    def _is_unauthorized_call(self, node: ast.Call, name: str) -> bool:
        if name == "triton.language.load":
            return False
        if name in _RISKY_FILE_CALLS or name in _RISKY_DECODE_CALLS:
            return True
        if name in {
            "importlib.import_module",
            "importlib.util.spec_from_file_location",
            "ctypes.CDLL",
            "ctypes.cdll.LoadLibrary",
            "marshal.loads",
            "pickle.loads",
            "base64.b64decode",
            "pty.spawn",
            "socket.socket",
            "urllib.request.urlopen",
            "requests.get",
            "requests.post",
            "requests.request",
            "torch.ops.load_library",
            "cpp_extension.load",
            "load_inline",
        }:
            return True
        if name in _RISKY_METHODS or name.endswith(
            tuple(f".{method}" for method in _RISKY_METHODS)
        ):
            return True
        if name.startswith("os.") and _is_process_attr(name.rsplit(".", 1)[-1]):
            return True
        if name.startswith("subprocess.") or name.startswith("socket."):
            return True
        return self._is_risky_getattr_call(node)

    def _is_precision_downgrade_call(self, node: ast.Call, name: str) -> bool:
        if name.endswith(tuple(f".{attr}" for attr in _PRECISION_ATTRS)):
            return True
        if name.endswith(".to"):
            return any(
                self._resolved_name(arg) in _PRECISION_DTYPE_NAMES for arg in node.args
            ) or any(
                self._resolved_name(keyword.value) in _PRECISION_DTYPE_NAMES
                for keyword in node.keywords
            )
        return False

    def _check_decorators(self, decorators: list[ast.expr]) -> None:
        for decorator in decorators:
            name = self._resolved_name(
                decorator.func if isinstance(decorator, ast.Call) else decorator
            )
            if name in {"functools.lru_cache", "lru_cache"}:
                self._add(
                    "semantic_output_cache", decorator, self._node_source(decorator)
                )

    def _is_risky_getattr_call(self, node: ast.Call) -> bool:
        if self._resolved_name(node.func) != "getattr" or len(node.args) < 2:
            return False
        if not isinstance(node.args[1], ast.Constant) or not isinstance(
            node.args[1].value, str
        ):
            return False
        attr = node.args[1].value
        base = self._resolved_name(node.args[0])
        return base == "os" and _is_process_attr(attr)

    def _resolved_name(self, node: ast.AST) -> str:
        return _resolve_alias(_dotted_name(node), self.aliases)

    def _add(self, rule_name: str, node: ast.AST, evidence: str) -> None:
        rule = _RULE_BY_NAME[rule_name]
        self.issues.append(
            SourceReviewIssue(
                path=self.path,
                rule=rule.rule,
                severity=rule.severity,
                message=rule.message,
                evidence=(evidence or self._node_source(node)).strip()[:120],
            )
        )

    def _node_source(self, node: ast.AST) -> str:
        return ast.get_source_segment(self.content, node) or ast.unparse(node)


def _target_is_cache(target: ast.AST) -> bool:
    if isinstance(target, ast.Name):
        return _name_is_cache_container(target.id)
    if isinstance(target, ast.Attribute):
        return _name_is_cache_container(target.attr)
    if isinstance(target, (ast.Tuple, ast.List)):
        return any(_target_is_cache(elt) for elt in target.elts)
    return False


def _name_is_cache_container(name: str) -> bool:
    normalized = name.lower()
    return normalized in {
        "cache",
        "_cache",
        "cache_dict",
        "_cache_dict",
        "memo",
        "_memo",
    } or normalized.endswith(("_cache_dict", "_memo"))


def _assignment_creates_semantic_cache(
    targets: Iterable[ast.AST], value: ast.AST
) -> bool:
    return any(_target_is_cache(target) for target in targets) and _value_is_cache(
        value
    )


def _value_is_cache(value: ast.AST) -> bool:
    if isinstance(value, (ast.Dict, ast.Set)):
        return True
    if isinstance(value, ast.Call):
        return _dotted_name(value.func) in {
            "dict",
            "set",
            "collections.defaultdict",
            "defaultdict",
            "OrderedDict",
            "collections.OrderedDict",
            "functools.lru_cache",
            "lru_cache",
        }
    return False


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call) and _dotted_name(node.func) == "__import__":
        if node.args and isinstance(node.args[0], ast.Constant):
            if isinstance(node.args[0].value, str):
                return node.args[0].value
    return ""


def _resolve_alias(name: str, aliases: dict[str, str]) -> str:
    if not name:
        return ""
    head, sep, tail = name.partition(".")
    resolved = aliases.get(head, head)
    return f"{resolved}{sep}{tail}" if sep else resolved


def _is_process_attr(attr: str) -> bool:
    return attr in _PROCESS_ATTRS or attr.startswith(_PROCESS_PREFIXES)


def _match_rule(
    path: str,
    content: str,
    rule: _SourceRule,
) -> Iterable[SourceReviewIssue]:
    if not rule.applies_to(path):
        return ()
    found: list[SourceReviewIssue] = []
    for match in rule.pattern.finditer(_strip_comments(content)):
        evidence = match.group(0).strip()
        found.append(
            SourceReviewIssue(
                path=path,
                rule=rule.rule,
                severity=rule.severity,
                message=rule.message,
                evidence=evidence[:120],
            )
        )
    return tuple(found)


def _strip_comments(content: str) -> str:
    """Remove simple line comments before pattern scanning."""
    lines = []
    for line in content.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#") or stripped.startswith("//"):
            continue
        lines.append(line)
    return "\n".join(lines)
