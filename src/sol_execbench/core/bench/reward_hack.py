# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, List

import torch

# ---------------------------------------------------------------------------
# Capture timing function identity at module load, before any user code runs.
# Used by check_monkey_patch() to detect post-import patching.
# ---------------------------------------------------------------------------
_ELAPSED_TIME_ADDR: int | None = None

try:
    import torch.cuda as _tc_init

    _ELAPSED_TIME_ADDR = id(_tc_init.Event.elapsed_time)
except Exception:
    pass


class RewardHackDetected(RuntimeError):
    """Raised when a reward-hacking pattern is detected in a submission."""


class SourceReviewSeverity(str, Enum):
    """Severity for static source review findings."""

    FLAG = "flag"
    BLOCK = "block"


@dataclass(frozen=True)
class SourceReviewIssue:
    """One static source review finding."""

    path: str
    rule: str
    severity: SourceReviewSeverity
    message: str
    evidence: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable issue payload."""
        return {
            "path": self.path,
            "rule": self.rule,
            "severity": self.severity.value,
            "message": self.message,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class SourceReview:
    """Static review result for a submitted solution."""

    issues: tuple[SourceReviewIssue, ...]

    @property
    def blocked(self) -> bool:
        """Whether any finding should reject execution."""
        return any(
            issue.severity == SourceReviewSeverity.BLOCK for issue in self.issues
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable review payload."""
        return {
            "blocked": self.blocked,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def format_blocking_message(self) -> str:
        """Return a compact human-readable blocking summary."""
        blocking = [
            f"{issue.path}:{issue.rule}: {issue.message} ({issue.evidence})"
            for issue in self.issues
            if issue.severity == SourceReviewSeverity.BLOCK
        ]
        evidence = json.dumps(self.to_dict(), sort_keys=True)
        return (
            "Static source review blocked submission: "
            + "; ".join(blocking)
            + f"; structured_evidence={evidence}"
        )


@dataclass(frozen=True)
class _SourceRule:
    rule: str
    severity: SourceReviewSeverity
    pattern: re.Pattern[str]
    message: str
    suffixes: tuple[str, ...] | None = None

    def applies_to(self, path: str) -> bool:
        return self.suffixes is None or path.endswith(self.suffixes)


_STATIC_RULES = (
    _SourceRule(
        "hidden_async_stream",
        SourceReviewSeverity.BLOCK,
        re.compile(
            r"\b(torch\.cuda\.Stream|torch\.cuda\.stream|wait_stream|hipStreamCreate|"
            r"cudaStreamCreate|hipStreamSynchronize|cudaStreamSynchronize)\b"
        ),
        "non-default stream or explicit stream synchronization can hide work from event timing",
    ),
    _SourceRule(
        "semantic_output_cache",
        SourceReviewSeverity.BLOCK,
        re.compile(
            r"(\bdata_ptr\s*\(|\blru_cache\b|\bglobal\s+[_A-Za-z0-9]*cache\b|"
            r"[_A-Za-z0-9]*cache\s*=\s*\{|\btobytes\s*\(|\bhashlib\b)"
        ),
        "data-pointer or content-keyed caching can reuse outputs across phases",
        suffixes=(".py",),
    ),
    _SourceRule(
        "unauthorized_file_or_loader",
        SourceReviewSeverity.BLOCK,
        re.compile(
            r"(\bopen\s*\(|\bPath\s*\(|\bread_text\s*\(|\bwrite_text\s*\(|"
            r"\bbase64\b|\bmarshal\.loads\b|\bpickle\.loads\b|\bctypes\.CDLL\b|"
            r"\bctypes\.cdll\b|\bdlopen\b|\btorch\.ops\.load_library\b|"
            r"\bload_inline\s*\(|\bcpp_extension\.load\s*\(|\bsubprocess\b|"
            r"\b__import__\s*\(|\bimportlib\.(import_module|util)\b|"
            r"\bgetattr\s*\(\s*os\s*,\s*['\"](system|popen|spawn[a-zA-Z_]*|exec[a-zA-Z_]*)['\"]\s*\)|"
            r"\bos\.(system|popen|spawn[a-zA-Z_]*|exec[a-zA-Z_]*)\s*\(|"
            r"\bpty\.spawn\s*\(|\bsocket\b|\burllib\b|\brequests\b)"
        ),
        "file I/O, embedded payload decoding, dynamic native loading, or process/network access is not allowed in submitted sources",
    ),
)

_PRECISION_DOWNGRADE_RULE = _SourceRule(
    "precision_downgrade",
    SourceReviewSeverity.BLOCK,
    re.compile(
        r"(\.half\s*\(|\.bfloat16\s*\(|\.to\s*\(\s*torch\.(float16|bfloat16)|"
        r"\btorch\.(float16|bfloat16)\b|\btl\.(float16|bfloat16)\b|"
        r"\.to\s*\(\s*tl\.(float16|bfloat16))"
    ),
    "precision downgrade is not allowed for float32 output contracts without explicit benchmark approval",
)

_RULE_BY_NAME = {
    rule.rule: rule for rule in (*_STATIC_RULES, _PRECISION_DOWNGRADE_RULE)
}

_PROCESS_ATTRS = {"system", "popen"}
_PROCESS_PREFIXES = ("spawn", "exec")
_RISKY_IMPORT_ROOTS = {
    "base64",
    "ctypes",
    "marshal",
    "pickle",
    "pty",
    "requests",
    "socket",
    "subprocess",
    "urllib",
}
_RISKY_FILE_CALLS = {"open", "Path", "pathlib.Path"}
_RISKY_DECODE_CALLS = {"eval", "exec", "compile", "__import__"}
_RISKY_METHODS = {"read_text", "write_text", "load_library", "load", "load_inline"}
_CACHE_METHODS = {"data_ptr", "tobytes"}
_PRECISION_ATTRS = {"half", "bfloat16"}
_PRECISION_DTYPE_NAMES = {"torch.float16", "torch.bfloat16", "tl.float16", "tl.bfloat16"}


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
    float32_contract = bool(output_dtypes) and any(
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
        if any(_target_is_cache(target) for target in node.targets):
            self._add("semantic_output_cache", node, self._node_source(node))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if _target_is_cache(node.target):
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
            self._add("unauthorized_file_or_loader", node, name or self._node_source(node))
        if self.float32_contract and self._is_precision_downgrade_call(node, name):
            self._add("precision_downgrade", node, name or self._node_source(node))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self.float32_contract and self._resolved_name(node) in _PRECISION_DTYPE_NAMES:
            self._add("precision_downgrade", node, self._node_source(node))
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
        return name in {"functools.lru_cache", "lru_cache", "hashlib.md5", "hashlib.sha1", "hashlib.sha256"} or name.endswith(
            tuple(f".{method}" for method in _CACHE_METHODS)
        )

    def _is_unauthorized_call(self, node: ast.Call, name: str) -> bool:
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
                self._add("semantic_output_cache", decorator, self._node_source(decorator))

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
        return "cache" in target.id.lower()
    if isinstance(target, ast.Attribute):
        return "cache" in target.attr.lower()
    if isinstance(target, (ast.Tuple, ast.List)):
        return any(_target_is_cache(elt) for elt in target.elts)
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


def check_monkey_patch() -> None:
    """Detect if torch.cuda.Event.elapsed_time has been patched.

    Compares the current function identity against the address captured at
    module load time.  Must be called before the timed section.

    Raises:
        RewardHackDetected: If the timing function has been replaced.
    """
    try:
        import torch.cuda as _tc

        if (
            _ELAPSED_TIME_ADDR is not None
            and id(_tc.Event.elapsed_time) != _ELAPSED_TIME_ADDR
        ):
            raise RewardHackDetected(
                "torch.cuda.Event.elapsed_time has been monkey-patched"
            )
    except RewardHackDetected:
        raise
    except Exception:
        pass


def check_thread_injection(threads_before: int, threads_after: int) -> None:
    """Detect if user code spawned background threads.

    Capture ``threading.active_count()`` before and after the user call and
    pass both values here.

    Raises:
        RewardHackDetected: If the thread count increased.
    """
    if threads_after > threads_before:
        raise RewardHackDetected(
            f"Thread injection detected: "
            f"{threads_after} threads after call vs {threads_before} before"
        )


def check_lazy_outputs(outputs: List[Any]) -> None:
    """Detect lazy/proxy tensors in the user output.

    Uses strict ``type()`` equality — not ``isinstance`` — so any subclass
    (including ``FakeTensor``) is rejected.

    Raises:
        RewardHackDetected: If any output is not exactly ``torch.Tensor``.
    """
    for t in outputs:
        if type(t) is not torch.Tensor:
            raise RewardHackDetected(
                f"Lazy evaluation detected: output is {type(t).__name__}, not torch.Tensor"
            )


def snapshot_critical_functions(namespace: dict, names: List[str]) -> dict[str, int]:
    """Capture ``id()`` of named functions from a namespace.

    Call this **before** user code is imported.  Pass the returned dict to
    :func:`check_eval_integrity` after user code runs.

    Args:
        namespace: The globals dict to snapshot (typically ``globals()``).
        names: Function names to capture.

    Returns:
        Mapping of name → ``id()`` for each name present in *namespace*.
    """
    return {name: id(namespace[name]) for name in names if name in namespace}


def check_eval_integrity(snapshot: dict[str, int], namespace: dict) -> None:
    """Verify that critical eval-driver functions have not been replaced.

    Compares the current ``id()`` of each snapshotted name against the
    value captured before user code was imported.

    Args:
        snapshot: The dict returned by :func:`snapshot_critical_functions`.
        namespace: The current globals dict to check.

    Raises:
        RewardHackDetected: If any function identity has changed.
    """
    for name, expected_id in snapshot.items():
        current = namespace.get(name)
        if current is None or id(current) != expected_id:
            raise RewardHackDetected(
                f"Eval driver integrity violated: '{name}' has been monkey-patched"
            )
