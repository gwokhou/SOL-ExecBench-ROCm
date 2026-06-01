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
        return "Static source review blocked submission: " + "; ".join(blocking)


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
        for rule in _STATIC_RULES:
            issues.extend(_match_rule(path, content, rule))
        if float32_contract:
            issues.extend(_match_rule(path, content, _PRECISION_DOWNGRADE_RULE))
    return SourceReview(issues=tuple(issues))


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
