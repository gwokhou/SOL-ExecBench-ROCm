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

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class RewardHackError(RuntimeError):
    """Raised when a reward-hacking pattern is detected in a submission."""


class SourceReviewSeverity(StrEnum):
    """Severity for static source review findings."""

    FLAG = "flag"
    BLOCK = "block"


@dataclass(frozen=True, slots=True, kw_only=True)
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
            "severity": self.severity,
            "message": self.message,
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceReview:
    """Static review result for a submitted solution."""

    issues: tuple[SourceReviewIssue, ...]

    @property
    def blocked(self) -> bool:
        """Whether any finding should reject execution."""
        return any(
            issue.severity == SourceReviewSeverity.BLOCK
            for issue in self.issues
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


@dataclass(frozen=True, slots=True, kw_only=True)
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
        rule="hidden_async_stream",
        severity=SourceReviewSeverity.BLOCK,
        pattern=re.compile(
            r"\b(torch\.cuda\.Stream|torch\.cuda\.stream|wait_stream|hipStreamCreate|"
            r"cudaStreamCreate|hipStreamSynchronize|cudaStreamSynchronize|"
            r"torch\.cuda\.CUDAGraph|torch\.cuda\.graph|make_graphed_callables|"
            r"capture_begin|capture_end|replay|hipGraph[A-Za-z0-9_]*|"
            r"cudaGraph[A-Za-z0-9_]*)\b",
        ),
        message="non-default streams or graph capture can hide work from event timing",
    ),
    _SourceRule(
        rule="semantic_output_cache",
        severity=SourceReviewSeverity.BLOCK,
        pattern=re.compile(
            r"(\bdata_ptr\s*\(|\blru_cache\b|\bfunctools\.cache\b|\bcached_property\b|"
            r"\bglobal\s+[_A-Za-z0-9]*cache\b|"
            r"[_A-Za-z0-9]*cache\s*=\s*\{|\btobytes\s*\(|\bhashlib\b)",
        ),
        message="data-pointer or content-keyed caching can reuse outputs across phases",
        suffixes=(".py",),
    ),
    _SourceRule(
        rule="parallel_execution",
        severity=SourceReviewSeverity.BLOCK,
        pattern=re.compile(
            r"\b(threading|_thread|multiprocessing|concurrent\.futures|"
            r"pthread_create|std::thread|hipLaunchHostFunc|"
            r"torch\.jit\.fork|jit\.fork)\b",
        ),
        message="submission-created workers can escape timed execution and cleanup",
    ),
    _SourceRule(
        rule="unauthorized_file_or_loader",
        severity=SourceReviewSeverity.BLOCK,
        pattern=re.compile(
            r"(\bopen\s*\(|\bPath\s*\(|\bread_text\s*\(|\bwrite_text\s*\(|"
            r"\bbase64\b|\bmarshal\.loads\b|\bpickle\.loads\b|\bctypes\.CDLL\b|"
            r"\bctypes\.cdll\b|\bdlopen\b|\btorch\.ops\.load_library\b|"
            r"\bload_inline\s*\(|\bcpp_extension\.load\s*\(|\bsubprocess\b|"
            r"\b__import__\s*\(|\bimportlib\.(import_module|util)\b|"
            r"\bgetattr\s*\(\s*os\s*,\s*['\"](system|popen|spawn[a-zA-Z_]*|exec[a-zA-Z_]*|"
            r"__dict__|__getattribute__)['\"]\s*\)|"
            r"\bos\.(system|popen|spawn[a-zA-Z_]*|exec[a-zA-Z_]*|"
            r"write|dup|dup2|dup3|fdopen|open|_exit|abort)\s*\(|"
            r"\bos\.__dict__\b|\bos\.__getattribute__\b|"
            r"\b__builtins__\b|"
            r"\b(mmap|fcntl)\b|"
            r"\btypes\.(FunctionType|CodeType|MethodType)\b|"
            r"\b(system|popen|fork)\s*\(|"
            r"\bpty\.spawn\s*\(|\bsocket\b|\burllib\b|\brequests\b)",
        ),
        message="file I/O, embedded payload decoding, dynamic native loading, or process/network access is not allowed in submitted sources",
    ),
)


_PRECISION_DOWNGRADE_RULE = _SourceRule(
    rule="precision_downgrade",
    severity=SourceReviewSeverity.FLAG,
    pattern=re.compile(
        r"(\.half\s*\(|\.bfloat16\s*\(|\.to\s*\(\s*torch\.(float16|bfloat16)|"
        r"\btorch\.(float16|bfloat16)\b|\btl\.(float16|bfloat16)\b|"
        r"\.to\s*\(\s*tl\.(float16|bfloat16))",
    ),
    message="precision downgrade is not allowed for float32 output contracts without explicit benchmark approval",
)


_RULE_BY_NAME = {
    rule.rule: rule for rule in (*_STATIC_RULES, _PRECISION_DOWNGRADE_RULE)
}


_PROCESS_ATTRS = {"system", "popen"}


_PROCESS_PREFIXES = ("spawn", "exec")


_RISKY_IMPORT_ROOTS = {
    "base64",
    "builtins",
    "ctypes",
    "fcntl",
    "marshal",
    "gc",
    "inspect",
    "mmap",
    "pickle",
    "pty",
    "requests",
    "socket",
    "subprocess",
    "sys",
    "sol_execbench",
    "solar",
    "urllib",
}


_PARALLEL_IMPORT_ROOTS = {
    "_thread",
    "concurrent",
    "multiprocessing",
    "threading",
}


_RISKY_FILE_CALLS = {"open", "Path", "pathlib.Path"}


_RISKY_DECODE_CALLS = {"eval", "exec", "compile", "__import__"}


_RISKY_METHODS = {
    "read_text",
    "write_text",
    "load_library",
    "load",
    "load_inline",
}


# ``os`` module calls that forge traces or duplicate trusted file descriptors
# (audit static-b8): the exact primitives a candidate uses to write forged
# JSONL to the preserved real-stdout fd.  Never required by a compute kernel.
_RISKY_OS_IO_CALLS = frozenset(
    {"write", "dup", "dup2", "dup3", "fdopen", "open"}
)


# ``os`` calls that short-circuit the process after emitting forged traces or
# before a reward-hack check runs (audit eval_driver.py:34).  A normal eval
# driver exits 0 only via its own ``os._exit``; a candidate never needs these.
_RISKY_OS_EXIT_CALLS = frozenset({"_exit", "abort"})


# Module roots whose direct use manipulates file descriptors, memory mappings,
# or advisory locks, bypassing the authorized-call allowlist (audit static-b8).
_RISKY_NATIVE_MODULE_ROOTS = frozenset({"mmap", "fcntl"})


# ``types`` attributes that construct callables/code objects from raw bytecode
# (audit static-b6), enabling marshal-based obfuscation to hide exploits.
_RISKY_TYPES_ATTRS = frozenset({"FunctionType", "CodeType", "MethodType"})


# ``os`` attribute lookups that re-expose process-execution or storage machinery
# after direct-name blocking (audit static-b7): ``os.__dict__`` and
# ``os.__getattribute__`` restore access to ``os.system``/``os.popen``.
_OS_DICT_ATTRS = frozenset({"__dict__", "__getattribute__"})


_CACHE_METHODS = {"data_ptr", "tobytes"}


# Attribute names that create or enter a non-default HIP stream when
# fetched indirectly via ``getattr`` (paper §4.4.1 concurrency exploit).
# Direct ``torch.cuda.Stream()`` calls are already caught by the static regex
# and the AST stream check. This set also covers dynamic stream lookup.
# obfuscation that bypasses those direct-name checks.
_INDIRECT_STREAM_ATTRS = {"Stream", "ExternalStream", "stream"}


_GRAPH_CALLS = {
    "torch.cuda.CUDAGraph",
    "torch.cuda.graph",
    "torch.cuda.make_graphed_callables",
}


_GRAPH_METHODS = {"capture_begin", "capture_end", "replay"}


# Resolved call targets that fork parallel TorchScript execution
# (paper §4.4.1 "JIT Forking" concurrency exploit). Sibling to the
# thread/stream vectors: the runtime thread-count guard cannot observe the
# C++ TorchScript thread that ``torch.jit.fork`` spawns, so the exploit is
# blocked statically rather than detected by ``check_thread_injection``.
_JIT_FORK_CALLS = {"torch.jit.fork"}


_PRECISION_ATTRS = {"half", "bfloat16"}


_PRECISION_DTYPE_NAMES = {
    "torch.float16",
    "torch.bfloat16",
    "tl.float16",
    "tl.bfloat16",
}
