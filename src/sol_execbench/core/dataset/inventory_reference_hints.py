# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Reference runtime hint classification for dataset inventory."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

from sol_execbench.core.data.definition import Definition


NVIDIA_RUNTIME_BLOCKER_HINTS = ("cupy", "cuda.c", "cuda runtime", "nvrtc")
NVIDIA_LEXICAL_FALSE_POSITIVE_HINTS = ("cublas", "cutlass")


@dataclasses.dataclass(frozen=True)
class ReferenceRuntimeHintEvidence:
    """Evidence record for a reference runtime hint match."""

    token: str
    match_kind: str
    context: str
    line_number: int | None = None


def classify_reference_runtime_hints(
    definition: Definition, reference_path: Path | None
) -> tuple[list[str], list[ReferenceRuntimeHintEvidence]]:
    """Classify reference runtime hints into blockers and false positives."""
    blocker_hints: list[str] = []
    false_positive_evidence: list[ReferenceRuntimeHintEvidence] = []

    lines: list[str] = []
    lines.extend(definition.reference.splitlines())
    if reference_path is not None and reference_path.is_file():
        lines.extend(reference_path.read_text(encoding="utf-8").splitlines())

    in_docstring = False
    import_patterns = {
        hint: re.compile(
            r"\b(?:from\s+\S+\s+)?import\s+[^#\n]*\b" + re.escape(hint) + r"\b",
            re.IGNORECASE,
        )
        for hint in NVIDIA_LEXICAL_FALSE_POSITIVE_HINTS
    }
    call_patterns = {
        hint: re.compile(
            rf"\b(?P<callee>[\w]*{re.escape(hint)}[\w]*)\s*\(",
            re.IGNORECASE,
        )
        for hint in NVIDIA_LEXICAL_FALSE_POSITIVE_HINTS
    }
    native_patterns = {
        hint: re.compile(rf"\b[\w]*{re.escape(hint)}[\w]*\.(cu|cuh)\b", re.IGNORECASE)
        for hint in NVIDIA_LEXICAL_FALSE_POSITIVE_HINTS
    }
    class_patterns = {
        hint: re.compile(
            rf"^\s*class\s+\w*{re.escape(hint)}\w*\b",
            re.IGNORECASE,
        )
        for hint in NVIDIA_LEXICAL_FALSE_POSITIVE_HINTS
    }
    variable_patterns = {
        hint: re.compile(rf"\b[a-z_][a-z0-9_]*{re.escape(hint)}[a-z0-9_]*\b")
        for hint in NVIDIA_LEXICAL_FALSE_POSITIVE_HINTS
    }

    for line_idx, line in enumerate(lines, start=1):
        lowered_line = line.lower()
        context = line.strip()[:200]
        stripped = line.strip()
        is_comment = stripped.startswith("#")
        is_docstring_line = (
            in_docstring or stripped.startswith('"""') or stripped.startswith("'''")
        )

        for hint in NVIDIA_RUNTIME_BLOCKER_HINTS:
            if hint in lowered_line:
                blocker_hints.append(hint)

        for hint in NVIDIA_LEXICAL_FALSE_POSITIVE_HINTS:
            if hint not in lowered_line:
                continue

            match_kind = "compatibility_label"
            is_blocker = False
            if import_patterns[hint].search(line):
                match_kind = "import"
                is_blocker = True
            elif call_match := call_patterns[hint].search(line):
                callee = call_match.group("callee")
                if callee and callee[:1].isupper():
                    match_kind = "constructor_name"
                else:
                    match_kind = "call"
                    is_blocker = True
            elif native_patterns[hint].search(lowered_line):
                match_kind = "native_source"
                is_blocker = True
            elif class_patterns[hint].search(line):
                match_kind = "class_name"
            elif variable_patterns[hint].search(lowered_line):
                match_kind = "variable_name"
            elif is_comment:
                match_kind = "comment"
            elif is_docstring_line:
                match_kind = "docstring"

            if is_blocker:
                blocker_hints.append(hint)
            else:
                false_positive_evidence.append(
                    ReferenceRuntimeHintEvidence(
                        token=hint,
                        match_kind=match_kind,
                        context=context,
                        line_number=line_idx,
                    )
                )

        quote_count = line.count('"""') + line.count("'''")
        if quote_count % 2 == 1:
            in_docstring = not in_docstring

    return sorted(set(blocker_hints)), false_positive_evidence
