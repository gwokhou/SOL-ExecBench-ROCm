# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Python 3.12/3.13 import compatibility for PEP 758 bare except tuples."""

from __future__ import annotations
import __future__

import ast
import io
import os
import sys
import tokenize
from collections.abc import Buffer, Sequence
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec, PathFinder, SourceFileLoader
from importlib.util import decode_source
from pathlib import Path
from types import CodeType, ModuleType

_COMPAT_PACKAGES = frozenset({"sol_execbench", "solar", "scripts"})
_SKIP_TOKENS = frozenset(
    {
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.COMMENT,
        tokenize.ENCODING,
    }
)
_OPEN_BRACKETS = frozenset({"(", "[", "{"})
_CLOSE_BRACKETS = frozenset({")", "]", "}"})


def _build_line_offsets(source: str) -> list[int]:
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _scan_except_span(
    tokens: Sequence[tokenize.TokenInfo],
    start_idx: int,
    offsets: Sequence[int],
) -> tuple[int, tuple[int, int] | None]:
    idx = start_idx + 1
    if (
        idx < len(tokens)
        and tokens[idx].type == tokenize.OP
        and tokens[idx].string == "*"
    ):
        idx += 1
    depth = 0
    has_comma = False
    has_as = False
    first_tok: tokenize.TokenInfo | None = None
    last_tok: tokenize.TokenInfo | None = None
    while idx < len(tokens):
        tok = tokens[idx]
        if tok.type == tokenize.ENDMARKER:
            break
        if depth == 0 and tok.type == tokenize.OP and tok.string == ":":
            if (
                first_tok is not None
                and last_tok is not None
                and has_comma
                and not has_as
            ):
                s_row, s_col = first_tok.start
                e_row, e_col = last_tok.end
                span = (offsets[s_row - 1] + s_col, offsets[e_row - 1] + e_col)
                return idx + 1, span
            return idx + 1, None
        if first_tok is None:
            first_tok = tok
        last_tok = tok
        if tok.type == tokenize.OP and tok.string in _OPEN_BRACKETS:
            depth += 1
        elif tok.type == tokenize.OP and tok.string in _CLOSE_BRACKETS:
            depth = max(0, depth - 1)
        elif depth == 0 and tok.type == tokenize.OP and tok.string == ",":
            has_comma = True
        elif depth == 0 and tok.type == tokenize.NAME and tok.string == "as":
            has_as = True
        idx += 1
    return idx, None


def rewrite_pep758_except_clauses(source: str) -> str:
    """Parenthesize PEP 758 ``except A, B:`` clauses while keeping line numbers."""
    if "except" not in source:
        return source
    try:
        tokens = [
            tok
            for tok in tokenize.generate_tokens(io.StringIO(source).readline)
            if tok.type not in _SKIP_TOKENS
        ]
    except tokenize.TokenError:
        return source
    offsets = _build_line_offsets(source)
    spans: list[tuple[int, int]] = []
    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]
        if tok.type == tokenize.NAME and tok.string == "except":
            idx, span = _scan_except_span(tokens, idx, offsets)
            if span is not None:
                spans.append(span)
        else:
            idx += 1
    if not spans:
        return source
    rewritten = source
    for start, end in reversed(spans):
        rewritten = (
            f"{rewritten[:start]}({rewritten[start:end]}){rewritten[end:]}"
        )
    return rewritten


class _PEP758CompatSourceLoader(SourceFileLoader):
    """SourceFileLoader that normalizes PEP 758 syntax before compilation."""

    __slots__ = ()

    def source_to_code(  # type: ignore[override]
        self,
        data: Buffer | str | ast.Module | ast.Expression | ast.Interactive,
        path: bytes | str | os.PathLike[str] = "<string>",
        *,
        _optimize: int = -1,
    ) -> CodeType:
        if isinstance(data, ast.AST):
            return super().source_to_code(data, path, _optimize=_optimize)
        text = data if isinstance(data, str) else decode_source(data)
        rewritten = rewrite_pep758_except_clauses(text)
        return compile(
            rewritten,
            os.fsdecode(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
            dont_inherit=True,
            optimize=_optimize,
        )


class _PEP758CompatFinder(MetaPathFinder):
    """MetaPathFinder wrapping local package loaders on Python < 3.14."""

    __slots__ = ()

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if fullname.partition(".")[0] not in _COMPAT_PACKAGES:
            return None
        spec = PathFinder.find_spec(fullname, path, target)
        if spec is None or not isinstance(spec.loader, SourceFileLoader):
            return None
        if not isinstance(spec.loader, _PEP758CompatSourceLoader):
            spec.loader = _PEP758CompatSourceLoader(
                spec.loader.name, spec.loader.path
            )
        return spec


def install_py312_pep758_import_hook(*, force: bool = False) -> bool:
    """Install the PEP 758 compatibility finder when running on Python < 3.14."""
    if not force and sys.version_info >= (3, 14):
        return False
    if any(isinstance(f, _PEP758CompatFinder) for f in sys.meta_path):
        return False
    sys.meta_path.insert(0, _PEP758CompatFinder())
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """Execute a target Python script with PEP 758 compatibility enabled."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit(
            "Usage: python -m scripts.mvp_v1.py312_compat <script.py> [args...]"
        )
    install_py312_pep758_import_hook(force=sys.version_info < (3, 14))
    script_path = Path(args[0]).resolve()
    sys.argv = [str(script_path), *args[1:]]
    source = rewrite_pep758_except_clauses(
        script_path.read_text(encoding="utf-8")
    )
    code = compile(
        source,
        str(script_path),
        "exec",
        flags=__future__.annotations.compiler_flag,
        dont_inherit=True,
    )
    globs = {
        "__name__": "__main__",
        "__file__": str(script_path),
        "__package__": None,
    }
    exec(code, globs)  # noqa: S102 -- CLI launcher for local repo scripts
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
