# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from sol_execbench.core.text_utils import (
    markdown_table_cell,
    ordered_unique,
    subprocess_text,
    text_tail,
)


def test_markdown_table_cell_escapes_pipes_backslashes_and_newlines() -> None:
    assert markdown_table_cell("a|b\\c\nnext\rline") == "a\\|b\\\\c next line"
    assert markdown_table_cell(None) == ""


def test_text_tail_normalizes_none_bytes_and_strings() -> None:
    assert text_tail(None) == ""
    assert text_tail(b"abcdef", limit=3) == "def"
    assert text_tail("abcdef", limit=4) == "cdef"


def test_subprocess_text_normalizes_none_bytes_and_strings() -> None:
    assert subprocess_text(None) == ""
    assert subprocess_text(b"a\xffb") == "a\ufffdb"
    assert subprocess_text("plain") == "plain"


def test_ordered_unique_preserves_first_seen_order() -> None:
    assert ordered_unique(["gfx1200", "gfx942", "gfx1200"]) == ["gfx1200", "gfx942"]
