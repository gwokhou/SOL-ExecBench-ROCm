# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from sol_execbench.core.text_utils import (
    normalize_ascii_alnum,
    ordered_unique,
    subprocess_text,
    text_tail,
)


def test_normalize_ascii_alnum_is_case_and_separator_insensitive() -> None:
    assert normalize_ascii_alnum("Start_Timestamp (ns)") == "starttimestampns"
    assert normalize_ascii_alnum(None) == ""
    assert normalize_ascii_alnum("μs") == "s"


def test_text_tail_normalizes_none_bytes_and_strings() -> None:
    assert text_tail(None) == ""
    assert text_tail(b"abcdef", limit=3) == "def"
    assert text_tail("abcdef", limit=4) == "cdef"


def test_subprocess_text_normalizes_none_bytes_and_strings() -> None:
    assert subprocess_text(None) == ""
    assert subprocess_text(b"a\xffb") == "a\ufffdb"
    assert subprocess_text("plain") == "plain"


def test_ordered_unique_preserves_first_seen_order() -> None:
    assert ordered_unique(["gfx1200", "gfx942", "gfx1200"]) == [
        "gfx1200",
        "gfx942",
    ]
