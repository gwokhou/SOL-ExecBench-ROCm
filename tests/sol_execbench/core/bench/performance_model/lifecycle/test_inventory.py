# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from sol_execbench.core.bench.performance_model.lifecycle.inventory import (
    inventory_regular_tree,
)


def test_inventory_is_sorted_by_relative_path_string(tmp_path: Path) -> None:
    """The inventory must be ordered by relative-path string.

    A regular file whose name prefixes a sibling directory (``solar.log``
    beside ``solar/``) exposes the divergence between path-object ordering and
    relative-path string ordering: ``.`` (0x2E) precedes ``/`` (0x2F) in string
    order, but a path component compares as a whole, so path-object ordering
    places the directory first. Lifecycle run-state validators enforce the
    string order, so the inventory must match it.
    """
    root = tmp_path / "collection"
    case = root / "cases" / "held_out" / "elementwise" / "case-00"
    case.mkdir(parents=True)
    # A file and a directory that share a name prefix: the collision case.
    (case / "solar.log").write_text("log", encoding="utf-8")
    (case / "solar").mkdir()
    (case / "solar" / "manifest.yaml").write_text("solar", encoding="utf-8")
    (case / "trace.jsonl").write_text("trace", encoding="utf-8")
    (case / "trace.jsonl.rocprofv3").mkdir()
    (case / "trace.jsonl.rocprofv3" / "pass_1").write_text(
        "counters", encoding="utf-8"
    )

    paths = [item.relative_path for item in inventory_regular_tree(root)]

    assert paths == sorted(paths)
    assert len(paths) == len(set(paths))
    # The file must precede the directory entry in string order.
    assert paths.index(
        "cases/held_out/elementwise/case-00/solar.log"
    ) < paths.index("cases/held_out/elementwise/case-00/solar/manifest.yaml")
