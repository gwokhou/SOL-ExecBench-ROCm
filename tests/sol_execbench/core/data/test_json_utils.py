# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_jsonl_file,
)


class ExampleModel(BaseModel):
    name: str
    count: int


def test_atomic_write_json_value_can_preserve_contractual_key_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "definition.json"

    atomic_write_json_value(
        path,
        {"inputs": {"v": {}, "a": {}, "max": {}}},
        sort_keys=False,
    )

    assert path.read_text(encoding="utf-8").find('"v"') < path.read_text(
        encoding="utf-8"
    ).find('"a"')


def test_load_jsonl_file_uses_pydantic_validation_and_skips_blank_lines(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps({"name": "a", "count": 1})
        + "\n\n"
        + json.dumps({"name": "b", "count": 2})
        + "\n",
        encoding="utf-8",
    )

    assert load_jsonl_file(ExampleModel, path) == [
        ExampleModel(name="a", count=1),
        ExampleModel(name="b", count=2),
    ]
