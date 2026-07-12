# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from sol_execbench.core.data.json_utils import (
    load_json_dict,
    load_json_value,
    load_jsonl_file,
    stable_model_checksum,
    stable_model_json,
)


class ExampleModel(BaseModel):
    name: str
    count: int


def test_stable_model_json_sorts_keys_and_adds_trailing_newline() -> None:
    model = ExampleModel(name="demo", count=3)

    assert stable_model_json(model) == '{\n  "count": 3,\n  "name": "demo"\n}\n'


def test_stable_model_checksum_ignores_selected_checksum_field() -> None:
    class ReportModel(BaseModel):
        name: str
        report_checksum: str | None = None

    with_checksum = ReportModel(name="demo", report_checksum="old")
    without_checksum = ReportModel(name="demo", report_checksum=None)

    assert stable_model_checksum(with_checksum, "report_checksum") == (
        stable_model_checksum(without_checksum, "report_checksum")
    )


def test_load_json_dict_requires_json_object(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected JSON object"):
        load_json_dict(path)

    assert load_json_value(path) == [1, 2, 3]


def test_load_jsonl_file_uses_pydantic_validation_and_skips_blank_lines(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps({"name": "a", "count": 1}) + "\n\n"
        + json.dumps({"name": "b", "count": 2})
        + "\n",
        encoding="utf-8",
    )

    assert load_jsonl_file(ExampleModel, path) == [
        ExampleModel(name="a", count=1),
        ExampleModel(name="b", count=2),
    ]
