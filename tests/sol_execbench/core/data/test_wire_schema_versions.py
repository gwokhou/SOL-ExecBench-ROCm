"""Boundary tests for canonical benchmark wire-schema versions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from sol_execbench.core import BenchmarkConfig
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.solution_instance import Solution
from sol_execbench.core.data.workload import Workload

SAMPLES = Path(__file__).resolve().parents[2] / "samples"


def _payload(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    value = (
        json.loads(next(item for item in text.splitlines() if item.strip()))
        if path.suffix == ".jsonl"
        else json.loads(text)
    )
    assert isinstance(value, dict)
    return value


@pytest.mark.parametrize(
    ("model", "path"),
    [
        (Definition, SAMPLES / "gqa_paged_decode" / "definition.json"),
        (Workload, SAMPLES / "gqa_paged_decode" / "workload.jsonl"),
        (Solution, SAMPLES / "gqa_paged_decode" / "solution.json"),
        (BenchmarkConfig, SAMPLES / "gqa_paged_decode" / "config.json"),
    ],
)
@pytest.mark.parametrize("replacement", [None, "future.v999"])
def test_canonical_reader_rejects_missing_or_wrong_schema_version(
    model: type[BaseModel],
    path: Path,
    replacement: str | None,
) -> None:
    payload = _payload(path)
    if replacement is None:
        payload.pop("schema_version")
    else:
        payload["schema_version"] = replacement

    with pytest.raises(ValueError, match="requires schema_version"):
        model.model_validate(payload)
