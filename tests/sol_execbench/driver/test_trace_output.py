"""Tests for parsing evaluation-driver trace output."""

import json

import pytest

from sol_execbench.core import Trace
from sol_execbench.core.integrity.schema_versions import (
    TRACE_SCHEMA_VERSION,
    WORKLOAD_SCHEMA_VERSION,
)
from sol_execbench.driver.trace_output import parse_trace_jsonl


def _trace_json(uuid: str = "wkl-0001") -> str:
    return json.dumps(
        {
            "schema_version": TRACE_SCHEMA_VERSION,
            "definition": "test_vecadd",
            "workload": {
                "schema_version": WORKLOAD_SCHEMA_VERSION,
                "axes": {},
                "inputs": {
                    "x": {"type": "random"},
                    "y": {"type": "random"},
                },
                "checks": [{"type": "numeric", "output": "z"}],
                "uuid": uuid,
            },
            "solution": "vecadd_python",
            "evaluation": {
                "status": "PASSED",
                "environment": {"hardware": "AMD Instinct MI300X (gfx942)"},
                "timestamp": "2026-01-01T00:00:00",
                "correctness": {
                    "max_absolute_error": 0.0,
                    "max_relative_error": 0.0,
                },
                "performance": {
                    "latency_ms": 0.1,
                    "reference_latency_ms": 0.2,
                    "speedup_factor": 2.0,
                },
            },
        },
    )


def test_parses_single_trace() -> None:
    traces = parse_trace_jsonl(_trace_json())

    assert len(traces) == 1
    assert isinstance(traces[0], Trace)
    assert traces[0].definition == "test_vecadd"


def test_parses_multiple_traces() -> None:
    stdout = _trace_json("wkl-0001") + "\n" + _trace_json("wkl-0002")

    assert len(parse_trace_jsonl(stdout)) == 2


def test_skips_non_json_lines() -> None:
    stdout = "some library noise\n" + _trace_json() + "\nmore noise\n"

    assert len(parse_trace_jsonl(stdout)) == 1


def test_returns_empty_for_no_traces() -> None:
    assert parse_trace_jsonl("no json here\njust noise\n") == []


@pytest.mark.parametrize("replacement", [None, "future.v999"])
def test_rejects_missing_or_wrong_trace_schema(
    replacement: str | None,
) -> None:
    payload = json.loads(_trace_json())
    if replacement is None:
        payload.pop("schema_version")
    else:
        payload["schema_version"] = replacement

    with pytest.raises(ValueError, match="requires schema_version"):
        parse_trace_jsonl(json.dumps(payload))
