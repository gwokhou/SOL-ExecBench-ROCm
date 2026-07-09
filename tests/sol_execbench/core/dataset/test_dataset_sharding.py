from __future__ import annotations

import pytest

from sol_execbench.core.dataset.sharding import (
    merge_dataset_shard_traces,
    plan_dataset_shards,
    workload_prefix_lines,
    workload_shard_paths,
)


def _workload(problem_id: str, uuid: str, row_index: int) -> dict:
    return {
        "problem_id": problem_id,
        "workload_uuid": uuid,
        "row_index": row_index,
    }


def _trace(uuid: str) -> dict:
    return {"workload": {"uuid": uuid}, "evaluation": {"status": "PASSED"}}


def test_plan_dataset_shards_is_deterministic_with_one_trace_ref_per_shard(tmp_path):
    workloads = [
        _workload("L1/a", "a0", 0),
        _workload("L1/b", "b0", 0),
        _workload("L1/c", "c0", 0),
        _workload("L1/d", "d0", 0),
        _workload("L1/e", "e0", 0),
    ]

    first = plan_dataset_shards(workloads, shard_count=2, output_dir=tmp_path)
    second = plan_dataset_shards(workloads, shard_count=2, output_dir=tmp_path)

    assert [plan.model_dump() for plan in first] == [
        plan.model_dump() for plan in second
    ]
    assert [plan.shard_id for plan in first] == [
        "shard-0000-of-0002",
        "shard-0001-of-0002",
    ]
    assert [plan.trace_ref for plan in first] == [
        "shard-0000-of-0002.traces.json",
        "shard-0001-of-0002.traces.json",
    ]
    assert [[workload.ordinal for workload in plan.workloads] for plan in first] == [
        [0, 2, 4],
        [1, 3],
    ]


def test_plan_dataset_shards_rejects_invalid_shard_count(tmp_path):
    with pytest.raises(ValueError, match="shard_count"):
        plan_dataset_shards([], shard_count=0, output_dir=tmp_path)


def test_merge_dataset_shard_traces_preserves_original_workload_order(tmp_path):
    plans = plan_dataset_shards(
        [
            _workload("L1/a", "a0", 0),
            _workload("L1/b", "b0", 0),
            _workload("L1/c", "c0", 0),
        ],
        shard_count=2,
        output_dir=tmp_path,
    )

    result = merge_dataset_shard_traces(
        plans,
        {
            "shard-0000-of-0002": [_trace("a0"), _trace("c0")],
            "shard-0001-of-0002": [_trace("b0")],
        },
    )

    assert result.complete is True
    assert [trace["workload"]["uuid"] for trace in result.traces] == [
        "a0",
        "b0",
        "c0",
    ]
    assert result.shard_trace_refs == {
        "shard-0000-of-0002": "shard-0000-of-0002.traces.json",
        "shard-0001-of-0002": "shard-0001-of-0002.traces.json",
    }


def test_merge_dataset_shard_traces_reports_duplicates_and_incomplete_shards(
    tmp_path,
):
    plans = plan_dataset_shards(
        [
            _workload("L1/a", "a0", 0),
            _workload("L1/b", "b0", 0),
            _workload("L1/c", "c0", 0),
        ],
        shard_count=2,
        output_dir=tmp_path,
    )

    result = merge_dataset_shard_traces(
        plans,
        {
            "shard-0000-of-0002": [_trace("a0"), _trace("a0")],
            "shard-0001-of-0002": [_trace("b0")],
        },
    )

    assert result.complete is False
    assert result.duplicate_workloads == (
        {
            "shard_id": "shard-0000-of-0002",
            "reason": "duplicate_workload",
            "problem_id": "L1/a",
            "workload_uuid": "a0",
            "row_index": 0,
            "ordinal": 0,
        },
    )
    assert result.incomplete_shards == (
        {
            "shard_id": "shard-0000-of-0002",
            "missing_workloads": [
                {
                    "problem_id": "L1/c",
                    "workload_uuid": "c0",
                    "row_index": 0,
                    "ordinal": 2,
                }
            ],
        },
    )


def test_workload_prefix_lines_reports_truncation(tmp_path):
    workload_path = tmp_path / "workload.jsonl"
    workload_path.write_text("a\nb\nc\n")

    lines, truncated = workload_prefix_lines(workload_path, 2)

    assert lines == ["a", "b"]
    assert truncated is True


def test_workload_shard_paths_splits_nonblank_lines(tmp_path):
    workload_path = tmp_path / "workload.jsonl"
    workload_path.write_text("a\n\nb\nc\n")

    shards = workload_shard_paths(
        workload_path,
        shard_size=2,
        output_dir=tmp_path / "out",
    )

    assert [path.name for path in shards] == [
        "workload_shard_0001.jsonl",
        "workload_shard_0002.jsonl",
    ]
    assert [path.read_text() for path in shards] == ["a\nb\n", "c\n"]


def test_workload_shard_paths_reuses_input_when_split_is_unneeded(tmp_path):
    workload_path = tmp_path / "workload.jsonl"
    workload_path.write_text("a\nb\n")

    assert workload_shard_paths(
        workload_path,
        shard_size=2,
        output_dir=tmp_path / "out",
    ) == [workload_path]
