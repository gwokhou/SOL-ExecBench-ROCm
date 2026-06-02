# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Deterministic dataset sharding helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from sol_execbench.core.dataset.evidence_refs import relative_ref


@dataclass(frozen=True)
class DatasetShardWorkload:
    """A workload assigned to a deterministic dataset shard."""

    ordinal: int
    problem_id: str
    workload_uuid: str | None
    row_index: int

    @property
    def key(self) -> tuple[str, str | None, int]:
        return (self.problem_id, self.workload_uuid, self.row_index)

    def model_dump(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "problem_id": self.problem_id,
            "workload_uuid": self.workload_uuid,
            "row_index": self.row_index,
        }


@dataclass(frozen=True)
class DatasetShardPlan:
    """One planned dataset shard and its trace output path."""

    shard_id: str
    shard_index: int
    shard_count: int
    trace_ref: str
    workloads: tuple[DatasetShardWorkload, ...]

    def model_dump(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "shard_index": self.shard_index,
            "shard_count": self.shard_count,
            "trace_ref": self.trace_ref,
            "workloads": [workload.model_dump() for workload in self.workloads],
        }


@dataclass(frozen=True)
class DatasetShardMergeResult:
    """Result of merging per-shard traces."""

    traces: tuple[dict[str, Any], ...]
    shard_trace_refs: dict[str, str]
    duplicate_workloads: tuple[dict[str, Any], ...]
    incomplete_shards: tuple[dict[str, Any], ...]

    @property
    def complete(self) -> bool:
        return not self.duplicate_workloads and not self.incomplete_shards


def _workload_uuid(ref: Mapping[str, Any]) -> str | None:
    uuid = ref.get("workload_uuid", ref.get("uuid"))
    return str(uuid) if uuid is not None else None


def _workload_row_index(ref: Mapping[str, Any]) -> int:
    return int(ref.get("row_index", 0))


def _workload_problem_id(ref: Mapping[str, Any]) -> str:
    value = ref.get("problem_id")
    if value is None:
        raise ValueError("workload ref is missing problem_id")
    return str(value)


def _shard_id(shard_index: int, shard_count: int) -> str:
    return f"shard-{shard_index:04d}-of-{shard_count:04d}"


def plan_dataset_shards(
    workload_refs: Sequence[Mapping[str, Any]],
    *,
    shard_count: int,
    output_dir: Path,
) -> tuple[DatasetShardPlan, ...]:
    """Return deterministic shard plans for workload refs in input order."""
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")

    shard_workloads: list[list[DatasetShardWorkload]] = [
        [] for _ in range(shard_count)
    ]
    for ordinal, ref in enumerate(workload_refs):
        workload = DatasetShardWorkload(
            ordinal=ordinal,
            problem_id=_workload_problem_id(ref),
            workload_uuid=_workload_uuid(ref),
            row_index=_workload_row_index(ref),
        )
        shard_workloads[ordinal % shard_count].append(workload)

    plans: list[DatasetShardPlan] = []
    for shard_index, workloads in enumerate(shard_workloads):
        shard_id = _shard_id(shard_index, shard_count)
        trace_path = output_dir / f"{shard_id}.traces.json"
        plans.append(
            DatasetShardPlan(
                shard_id=shard_id,
                shard_index=shard_index,
                shard_count=shard_count,
                trace_ref=relative_ref(trace_path, output_dir),
                workloads=tuple(workloads),
            )
        )
    return tuple(plans)


def _trace_uuid(trace: Mapping[str, Any]) -> str | None:
    workload = trace.get("workload") or {}
    uuid = workload.get("uuid")
    return str(uuid) if uuid is not None else None


def _missing_workload_payload(workload: DatasetShardWorkload) -> dict[str, Any]:
    return {
        "problem_id": workload.problem_id,
        "workload_uuid": workload.workload_uuid,
        "row_index": workload.row_index,
        "ordinal": workload.ordinal,
    }


def merge_dataset_shard_traces(
    plans: Sequence[DatasetShardPlan],
    shard_traces: Mapping[str, Sequence[Mapping[str, Any]]],
) -> DatasetShardMergeResult:
    """Merge shard traces by original workload order with diagnostics."""
    ordered: list[tuple[int, dict[str, Any]]] = []
    duplicate_workloads: list[dict[str, Any]] = []
    incomplete_shards: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None, int]] = set()
    shard_trace_refs = {plan.shard_id: plan.trace_ref for plan in plans}

    for plan in plans:
        traces = list(shard_traces.get(plan.shard_id, ()))
        unmatched = {workload.key: workload for workload in plan.workloads}
        uuid_to_workload = {
            workload.workload_uuid: workload
            for workload in plan.workloads
            if workload.workload_uuid is not None
        }

        for trace_index, trace in enumerate(traces):
            workload = uuid_to_workload.get(_trace_uuid(trace))
            if workload is None and trace_index < len(plan.workloads):
                workload = plan.workloads[trace_index]
            if workload is None:
                duplicate_workloads.append(
                    {
                        "shard_id": plan.shard_id,
                        "reason": "unexpected_trace",
                        "trace_index": trace_index,
                    }
                )
                continue

            if workload.key in seen:
                duplicate_workloads.append(
                    {
                        "shard_id": plan.shard_id,
                        "reason": "duplicate_workload",
                        **_missing_workload_payload(workload),
                    }
                )
                continue

            seen.add(workload.key)
            unmatched.pop(workload.key, None)
            ordered.append((workload.ordinal, dict(trace)))

        if unmatched:
            incomplete_shards.append(
                {
                    "shard_id": plan.shard_id,
                    "missing_workloads": [
                        _missing_workload_payload(workload)
                        for workload in sorted(
                            unmatched.values(), key=lambda item: item.ordinal
                        )
                    ],
                }
            )

    return DatasetShardMergeResult(
        traces=tuple(trace for _, trace in sorted(ordered, key=lambda item: item[0])),
        shard_trace_refs=shard_trace_refs,
        duplicate_workloads=tuple(duplicate_workloads),
        incomplete_shards=tuple(incomplete_shards),
    )
