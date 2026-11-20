"""Post-C3 full-matrix summarizer, invariant checker, and device-free auditor."""

from __future__ import annotations

import argparse
import os
import re
import statistics
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from shutil import copyfile
from subprocess import TimeoutExpired
from typing import TYPE_CHECKING, Any

import torch

from scripts.internal.solar import c2_c3_campaign
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_file,
    load_json_value,
)
from sol_execbench.core.dataset.target_view import scoped_artifact
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded
from sol_execbench.core.process.termination import defer_termination
from sol_execbench.core.solar_bridge import target_coverage
from sol_execbench.core.solar_bridge.coverage_models import (
    CoverageAttempt,
    CoverageOptions,
    CoverageRow,
    CoverageStatus,
    TargetCoverageReport,
)
from sol_execbench.core.solar_bridge.coverage_plan import (
    load_coverage_target,
    new_coverage_report,
)
from sol_execbench.core.solar_bridge.coverage_resume import (
    authenticate_resumed_attempts,
)
from sol_execbench.tools.docker_process import _remove_owned_container
from solar.analysis.orojenesis.runner import OrojenesisRunner
from solar.errors import MAPPER_TIMEOUT_REASON_CODE
from solar.ir.contracts import IRPath

if TYPE_CHECKING:
    from scripts.internal.solar.c2_c3_host import HostConfig

PRESERVED_SMOKE_SHA256: dict[str, str] = {
    "llm-core/capacity-evidence.json": (
        "cb47591cdaf8a101bc37897d79b861c1590a6b483a85c60f8ffe19ee78e5d6bb"
    ),
    "llm-core/c2-summary.json": (
        "dbf06a42a505a1592c4b19972e9b758723331c2ec850ba64241b8e230d645293"
    ),
    "llm-core/smoke-summary.json": (
        "c43d5b9c0753cda33044ebca6f5bc5aea0fa2835a8bfc131523d23f7fd1dc7d9"
    ),
    "llm-core/exposure-ledger.json": (
        "ea350f98b9640fd40f1d2566038ab1e36e508dc3a71771a5f1df6d8fa601ad35"
    ),
    "llm-core/cpu-audit.json": (
        "851acbfec7ee1219ceb59a0b788639712add40b1d20e557c8ef6845ec45cbd3"
    ),
}
_MIB = 1024**2
_GIB = 1024**3
_AUDIT_TIMEOUT_SECONDS = 1800.0
_AUDIT_CAPTURE_BYTES = 32768
_UNSUPPORTED_OP_RE = re.compile(r"Unsupported operation:\s*([A-Za-z0-9_.-]+)")
_MISSING_RULE_RE = re.compile(
    r"operation '([^']+)' has no amd_resource_v3 rule"
)


class AuditMode(StrEnum):
    """Supported execution modes for final C3 verification."""

    HOST_AUDIT = "host-audit"
    CONTAINER_AUDIT = "container-audit"


@dataclass(frozen=True, slots=True, kw_only=True)
class _GapBucket:
    gap_id: str
    m0_category: str
    m0_subreason: str
    blocking_layer: str
    sample_row_key: str
    sample_semantic_id: str
    sample_slot: str
    sample_artifact_dir: str | None
    sample_message: str
    recommended_followup_scope: str
    l3d_eligible_layer: bool
    ir_paths: set[str] = field(default_factory=set)
    definitions: set[str] = field(default_factory=set)
    make_fx_definitions: set[str] = field(default_factory=set)
    torchview_definitions: set[str] = field(default_factory=set)
    slots: set[str] = field(default_factory=set)
    row_indices: list[int] = field(default_factory=list)


@dataclass(frozen=True, slots=True, kw_only=True)
class _DefConsistencyAcc:
    make_fx_statuses: dict[str, str] = field(default_factory=dict)
    torchview_statuses: dict[str, str] = field(default_factory=dict)
    common_passed_slots: list[str] = field(default_factory=list)
    flops_equal: list[bool] = field(default_factory=list)
    resource_work_equal: list[bool] = field(default_factory=list)
    external_read_bytes_equal: list[bool] = field(default_factory=list)


def verify_preserved_artifacts(bundle: Path) -> dict[str, str]:
    """Ensure C2 and C3 smoke artifacts remained byte-identical after resume."""
    manifest_path = bundle / "host-run/smoke-preserved-sha256.json"
    if manifest_path.is_file():
        raw = load_json_value(manifest_path)
        if not isinstance(raw, dict) or set(raw) != set(PRESERVED_SMOKE_SHA256):
            raise RuntimeError(f"invalid preserved manifest: {manifest_path}")
        expected_map = {str(k): str(v) for k, v in raw.items()}
    else:
        expected_map = PRESERVED_SMOKE_SHA256
    verified: dict[str, str] = {}
    for rel, expected in expected_map.items():
        path = bundle / rel
        if not path.is_file():
            raise RuntimeError(f"missing preserved C2/smoke artifact: {rel}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"preserved artifact modified {rel}: {actual} != {expected}"
            )
        verified[rel] = actual
    return verified


def classify_m0_reporting(row: CoverageRow) -> tuple[str, str]:
    """Map one coverage row onto M0-C0 Section 5 report categories."""
    if row.status is CoverageStatus.PASSED:
        return "passed", "passed"
    if row.status is CoverageStatus.NOT_REACHED:
        return "not_reached", row.reason
    if row.status is CoverageStatus.BUDGET_EXHAUSTED:
        return "budget_exhausted", row.reason
    if row.status is CoverageStatus.INFRASTRUCTURE_ERROR:
        return "infrastructure_error", row.reason
    if row.status is CoverageStatus.UNSUPPORTED:
        return "unsupported_or_unproved", row.reason
    if row.status is CoverageStatus.UNPROVED:
        attempt = row.attempts[-1] if row.attempts else None
        readiness = (attempt.readiness or {}) if attempt else {}
        r_msg = str(readiness.get("message") or "").lower()
        if row.reason == "conversion_not_proven" and "out of memory" in r_msg:
            return "unsupported_or_unproved", "conversion_not_proven_device_oom"
        return "unsupported_or_unproved", row.reason
    return _classify_failed_row(row)


def _classify_failed_row(row: CoverageRow) -> tuple[str, str]:
    attempt = row.attempts[-1] if row.attempts else None
    readiness = (attempt.readiness or {}) if attempt else {}
    formal = (attempt.formal or {}) if attempt else {}
    r_reason = str(readiness.get("reason_code") or row.reason or "")
    f_reason = str(formal.get("reason_code") or row.reason or "")
    r_msg = str(readiness.get("message") or "")
    f_msg = str(formal.get("message") or "")
    if (
        r_reason == "graph_extraction_failed"
        and "out of memory" in r_msg.lower()
    ):
        return "budget_exhausted", "graph_extraction_device_oom"
    if (
        r_reason == "strict_conversion_failed"
        and "Unsupported operation" in r_msg
    ):
        return "unsupported_or_unproved", "ir_unsupported_op"
    if r_reason == "exact_replay_failed":
        return "failed", "exact_replay_failed"
    if r_reason == "numerical_equivalence_failed":
        return "failed", "numerical_equivalence_failed"
    if r_reason == "gradient_verification_failed":
        return "failed", "gradient_verification_failed"
    if (
        f_reason == "toolchain_unavailable"
        and attempt
        and attempt.wall_seconds >= 7200.0
    ):
        return (
            "budget_exhausted",
            "mapper_timeout_surfaced_as_toolchain_unavailable",
        )
    if "has no amd_resource_v3 rule" in f_msg:
        return "unsupported_or_unproved", "missing_resource_rule"
    if f_msg.startswith("external reads are unproved:"):
        return "unsupported_or_unproved", "external_reads_unproved"
    if "did not produce a complete tile-aware bound" in f_msg:
        return "unsupported_or_unproved", "incomplete_tile_aware_bound"
    if "unproved" in f_msg.lower() or "unsupported" in f_msg.lower():
        return "unsupported_or_unproved", f_reason or "formal_unproved"
    return "failed", row.reason or "other_failed"


def _failure_signature(
    row: CoverageRow, m0_subreason: str
) -> tuple[str, str, str]:
    if row.status is CoverageStatus.BUDGET_EXHAUSTED:
        if row.reason in {MAPPER_TIMEOUT_REASON_CODE, "orojenesis_timeout"}:
            return (
                "formal_analysis:mapper_timeout_7200s",
                "formal_analysis",
                "orojenesis_mapspace_pruning_and_timeout_code",
            )
        return (
            "outer_bridge:activity_budget_worker_timeout",
            "outer_bridge",
            "campaign_activity_window_12h",
        )
    if m0_subreason == "conversion_not_proven_device_oom":
        return (
            "conversion_verification:conversion_not_proven_device_oom",
            "conversion_verification",
            "verification_replay_memory_budget",
        )
    if m0_subreason == "graph_extraction_device_oom":
        return (
            "graph_extraction:device_oom_eager_trace",
            "graph_extraction",
            "torchview_eager_extraction_memory",
        )
    if m0_subreason == "mapper_timeout_surfaced_as_toolchain_unavailable":
        return (
            "formal_analysis:mapper_timeout_7200s",
            "formal_analysis",
            "orojenesis_mapspace_pruning_and_timeout_code",
        )
    return _classify_standard_failure_signature(row, m0_subreason)


def _classify_standard_failure_signature(
    row: CoverageRow, m0_subreason: str
) -> tuple[str, str, str]:
    attempt = row.attempts[-1] if row.attempts else None
    readiness = (attempt.readiness or {}) if attempt else {}
    formal = (attempt.formal or {}) if attempt else {}
    if attempt and attempt.formal is not None:
        layer = str(formal.get("stage") or "formal_analysis")
        msg = str(formal.get("message") or row.reason)
        op_rule = _MISSING_RULE_RE.search(msg)
        if op_rule:
            gap_id = f"{layer}:missing_amd_resource_v3_rule:{op_rule.group(1)}"
            return gap_id, layer, "native_op_rule_or_lowering"
        if msg.startswith("external reads are unproved:"):
            detail = msg.split(":", 1)[1].strip()
            return (
                f"{layer}:external_reads_unproved:{detail}",
                layer,
                "external_read_proof",
            )
        if "did not produce a complete tile-aware bound" in msg:
            return (
                f"{layer}:incomplete_tile_aware_bound",
                layer,
                "tile_aware_bound_mapping",
            )
        return f"{layer}:{m0_subreason}", layer, "formal_investigation"
    layer = str(readiness.get("failure_stage") or "readiness")
    msg = str(readiness.get("message") or row.reason)
    unsup = _UNSUPPORTED_OP_RE.search(msg)
    if unsup:
        return (
            f"{layer}:unsupported_op:{unsup.group(1)}",
            layer,
            "ir_frontend_adapter_op_support",
        )
    if m0_subreason in {"exact_replay_failed", "numerical_equivalence_failed"}:
        return (
            f"{layer}:{m0_subreason}",
            layer,
            "ir_conversion_forward_numerics",
        )
    if m0_subreason == "gradient_verification_failed":
        return (
            f"{layer}:gradient_numerical_mismatch",
            layer,
            "ir_conversion_gradient_numerics",
        )
    return f"{layer}:{m0_subreason}", layer, "readiness_investigation"


def _row_attempt_dir(index: int, row: CoverageRow) -> str | None:
    if not row.attempts:
        return None
    return f"row-{index:03d}/attempt-{len(row.attempts) - 1:02d}"


def build_gap_inventory(rows: list[CoverageRow]) -> list[dict[str, object]]:
    """Rank executed coverage gaps deduplicated by unique semantic definition."""
    buckets: dict[str, _GapBucket] = {}
    for index, row in enumerate(rows):
        if row.status in {CoverageStatus.PASSED, CoverageStatus.NOT_REACHED}:
            continue
        m0_cat, m0_sub = classify_m0_reporting(row)
        gap_id, layer, followup = _failure_signature(row, m0_sub)
        bucket = buckets.get(gap_id)
        if bucket is None:
            attempt = row.attempts[-1] if row.attempts else None
            formal = (attempt.formal or {}) if attempt else {}
            readiness = (attempt.readiness or {}) if attempt else {}
            msg = (
                formal.get("message") or readiness.get("message") or row.reason
            )
            bucket = _GapBucket(
                gap_id=gap_id,
                m0_category=m0_cat,
                m0_subreason=m0_sub,
                blocking_layer=layer,
                sample_row_key=row.key,
                sample_semantic_id=row.semantic_id,
                sample_slot=row.slot,
                sample_artifact_dir=_row_attempt_dir(index, row),
                sample_message=str(msg),
                recommended_followup_scope=followup,
                l3d_eligible_layer=(
                    layer in {"graph_extraction", "ir_conversion"}
                    and m0_sub != "graph_extraction_device_oom"
                ),
            )
            buckets[gap_id] = bucket
        bucket.row_indices.append(index)
        bucket.ir_paths.add(row.ir_path.value)
        bucket.definitions.add(row.semantic_id)
        bucket.slots.add(row.slot)
        if row.ir_path is IRPath.MAKE_FX_ATEN:
            bucket.make_fx_definitions.add(row.semantic_id)
        else:
            bucket.torchview_definitions.add(row.semantic_id)
    ordered = sorted(
        buckets.values(),
        key=lambda b: (
            -len(b.definitions),
            -len(b.make_fx_definitions),
            -len(b.row_indices),
            b.gap_id,
        ),
    )
    return [_finalize_gap(b) for b in ordered]


def _finalize_gap(bucket: _GapBucket) -> dict[str, object]:
    ir_paths = sorted(bucket.ir_paths)
    defs = sorted(bucket.definitions)
    both_paths_frontend = bucket.l3d_eligible_layer and len(ir_paths) == 2
    return {
        "gap_id": bucket.gap_id,
        "m0_category": bucket.m0_category,
        "m0_subreason": bucket.m0_subreason,
        "blocking_layer": bucket.blocking_layer,
        "ir_paths_affected": ir_paths,
        "affected_definitions_count": len(defs),
        "make_fx_affected_definitions_count": len(bucket.make_fx_definitions),
        "torchview_affected_definitions_count": len(
            bucket.torchview_definitions
        ),
        "affected_rows_count": len(bucket.row_indices),
        "affected_slots": sorted(bucket.slots),
        "affected_definitions": defs,
        "sample_row_key": bucket.sample_row_key,
        "sample_semantic_id": bucket.sample_semantic_id,
        "sample_slot": bucket.sample_slot,
        "sample_artifact_dir": bucket.sample_artifact_dir,
        "sample_message": bucket.sample_message,
        "l3d_frontend_candidate": both_paths_frontend,
        "recommended_followup_scope": bucket.recommended_followup_scope,
    }


def _verification_dimension_counts(
    rows: list[CoverageRow],
) -> dict[str, dict[str, int]]:
    counters: dict[str, Counter[str]] = {
        "numerical": Counter(),
        "gradient": Counter(),
        "effects": Counter(),
    }
    for row in rows:
        attempt = row.attempts[-1] if row.attempts else None
        if attempt is None:
            for c in counters.values():
                c["not_reached"] += 1
            continue
        _tally_attempt_verification(attempt, counters)
    return {k: dict(sorted(v.items())) for k, v in counters.items()}


def _tally_attempt_verification(
    attempt: CoverageAttempt, counters: dict[str, Counter[str]]
) -> None:
    verif = attempt.observations.get("verification")
    if isinstance(verif, dict):
        counters["numerical"][
            _extract_status(verif.get("numerical"), "not_observable")
        ] += 1
        counters["effects"][
            _extract_status(verif.get("effects"), "not_observable")
        ] += 1
        counters["gradient"][
            _summarize_gradient_cases(verif.get("gradient"))
        ] += 1
        return
    readiness = attempt.readiness or {}
    r_reason = str(readiness.get("reason_code") or attempt.reason or "")
    r_stage = str(readiness.get("failure_stage") or "")
    if r_stage in {"graph_extraction", "ir_conversion"}:
        for c in counters.values():
            c["not_reached"] += 1
        return
    if r_reason in {"exact_replay_failed", "numerical_equivalence_failed"}:
        counters["numerical"]["failed"] += 1
        counters["gradient"]["not_reached"] += 1
        counters["effects"]["not_reached"] += 1
        return
    if r_reason == "gradient_verification_failed":
        counters["numerical"]["not_observable"] += 1
        counters["gradient"]["failed"] += 1
        counters["effects"]["not_observable"] += 1
        return
    for c in counters.values():
        c["not_observable"] += 1


def _extract_status(obj: object, fallback: str) -> str:
    if isinstance(obj, dict):
        status = obj.get("status")
        if status:
            return str(status)
    return fallback


def _summarize_gradient_cases(cases: object) -> str:
    if not isinstance(cases, list) or not cases:
        return "not_observable"
    statuses = {str(c.get("status")) for c in cases if isinstance(c, dict)}
    if not statuses:
        return "not_observable"
    if statuses == {CoverageStatus.PASSED.value}:
        return CoverageStatus.PASSED.value
    if CoverageStatus.FAILED.value in statuses:
        return CoverageStatus.FAILED.value
    if CoverageStatus.NOT_APPLICABLE.value in statuses:
        return CoverageStatus.NOT_APPLICABLE.value
    return ",".join(sorted(statuses))


def compare_two_paths(
    rows: list[CoverageRow], details: list[dict[str, object]]
) -> dict[str, Any]:
    """Compare both IR paths per ``(semantic_id, slot, workload_uuid)`` pair."""
    pairs_by_workload: dict[
        tuple[str, str, str], dict[str, tuple[CoverageRow, dict[str, object]]]
    ] = {}
    for row, detail in zip(rows, details, strict=True):
        wkey = (row.semantic_id, row.slot, row.workload_uuid)
        pairs_by_workload.setdefault(wkey, {})[row.ir_path.value] = (
            row,
            detail,
        )
    status_counts: Counter[str] = Counter()
    by_role_counts: dict[str, Counter[str]] = {}
    by_slot_counts: dict[str, Counter[str]] = {}
    common_passed: list[dict[str, object]] = []
    by_def: dict[str, _DefConsistencyAcc] = {}
    for (sem_id, slot, _uuid), paths in sorted(pairs_by_workload.items()):
        mfx_row, mfx_det = paths[IRPath.MAKE_FX_ATEN.value]
        tv_row, tv_det = paths[IRPath.TORCHVIEW_EXTENDED_EINSUM.value]
        pair_label = f"{mfx_row.status.value}__vs__{tv_row.status.value}"
        status_counts[pair_label] += 1
        by_role_counts.setdefault(mfx_row.role, Counter())[pair_label] += 1
        by_slot_counts.setdefault(slot, Counter())[pair_label] += 1
        acc = by_def.setdefault(sem_id, _DefConsistencyAcc())
        acc.make_fx_statuses[slot] = mfx_row.status.value
        acc.torchview_statuses[slot] = tv_row.status.value
        if (
            mfx_row.status is CoverageStatus.PASSED
            and tv_row.status is CoverageStatus.PASSED
        ):
            diff = _build_common_pair_entry(
                sem_id, slot, mfx_row, mfx_det, tv_det
            )
            common_passed.append(diff)
            acc.common_passed_slots.append(slot)
            acc.flops_equal.append(bool(diff["flops_equal"]))
            acc.resource_work_equal.append(bool(diff["resource_work_equal"]))
            acc.external_read_bytes_equal.append(
                bool(diff["external_read_bytes_equal"])
            )
    return {
        "total_workload_pairs": len(pairs_by_workload),
        "status_pair_counts": dict(sorted(status_counts.items())),
        "by_role_status_pair_counts": {
            k: dict(sorted(v.items()))
            for k, v in sorted(by_role_counts.items())
        },
        "by_slot_status_pair_counts": {
            k: dict(sorted(v.items()))
            for k, v in sorted(by_slot_counts.items())
        },
        "common_passed_pair_count": len(common_passed),
        "common_passed_pairs": common_passed,
        "by_definition_consistency": _finalize_def_consistency(by_def),
    }


def _build_common_pair_entry(
    semantic_id: str,
    slot: str,
    mfx_row: CoverageRow,
    mfx_det: dict[str, object],
    tv_det: dict[str, object],
) -> dict[str, object]:
    base = c2_c3_campaign._compare_passed_pair(semantic_id, mfx_det, tv_det)
    mfx_tot = mfx_det.get("total")
    tv_tot = tv_det.get("total")
    mfx_rw = mfx_tot.get("resource_work") if isinstance(mfx_tot, dict) else {}
    tv_rw = tv_tot.get("resource_work") if isinstance(tv_tot, dict) else {}
    mfx_map = mfx_rw if isinstance(mfx_rw, dict) else {}
    tv_map = tv_rw if isinstance(tv_rw, dict) else {}
    all_rw_keys = sorted({str(k) for k in mfx_map} | {str(k) for k in tv_map})
    differing_rw = [k for k in all_rw_keys if mfx_map.get(k) != tv_map.get(k)]
    return {
        **base,
        "slot": slot,
        "role": mfx_row.role,
        "workload_uuid": mfx_row.workload_uuid,
        "make_fx_macs": mfx_tot.get("macs")
        if isinstance(mfx_tot, dict)
        else None,
        "torchview_macs": tv_tot.get("macs")
        if isinstance(tv_tot, dict)
        else None,
        "macs_equal": (
            mfx_tot.get("macs") == tv_tot.get("macs")
            if isinstance(mfx_tot, dict) and isinstance(tv_tot, dict)
            else None
        ),
        "differing_resource_work_keys": differing_rw,
    }


def _finalize_def_consistency(
    by_def: dict[str, _DefConsistencyAcc],
) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for sem_id, acc in sorted(by_def.items()):
        out[sem_id] = {
            "make_fx_passed_slots_count": sum(
                v == CoverageStatus.PASSED.value
                for v in acc.make_fx_statuses.values()
            ),
            "torchview_passed_slots_count": sum(
                v == CoverageStatus.PASSED.value
                for v in acc.torchview_statuses.values()
            ),
            "slot_status_uniform_make_fx": len(
                set(acc.make_fx_statuses.values())
            )
            == 1,
            "slot_status_uniform_torchview": len(
                set(acc.torchview_statuses.values())
            )
            == 1,
            "make_fx_statuses_by_slot": acc.make_fx_statuses,
            "torchview_statuses_by_slot": acc.torchview_statuses,
            "common_passed_slots": acc.common_passed_slots,
            "all_common_flops_equal": (
                all(acc.flops_equal) if acc.flops_equal else None
            ),
            "all_common_external_reads_equal": (
                all(acc.external_read_bytes_equal)
                if acc.external_read_bytes_equal
                else None
            ),
            "all_common_resource_work_equal": (
                all(acc.resource_work_equal)
                if acc.resource_work_equal
                else None
            ),
        }
    return out


def _group_counts(
    rows: list[CoverageRow], key_fn: Callable[[CoverageRow], str]
) -> dict[str, dict[str, int]]:
    groups: dict[str, Counter[str]] = {}
    for row in rows:
        groups.setdefault(key_fn(row), Counter())[row.status.value] += 1
    return {k: dict(sorted(v.items())) for k, v in sorted(groups.items())}


def _nested_group_counts(
    rows: list[CoverageRow], outer_fn: Callable[[CoverageRow], str]
) -> dict[str, dict[str, dict[str, int]]]:
    groups: dict[str, dict[str, Counter[str]]] = {}
    for row in rows:
        inner = groups.setdefault(outer_fn(row), {})
        inner.setdefault(row.ir_path.value, Counter())[row.status.value] += 1
    return {
        ok: {ik: dict(sorted(c.items())) for ik, c in sorted(iv.items())}
        for ok, iv in sorted(groups.items())
    }


def _wall_stats(walls: list[float]) -> dict[str, float | int | None]:
    if not walls:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "mean": None,
            "max": None,
            "total": 0.0,
        }
    return {
        "count": len(walls),
        "min": min(walls),
        "median": statistics.median(walls),
        "mean": statistics.fmean(walls),
        "max": max(walls),
        "total": sum(walls),
    }


def _wall_by_slot(
    rows: list[CoverageRow],
) -> dict[str, dict[str, float | int | None]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row.attempts:
            grouped.setdefault(row.slot, []).append(
                row.attempts[-1].wall_seconds
            )
    return {slot: _wall_stats(vals) for slot, vals in sorted(grouped.items())}


def _stage_costs_by_slot(
    rows: list[CoverageRow], raw_details: list[dict[str, object]]
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row, det in zip(rows, raw_details, strict=True):
        grouped.setdefault(row.slot, []).append(det)
    return {
        slot: c2_c3_campaign._stage_cost_summary(slot_rows)
        for slot, slot_rows in sorted(grouped.items())
    }


def _formal_distribution_counts(
    rows: list[CoverageRow],
) -> dict[str, dict[str, int]]:
    bound_kinds: Counter[str] = Counter()
    limiting_resources: Counter[str] = Counter()
    for row in rows:
        if not row.attempts or row.attempts[-1].formal is None:
            continue
        formal = row.attempts[-1].formal or {}
        if formal.get("bound_kind") is not None:
            bound_kinds[str(formal["bound_kind"])] += 1
        if formal.get("limiting_resource") is not None:
            limiting_resources[str(formal["limiting_resource"])] += 1
    return {
        "bound_kind_counts": dict(sorted(bound_kinds.items())),
        "limiting_resource_counts": dict(sorted(limiting_resources.items())),
    }


def _compact_row_record(
    index: int, row: CoverageRow, det: dict[str, object]
) -> dict[str, object]:
    m0_cat, m0_sub = classify_m0_reporting(row)
    attempt = row.attempts[-1] if row.attempts else None
    formal = (attempt.formal or {}) if attempt else {}
    readiness = (attempt.readiness or {}) if attempt else {}
    tot = det.get("total")
    ext = det.get("external_reads")
    return {
        "index": index,
        "key": row.key,
        "semantic_id": row.semantic_id,
        "workload_uuid": row.workload_uuid,
        "slot": row.slot,
        "role": row.role,
        "family": row.family,
        "ir_path": row.ir_path.value,
        "status": row.status.value,
        "reason": row.reason,
        "m0_category": m0_cat,
        "m0_subreason": m0_sub,
        "wall_seconds": attempt.wall_seconds if attempt else 0.0,
        "artifact_dir": _row_attempt_dir(index, row),
        "readiness_status": readiness.get("status"),
        "readiness_failure_stage": readiness.get("failure_stage"),
        "readiness_reason_code": readiness.get("reason_code"),
        "readiness_message": readiness.get("message"),
        "formal_stage": formal.get("stage"),
        "formal_reason_code": formal.get("reason_code"),
        "formal_message": formal.get("message"),
        "formal_bound_kind": formal.get("bound_kind"),
        "formal_limiting_resource": formal.get("limiting_resource"),
        "lower_bound_seconds": formal.get("lower_bound_seconds"),
        "flops": tot.get("flops") if isinstance(tot, dict) else None,
        "external_read_bytes": ext.get("bytes")
        if isinstance(ext, dict)
        else None,
    }


def _m0_reporting_counts(
    rows: list[CoverageRow],
) -> tuple[dict[str, int], dict[str, int]]:
    m0_counts: Counter[str] = Counter()
    m0_sub_counts: Counter[str] = Counter()
    for row in rows:
        cat, sub = classify_m0_reporting(row)
        m0_counts[cat] += 1
        m0_sub_counts[f"{cat}:{sub}"] += 1
    return dict(sorted(m0_counts.items())), dict(sorted(m0_sub_counts.items()))


def _build_budget_stop_summary(
    rows: list[CoverageRow],
    report: TargetCoverageReport,
    options: CoverageOptions,
    *,
    timeout_transient_drifts: list[dict[str, object]] | None = None,
) -> dict[str, Any]:
    not_reached = [
        {
            "index": i,
            "semantic_id": r.semantic_id,
            "slot": r.slot,
            "ir_path": r.ir_path.value,
            "reason": r.reason,
        }
        for i, r in enumerate(rows)
        if r.status is CoverageStatus.NOT_REACHED
    ]
    boundary_timeouts = [
        {
            "index": i,
            "semantic_id": r.semantic_id,
            "slot": r.slot,
            "ir_path": r.ir_path.value,
            "wall_seconds": r.attempts[-1].wall_seconds,
            "reason": r.reason,
        }
        for i, r in enumerate(rows)
        if r.status is CoverageStatus.BUDGET_EXHAUSTED and r.attempts
    ]
    mapper_timeouts = [
        {
            "index": i,
            "semantic_id": r.semantic_id,
            "slot": r.slot,
            "ir_path": r.ir_path.value,
            "wall_seconds": r.attempts[-1].wall_seconds,
            "reason": r.reason,
        }
        for i, r in enumerate(rows)
        if r.attempts
        and r.reason == "toolchain_unavailable"
        and r.attempts[-1].wall_seconds >= options.mapper_seconds
    ]
    def_counts: dict[str, Counter[str]] = {}
    for row in rows:
        bucket = def_counts.setdefault(row.semantic_id, Counter())
        bucket["total"] += 1
        if row.attempts:
            bucket["executed"] += 1
    fully_exec = sorted(
        sem for sem, c in def_counts.items() if c["executed"] == c["total"]
    )
    partial_exec = sorted(
        sem for sem, c in def_counts.items() if 2 < c["executed"] < c["total"]
    )
    smoke_only = sorted(
        sem for sem, c in def_counts.items() if c["executed"] == 2
    )
    unstarted = sorted(
        sem for sem, c in def_counts.items() if c["executed"] == 0
    )
    return {
        "stopped_by_activity_budget": report.elapsed_seconds
        >= options.activity_seconds,
        "activity_budget_seconds": options.activity_seconds,
        "elapsed_seconds": report.elapsed_seconds,
        "executed_rows_count": sum(bool(r.attempts) for r in rows),
        "not_reached_rows_count": len(not_reached),
        "first_not_reached_row": not_reached[0] if not_reached else None,
        "boundary_timeout_rows": boundary_timeouts,
        "mapper_timeout_toolchain_rows": mapper_timeouts,
        "fully_executed_definitions": fully_exec,
        "partially_executed_definitions": partial_exec,
        "smoke_only_definitions": smoke_only,
        "unstarted_definitions": unstarted,
        "timeout_transient_artifact_drifts": timeout_transient_drifts or [],
    }


def build_c3_summary(
    paths: c2_c3_campaign.CampaignPaths,
    report: TargetCoverageReport,
    options: CoverageOptions,
    *,
    timeout_transient_drifts: list[dict[str, object]] | None = None,
) -> dict[str, Any]:
    """Build complete verified 360-row C3 summary JSON without modifying artifacts."""
    rows = list(report.rows)
    raw_details = [c2_c3_campaign._summarize_row(row) for row in rows]
    evidence_bytes = c2_c3_campaign._tree_bytes(paths.coverage_dir)
    m0_counts, m0_sub_counts = _m0_reporting_counts(rows)
    executed_walls = [r.attempts[-1].wall_seconds for r in rows if r.attempts]
    formal_metrics = {
        **c2_c3_campaign._formal_metrics_counts(raw_details),
        **_formal_distribution_counts(rows),
    }
    return {
        "spec_id": report.spec_id.value,
        "performance_authority": False,
        "identity_sha256": report.identity_sha256,
        "total_rows": len(rows),
        "executed_rows_count": sum(bool(r.attempts) for r in rows),
        "unique_definitions_count": len({r.semantic_id for r in rows}),
        "unique_workloads_count": len({r.workload_uuid for r in rows}),
        "matrix_complete": report.matrix_complete,
        "costs_observed": report.costs_observed,
        "complete": report.complete,
        "elapsed_seconds": report.elapsed_seconds,
        "smoke_elapsed_seconds": report.smoke_elapsed_seconds,
        "non_smoke_elapsed_seconds": report.elapsed_seconds
        - report.smoke_elapsed_seconds,
        "activity_budget_seconds": options.activity_seconds,
        "total_evidence_bytes": evidence_bytes,
        "total_evidence_mib": evidence_bytes / _MIB,
        "total_evidence_gib": evidence_bytes / _GIB,
        "evidence_budget_gib": options.evidence_bytes / _GIB,
        "budget_stop_summary": _build_budget_stop_summary(
            rows,
            report,
            options,
            timeout_transient_drifts=timeout_transient_drifts,
        ),
        "row_wall_seconds": _wall_stats(executed_walls),
        "row_wall_seconds_by_slot": _wall_by_slot(rows),
        "overall_status_counts": dict(
            sorted(Counter(r.status.value for r in rows).items())
        ),
        "m0_reporting_status_counts": m0_counts,
        "m0_reporting_subreason_counts": m0_sub_counts,
        "by_ir_path": _group_counts(rows, lambda r: r.ir_path.value),
        "by_family": _group_counts(rows, lambda r: r.family),
        "by_role": _group_counts(rows, lambda r: r.role),
        "by_slot": _group_counts(rows, lambda r: r.slot),
        "by_family_and_ir_path": _nested_group_counts(rows, lambda r: r.family),
        "by_role_and_ir_path": _nested_group_counts(rows, lambda r: r.role),
        "by_slot_and_ir_path": _nested_group_counts(rows, lambda r: r.slot),
        "readiness_stage_counts": c2_c3_campaign._readiness_stage_counts(
            raw_details
        ),
        "verification_dimension_counts": _verification_dimension_counts(rows),
        "formal_metrics_counts": formal_metrics,
        "reason_counts": c2_c3_campaign._reason_breakdown(raw_details),
        "stage_cost_summary": c2_c3_campaign._stage_cost_summary(raw_details),
        "stage_cost_summary_by_slot": _stage_costs_by_slot(rows, raw_details),
        "two_path_comparison": compare_two_paths(rows, raw_details),
        "gap_inventory": build_gap_inventory(rows),
        "make_fx_real_retry_observability": {
            "observable_in_current_artifacts": False,
            "reason": (
                "make_fx_backend falls back on DynamicOutputShapeException/"
                "DataDependentOutputException without tagging operator_graph.yaml"
            ),
        },
        "rows_compact": [
            _compact_row_record(i, r, d)
            for i, (r, d) in enumerate(zip(rows, raw_details, strict=True))
        ],
    }


def build_c3_exposure_ledger(
    *,
    c2_summary: dict[str, object],
    report: TargetCoverageReport,
    freeze_sha256: str,
) -> dict[str, object]:
    """Emit the complete M0-C0 exposure ledger across smoke, development, and holdout."""
    uuids_by_role: dict[str, set[str]] = {}
    exposed_by_role: Counter[str] = Counter()
    unexposed_by_role: Counter[str] = Counter()
    for row in report.rows:
        if row.attempts:
            exposed_by_role[row.role] += 1
            uuids_by_role.setdefault(row.role, set()).add(row.workload_uuid)
        else:
            unexposed_by_role[row.role] += 1
    return {
        "schema_role": "llm_core_c3_final_exposure_ledger",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "freeze_sha256": freeze_sha256,
        "report_identity_sha256": report.identity_sha256,
        "target_view_manifest_sha256": c2_summary[
            "target_view_manifest_sha256"
        ],
        "workload_view_digest": c2_summary["workload_view_digest"],
        "exposed_rows_count": sum(exposed_by_role.values()),
        "unexposed_rows_count": sum(unexposed_by_role.values()),
        "exposed_rows_by_role": dict(sorted(exposed_by_role.items())),
        "unexposed_rows_by_role": dict(sorted(unexposed_by_role.items())),
        "exposed_workload_uuids_by_role": {
            k: sorted(v) for k, v in sorted(uuids_by_role.items())
        },
        "policy": (
            "All exposed LLM_CORE_V2 development and holdout workloads cannot serve as "
            "independent unexposed holdouts for subsequent fixes trained on these failures."
        ),
    }


def _verify_row_attempt_hashes(
    index: int, row: CoverageRow, coverage_dir: Path
) -> list[dict[str, object]]:
    drifts: list[dict[str, object]] = []
    for attempt in row.attempts:
        if attempt.reason == "interrupted_in_progress":
            raise ValueError(
                "interrupted worker requires an explicit new campaign; elapsed budget is unknown"
            )
        for relative, recorded_sha in attempt.artifacts.items():
            path = scoped_artifact(coverage_dir, relative)
            if not path.is_file():
                raise ValueError(
                    f"coverage resume artifact missing: {relative}"
                )
            actual_sha = sha256_file(path)
            if actual_sha == recorded_sha:
                continue
            if (
                attempt.status is CoverageStatus.BUDGET_EXHAUSTED
                and "/.formal." in f"/{relative}"
            ):
                drifts.append(
                    {
                        "row_index": index,
                        "semantic_id": row.semantic_id,
                        "slot": row.slot,
                        "ir_path": row.ir_path.value,
                        "artifact": relative,
                        "recorded_sha256": recorded_sha,
                        "disk_sha256": actual_sha,
                        "disk_bytes": path.stat().st_size,
                    }
                )
                continue
            raise ValueError(f"coverage resume artifact drift: {relative}")
    return drifts


def _verify_report_and_artifact_integrity(
    report: TargetCoverageReport,
    expected: TargetCoverageReport,
    coverage_dir: Path,
) -> list[dict[str, object]]:
    if (
        report.identity != expected.identity
        or report.identity_sha256 != expected.identity_sha256
    ):
        raise ValueError(
            "coverage resume identity differs from current sources, runtime, view or budgets"
        )
    if [r.key for r in report.rows] != [r.key for r in expected.rows]:
        raise ValueError("coverage resume row inventory mismatch")
    drifts: list[dict[str, object]] = []
    resolved_dir = coverage_dir.resolve()
    for index, (row, planned) in enumerate(
        zip(report.rows, expected.rows, strict=True)
    ):
        if row.model_dump(
            mode="json", exclude={"status", "reason", "attempts"}
        ) != planned.model_dump(
            mode="json", exclude={"status", "reason", "attempts"}
        ):
            raise ValueError("coverage resume row metadata mismatch")
        drifts.extend(_verify_row_attempt_hashes(index, row, resolved_dir))
    return drifts


def run_container_audit(
    paths: c2_c3_campaign.CampaignPaths,
    expected_freeze_sha256: str,
    *,
    audit_out: Path,
) -> dict[str, object]:
    """Verify full 360-row C3 coverage report inside a device-free container."""
    if torch.cuda.is_initialized():
        raise RuntimeError("device-free C3 audit unexpectedly initialized GPU")
    if os.environ.get("ROCR_VISIBLE_DEVICES", "") != "":
        raise RuntimeError(
            "ROCR_VISIBLE_DEVICES must be empty in device-free audit"
        )
    spec = c2_c3_campaign.verify_environment(
        paths, expected_freeze_sha256, online=False
    )
    c2_summary = c2_c3_campaign._require_verified_c2(paths)
    preserved = verify_preserved_artifacts(paths.outputs)
    cli_output = c2_c3_campaign._run_readonly_cli_preflight(paths)
    target = load_coverage_target(
        paths.manifest, paths.target_view_manifest, paths.descriptor
    )
    options = c2_c3_campaign.clamped_coverage_options(Path(spec["mapper_home"]))
    expected = new_coverage_report(
        target,
        options,
        OrojenesisRunner(options.orojenesis_home).toolchain_identity,
    )
    report = load_json_file(TargetCoverageReport, paths.coverage_report_path)
    transient_drifts = _verify_report_and_artifact_integrity(
        report, expected, paths.coverage_dir
    )
    standard_resume_ok = True
    try:
        target_coverage._resume_or_create(
            paths.coverage_dir, expected, resume=True
        )
    except ValueError as exc:
        if transient_drifts and "coverage resume artifact drift" in str(exc):
            standard_resume_ok = False
        else:
            raise
    authenticate_resumed_attempts(target, report, paths.coverage_dir, options)
    _verify_denominator_360(report)
    summary = build_c3_summary(
        paths, report, options, timeout_transient_drifts=transient_drifts
    )
    ledger = build_c3_exposure_ledger(
        c2_summary=c2_summary,
        report=report,
        freeze_sha256=expected_freeze_sha256,
    )
    audit_out.mkdir(parents=True, exist_ok=True)
    summary_path = audit_out / "c3-summary.json"
    ledger_path = audit_out / "c3-exposure-ledger.json"
    atomic_write_json_value(summary_path, summary)
    atomic_write_json_value(ledger_path, ledger)
    payload = _build_final_audit_payload(
        expected_freeze_sha256=expected_freeze_sha256,
        cli_output=cli_output,
        report=report,
        expected=expected,
        preserved=preserved,
        transient_drifts=transient_drifts,
        standard_resume_ok=standard_resume_ok,
        coverage_report_path=paths.coverage_report_path,
        summary_path=summary_path,
        ledger_path=ledger_path,
    )
    atomic_write_json_value(audit_out / "c3-final-audit.json", payload)
    return payload


def _build_final_audit_payload(
    *,
    expected_freeze_sha256: str,
    cli_output: str,
    report: TargetCoverageReport,
    expected: TargetCoverageReport,
    preserved: dict[str, str],
    transient_drifts: list[dict[str, object]],
    standard_resume_ok: bool,
    coverage_report_path: Path,
    summary_path: Path,
    ledger_path: Path,
) -> dict[str, object]:
    return {
        "audit_passed": True,
        "performance_authority": False,
        "gpu_initialized": torch.cuda.is_initialized(),
        "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "freeze_sha256": expected_freeze_sha256,
        "cli_preflight_output": cli_output,
        "report_schema_version": report.schema_version,
        "report_identity_sha256": report.identity_sha256,
        "expected_identity_sha256": expected.identity_sha256,
        "resume_identity_matches_full_audit": (
            report.identity == expected.identity
            and report.identity_sha256 == expected.identity_sha256
        ),
        "authenticated_rows_count": sum(bool(r.attempts) for r in report.rows),
        "matrix_complete": report.matrix_complete,
        "costs_observed": report.costs_observed,
        "complete": report.complete,
        "non_timeout_transient_artifact_mismatches": 0,
        "timeout_transient_artifact_drifts": transient_drifts,
        "standard_resume_blocked_by_timeout_transient_drift": (
            not standard_resume_ok
        ),
        "preserved_artifacts_sha256": preserved,
        "verified_outputs_sha256": {
            "llm-core/coverage/target-coverage.json": sha256_file(
                coverage_report_path
            ),
            "llm-core/c3-summary.json": sha256_file(summary_path),
            "llm-core/c3-exposure-ledger.json": sha256_file(ledger_path),
        },
    }


def _verify_denominator_360(report: TargetCoverageReport) -> None:
    if len(report.rows) != 360:
        raise ValueError(f"expected 360 coverage rows, got {len(report.rows)}")
    if len({r.semantic_id for r in report.rows}) != 20:
        raise ValueError("expected 20 unique semantic definitions in C3 report")
    if len({r.workload_uuid for r in report.rows}) != 180:
        raise ValueError("expected 180 unique workloads in C3 report")
    if {r.ir_path.value for r in report.rows} != {
        IRPath.MAKE_FX_ATEN.value,
        IRPath.TORCHVIEW_EXTENDED_EINSUM.value,
    }:
        raise ValueError("unexpected IR paths in C3 report")


def run_host_audit(bundle: Path, expected_freeze_sha: str) -> int:
    """Validate C3 resume closeout and launch the isolated device-free audit container."""
    from scripts.internal.solar import c2_c3_host

    cfg = c2_c3_host._load_validated_config(bundle, expected_freeze_sha)
    closeout_path = cfg.bundle / "host-run/c3-resume-closeout.json"
    if not closeout_path.is_file():
        raise RuntimeError(f"missing {closeout_path}")
    closeout = load_json_value(closeout_path)
    if closeout.get("clean") is not True or closeout.get("returncode") != 0:
        raise RuntimeError(f"c3-resume closeout is not clean/zero: {closeout}")
    verify_preserved_artifacts(cfg.bundle)
    audit_out = cfg.bundle / "audit-c3-out"
    audit_out.mkdir(parents=True, exist_ok=True)
    receipt = cfg.bundle / "host-run/c3-final-audit.json"
    cid = receipt.with_suffix(".cid")
    argv = _host_docker_command(cfg, audit_out, cid)
    code = _exec_audit_container(argv, receipt, cid, closeout)
    if code == 0:
        for name in (
            "c3-summary.json",
            "c3-exposure-ledger.json",
            "c3-final-audit.json",
        ):
            src = audit_out / name
            if not src.is_file():
                raise RuntimeError(f"container audit did not emit {name}")
            copyfile(src, cfg.bundle / "llm-core" / name)
        verify_preserved_artifacts(cfg.bundle)
    return code


def _host_docker_command(
    cfg: HostConfig, audit_out: Path, cid: Path
) -> list[str]:
    from scripts.internal.solar import c2_c3_host

    spec = cfg.spec
    affinity = "--cpuset-cpus=" + ",".join(
        map(str, spec["budgets"]["required_cpu_affinity"])
    )
    mem = spec["budgets"]["worker_memory_bytes"]
    mounts = [
        *c2_c3_host._mount(cfg.execution_checkout, c2_c3_host.CONTAINER_REPO),
        *c2_c3_host._mount(cfg.mapper_home, str(cfg.mapper_home)),
        *c2_c3_host._mount(cfg.bundle / "capsule", "/research"),
        *c2_c3_host._mount(
            Path(__file__).resolve(), "/audit-src/c3_final_audit.py"
        ),
        *c2_c3_host._mount(cfg.bundle, c2_c3_host.CONTAINER_OUTPUTS),
        *c2_c3_host._mount(audit_out, "/audit-c3-out", writable=True),
    ]
    env = c2_c3_host._env_flags(
        {
            "PATH": "/venv/bin:/usr/local/bin:/usr/bin:/bin",
            c2_c3_host.CONTAINER_IMAGE_ID_ENV: spec["image_id"],
            "PYTHONPATH": f"{c2_c3_host.CONTAINER_REPO}/src:/research:/audit-src",
            "PYTHONDONTWRITEBYTECODE": "1",
            "SOL_EXECBENCH_SANDBOXED": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "ROCR_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
        }
    )
    return [
        "docker",
        "run",
        "--pull=never",
        "--rm",
        "--cidfile",
        str(cid),
        "--init",
        "--read-only",
        "--network=none",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=256",
        f"--memory={mem}",
        f"--memory-swap={mem}",
        affinity,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--workdir",
        c2_c3_host.CONTAINER_REPO,
        "--entrypoint",
        c2_c3_host.CONTAINER_PYTHON,
        "--tmpfs",
        "/tmp:rw,exec,nosuid,nodev,size=128m",
        *mounts,
        *env,
        spec["image_id"],
        "-B",
        "-m",
        "c3_final_audit",
        AuditMode.CONTAINER_AUDIT.value,
        "--expected-freeze-sha",
        cfg.freeze_sha256,
        "--audit-out",
        "/audit-c3-out",
    ]


def _exec_audit_container(
    argv: list[str], receipt: Path, cid: Path, closeout: dict[str, object]
) -> int:
    if cid.exists():
        with defer_termination():
            _remove_owned_container("docker", cid)
            cid.unlink(missing_ok=True)
    started = time.monotonic()
    result = None
    timed_out = False
    try:
        result = run_in_process_group_bounded(
            argv,
            timeout=_AUDIT_TIMEOUT_SECONDS,
            max_capture_bytes=_AUDIT_CAPTURE_BYTES,
        )
        code = result.returncode
    except TimeoutExpired:
        timed_out = True
        code = 124
    finally:
        atomic_write_json_value(
            receipt,
            {
                "action": "c3-final-audit",
                "returncode": code if result is not None or timed_out else None,
                "timed_out": timed_out,
                "wall_seconds": time.monotonic() - started,
                "c3_resume_returncode": closeout.get("returncode"),
                "c3_resume_wall_seconds": closeout.get("wall_seconds"),
                "c3_resume_clean": closeout.get("clean"),
                "stdout": result.stdout if result else None,
                "stderr": result.stderr if result else None,
            },
        )
        if cid.exists():
            with defer_termination():
                _remove_owned_container("docker", cid)
                cid.unlink(missing_ok=True)
    return code


def main(argv: Sequence[str] | None = None) -> int:
    """Run either host orchestration or in-container C3 verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=tuple(m.value for m in AuditMode))
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--repo", type=Path, default=Path("/sol-execbench"))
    parser.add_argument("--outputs", type=Path, default=Path("/outputs"))
    parser.add_argument(
        "--freeze", type=Path, default=Path("/outputs/freeze.json")
    )
    parser.add_argument("--expected-freeze-sha", required=True)
    parser.add_argument("--audit-out", type=Path, default=Path("/audit-c3-out"))
    args = parser.parse_args(argv)
    mode = AuditMode(args.mode)
    if mode is AuditMode.HOST_AUDIT:
        bundle = args.bundle or (
            Path(__file__).resolve().parents[3]
            / "data/outputs/solar-c2-c3-20261119"
        )
        return run_host_audit(bundle, args.expected_freeze_sha)
    paths = c2_c3_campaign.CampaignPaths(
        repo=args.repo, outputs=args.outputs, freeze=args.freeze
    )
    run_container_audit(
        paths, args.expected_freeze_sha, audit_out=args.audit_out
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
