"""Verify E2 successor inputs and readiness before any holdout can be opened."""

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.internal.solar.e2_contracts import (
    EVIDENCE_BUDGET_BYTES,
    GPU_BUDGET_SECONDS,
    ORACLE_BUDGET_SECONDS,
)
from scripts.internal.solar.e2_successor_ledger import Phase, planned_sessions
from sol_execbench.core.bench.clock_ownership import ClockDevice
from sol_execbench.core.data.json_utils import load_json_value
from sol_execbench.core.integrity import sha256_file, stable_json_checksum
from sol_execbench.core.scoring.release_environment import (
    current_release_execution_identity,
    verify_release_source_state,
)
from solar.verification.execution import require_current_semantic_runtime

FREEZE_KIND = "e2-successor-execution-v1"
_ROLE_NAMES = frozenset(
    {
        "protocol",
        "original_spec",
        "count_erratum",
        "b2_qualification",
        "baseline_budget",
        "compact_budget",
        "development_launch",
        "development_capsule",
        "development_measurement",
        "gpu_interruptions",
        "offline_term",
        "offline_kill",
    }
)
_BUDGETS = {
    "gpu_seconds": GPU_BUDGET_SECONDS,
    "oracle_cpu_seconds": ORACLE_BUDGET_SECONDS,
    "evidence_bytes": EVIDENCE_BUDGET_BYTES,
}
_CAPTURE_COMPONENTS = tuple(
    f"scripts/internal/solar/{name}.py"
    for name in (
        "e2_archive_arrays",
        "e2_archive_spool",
        "e2_contracts",
        "e2_evidence_budget",
        "e2_input_replay",
        "e2_inputs",
        "e2_observation",
        "e2_offline_storage",
        "e2_oracle_budget",
        "e2_postprocess",
        "e2_runner",
        "e2_service",
        "e2_staging",
        "e2_successor_campaign",
        "e2_successor_capture",
        "e2_successor_entry",
        "e2_successor_exposure",
        "e2_successor_ledger",
        "e2_successor_runner",
        "e2_successor_offline",
    )
)


@dataclass(frozen=True, slots=True, kw_only=True)
class VerifiedSuccessorFreeze:
    """Share the identity already verified in the selected runtime."""

    path: Path
    sha256: str
    schedule_sha256: str
    execution_commit: str
    image_id: str
    device: ClockDevice
    activity: Path


def schedule_sha256() -> str:
    """Bind the fixed 1,440 ordered metadata rows without generating any inputs."""
    return stable_json_checksum(
        [item.metadata() for item in planned_sessions(Phase.ONLINE)]
    )


def _bound_artifacts(root: Path, raw: dict[str, Any]) -> dict[str, Path]:
    if set(raw) != _ROLE_NAMES:
        raise ValueError(
            "E2 successor freeze must bind every preregistration and readiness artifact"
        )
    bound = {}
    scope = root.resolve(strict=True)
    for role, binding in raw.items():
        relative = Path(binding["path"])
        if relative.is_absolute():
            raise ValueError("E2 readiness artifacts require relative paths")
        path = (scope / relative).resolve(strict=True)
        if (
            not path.is_relative_to(scope)
            or sha256_file(path) != binding["sha256"]
        ):
            raise ValueError(f"E2 readiness artifact changed or moved: {role}")
        bound[role] = path
    return bound


def _verify_capsule(capsule: Path, expected: dict[str, Any]) -> None:
    actual = {
        path.relative_to(capsule).as_posix(): sha256_file(path)
        for path in capsule.rglob("*.py")
    }
    if expected != actual or not set(_CAPTURE_COMPONENTS) <= actual.keys():
        raise ValueError(
            "E2 frozen research capsule changed or omitted capture components"
        )


def _verify_qualification(path: Path, execution: str) -> None:
    b2 = load_json_value(path)
    expected = [("static", 163), ("canary", 112), ("full", 163)]
    if (
        b2["status"] != "passed"
        or b2["execution_commit"] != execution
        or b2["representative_slots"] != 8
        or [(gate["stage"], gate["items"]) for gate in b2["gates"]] != expected
        or any(gate["source_revision"] != execution for gate in b2["gates"])
    ):
        raise ValueError(
            "E2 native prerequisite is not the full B2 chain for this execution commit"
        )


def _verify_budget(bound: dict[str, Path]) -> None:
    baseline_path = bound["baseline_budget"]
    baseline = load_json_value(baseline_path)
    compact = load_json_value(bound["compact_budget"])
    if (
        baseline["provisional"] is not False
        or baseline["input_rows"] != 1440
        or compact["input_rows"] != 1440
        or compact["exact_source_members_checked"] is not True
        or compact["prior_forecast_sha256"] != sha256_file(baseline_path)
    ):
        raise ValueError(
            "E2 budget proof did not reconcile the original denominator"
        )
    gpu = baseline["scenarios"]["pressure_plus25"][
        "gpu_if_offline_deferred_seconds"
    ]
    storage = compact["scenarios"]["pressure_plus25"]
    if (
        not 0 < gpu <= GPU_BUDGET_SECONDS
        or not 0 < storage["online_peak_bytes"] <= EVIDENCE_BUDGET_BYTES
        or not 0
        < storage["offline_peak_bytes_with_one_pending_archive_reserve"]
        <= EVIDENCE_BUDGET_BYTES
    ):
        raise ValueError(
            "E2 full-roster budget proof exceeds the frozen limits"
        )


def _verify_interruptions(bound: dict[str, Path]) -> None:
    rows = load_json_value(bound["gpu_interruptions"])
    expected = [("term", 189), ("kill", 0), ("capture-quota", 189)]
    if (
        [(row["fault"], row["verified_raw_calls"]) for row in rows] != expected
        or [row["raw_prefix_lost"] for row in rows] != [0, 189, 0]
        or any(
            row["gpu_closeout_clean"] is not True
            or row["actual_final_verdict"] is not None
            for row in rows
        )
    ):
        raise ValueError("E2 near-peak GPU interruption proof is incomplete")
    (term,) = load_json_value(bound["offline_term"])
    kill = load_json_value(bound["offline_kill"])
    if (
        term["fault"] != "term"
        or term["integrity_check_passed"] is not True
        or term["observed"]["verified_committed_calls"] != 188
        or term["observed"]["pending_writes"] != 1
        or kill["state"] != "completed"
        or kill["verified_committed_calls"] != 188
        or kill["pending_writes"] != 1
        or kill["observed_process_exit"] != 137
        or kill["final_verdict"] is not None
    ):
        raise ValueError(
            "E2 near-peak offline interruption proof is incomplete"
        )


def _verify_measured_capture(
    bound: dict[str, Path], spec: dict[str, Any]
) -> None:
    plan_path = bound["development_capsule"]
    plan = load_json_value(plan_path)
    launch = load_json_value(bound["development_launch"])
    measured = load_json_value(bound["development_measurement"])
    if (
        launch["prepared_sha256"] != sha256_file(plan_path)
        or launch["source_commit"] != spec["execution_commit"]
        or launch["image_id"] != spec["image_id"]
        or launch["device"] != spec["device"]
        or plan["development_only"] is not True
        or measured["holdout_materialized"] is not False
        or measured["full_path_sessions"] != 8
        or measured["verified_input_hashes"] != 1520
        or measured["forecast_sha256"] != sha256_file(bound["compact_budget"])
        or [row["session_id"] for row in measured["rows"]] != plan["roster"]
        or any(
            row["full_path"] is not True
            or row["verified_input_count"] != 190
            or row["reference_calls"] != 371
            or row["published_after_candidate_exit"] is not True
            for row in measured["rows"]
        )
    ):
        raise ValueError(
            "E2 measured compact capture proof does not match the chosen execution"
        )
    rows = measured["rows"]
    for row in rows:
        cpu = row["oracle_cpu_seconds"]
        if (
            not math.isfinite(cpu)
            or not 0 <= cpu <= ORACLE_BUDGET_SECONDS
            or not 0
            < row["exclusive_wall_seconds"]
            <= row["online_pressure_prediction_seconds"]
            or not 0
            < row["retained_allocated_bytes"]
            <= row["online_pressure_prediction_bytes"]
        ):
            raise ValueError(
                "E2 measured cost no longer fits the verified pressure forecast"
            )
    deferred = [row["offline"] for row in rows if row["mode"] == "incumbent"]
    if len(deferred) != 4 or any(
        row["state"] != "completed"
        or row["gpu_initialized"] is not False
        or row["committed"] != 190
        or row["available_observations"] != 190
        for row in deferred
    ):
        raise ValueError(
            "E2 current capture evidence lacks its complete deferred replays"
        )
    for name in _CAPTURE_COMPONENTS:
        digest = plan["capsule_sha256"].get(name)
        if digest is None or spec["capsule_sha256"][name] != digest:
            raise ValueError(
                f"E2 capture component changed since the development proof: {name}"
            )


def verify_frozen_successor(
    path: Path, *, repository: Path, capsule: Path, evidence: Path
) -> VerifiedSuccessorFreeze:
    """Check source, runtime, full-roster budget and readiness without opening holdout."""
    path_sha256 = sha256_file(path)
    spec = load_json_value(path)
    schedule_digest = schedule_sha256()
    if (
        spec["research_format"] != FREEZE_KIND
        or spec["schedule_sha256"] != schedule_digest
        or spec["budgets"] != _BUDGETS
        or spec["performance_authority"] is not False
    ):
        raise ValueError("E2 successor experiment envelope changed")
    execution = spec["execution_commit"]
    image = spec["image_id"]
    name = spec["activity_directory"]
    if re.fullmatch(r"[a-z0-9][a-z0-9-]+", name) is None:
        raise ValueError("E2 successor requires one frozen activity directory")
    activity = evidence.resolve(strict=True) / name
    if activity.is_symlink():
        raise ValueError("E2 activity directory must be owned by the execution")
    device = ClockDevice(uuid=spec["device"]["uuid"], bdf=spec["device"]["bdf"])
    if re.fullmatch(r"[0-9a-f]{40}", execution) is None:
        raise ValueError("E2 successor requires a committed base")
    bound = _bound_artifacts(evidence, spec["evidence_bindings"])
    _verify_capsule(capsule, spec["capsule_sha256"])
    _verify_qualification(bound["b2_qualification"], execution)
    _verify_budget(bound)
    _verify_interruptions(bound)
    _verify_measured_capture(bound, spec)
    verify_release_source_state(repository, expected_revision=execution)
    if (
        current_release_execution_identity(execution).container_image_id
        != image
    ):
        raise ValueError("E2 successor runtime image changed")
    require_current_semantic_runtime(spec["container_semantic_runtime"])
    if sha256_file(path) != path_sha256:
        raise ValueError("E2 successor freeze changed during verification")
    return VerifiedSuccessorFreeze(
        path=path,
        sha256=path_sha256,
        schedule_sha256=schedule_digest,
        execution_commit=execution,
        image_id=image,
        device=device,
        activity=activity,
    )
