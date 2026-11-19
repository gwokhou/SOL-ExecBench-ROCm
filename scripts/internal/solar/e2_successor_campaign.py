"""Bounded serial phase execution for a separately frozen successor."""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from subprocess import SubprocessError
from typing import Any

from scripts.internal.solar.e2_contracts import (
    CORRECTNESS_TIMEOUT_SECONDS,
    COST_TIMEOUT_SECONDS,
    EVIDENCE_BUDGET_BYTES,
    GPU_BUDGET_SECONDS,
    ORACLE_BUDGET_SECONDS,
    Activity,
)
from scripts.internal.solar.e2_evidence_budget import (
    admit_evidence_writer,
    admit_offline_writer,
    retained_evidence,
)
from scripts.internal.solar.e2_successor_ledger import (
    AttemptState,
    Phase,
    PlannedSession,
    SuccessorLedger,
    planned_sessions,
)
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_value,
)
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.process.termination import cleanup_on_termination

_ONLINE_WRITER_BYTES = 1024**3 + 16 * 1024**2
_PHASE_RESULT = "phase-result.json"
_HANDBACK = "bound-gpu-handback.json"
ONLINE_CLOSEOUT_RESERVE_SECONDS = 60


@dataclass(frozen=True, slots=True, kw_only=True)
class AttemptAllowance:
    """Limits the phase controller grants to a single trusted worker."""

    wall_seconds: float
    oracle_cpu_seconds: float
    output_bytes: int


@dataclass(frozen=True, slots=True, kw_only=True)
class AttemptOutcome:
    """One observed verdict, optional completed evidence and any stopping cause."""

    verdict: str | None
    evidence: Path | None
    oracle_cpu_seconds: float | None
    stop_reason: str | None = None


ExecuteAttempt = Callable[
    [PlannedSession, Path, AttemptAllowance], AttemptOutcome
]


def _phase_result(ledger: SuccessorLedger, phase: Phase) -> Path:
    return ledger.root / phase / _PHASE_RESULT


def bind_gpu_handback(ledger: SuccessorLedger, closeout: Path) -> None:
    """Join the drained native lease receipt to this activity's online result."""
    if closeout.parent != ledger.root:
        raise ValueError("native handback must belong to the current activity")
    data = load_json_value(closeout)
    if (
        data["clean"] is not True
        or data["gpu_process_count"] != 0
        or [row["perf_level"] for row in data["clocks"]["gpu_data"]]
        != ["AMDSMI_DEV_PERF_LEVEL_AUTO"]
        or len(data["locks"]) != 2
        or {row["level"] for row in data["locks"]}
        != {"L1-ownership", "L2-execution"}
        or any(row["held"] is not False for row in data["locks"])
    ):
        raise ValueError("native GPU handback is incomplete")
    bound = ledger.root / _HANDBACK
    if bound.exists():
        raise FileExistsError(bound)
    atomic_write_json_value(
        bound,
        {
            "native_closeout": closeout.name,
            "native_closeout_sha256": sha256_file(closeout),
            "online_result_sha256": sha256_file(
                _phase_result(ledger, Phase.ONLINE)
            ),
            "roster_sha256": ledger.roster_sha256,
        },
    )


def _offline_oracle_spend(ledger: SuccessorLedger) -> float:
    handback = load_json_value(ledger.root / _HANDBACK)
    online = _phase_result(ledger, Phase.ONLINE)
    native = ledger.root / handback["native_closeout"]
    if (
        handback["roster_sha256"] != ledger.roster_sha256
        or handback["online_result_sha256"] != sha256_file(online)
        or handback["native_closeout_sha256"] != sha256_file(native)
    ):
        raise ValueError("E2 offline handback binding does not match")
    data = load_json_value(online)
    if data["state"] != "completed":
        raise ValueError(
            "E2 offline phase requires the completed online roster"
        )
    rows = ledger.reconcile(Phase.ONLINE)
    if (
        data["roster_sha256"] != ledger.roster_sha256
        or data["completed_attempts"] != len(rows)
        or any(row["state"] != AttemptState.COMPLETED for row in rows)
    ):
        raise ValueError(
            "E2 online completion is inconsistent with the attempt receipts"
        )
    cpu_seconds = float(data["total_oracle_cpu_seconds"])
    if (
        data["oracle_cpu_spend_complete"] is not True
        or not 0 <= cpu_seconds <= ORACLE_BUDGET_SECONDS
    ):
        raise ValueError("E2 online oracle CPU accounting is incomplete")
    return cpu_seconds


def _allowance(
    ledger: SuccessorLedger,
    phase: Phase,
    item: PlannedSession,
    deadline: float | None,
    cpu_spent: float,
) -> AttemptAllowance:
    wall = float(
        COST_TIMEOUT_SECONDS
        if item.session.activity is Activity.COST
        else CORRECTNESS_TIMEOUT_SECONDS
    )
    if deadline is not None:
        wall = min(wall, deadline - time.monotonic())
    remaining_cpu = ORACLE_BUDGET_SECONDS - cpu_spent
    if wall <= 0:
        raise TimeoutError("GPU activity wall budget exhausted")
    if remaining_cpu <= 0:
        raise TimeoutError("cumulative oracle CPU budget exhausted")
    if phase is Phase.ONLINE:
        admit_evidence_writer(
            ledger.root, writer_limit_bytes=_ONLINE_WRITER_BYTES
        )
        output_bytes = _ONLINE_WRITER_BYTES
    else:
        output_bytes = admit_offline_writer(ledger.root)
    if deadline is not None:
        wall = min(wall, deadline - time.monotonic())
        if wall <= 0:
            raise TimeoutError("GPU activity wall budget exhausted")
    return AttemptAllowance(
        wall_seconds=wall,
        oracle_cpu_seconds=remaining_cpu,
        output_bytes=output_bytes,
    )


def _run_attempt(
    ledger: SuccessorLedger,
    phase: Phase,
    item: PlannedSession,
    allowance: AttemptAllowance,
    execute: ExecuteAttempt,
) -> AttemptOutcome:
    ledger.start(phase, item)
    stage = ledger.root / phase / item.directory
    start = time.monotonic()
    try:
        outcome = execute(item, stage / "evidence", allowance)
        spent = outcome.oracle_cpu_seconds
        if spent is None and outcome.stop_reason is None:
            raise ValueError("a completed E2 attempt requires CPU accounting")
        if spent is not None and (not math.isfinite(spent) or spent < 0):
            raise ValueError("invalid actual oracle CPU accounting")
        reason = outcome.stop_reason
        if spent is not None and spent > allowance.oracle_cpu_seconds:
            reason = reason or "oracle_cpu_allowance_exceeded"
        elif time.monotonic() - start > allowance.wall_seconds:
            reason = reason or "attempt_wall_allowance_exceeded"
        ledger.finish(
            phase,
            item,
            verdict=outcome.verdict,
            stop_reason=reason,
            evidence=outcome.evidence,
        )
        return AttemptOutcome(
            verdict=outcome.verdict,
            evidence=outcome.evidence,
            oracle_cpu_seconds=spent,
            stop_reason=reason,
        )
    except (
        OSError,
        RuntimeError,
        ValueError,
        SubprocessError,
        SystemExit,
    ) as exc:
        reason = f"{type(exc).__name__}: {exc}"
        ledger.finish(
            phase, item, verdict=None, stop_reason=reason, evidence=None
        )
        return AttemptOutcome(
            verdict=None,
            evidence=None,
            oracle_cpu_seconds=None,
            stop_reason=reason,
        )


def execute_phase(
    ledger: SuccessorLedger,
    phase: Phase,
    execute: ExecuteAttempt,
    *,
    online_owner_started_monotonic: float | None = None,
) -> dict[str, Any]:
    """Run every planned attempt once, halting at a recorded failure."""
    if phase is Phase.ONLINE:
        if (
            online_owner_started_monotonic is None
            or not 0 < online_owner_started_monotonic <= time.monotonic()
        ):
            raise ValueError(
                "online phase requires the actual native owner start"
            )
        deadline = (
            online_owner_started_monotonic
            + GPU_BUDGET_SECONDS
            - ONLINE_CLOSEOUT_RESERVE_SECONDS
        )
        cpu_spent = 0.0
    else:
        deadline = None
        cpu_spent = _offline_oracle_spend(ledger)
    plan = tuple(planned_sessions(phase))
    ledger.begin_phase(phase)
    completed = 0
    oracle_cpu_spend_complete = True
    stop_reason = None
    try:
        with cleanup_on_termination():
            for item in plan:
                allowance = _allowance(ledger, phase, item, deadline, cpu_spent)
                outcome = _run_attempt(ledger, phase, item, allowance, execute)
                if outcome.oracle_cpu_seconds is None:
                    oracle_cpu_spend_complete = False
                else:
                    cpu_spent += outcome.oracle_cpu_seconds
                stop_reason = outcome.stop_reason
                if stop_reason:
                    break
                completed += 1
    except (OSError, RuntimeError, ValueError, SystemExit) as exc:
        stop_reason = f"{type(exc).__name__}: {exc}"
    logical_bytes = allocated_bytes = None
    try:
        retained = retained_evidence(ledger.root)
        logical_bytes = retained.logical_bytes
        allocated_bytes = retained.allocated_bytes
        if max(logical_bytes, allocated_bytes) > EVIDENCE_BUDGET_BYTES:
            stop_reason = stop_reason or "evidence_budget_exhausted"
    except (OSError, ValueError) as exc:
        stop_reason = stop_reason or f"{type(exc).__name__}: {exc}"
    result = {
        "state": "completed"
        if completed == len(plan) and stop_reason is None
        else "INCOMPLETE",
        "phase": phase.value,
        "roster_sha256": ledger.roster_sha256,
        "completed_attempts": completed,
        "planned_attempts": len(plan),
        "total_oracle_cpu_seconds": cpu_spent,
        "oracle_cpu_spend_complete": oracle_cpu_spend_complete,
        "retained_logical_bytes": logical_bytes,
        "retained_allocated_bytes": allocated_bytes,
        "stop_reason": stop_reason,
        "performance_authority": False,
    }
    atomic_write_json_value(_phase_result(ledger, phase), result)
    return result
