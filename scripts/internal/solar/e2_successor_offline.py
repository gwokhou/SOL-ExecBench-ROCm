"""Diagnose retained E2 cost calls in a device-free process using split storage."""

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from scripts.internal.solar.e2_contracts import (
    MAX_SESSION_OBSERVATIONS,
    ORACLE_BUDGET_SECONDS,
    Mode,
    Session,
)
from scripts.internal.solar.e2_observation import diagnose
from scripts.internal.solar.e2_offline_storage import (
    commit_diagnostic_bundle,
    encode_diagnostic_arrays,
)
from scripts.internal.solar.e2_oracle_budget import OracleCpuBudget
from scripts.internal.solar.e2_postprocess import decode_observation
from sol_execbench.core.bench.judge.contracts import (
    JudgeInstance,
    load_case_problem,
)
from sol_execbench.core.bench.judge.interval_oracle import OracleUncertainError
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.process.termination import cleanup_on_termination


@dataclass(frozen=True, slots=True, kw_only=True)
class OfflineReplay:
    """Bind a completed raw session to a distinct and bounded diagnosis directory."""

    repository: Path
    source: Path
    output: Path
    session: Session
    output_allowance_bytes: int
    oracle_cpu_allowance_seconds: float = ORACLE_BUDGET_SECONDS


def replay_incumbent(replay: OfflineReplay) -> dict[str, Any]:
    """Recompute the unchanged E2 oracle over each actual raw capture."""
    if replay.session.mode is not Mode.INCUMBENT:
        raise ValueError("offline replay requires an incumbent cost session")
    if replay.output_allowance_bytes <= 0 or torch.cuda.is_initialized():
        raise ValueError(
            "offline replay requires a positive allowance and no GPU"
        )
    raw = replay.source / "judge-observations"
    manifest = raw / "observations.json"
    payload = json.loads(manifest.read_text())
    closed = json.loads((replay.source / "session.json").read_text())
    if (
        closed["session_id"] != replay.session.identity
        or closed["observations"] != payload
    ):
        raise ValueError("offline source session identity changed")
    if payload["stop_reason"] or payload["pending_correctness"]:
        raise ValueError("offline replay requires the closed incumbent prefix")
    rows = payload["observations"]
    if len(rows) > MAX_SESSION_OBSERVATIONS:
        raise ValueError("offline replay exceeded the frozen call bound")
    definition, workload = load_case_problem(
        replay.repository,
        replace(replay.session.case, instance=JudgeInstance.DEVELOPMENT_0),
    )
    replay.output.mkdir(exist_ok=False)
    spent, committed = 0, 0
    oracle_budget = OracleCpuBudget(replay.oracle_cpu_allowance_seconds)
    started = time.monotonic()
    stop_reason = None
    try:
        with cleanup_on_termination():
            for index, row in enumerate(rows):
                artifact = f"call-{index:03d}.npz"
                if row["artifact"] != artifact:
                    raise ValueError("offline source call ordering changed")
                source = raw / artifact
                if sha256_file(source) != row["artifact_sha256"]:
                    raise ValueError("offline source archive identity changed")
                observation = decode_observation(source, row)
                with oracle_budget.charge():
                    classification, arrays = diagnose(
                        replay.session.case,
                        replay.session.strategy,
                        observation,
                        definition,
                        workload,
                    )
                spent += commit_diagnostic_bundle(
                    replay.output,
                    index,
                    encode_diagnostic_arrays(arrays),
                    {**classification, "input_sha256": row["input_sha256"]},
                    source_sha256=row["artifact_sha256"],
                    remaining_bytes=replay.output_allowance_bytes - spent,
                )
                committed += 1
    except (OSError, OracleUncertainError, SystemExit, ValueError) as exc:
        stop_reason = f"{type(exc).__name__}: {exc}"
    result = {
        "session_id": replay.session.identity,
        "source_receipt_sha256": sha256_file(manifest),
        "available_observations": len(rows),
        "committed": committed,
        "allocated_upper_estimate_bytes": spent,
        "oracle_cpu_seconds": oracle_budget.spent_seconds,
        "wall_seconds": time.monotonic() - started,
        "gpu_initialized": torch.cuda.is_initialized(),
        "stop_reason": stop_reason,
        "performance_authority": False,
    }
    atomic_write_json_value(replay.output / "result.json", result)
    return result
