"""Independently derive E2 input identities from closed-session receipts."""

import re
from pathlib import Path
from typing import Any

from scripts.internal.solar.e2_contracts import (
    MAX_SESSION_OBSERVATIONS,
    Session,
)
from scripts.internal.solar.e2_staging import config
from sol_execbench.core.bench.custom_inputs import derive_custom_input_seed
from sol_execbench.core.bench.judge.inputs import generate_seeded_inputs
from sol_execbench.core.bench.judge.storage import tensor_identity
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload

_NONCE_PATTERN = re.compile(r"[0-9a-f]{64}")


def _variation_index(
    observation: dict[str, Any], *, iterations: int, trials: int
) -> int | None:
    operation = observation["operation"]
    if operation == "correctness":
        return None
    if operation != "timing_iteration":
        raise ValueError(
            f"E2 input replay does not recognize operation {operation!r}"
        )
    trial, step = observation["trial_index"], observation["iteration_index"]
    if not 0 <= trial < trials or not 0 <= step < iterations:
        raise ValueError(
            "E2 input replay coordinate is outside the timing schedule"
        )
    return trial * iterations + step


def verify_session_inputs(
    staging: Path,
    session: Session,
    observations: dict[str, Any],
    *,
    private_seed: int,
) -> tuple[str, ...]:
    """Verify a bounded observed prefix without importing reference or candidate code."""
    nonce = observations["private_nonce_after_channel_close"]
    if not isinstance(nonce, str) or not _NONCE_PATTERN.fullmatch(nonce):
        raise ValueError("E2 private nonce has an unexpected format")
    rows = observations["observations"]
    if len(rows) > MAX_SESSION_OBSERVATIONS:
        raise ValueError("E2 input replay exceeded the call bound")
    definition = Definition.model_validate_json(
        (staging / "definition.json").read_text()
    )
    (workload_json,) = (staging / "workload.jsonl").read_text().splitlines()
    workload = Workload.model_validate_json(workload_json)
    axes = definition.get_resolved_axes_values(workload.axes)
    benchmark = config(session)
    digests = []
    for row in rows:
        variation = _variation_index(
            row,
            iterations=benchmark.iterations + benchmark.warmup_runs,
            trials=benchmark.trials,
        )
        seed = derive_custom_input_seed(
            definition,
            workload,
            row_index=row["row_index"],
            base_seed=private_seed,
            round_index=row["round_index"],
            run_nonce=nonce,
            variation_index=variation,
        )
        inputs = generate_seeded_inputs(
            session.case.task, session.case.family, axes, seed
        )
        identity = tensor_identity(inputs.values)
        if identity != row["input_sha256"]:
            raise ValueError("E2 session input identity did not reproduce")
        digests.append(identity)
    if len(set(digests)) != len(digests):
        raise ValueError("E2 session repeated an input hash")
    return tuple(digests)
