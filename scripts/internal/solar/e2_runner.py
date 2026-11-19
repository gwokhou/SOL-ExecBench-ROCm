"""One E2 session through incumbent isolation, clocks, traces and termination."""

import json
import os
import time
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.internal.solar.e2_contracts import (
    CORRECTNESS_TIMEOUT_SECONDS,
    COST_TIMEOUT_SECONDS,
    PRIVATE_FILE,
    Activity,
    Session,
)
from scripts.internal.solar.e2_staging import instrument_drivers, package
from sol_execbench.cli.evaluation.profile_mode import ProfileMode
from sol_execbench.cli.evaluation.runtime import (
    EvaluationRuntimeResult,
    EvaluationRuntimeSuccess,
    run_evaluation_runtime,
)
from sol_execbench.core.bench.gpu_execution import native_gpu_execution
from sol_execbench.core.bench.judge.candidate_catalog import candidate_source
from sol_execbench.core.bench.reward_hack.static_review import (
    review_solution_sources,
)
from sol_execbench.core.data.dtypes import dtype_str_to_torch_dtype
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.data.trace import EvaluationStatus
from sol_execbench.core.integrity import sha256_bytes
from sol_execbench.core.process.environment import (
    ENV_SOL_EXECBENCH_DEVICE,
    ENV_SOL_EXECBENCH_INPUT_NONCE,
    ENV_SOL_EXECBENCH_SANDBOXED,
)
from sol_execbench.driver.problem_packager import ProblemPackager


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionStaging:
    """Trusted experiment hooks attached before staging any candidate."""

    workspace: Callable[[Path], AbstractContextManager[dict[str, Any]]]
    instrument: Callable[[Path], None]


def run_session(
    repository: Path,
    output: Path,
    session: Session,
    *,
    private_seed: int,
    nonce: str,
    remaining_seconds: float,
    staging: SessionStaging | None = None,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Execute once; preserve failed verdicts and never compensate with retries."""
    if os.environ.get(ENV_SOL_EXECBENCH_SANDBOXED) != "1":
        raise RuntimeError("E2 requires the hardened container")
    if device not in {"cpu", "cuda:0"}:
        raise ValueError(
            "E2 requires the selected GPU or the development CPU fixture"
        )
    start = time.monotonic()
    limit = (
        COST_TIMEOUT_SECONDS
        if session.activity is Activity.COST
        else CORRECTNESS_TIMEOUT_SECONDS
    )
    allowance = min(limit, remaining_seconds)
    if allowance < 1:
        raise TimeoutError("E2 session has no remaining evaluation window")
    deadline = start + allowance
    output.mkdir(parents=True, exist_ok=False)
    old = {
        name: os.environ.get(name)
        for name in (ENV_SOL_EXECBENCH_DEVICE, ENV_SOL_EXECBENCH_INPUT_NONCE)
    }
    os.environ[ENV_SOL_EXECBENCH_DEVICE] = device
    os.environ[ENV_SOL_EXECBENCH_INPUT_NONCE] = nonce
    try:
        with (
            staging.workspace(output)
            if staging
            else nullcontext({}) as workspace,
            native_gpu_execution(
                device, timeout_seconds=min(60, allowance), manage_clocks=True
            ),
            package(
                repository,
                output,
                session,
                workload_uuid=str(uuid.uuid4()),
                private_seed=private_seed,
            ) as packager,
        ):
            result = _evaluate_staged(
                packager,
                session,
                output,
                device=device,
                start=start,
                deadline=deadline,
                instrument=staging.instrument
                if staging
                else instrument_drivers,
            )
        result.update(workspace)
    finally:
        (output / PRIVATE_FILE).unlink(missing_ok=True)
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    result["session_wall_seconds"] = time.monotonic() - start
    if result["session_wall_seconds"] > allowance and not result["stop_reason"]:
        result["stop_reason"] = "session_wall_budget_exhausted"
    atomic_write_json_value(output / "session.json", result)
    return result


def _evaluate_staged(
    packager: ProblemPackager,
    session: Session,
    output: Path,
    *,
    device: str,
    start: float,
    deadline: float,
    instrument: Callable[[Path], None],
) -> dict[str, Any]:
    if device == "cpu":
        packager.config.lock_clocks = False
        (output / "config.json").write_text(packager.config.model_dump_json())
    command = packager.execute()
    instrument(output)
    prepared = time.monotonic() - start
    timeout = int(deadline - time.monotonic())
    if timeout <= 0:
        raise TimeoutError(
            "E2 session spent its allowance before candidate launch"
        )
    runtime = run_evaluation_runtime(
        packager,
        eval_cmd=command,
        staging_dir=output,
        output_file=output / "trace.jsonl",
        timeout=timeout,
        profile=ProfileMode.NONE,
        expected_trace_count=1,
    )
    result = _session_result(session, output, runtime)
    _record_preparation(result, packager, prepared)
    return result


def _session_result(
    session: Session, output: Path, runtime: EvaluationRuntimeResult
) -> dict[str, Any]:
    (output / "stdout.log").write_text(runtime.stdout)
    (output / "stderr.log").write_text(runtime.stderr)
    result = {
        **session.metadata(),
        "performance_authority": False,
        "candidate_sha256": sha256_bytes(
            candidate_source(session.case.task, session.strategy).encode()
        ),
        "returncode": runtime.returncode,
        "stop_reason": None,
        "original_evaluator_verdict": "not_reached",
    }
    if not isinstance(runtime, EvaluationRuntimeSuccess):
        result["stop_reason"] = runtime.reason.value
    else:
        trace = runtime.traces[0]
        atomic_write_json_value(
            output / "trace.json", trace.model_dump(mode="json")
        )
        if trace.evaluation is None:
            result["stop_reason"] = "evaluation_missing"
        else:
            result["original_evaluator_verdict"] = trace.evaluation.status.value
            if trace.evaluation.status in {
                EvaluationStatus.RUNTIME_ERROR,
                EvaluationStatus.INVALID_REFERENCE,
                EvaluationStatus.TIMEOUT,
            }:
                result["stop_reason"] = (
                    "runtime_or_reference_failure_requires_review"
                )
    observation = output / "judge-observations/observations.json"
    if observation.exists():
        payload = json.loads(observation.read_text())
        result["observations"] = payload
        if payload["stop_reason"]:
            result["stop_reason"] = payload["stop_reason"]
        if payload["pending_correctness"]:
            result["stop_reason"] = "unobserved_correctness_output"
    else:
        result["stop_reason"] = (
            result["stop_reason"] or "worker_evidence_missing"
        )
    client_cost = output / "judge-client-cost.json"
    result["client_cost"] = (
        json.loads(client_cost.read_text()) if client_cost.exists() else None
    )
    return result


def _record_preparation(
    result: dict[str, Any], packager: ProblemPackager, prepared: float
) -> None:
    result["static_review"] = review_solution_sources(
        packager.solution,
        output_dtypes={
            name: dtype_str_to_torch_dtype(spec.dtype)
            for name, spec in packager.definition.outputs.items()
        },
    ).to_dict()
    result["original_checks"] = [
        item.model_dump(mode="json") for item in packager.workloads[0].checks
    ]
    result["prepare_wall_seconds"] = prepared
    result["compile_disposition"] = "not_applicable_frozen_pytorch_sources"
    result["compile_seconds"] = 0.0
