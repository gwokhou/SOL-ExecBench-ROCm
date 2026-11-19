"""CPU evidence for E2 instrumentation; no holdout key or input materialization."""

import base64
import json
import os
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save

from scripts.internal.solar import e2_campaign
from scripts.internal.solar.e2_contracts import (
    OBSERVATION_OPERATION,
    PRIVATE_FILE,
    Activity,
    Mode,
    Session,
    schedule,
)
from scripts.internal.solar.e2_postprocess import diagnose_incumbent
from scripts.internal.solar.e2_runner import _session_result
from scripts.internal.solar.e2_service import ObservationService
from scripts.internal.solar.e2_staging import (
    LayerNormReference,
    instrument_drivers,
    package,
)
from sol_execbench.cli.evaluation.runtime import EvaluationRuntimeSuccess
from sol_execbench.core.bench.judge.candidate_catalog import candidate_callable
from sol_execbench.core.bench.judge.contracts import (
    JudgeCase,
    JudgeFamily,
    JudgeInstance,
    JudgeStrategy,
    JudgeTask,
)
from sol_execbench.core.bench.reference_service import ReferenceRequestError
from sol_execbench.core.data.trace import EvaluationStatus
from sol_execbench.core.process import run_in_process_group_to_files
from tests.sol_execbench_type_helpers import make_trace, make_workload

ROOT = Path(__file__).resolve().parents[4]


def _session(
    task=JudgeTask.SUM, family=JudgeFamily.R0, strategy=JudgeStrategy.REFERENCE
):
    return Session(
        case=JudgeCase(
            task=task,
            family=family,
            workload_index=0,
            instance=JudgeInstance.DEVELOPMENT_0,
        ),
        strategy=strategy,
        repeat=0,
        mode=Mode.DIAGNOSTIC,
        activity=Activity.CORRECTNESS,
    )


def test_frozen_denominators_and_cost_order_without_materialization():
    sessions = schedule()
    assert len({session.identity for session in sessions}) == 1440
    assert Counter(session.activity for session in sessions) == {
        Activity.CORRECTNESS: 1296,
        Activity.COST: 144,
    }
    correctness = [s for s in sessions if s.activity is Activity.CORRECTNESS]
    units = Counter((s.case.logical_id, s.strategy) for s in correctness)
    assert len(units) == 432
    assert set(units.values()) == {3}
    assert (
        sum(s.case.instance is JudgeInstance.HOLDOUT for s in correctness)
        == 432
    )
    costs = [s for s in sessions if s.activity is Activity.COST]
    for start in range(0, len(costs), 6):
        assert [s.mode for s in costs[start : start + 6]] == [
            Mode.INCUMBENT,
            Mode.DIAGNOSTIC,
            Mode.DIAGNOSTIC,
            Mode.INCUMBENT,
            Mode.INCUMBENT,
            Mode.DIAGNOSTIC,
        ]


@pytest.mark.parametrize(
    ("task", "family"),
    [
        pytest.param(JudgeTask.GEMM, JudgeFamily.G2, id="gemm-semantic-anchor"),
        pytest.param(JudgeTask.SUM, JudgeFamily.R0, id="sum-semantic-tail"),
        pytest.param(
            JudgeTask.LAYERNORM, JudgeFamily.L3, id="layernorm-affine"
        ),
    ],
)
def test_private_service_retains_rejected_actual_output_once(
    tmp_path, task, family
):
    session = _session(task, family, JudgeStrategy.SEMANTIC)
    with package(
        ROOT,
        tmp_path,
        session,
        workload_uuid="cpu-observer-case",
        private_seed=1701,
    ):
        service = ObservationService(
            tmp_path, device="cpu", input_nonce="e2" * 32
        )
        assert not (tmp_path / PRIVATE_FILE).exists()
        request = {
            "operation": "correctness",
            "row_index": 0,
            "round_index": 0,
            "workload_uuid": "cpu-observer-case",
        }
        _, reference, _, _ = service.handle(request)
        actual = candidate_callable(task, JudgeStrategy.SEMANTIC)(
            *reference.inputs
        )
        observation = {
            "operation": OBSERVATION_OPERATION,
            "row_index": 0,
            "round_index": 0,
            "checks_seconds": 0.01,
            "numeric_passed": False,
            "payload": base64.b64encode(
                save({"output": actual.contiguous()})
            ).decode(),
        }
        _, response, _, _ = service.handle(observation)
        assert response.inputs == []
        assert response.outputs == []
        assert service.reference_calls == 1
        assert service.rows[0]["original_numeric_passed"] is False
        assert service.rows[0]["witness_certified"] is True
        assert not (tmp_path / "judge-observations").exists()
        with pytest.raises(ReferenceRequestError, match="identity mismatch"):
            service.handle(observation)
        service.close()
        assert len(list((tmp_path / "judge-observations").glob("*.npz"))) == 1


@pytest.mark.parametrize(
    ("mode", "activity", "layernorm_reference"),
    [
        *(
            pytest.param(
                mode,
                activity,
                LayerNormReference.INCUMBENT,
                id=f"{mode}-{activity}",
            )
            for mode in Mode
            for activity in Activity
        ),
        pytest.param(
            Mode.DIAGNOSTIC,
            Activity.CORRECTNESS,
            LayerNormReference.CENTERED_FP64_V1,
            id="layernorm-centered-fp64",
        ),
    ],
)
def test_cpu_subprocess_uses_original_gate_and_exact_call_counts(
    tmp_path, mode, activity, layernorm_reference
):
    case_session = (
        _session(JudgeTask.LAYERNORM, JudgeFamily.L1)
        if layernorm_reference is LayerNormReference.CENTERED_FP64_V1
        else _session()
    )
    session = replace(case_session, mode=mode, activity=activity)
    with package(
        ROOT,
        tmp_path,
        session,
        workload_uuid="cpu-subprocess",
        private_seed=1701,
        layernorm_reference=layernorm_reference,
    ) as packager:
        packager.config.lock_clocks = False
        (tmp_path / "config.json").write_text(packager.config.model_dump_json())
        command = packager.execute()
        instrument_drivers(tmp_path)
        result = run_in_process_group_to_files(
            command,
            tmp_path / "stdout",
            tmp_path / "stderr",
            cwd=tmp_path,
            timeout=120,
            env={
                **os.environ,
                "SOL_EXECBENCH_DEVICE": "cpu",
                "SOL_EXECBENCH_ALLOW_CPU_TIMING": "1",
                "ROCR_VISIBLE_DEVICES": "",
                "HIP_VISIBLE_DEVICES": "",
                "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}",
            },
        )
        assert result.returncode == 0, (tmp_path / "stderr").read_text()
        traces = packager.convert_stdout_to_traces(
            (tmp_path / "stdout").read_text()
        )
        assert len(traces) == 1
        assert traces[0].evaluation is not None
        assert traces[0].evaluation.status.value == "PASSED"
        payload = json.loads(
            (tmp_path / "judge-observations/observations.json").read_text()
        )
        assert len(payload["observations"]) == (
            190 if activity is Activity.COST else 11
        )
        assert payload["reference_calls"] == (
            371 if activity is Activity.COST else 12
        )
        assert payload["stop_reason"] is None
        assert payload["pending_correctness"] is False
        assert not (tmp_path / PRIVATE_FILE).exists()
        if mode is Mode.DIAGNOSTIC:
            assert all(
                row["original_numeric_passed"]
                for row in payload["observations"]
            )
        else:
            assert all(
                row["diagnostic_status"] == "deferred_incumbent_mode"
                for row in payload["observations"]
            )
            offline = diagnose_incumbent(ROOT, tmp_path, session)
            assert offline["stop_reason"] is None
            assert len(offline["observations"]) == len(payload["observations"])
            assert all(
                row["original_numeric_passed"]
                for row in offline["observations"]
            )
    assert not torch.cuda.is_initialized()


def test_activity_stop_retains_every_unreached_denominator(
    tmp_path, monkeypatch
):
    rows = [
        {**session.metadata(), "state": "not_reached", "verdict": "not_reached"}
        for session in schedule()
    ]
    attempts = []

    def stopped_session(*args, **kwargs):
        attempts.append(args[2].identity)
        return {
            "original_evaluator_verdict": "not_reached",
            "stop_reason": "oracle_uncertain",
        }

    monkeypatch.setattr(e2_campaign, "run_session", stopped_session)
    reason, seconds = e2_campaign._run_schedule(
        ROOT, tmp_path, b"", float("inf"), rows
    )
    assert reason == "oracle_uncertain"
    assert seconds == 0
    assert len(attempts) == 1
    assert rows[0]["state"] == "incomplete"
    assert sum(row["state"] == "not_reached" for row in rows) == 1439
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert ledger["status"] == "INCOMPLETE"
    assert ledger["logical_units"] == 432
    assert ledger["correctness_sessions"] == 1296
    assert ledger["cost_sessions"] == 144


@pytest.mark.parametrize(
    ("status", "stops"),
    [
        pytest.param(
            EvaluationStatus.RUNTIME_ERROR, True, id="runtime-error-stops"
        ),
        pytest.param(
            EvaluationStatus.INVALID_REFERENCE, True, id="reference-error-stops"
        ),
        pytest.param(EvaluationStatus.TIMEOUT, True, id="trace-timeout-stops"),
        pytest.param(
            EvaluationStatus.REWARD_HACK, False, id="policy-rejection-retained"
        ),
        pytest.param(
            EvaluationStatus.INCORRECT_NUMERICAL,
            False,
            id="numeric-rejection-retained",
        ),
    ],
)
def test_zero_exit_trace_failures_stop_activity_without_dropping_verdict(
    tmp_path, status, stops
):
    observation_dir = tmp_path / "judge-observations"
    observation_dir.mkdir()
    (observation_dir / "observations.json").write_text(
        json.dumps(
            {
                "observations": [],
                "stop_reason": None,
                "pending_correctness": False,
            }
        )
    )
    trace = make_trace(
        definition="diagnostic-fixture",
        solution="diagnostic-fixture",
        workload=make_workload(uuid="fixture", axes={}, inputs={}),
        evaluation={
            "status": status,
            "environment": {"hardware": "cpu"},
            "timestamp": "2026-11-08T00:00:00Z",
            "correctness": {}
            if status is EvaluationStatus.INCORRECT_NUMERICAL
            else None,
        },
    )
    runtime = EvaluationRuntimeSuccess(
        traces=[trace],
        stdout=trace.model_dump_json(),
        stderr="",
        filtered_stderr="",
        returncode=0,
        profile_result=None,
    )
    result = _session_result(_session(), tmp_path, runtime)
    assert result["original_evaluator_verdict"] == status.value
    assert (result["stop_reason"] is not None) is stops
