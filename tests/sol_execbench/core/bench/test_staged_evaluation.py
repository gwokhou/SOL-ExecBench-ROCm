from __future__ import annotations

from io import StringIO
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from sol_execbench.core.bench import (
    eval_correctness,
    eval_workload_execution,
    eval_workload_runner,
)
from sol_execbench.core.bench.config import BenchmarkConfig
from sol_execbench.core.bench.eval_timing import SolutionTimingResult
from sol_execbench.core.platform.runtime import CacheClearPolicy
from sol_execbench.core.bench.eval_trace_helpers import WorkloadTraceEmitter
from sol_execbench.core.bench.evaluation_requests import WorkloadEvaluationRequest
from sol_execbench.core.bench.reference_protocol import (
    ReferenceCase,
    ReferenceExecutionError,
    ReferenceFailureKind,
    ReferenceProtocolError,
    ReferenceTimingCase,
)
from sol_execbench.core.bench.reward_hack import RewardHackDetected
from sol_execbench.core.data.trace import Correctness, EvaluationStatus
from sol_execbench_type_helpers import make_workload


class RecordingEmitter:
    def __init__(self) -> None:
        self.events: list[tuple[EvaluationStatus, dict[str, Any]]] = []

    def emit_status(self, workload, status, **kwargs) -> None:
        del workload
        self.events.append((status, kwargs))

    def emit_status_for_workloads(self, workloads, status, **kwargs) -> None:
        for workload in workloads:
            self.emit_status(workload, status, **kwargs)


class ReferenceClientStub:
    def __init__(self, correctness: Any = None, timing: Any = None) -> None:
        self.correctness = correctness
        self.timing = timing
        self.correctness_calls = 0

    def correctness_case(self, **kwargs):
        self.correctness_calls += 1
        if isinstance(self.correctness, BaseException):
            raise self.correctness
        return self.correctness

    def timing_case(self, **kwargs):
        if isinstance(self.timing, BaseException):
            raise self.timing
        return self.timing


def _request(
    client: ReferenceClientStub,
    *,
    check_integrity=lambda snapshot, globals_: None,
    lock_clocks: bool = False,
) -> WorkloadEvaluationRequest:
    definition = SimpleNamespace(
        name="demo",
        get_resolved_axes_values=lambda axes: {},
    )
    dependencies = SimpleNamespace(
        reference_client=client,
        user_fn=lambda value: value,
        integrity_snapshot={},
        check_integrity=check_integrity,
        driver_globals={},
        real_stdout=StringIO(),
    )
    return cast(
        WorkloadEvaluationRequest,
        SimpleNamespace(
            definition=definition,
            workloads=[make_workload(uuid="workload-1", axes={}, inputs={})],
            solution_name="candidate",
            device="cpu",
            output_names=["output"],
            output_dtypes_torch={"output": torch.float32},
            bench_config=BenchmarkConfig(
                warmup_runs=0,
                iterations=1,
                trials=1,
                lock_clocks=lock_clocks,
            ),
            destination_passing_style=False,
            dependencies=dependencies,
        ),
    )


def _emitter() -> tuple[RecordingEmitter, WorkloadTraceEmitter]:
    emitter = RecordingEmitter()
    return emitter, cast(WorkloadTraceEmitter, emitter)


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (ReferenceFailureKind.INPUT_GENERATION, EvaluationStatus.RUNTIME_ERROR),
        (ReferenceFailureKind.REFERENCE_EXECUTION, EvaluationStatus.INVALID_REFERENCE),
    ],
)
def test_correctness_rounds_classify_reference_failures(kind, expected) -> None:
    client = ReferenceClientStub(
        correctness=ReferenceExecutionError("reference failed", kind=kind)
    )
    request = _request(client)
    recording, emitter = _emitter()

    result = eval_correctness.run_correctness_rounds(
        request=request,
        workload=request.workloads[0],
        row_index=0,
        emitter=emitter,
    )

    assert result.failed is True
    assert recording.events[0][0] is expected


def test_correctness_rounds_classify_protocol_failure() -> None:
    request = _request(
        ReferenceClientStub(correctness=ReferenceProtocolError("bad frame"))
    )
    recording, emitter = _emitter()

    eval_correctness.run_correctness_rounds(
        request=request,
        workload=request.workloads[0],
        row_index=0,
        emitter=emitter,
    )

    status, details = recording.events[0]
    assert status is EvaluationStatus.RUNTIME_ERROR
    assert "Trusted reference IPC failed" in details["extra_msg"]


def test_correctness_rounds_classify_user_exception(monkeypatch) -> None:
    tensor = torch.ones(2)
    request = _request(
        ReferenceClientStub(correctness=ReferenceCase([tensor], [tensor]))
    )
    recording, emitter = _emitter()

    def fail(*args, **kwargs):
        raise RuntimeError("candidate failed")

    monkeypatch.setattr(eval_correctness, "call_and_collect_outputs", fail)

    result = eval_correctness.run_correctness_rounds(
        request=request,
        workload=request.workloads[0],
        row_index=0,
        emitter=emitter,
    )

    assert result.failed is True
    assert recording.events[0][0] is EvaluationStatus.RUNTIME_ERROR
    assert "candidate failed" in recording.events[0][1]["extra_msg"]


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (torch.ones(3), EvaluationStatus.INCORRECT_SHAPE),
        (torch.ones(2), EvaluationStatus.INCORRECT_NUMERICAL),
    ],
)
def test_correctness_rounds_reject_bad_candidate_outputs(
    candidate, expected, monkeypatch
) -> None:
    reference = torch.zeros(2)
    request = _request(
        ReferenceClientStub(correctness=ReferenceCase([reference], [reference]))
    )
    recording, emitter = _emitter()
    monkeypatch.setattr(
        eval_correctness,
        "call_and_collect_outputs",
        lambda *args, **kwargs: [candidate],
    )

    result = eval_correctness.run_correctness_rounds(
        request=request,
        workload=request.workloads[0],
        row_index=0,
        emitter=emitter,
    )

    assert result.failed is True
    assert recording.events[0][0] is expected


def test_correctness_rounds_execute_all_ten_cases(monkeypatch) -> None:
    tensor = torch.ones(2)
    client = ReferenceClientStub(correctness=ReferenceCase([tensor], [tensor]))
    request = _request(client)
    recording, emitter = _emitter()
    monkeypatch.setattr(
        eval_correctness, "call_and_collect_outputs", lambda *args, **kwargs: [tensor]
    )

    result = eval_correctness.run_correctness_rounds(
        request=request,
        workload=request.workloads[0],
        row_index=0,
        emitter=emitter,
    )

    assert result.failed is False
    assert result.threads_before is not None
    assert client.correctness_calls == 10
    assert recording.events == []


def test_correctness_thread_baseline_precedes_first_candidate_call(monkeypatch) -> None:
    tensor = torch.ones(2)
    client = ReferenceClientStub(correctness=ReferenceCase([tensor], [tensor]))
    request = _request(client)
    _recording, emitter = _emitter()
    candidate_calls = 0
    call_order: list[str] = []

    def call_candidate(*args, **kwargs):
        del args, kwargs
        nonlocal candidate_calls
        candidate_calls += 1
        call_order.append("candidate")
        return [tensor]

    def active_count():
        call_order.append("baseline")
        return 7

    monkeypatch.setattr(eval_correctness.threading, "active_count", active_count)
    monkeypatch.setattr(eval_correctness, "call_and_collect_outputs", call_candidate)

    result = eval_correctness.run_correctness_rounds(
        request=request,
        workload=request.workloads[0],
        row_index=0,
        emitter=emitter,
    )

    assert candidate_calls == 10
    assert result.threads_before == 7
    assert call_order == ["baseline", *("candidate" for _ in range(10))]


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (
            ReferenceExecutionError(
                "input failed", kind=ReferenceFailureKind.INPUT_GENERATION
            ),
            EvaluationStatus.RUNTIME_ERROR,
        ),
        (
            ReferenceExecutionError(
                "reference failed", kind=ReferenceFailureKind.REFERENCE_EXECUTION
            ),
            EvaluationStatus.INVALID_REFERENCE,
        ),
        (ReferenceProtocolError("bad frame"), EvaluationStatus.RUNTIME_ERROR),
    ],
)
def test_timing_case_classifies_reference_failures(failure, expected) -> None:
    request = _request(ReferenceClientStub(timing=failure))
    recording, emitter = _emitter()

    result = eval_workload_execution._load_timing_case(
        request, emitter, request.workloads[0], 0
    )

    assert result is None
    assert recording.events[0][0] is expected


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RewardHackDetected("timed output changed"), EvaluationStatus.REWARD_HACK),
        (RuntimeError("timer failed"), EvaluationStatus.RUNTIME_ERROR),
    ],
)
def test_measure_and_emit_classifies_timing_errors(
    error, expected, monkeypatch
) -> None:
    tensor = torch.ones(2)
    request = _request(ReferenceClientStub())
    recording, emitter = _emitter()
    monkeypatch.setattr(
        eval_workload_execution,
        "measure_solution_latency",
        lambda **kwargs: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(eval_workload_execution, "_release_device_cache", lambda: None)

    eval_workload_execution._measure_and_emit(
        request,
        emitter,
        request.workloads[0],
        {},
        ReferenceTimingCase([tensor], [tensor], 2.0),
        1,
        Correctness(),
    )

    assert recording.events[0][0] is expected


def test_measure_and_emit_records_validated_timing(monkeypatch) -> None:
    tensor = torch.ones(2)
    request = _request(ReferenceClientStub())
    recording, emitter = _emitter()
    monkeypatch.setattr(
        eval_workload_execution,
        "measure_solution_latency",
        lambda **kwargs: SolutionTimingResult(
            1.25,
            (4,),
            CacheClearPolicy(
                detected_l2_bytes=4 * 1024**2,
                clear_buffer_bytes=8 * 1024**2,
                source="torch_device_properties",
            ),
        ),
    )
    monkeypatch.setattr(eval_workload_execution, "_release_device_cache", lambda: None)
    monkeypatch.setattr(
        eval_workload_execution, "emit_reward_hack_if_detected", lambda **kwargs: False
    )

    eval_workload_execution._measure_and_emit(
        request,
        emitter,
        request.workloads[0],
        {},
        ReferenceTimingCase([tensor], [tensor], 2.0, "reference diagnostic"),
        1,
        Correctness(),
    )

    status, details = recording.events[0]
    assert status is EvaluationStatus.PASSED
    assert details["performance"].latency_ms == 1.25
    assert details["performance"].timed_iterations == 4
    cache_clear = details["performance"].cache_clear
    assert cache_clear is not None
    assert cache_clear.detected_l2_bytes == 4 * 1024**2
    assert cache_clear.clear_buffer_bytes == 8 * 1024**2
    assert details["extra_msg"] == "reference diagnostic"


def test_preflight_rejects_integrity_failure() -> None:
    def fail(snapshot, globals_):
        raise RewardHackDetected("driver patched")

    request = _request(ReferenceClientStub(), check_integrity=fail)
    recording, emitter = _emitter()

    assert eval_workload_runner._preflight_succeeds(request, emitter, False) is False
    assert recording.events[0][0] is EvaluationStatus.REWARD_HACK


def test_preflight_requires_requested_clock_lock() -> None:
    request = _request(ReferenceClientStub(), lock_clocks=True)
    recording, emitter = _emitter()

    assert eval_workload_runner._preflight_succeeds(request, emitter, False) is False
    assert recording.events[0][0] is EvaluationStatus.RUNTIME_ERROR


def test_evaluate_workloads_seeds_and_dispatches_each_row(monkeypatch) -> None:
    request = _request(ReferenceClientStub())
    second = make_workload(uuid="workload-2", axes={}, inputs={})
    request.workloads.append(second)
    observed: dict[str, Any] = {"rows": []}
    monkeypatch.setattr(eval_workload_runner, "are_clocks_locked", lambda: True)
    monkeypatch.setattr(
        eval_workload_runner,
        "set_evaluation_seed",
        lambda seed: observed.update(seed=seed),
    )
    monkeypatch.setattr(
        eval_workload_runner,
        "evaluate_one_workload",
        lambda request, emitter, row_index, workload: observed["rows"].append(
            (row_index, workload.uuid)
        ),
    )

    eval_workload_runner.evaluate_workloads(request)

    assert observed == {
        "seed": request.bench_config.seed,
        "rows": [(0, "workload-1"), (1, "workload-2")],
    }
