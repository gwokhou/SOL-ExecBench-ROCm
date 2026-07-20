from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from sol_execbench.core.bench import eval_timing
from sol_execbench.core.bench.evaluation_requests import WorkloadEvaluationRequest
from sol_execbench.core.bench.eval_runtime import TimingResult
from sol_execbench.core.bench.reward_hack import RewardHackDetected
from sol_execbench.core.data.workload import Workload
from sol_execbench_type_helpers import make_workload


def _request(*, destination_passing_style: bool = False):
    integrity_calls: list[tuple[dict[str, int], dict[str, object]]] = []

    def check_integrity(snapshot, driver_globals) -> None:
        integrity_calls.append((snapshot, driver_globals))

    request = cast(
        WorkloadEvaluationRequest,
        SimpleNamespace(
            definition=object(),
            device="cpu",
            output_names=["output"],
            output_dtypes_torch={"output": torch.float32},
            destination_passing_style=destination_passing_style,
            bench_config=SimpleNamespace(
                warmup_runs=2,
                iterations=3,
                min_measurement_time_seconds=0.25,
            ),
            dependencies=SimpleNamespace(
                user_fn=lambda value: value,
                check_integrity=check_integrity,
                integrity_snapshot={"reference": 1},
                driver_globals={"driver": object()},
            ),
        ),
    )
    return request, integrity_calls


def _workload() -> Workload:
    return make_workload(
        uuid="timing",
        axes={},
        inputs={},
        tolerance={
            "max_atol": 0.01,
            "max_rtol": 0.01,
            "required_matched_ratio": 1.0,
        },
    )


def test_solution_timing_records_actual_iterations_for_each_trial(monkeypatch):
    request = cast(
        WorkloadEvaluationRequest,
        SimpleNamespace(bench_config=SimpleNamespace(trials=3)),
    )
    workload = cast(Workload, object())
    trial_results = iter(
        (
            TimingResult(latency_ms=1.0, timed_iterations=2),
            TimingResult(latency_ms=2.0, timed_iterations=3),
            TimingResult(latency_ms=3.0, timed_iterations=2),
        )
    )
    monkeypatch.setattr(
        eval_timing, "_build_timed_output_validator", lambda **kwargs: lambda *_: None
    )
    monkeypatch.setattr(
        eval_timing, "_measure_solution_trial", lambda *args: next(trial_results)
    )

    result = eval_timing.measure_solution_latency(
        request=request,
        workload=workload,
        resolved_axes={},
        inputs=[],
        expected_outputs=[],
    )

    assert result.latency_ms == pytest.approx(2.0)
    assert result.timed_iterations_per_trial == (2, 3, 2)
    assert result.uniform_timed_iterations == 0


def test_solution_timing_reports_uniform_actual_iteration_count(monkeypatch):
    request = cast(
        WorkloadEvaluationRequest,
        SimpleNamespace(bench_config=SimpleNamespace(trials=2)),
    )
    workload = cast(Workload, object())
    monkeypatch.setattr(
        eval_timing, "_build_timed_output_validator", lambda **kwargs: lambda *_: None
    )
    monkeypatch.setattr(
        eval_timing,
        "_measure_solution_trial",
        lambda *args: TimingResult(latency_ms=1.0, timed_iterations=4),
    )

    result = eval_timing.measure_solution_latency(
        request=request,
        workload=workload,
        resolved_axes={},
        inputs=[],
        expected_outputs=[],
    )

    assert result.timed_iterations_per_trial == (4, 4)
    assert result.uniform_timed_iterations == 4


def test_measure_solution_trial_passes_outputs_and_benchmark_config(monkeypatch):
    request, _ = _request(destination_passing_style=True)
    allocated = [torch.empty(4)]
    observed: dict[str, object] = {}
    monkeypatch.setattr(eval_timing, "allocate_outputs", lambda *_args: allocated)

    def measure(fn, inputs, outputs, device, **kwargs):
        observed.update(
            fn=fn,
            inputs=inputs,
            outputs=outputs,
            device=device,
            **kwargs,
        )
        return TimingResult(latency_ms=1.5, timed_iterations=3)

    monkeypatch.setattr(eval_timing, "measure_latency", measure)

    def validator(*_args) -> None:
        return None

    result = eval_timing._measure_solution_trial(
        request, {"N": 4}, [torch.ones(4)], validator
    )

    assert result.latency_ms == 1.5
    assert observed["outputs"] is allocated
    assert observed["warmup"] == 2
    assert observed["rep"] == 3
    assert observed["min_measurement_time_seconds"] == 0.25
    assert observed["validator"] is validator


def test_measure_solution_trial_maps_timing_failure(monkeypatch):
    request, _ = _request()
    monkeypatch.setattr(
        eval_timing,
        "measure_latency",
        lambda *_args, **_kwargs: TimingResult(
            latency_ms=0.0,
            failure="timing failed",
        ),
    )

    with pytest.raises(RuntimeError, match="timing failed"):
        eval_timing._measure_solution_trial(request, {}, [], lambda *_args: None)


def test_timed_validator_accepts_matching_functional_output() -> None:
    request, integrity_calls = _request()
    expected = [torch.ones(4)]
    validator = eval_timing._build_timed_output_validator(
        request=request,
        workload=_workload(),
        inputs=[torch.zeros(4)],
        expected=expected,
    )

    validator([torch.zeros(4)], torch.ones(4))

    assert integrity_calls == [({"reference": 1}, request.dependencies.driver_globals)]


@pytest.mark.parametrize(
    ("actual", "message"),
    [
        (torch.ones(3), "invalid output shape or dtype"),
        (torch.zeros(4), "differs from the reference"),
    ],
)
def test_timed_validator_rejects_shape_and_numerical_changes(actual, message) -> None:
    request, _ = _request()
    validator = eval_timing._build_timed_output_validator(
        request=request,
        workload=_workload(),
        inputs=[torch.zeros(4)],
        expected=[torch.ones(4)],
    )

    with pytest.raises(RewardHackDetected, match=message):
        validator([torch.zeros(4)], actual)


def test_timed_outputs_reads_destination_passing_buffer() -> None:
    request, _ = _request(destination_passing_style=True)
    input_tensor = torch.zeros(4)
    output_tensor = torch.ones(4)

    result = eval_timing._timed_outputs(
        request,
        [input_tensor],
        [input_tensor, output_tensor],
        None,
    )

    assert len(result) == 1
    assert result[0] is output_tensor
