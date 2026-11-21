# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""P0/P1 coverage for all-round structure checks and input cloning in eval_correctness."""

from __future__ import annotations

import json
from io import StringIO
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from sol_execbench_type_helpers import make_definition, make_workload

from sol_execbench.core.bench.config import BenchmarkConfig
from sol_execbench.core.bench.eval_correctness import run_correctness_rounds
from sol_execbench.core.bench.eval_trace_helpers import WorkloadTraceEmitter
from sol_execbench.core.bench.evaluation_requests import (
    EvaluationDependencies,
    WorkloadEvaluationRequest,
)
from sol_execbench.core.bench.reference_protocol import (
    ReferenceCase,
    ReferenceClient,
)
from sol_execbench.core.bench.reward_hack import snapshot_critical_functions
from sol_execbench.core.data.dtypes import dtype_str_to_torch_dtype
from sol_execbench.core.data.trace import EvaluationStatus


def _run_rounds_with_fn(definition, workload, user_fn, make_case):
    stream = StringIO()
    emitter = WorkloadTraceEmitter(
        definition=definition,
        solution_name="sol",
        device="cpu",
        clock_status_msg=None,
        clocks_locked=False,
        timing_protocol="events_v4",
        real_stdout=stream,
    )
    ref_client = cast(
        ReferenceClient,
        SimpleNamespace(
            correctness_case=lambda *, round_index, **_kw: make_case(
                round_index
            ),
        ),
    )
    output_dtypes = {
        k: dtype_str_to_torch_dtype(v.dtype)
        for k, v in definition.outputs.items()
    }
    request = WorkloadEvaluationRequest(
        definition=definition,
        workloads=[workload],
        solution_name="sol",
        device="cpu",
        output_names=list(definition.outputs),
        output_dtypes_torch=output_dtypes,
        bench_config=BenchmarkConfig(lock_clocks=False),
        destination_passing_style=False,
        dependencies=EvaluationDependencies(
            reference_client=ref_client,
            user_fn=user_fn,
            integrity_snapshot=snapshot_critical_functions({}, []),
            check_integrity=lambda *_a: None,
            driver_globals={},
            real_stdout=stream,
        ),
    )
    result = run_correctness_rounds(
        request=request,
        workload=workload,
        row_index=0,
        emitter=emitter,
    )
    emitted: list[tuple[EvaluationStatus, str | None]] = []
    for line in stream.getvalue().splitlines():
        if line.strip():
            payload = json.loads(line)
            ev = payload["evaluation"]
            emitted.append((EvaluationStatus(ev["status"]), ev.get("log")))
    return result, emitted


def test_lazy_tensor_subclass_on_later_round_is_rejected() -> None:
    class _LazyRoundFive(torch.Tensor):
        pass

    definition = make_definition(
        name="vecadd",
        op_type="elementwise",
        axes={"n": {"type": "const", "value": 4}},
        inputs={"x": {"shape": ["n"], "dtype": "float32"}},
        outputs={"z": {"shape": ["n"], "dtype": "float32"}},
        reference="def run(x): return x + 1\n",
    )
    workload = make_workload(
        uuid="wkl-round5",
        axes={},
        inputs={"x": {"type": "random"}},
        checks=[{"type": "numeric", "output": "z"}],
    )
    calls = {"n": 0}

    def tricky(x: torch.Tensor) -> torch.Tensor:
        idx = calls["n"]
        calls["n"] += 1
        out = x + 1
        return out.as_subclass(_LazyRoundFive) if idx == 5 else out

    result, emitted = _run_rounds_with_fn(
        definition,
        workload,
        tricky,
        lambda r: ReferenceCase(
            inputs=[torch.full((4,), float(r))],
            outputs=[torch.full((4,), float(r + 1))],
        ),
    )
    assert result.failed is True
    assert calls["n"] == 6
    assert emitted[-1][0] is EvaluationStatus.REWARD_HACK
    assert "_LazyRoundFive" in (emitted[-1][1] or "")


@pytest.mark.parametrize(
    ("bad_fn", "expected_status"),
    [
        (
            lambda x, n: x[:2] if n == 4 else x + 1,
            EvaluationStatus.INCORRECT_SHAPE,
        ),
        (
            lambda x, n: (x + 1).half() if n == 4 else x + 1,
            EvaluationStatus.INCORRECT_DTYPE,
        ),
    ],
)
def test_shape_or_dtype_mismatch_on_later_round_is_classified_cleanly(
    bad_fn,
    expected_status: EvaluationStatus,
) -> None:
    definition = make_definition(
        name="vecadd",
        op_type="elementwise",
        axes={"n": {"type": "const", "value": 4}},
        inputs={"x": {"shape": ["n"], "dtype": "float32"}},
        outputs={"z": {"shape": ["n"], "dtype": "float32"}},
        reference="def run(x): return x + 1\n",
    )
    workload = make_workload(
        uuid="wkl-shape",
        axes={},
        inputs={"x": {"type": "random"}},
        checks=[{"type": "numeric", "output": "z"}],
    )
    calls = {"n": 0}

    def wrapper(x: torch.Tensor) -> torch.Tensor:
        idx = calls["n"]
        calls["n"] += 1
        return bad_fn(x, idx)

    result, emitted = _run_rounds_with_fn(
        definition,
        workload,
        wrapper,
        lambda r: ReferenceCase(
            inputs=[torch.full((4,), float(r))],
            outputs=[torch.full((4,), float(r + 1))],
        ),
    )
    assert result.failed is True
    assert emitted[-1][0] is expected_status


def test_inplace_gating_zeroing_cannot_force_ties_in_topk_routing() -> None:
    definition = make_definition(
        name="routing",
        op_type="routing",
        axes={
            "T": {"type": "const", "value": 1},
            "E": {"type": "const", "value": 4},
            "K": {"type": "const", "value": 2},
        },
        inputs={"gating": {"shape": ["T", "E"], "dtype": "float32"}},
        outputs={
            "weights": {"shape": ["T", "K"], "dtype": "float32"},
            "ids": {"shape": ["T", "K"], "dtype": "int32"},
        },
        reference="def run(gating): pass\n",
    )
    workload = make_workload(
        uuid="wkl-topk-inplace",
        axes={},
        inputs={"gating": {"type": "random"}},
        checks=[
            {
                "type": "topk_routing",
                "ids_output": "ids",
                "weights_output": "weights",
                "gating_input": "gating",
                "topk": 2,
                "tie_atol": 1e-4,
                "weight_atol": 1e-2,
                "max_mismatch_ratio": 0.0,
            }
        ],
    )

    def evil_router(gating: torch.Tensor):
        # Attempt to zero gating in-place so all experts tie at 0.25!
        gating.zero_()
        return (
            torch.tensor([[0.25, 0.25]], dtype=torch.float32),
            torch.tensor([[2, 3]], dtype=torch.int32),
        )

    gating_src = torch.tensor([[10.0, 9.0, -10.0, -10.0]], dtype=torch.float32)
    probs = torch.softmax(gating_src, dim=-1)
    result, emitted = _run_rounds_with_fn(
        definition,
        workload,
        evil_router,
        lambda _r: ReferenceCase(
            inputs=[gating_src.clone()],
            outputs=[
                probs[:, :2].clone(),
                torch.tensor([[0, 1]], dtype=torch.int32),
            ],
        ),
    )
    assert result.failed is True
    assert emitted[-1][0] is EvaluationStatus.INCORRECT_NUMERICAL
