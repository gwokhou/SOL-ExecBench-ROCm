from __future__ import annotations

from collections import Counter

from sol_execbench.core.bench.agent_feedback_items import trace_feedback_items
from sol_execbench.core.data.trace import EvaluationStatus


def test_trace_feedback_items_reports_all_passed_when_no_failure_items() -> None:
    items = trace_feedback_items(Counter({EvaluationStatus.PASSED: 2}))
    payload = [item.model_dump(mode="json") for item in items]

    assert payload == [
        {
            "code": "all_evaluated_traces_passed",
            "severity": "info",
            "bottleneck": "unknown",
            "message": (
                "All evaluated traces passed; no failure-specific diagnostic "
                "is available."
            ),
            "recommendation": (
                "Use optional profiling or static evidence for next-step "
                "performance diagnosis."
            ),
            "source_refs": [
                {
                    "kind": "trace",
                    "label": "canonical_trace_jsonl",
                    "status": None,
                }
            ],
        }
    ]


def test_trace_feedback_items_reports_compile_runtime_timeout_and_policy_failures() -> (
    None
):
    items = trace_feedback_items(
        Counter(
            {
                EvaluationStatus.COMPILE_ERROR: 1,
                EvaluationStatus.RUNTIME_ERROR: 2,
                EvaluationStatus.TIMEOUT: 3,
                EvaluationStatus.REWARD_HACK: 4,
            }
        )
    )

    assert [
        (item.code, item.severity.value, item.bottleneck.value) for item in items
    ] == [
        ("compile_error", "action", "compile_failure"),
        ("reward_hack", "action", "policy_violation"),
        ("runtime_error", "action", "runtime_failure"),
        ("timeout", "action", "timeout"),
    ]
    assert all(item.source_refs[0].label == "canonical_trace_jsonl" for item in items)


def test_trace_feedback_items_reports_correctness_and_reference_failures() -> None:
    items = trace_feedback_items(
        Counter(
            {
                EvaluationStatus.INCORRECT_NUMERICAL: 1,
                EvaluationStatus.INCORRECT_SHAPE: 1,
                EvaluationStatus.INCORRECT_DTYPE: 1,
                EvaluationStatus.INVALID_REFERENCE: 1,
            }
        )
    )

    assert [
        (item.code, item.severity.value, item.bottleneck.value) for item in items
    ] == [
        ("incorrect_dtype", "action", "interface_correctness"),
        ("incorrect_numerical", "action", "numerical_correctness"),
        ("incorrect_shape", "action", "interface_correctness"),
        ("invalid_reference", "warning", "reference_failure"),
    ]
