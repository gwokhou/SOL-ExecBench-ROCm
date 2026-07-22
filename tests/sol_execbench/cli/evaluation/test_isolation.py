from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from types import SimpleNamespace
from typing import cast

import pytest

from sol_execbench.cli.evaluation import phases
from sol_execbench.cli.evaluation.phases import require_execution_isolation
from sol_execbench.cli.evaluation.requests import EvaluationRequest
from sol_execbench.cli.protocol import CliFailure


def test_untrusted_evaluation_requires_container(monkeypatch):
    monkeypatch.delenv("SOL_EXECBENCH_SANDBOXED", raising=False)
    request = cast(EvaluationRequest, SimpleNamespace(unsafe_local_execution=False))

    with pytest.raises(CliFailure, match="requires the hardened container"):
        require_execution_isolation(request)


def test_explicit_unsafe_local_override_is_diagnostic_escape_hatch(monkeypatch):
    monkeypatch.delenv("SOL_EXECBENCH_SANDBOXED", raising=False)
    request = cast(EvaluationRequest, SimpleNamespace(unsafe_local_execution=True))

    require_execution_isolation(request)


def test_container_marker_allows_evaluation(monkeypatch):
    monkeypatch.setenv("SOL_EXECBENCH_SANDBOXED", "1")
    request = cast(EvaluationRequest, SimpleNamespace(unsafe_local_execution=False))

    require_execution_isolation(request)


def test_unsafe_local_marker_is_serialized_with_gpu_boundary(monkeypatch):
    monkeypatch.delenv("SOL_EXECBENCH_SANDBOXED", raising=False)
    monkeypatch.delenv("SOL_EXECBENCH_UNSAFE_LOCAL_EXECUTION", raising=False)
    request = cast(
        EvaluationRequest,
        SimpleNamespace(unsafe_local_execution=True, timeout=5, device="cuda:1"),
    )
    first_inside = threading.Event()
    second_attempted = threading.Event()
    observed: list[str | None] = []

    @contextmanager
    def fake_gpu_lock(**kwargs):
        del kwargs
        if threading.current_thread().name == "second":
            second_attempted.set()
        yield

    monkeypatch.setattr(phases, "acquire_gpu_lock", fake_gpu_lock)

    def evaluate_first():
        with phases.evaluation_execution_boundary(request):
            observed.append(os.environ.get("SOL_EXECBENCH_UNSAFE_LOCAL_EXECUTION"))
            first_inside.set()
            assert second_attempted.wait(timeout=2)

    def evaluate_second():
        with phases.evaluation_execution_boundary(request):
            observed.append(os.environ.get("SOL_EXECBENCH_UNSAFE_LOCAL_EXECUTION"))

    first = threading.Thread(target=evaluate_first, name="first")
    second = threading.Thread(target=evaluate_second, name="second")
    first.start()
    assert first_inside.wait(timeout=2)
    second.start()
    first.join(timeout=2)
    second.join(timeout=2)

    assert observed == ["1", "1"]
    assert "SOL_EXECBENCH_UNSAFE_LOCAL_EXECUTION" not in os.environ
    assert "SOL_EXECBENCH_DEVICE" not in os.environ
