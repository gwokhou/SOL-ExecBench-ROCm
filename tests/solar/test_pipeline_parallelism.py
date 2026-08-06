from __future__ import annotations

import fcntl
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from solar.contracts import (
    AnalysisExecutionPolicy,
    AnalysisRequest,
    SolarStage,
)
from solar.pipeline import analysis as pipeline
from solar.rocm.architecture import ArchitectureProfile


def test_device_stages_serialize_but_formal_analysis_overlaps(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first_device_entered = threading.Event()
    release_first_device = threading.Event()
    second_device_entered = threading.Event()
    formal_barrier = threading.Barrier(2)

    def extract(request, _staging):
        if request.name == "first":
            first_device_entered.set()
            assert release_first_device.wait(timeout=2)
        else:
            second_device_entered.set()
        return object()

    monkeypatch.setattr(pipeline, "extract_request_graph", extract)
    monkeypatch.setattr(
        pipeline,
        "convert_request_graph",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        pipeline,
        "verify_request_graph",
        lambda *_args: None,
    )

    def analyze(*_args):
        formal_barrier.wait(timeout=2)
        return {"status": "complete"}

    monkeypatch.setattr(pipeline, "analyze_request_graph", analyze)
    policy = AnalysisExecutionPolicy(
        device_stage_lock_path=tmp_path / "device.lock",
        device_stage_lock_timeout_seconds=2,
    )
    requests = tuple(
        cast(
            AnalysisRequest,
            SimpleNamespace(name=name, execution_policy=policy),
        )
        for name in ("first", "second")
    )
    profile = cast(ArchitectureProfile, SimpleNamespace())

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            pipeline.run_pipeline,
            requests[0],
            profile,
            tmp_path / "first",
        )
        assert first_device_entered.wait(timeout=2)
        second = executor.submit(
            pipeline.run_pipeline,
            requests[1],
            profile,
            tmp_path / "second",
        )
        assert not second_device_entered.wait(timeout=0.1)
        release_first_device.set()
        assert first.result(timeout=2).analysis == {"status": "complete"}
        assert second.result(timeout=2).analysis == {"status": "complete"}

    assert second_device_entered.is_set()


def test_device_stage_lock_wait_is_bounded(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "device.lock"
    policy = AnalysisExecutionPolicy(
        device_stage_lock_path=lock_path,
        device_stage_lock_timeout_seconds=0.01,
    )
    request = cast(
        AnalysisRequest,
        SimpleNamespace(execution_policy=policy),
    )
    profile = cast(ArchitectureProfile, SimpleNamespace())

    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(pipeline.PipelineStageError) as captured:
            pipeline.run_pipeline(
                request,
                profile,
                tmp_path / "staging",
            )

    assert captured.value.stage is SolarStage.GRAPH_EXTRACTION
    assert isinstance(captured.value.error, TimeoutError)


def test_device_cleanup_runs_before_cpu_analysis(
    tmp_path: Path,
    monkeypatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        pipeline,
        "extract_request_graph",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        pipeline,
        "convert_request_graph",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        pipeline,
        "verify_request_graph",
        lambda *_args: events.append("verify"),
    )
    monkeypatch.setattr(
        pipeline,
        "analyze_request_graph",
        lambda *_args: events.append("analyze") or {},
    )
    request = cast(
        AnalysisRequest,
        SimpleNamespace(
            execution_policy=AnalysisExecutionPolicy(
                device_stage_cleanup=lambda: events.append("cleanup"),
            ),
        ),
    )

    pipeline.run_pipeline(
        request,
        cast(ArchitectureProfile, SimpleNamespace()),
        tmp_path,
    )

    assert events == ["verify", "cleanup", "analyze"]


def test_execution_policy_requires_positive_lock_timeout() -> None:
    with pytest.raises(ValueError, match="timeout must be positive"):
        AnalysisExecutionPolicy(device_stage_lock_timeout_seconds=0)
