"""Deterministic quota checks; no real processes are killed or GPU queried."""

import threading

import pytest

from sol_execbench.core.data.json_utils import load_json_value
from sol_execbench.core.process import resource_guard


@pytest.mark.parametrize(
    "memory,files,reason",
    [
        (100, 0, "memory_budget_exhausted"),
        (1, 100, "evidence_budget_exhausted"),
    ],
    ids=["rss-quota", "evidence-quota"],
)
def test_quota_retains_cause_before_group_termination(
    tmp_path, monkeypatch, memory, files, reason
):
    budget = resource_guard.WorkerResourceBudget(
        memory_bytes=100, evidence_root=tmp_path, evidence_bytes=100
    )
    monkeypatch.setattr(resource_guard, "process_group_rss", lambda _: memory)
    monkeypatch.setattr(resource_guard, "evidence_size", lambda *_: files)
    killed = []
    monkeypatch.setattr(
        resource_guard.os,
        "killpg",
        lambda group, signal: killed.append((group, signal)),
    )
    failure = tmp_path / "failure.json"
    resource_guard._watch_budget(budget, failure, 123, threading.Event())
    assert load_json_value(failure) == {"reason": reason}
    assert killed == [(123, resource_guard.signal.SIGTERM)]


def test_missing_observation_fails_closed(tmp_path, monkeypatch):
    def denied(_):
        raise PermissionError("synthetic denied observation")

    monkeypatch.setattr(resource_guard, "process_group_rss", denied)
    monkeypatch.setattr(resource_guard.os, "killpg", lambda *_: None)
    budget = resource_guard.WorkerResourceBudget(
        memory_bytes=100, evidence_root=tmp_path, evidence_bytes=100
    )
    failure = tmp_path / "failure.json"
    resource_guard._watch_budget(budget, failure, 123, threading.Event())
    assert load_json_value(failure)["reason"] == "resource_observation_failed"
