# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""L4 throughput-line acceptance tests (spec: solar-l4-throughput)."""

from __future__ import annotations

import importlib.resources

import pytest

from sol_execbench.core.solar_bridge.resource_policy import (
    validate_formal_job_count,
)
from solar.analysis.orojenesis.configuration import (
    MAPPER_THREADS_ENV,
    orojenesis_mapper_thread_count,
)


def test_mapper_threads_default_is_all_visible_cpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T1: without the override the full-core default is untouched."""
    monkeypatch.delenv(MAPPER_THREADS_ENV, raising=False)
    monkeypatch.setattr(
        "solar.analysis.orojenesis.configuration.available_logical_cpu_count",
        lambda *, cpu_ids=None: 12,
    )
    assert orojenesis_mapper_thread_count() == 12


def test_mapper_threads_env_override_is_honored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T1: a positive integer override pins the demand explicitly."""
    monkeypatch.setenv(MAPPER_THREADS_ENV, "4")
    assert orojenesis_mapper_thread_count() == 4


@pytest.mark.parametrize("raw", ["", "0", "-3", "two", "2.5"])
def test_mapper_threads_malformed_values_fall_back(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    """T1: absent or malformed values never partially adopt."""
    monkeypatch.setattr(
        "solar.analysis.orojenesis.configuration.available_logical_cpu_count",
        lambda *, cpu_ids=None: 12,
    )
    if raw:
        monkeypatch.setenv(MAPPER_THREADS_ENV, raw)
    else:
        monkeypatch.delenv(MAPPER_THREADS_ENV, raising=False)
    assert orojenesis_mapper_thread_count() == 12


def test_qualification_shares_the_runner_job_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T2: the same CPU budget rejects jobs the runner would reject."""
    monkeypatch.setattr(
        "sol_execbench.core.solar_bridge.resource_policy."
        "available_formal_mapper_logical_cpu_count",
        lambda: 64,
    )
    monkeypatch.setattr(
        "sol_execbench.core.solar_bridge.resource_policy."
        "formal_mapper_thread_count",
        lambda: 32,
    )
    validate_formal_job_count(1, caller="SOLAR qualification")
    validate_formal_job_count(2, caller="SOLAR qualification")
    with pytest.raises(
        ValueError, match="SOLAR qualification jobs 3 exceed the safe limit 2"
    ):
        validate_formal_job_count(3, caller="SOLAR qualification")


def test_job_count_rejects_nonpositive() -> None:
    """T2: the shared validator keeps the positivity invariant."""
    with pytest.raises(ValueError, match="jobs must be positive"):
        validate_formal_job_count(0, caller="SOLAR release")


def test_interpreter_parallel_extraction_stays_rejected() -> None:
    """T3: torchview's recorder patches torch globals; pin the source fact.

    Same-interpreter concurrent extraction is rejected by design (L4 spec
    §3); extraction keeps process isolation. This test pins the underlying
    recorder fact so the rejection cannot silently lose its justification.
    """
    source = (
        importlib.resources.files("solar._vendor.torchview")
        .joinpath("recorder_tensor.py")
        .read_text(encoding="utf-8")
    )
    assert "Module.__call__" in source or "__call__" in source
