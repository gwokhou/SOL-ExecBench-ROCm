"""CPU-safe coverage for the focused dual-path SOLAR runner."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def _load_focus(load_script: Any):
    return load_script("scripts/internal/solar/run_cross_path_focus.py")


def _load_corpus(focus):
    return focus.AKACorpusManifest.load(Path("problems/AMD_AKA/manifest.yaml"))


def test_focus_check_validates_exact_current_8_problem_41_workload_inventory(
    load_script: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    focus = _load_focus(load_script)

    assert focus.main(["--check"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["problems"] == 8
    assert payload["workloads"] == 41
    assert payload["path_workloads"] == 82
    assert {item["problem"] for item in payload["selection"]} == set(
        focus.FOCUS_WORKLOAD_COUNTS
    )


def test_focus_inventory_rejects_missing_and_duplicate_workloads(
    load_script: Any,
) -> None:
    focus = _load_focus(load_script)
    corpus = _load_corpus(focus)
    selected = focus._focus_entries(corpus)
    missing = replace(
        corpus,
        entries=tuple(
            entry
            for entry in corpus.entries
            if entry.relative_problem_dir != selected[0].relative_problem_dir
        ),
    )
    with pytest.raises(ValueError, match="lacks focused problems"):
        focus._focus_entries(missing)

    duplicate_entry = replace(
        selected[0],
        workload_uuids=(
            selected[0].workload_uuids[0],
            *selected[0].workload_uuids[:-1],
        ),
    )
    duplicate = replace(
        corpus,
        entries=tuple(
            duplicate_entry if entry is selected[0] else entry
            for entry in corpus.entries
        ),
    )
    with pytest.raises(ValueError, match="repeat UUIDs"):
        focus._focus_entries(duplicate)


def test_focus_runner_executes_fixed_order_resumes_and_compares_last(
    load_script: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    focus = _load_focus(load_script)
    corpus = _load_corpus(focus)
    entries = focus._focus_entries(corpus)
    first = entries[0]
    existing = (
        tmp_path
        / focus.IRPath.MAKE_FX_ATEN.value
        / first.relative_problem_dir
        / first.workload_uuids[0]
    )
    existing.mkdir(parents=True)
    calls = []
    comparison = []

    def fake_worker(request, *, timeout_seconds):
        calls.append((request, timeout_seconds))
        Path(request.output_dir).mkdir(parents=True)
        return SimpleNamespace(is_formal_publication=True)

    def fake_compare(left, right, output):
        comparison.append((left, right, output, len(calls)))

    monkeypatch.setattr(focus, "run_solar_worker", fake_worker)
    monkeypatch.setattr(focus, "compare_solar_ir_paths", fake_compare)

    result = focus.run_focus(
        corpus,
        output_root=tmp_path,
        orojenesis_home=tmp_path / "orojenesis",
        timeout_seconds=123,
        resume=True,
    )

    assert result.generated == 81
    assert result.resumed == 1
    assert len(calls) == 81
    assert {call[0].ir_path for call in calls} == set(focus.IR_PATHS)
    assert all(call[1] == 123 for call in calls)
    assert comparison == [
        (
            tmp_path / focus.IRPath.MAKE_FX_ATEN.value,
            tmp_path / focus.IRPath.TORCHVIEW_EXTENDED_EINSUM.value,
            tmp_path / "path-comparison.json",
            81,
        )
    ]


def test_focus_runner_fails_closed_before_comparison(
    load_script: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    focus = _load_focus(load_script)
    corpus = _load_corpus(focus)
    compared = []

    monkeypatch.setattr(
        focus,
        "run_solar_worker",
        lambda *args, **kwargs: SimpleNamespace(
            is_formal_publication=False,
            stage="verification",
            reason_code="failed",
            message="not formal",
        ),
    )
    monkeypatch.setattr(
        focus,
        "compare_solar_ir_paths",
        lambda *args: compared.append(args),
    )

    with pytest.raises(RuntimeError, match="not formal"):
        focus.run_focus(
            corpus,
            output_root=tmp_path,
            orojenesis_home=tmp_path / "orojenesis",
        )
    assert compared == []
