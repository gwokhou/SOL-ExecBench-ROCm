from __future__ import annotations

import json
from pathlib import Path

import click

from sol_execbench.cli import problem_io
from sol_execbench.core import BenchmarkConfig


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))


def _definition_payload() -> dict:
    return {
        "name": "toy_vecadd",
        "axes": {"n": {"type": "const", "value": 1}},
        "inputs": {
            "x": {"shape": ["n"], "dtype": "float32"},
            "y": {"shape": ["n"], "dtype": "float32"},
        },
        "outputs": {"z": {"shape": ["n"], "dtype": "float32"}},
        "reference": "import torch\ndef run(x, y):\n    return x + y\n",
    }


def _workload_payload(uuid: str = "w0") -> dict:
    return {
        "uuid": uuid,
        "axes": {},
        "inputs": {"x": {"type": "random"}, "y": {"type": "random"}},
    }


def _solution_payload() -> dict:
    return {
        "name": "candidate",
        "definition": "toy_vecadd",
        "author": "agent",
        "spec": {
            "languages": ["pytorch"],
            "target_hardware": ["gfx1200"],
            "entry_point": "solution.py::run",
        },
        "sources": [{"path": "solution.py"}],
    }


def test_load_solution_resolves_source_content_relative_to_solution_json(
    tmp_path: Path,
) -> None:
    solution_path = tmp_path / "solution.json"
    (tmp_path / "solution.py").write_text("def run(x, y):\n    return x + y\n")
    _write_json(solution_path, _solution_payload())

    solution = problem_io._load_solution(solution_path)

    assert solution.name == "candidate"
    assert solution.sources[0].content == "def run(x, y):\n    return x + y\n"


def test_load_workloads_skips_blank_lines(tmp_path: Path) -> None:
    workload_path = tmp_path / "workload.jsonl"
    workload_path.write_text(
        json.dumps(_workload_payload("w0"))
        + "\n\n"
        + json.dumps(_workload_payload("w1"))
    )

    workloads = problem_io._load_workloads(workload_path)

    assert [workload.uuid for workload in workloads] == ["w0", "w1"]


def test_load_config_defaults_when_missing() -> None:
    config = problem_io._load_config(None)

    assert isinstance(config, BenchmarkConfig)


def test_load_config_reads_json(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_json(config_path, {"warmup_runs": 3, "iterations": 7})

    config = problem_io._load_config(config_path)

    assert config.warmup_runs == 3
    assert config.iterations == 7


def test_resolve_problem_dir_finds_optional_config_and_solution(tmp_path: Path) -> None:
    problem_dir = tmp_path / "problem"
    problem_dir.mkdir()
    for name in ("definition.json", "workload.jsonl", "config.json", "solution.json"):
        (problem_dir / name).write_text("{}")

    definition, workload, config, solution = problem_io._resolve_problem_dir(
        problem_dir
    )

    assert definition == problem_dir / "definition.json"
    assert workload == problem_dir / "workload.jsonl"
    assert config == problem_dir / "config.json"
    assert solution == problem_dir / "solution.json"


def test_resolve_problem_dir_rejects_missing_definition(tmp_path: Path) -> None:
    problem_dir = tmp_path / "problem"
    problem_dir.mkdir()
    (problem_dir / "workload.jsonl").write_text("")

    try:
        problem_io._resolve_problem_dir(problem_dir)
    except click.ClickException as exc:
        assert "definition.json not found" in str(exc)
    else:
        raise AssertionError("expected ClickException")


def test_resolve_problem_inputs_uses_problem_dir_defaults(tmp_path: Path) -> None:
    problem_dir = tmp_path / "problem"
    problem_dir.mkdir()
    definition = problem_dir / "definition.json"
    workload = problem_dir / "workload.jsonl"
    config = problem_dir / "config.json"
    solution = problem_dir / "solution.json"
    definition.write_text("{}")
    workload.write_text("")
    config.write_text("{}")
    solution.write_text("{}")

    resolved = problem_io.resolve_problem_inputs(
        problem_dir=problem_dir,
        definition_file=None,
        workload_file=None,
        solution_file=None,
        config_file=None,
    )

    assert resolved.definition_file == definition
    assert resolved.workload_file == workload
    assert resolved.solution_file == solution
    assert resolved.config_file == config


def test_resolve_problem_inputs_allows_explicit_definition_when_problem_dir_default_missing(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "problem"
    problem_dir.mkdir()
    workload = problem_dir / "workload.jsonl"
    solution = tmp_path / "solution.json"
    explicit_definition = tmp_path / "definition.json"
    workload.write_text("")
    solution.write_text("{}")
    explicit_definition.write_text("{}")

    resolved = problem_io.resolve_problem_inputs(
        problem_dir=problem_dir,
        definition_file=explicit_definition,
        workload_file=None,
        solution_file=solution,
        config_file=None,
    )

    assert resolved.definition_file == explicit_definition
    assert resolved.workload_file == workload
    assert resolved.solution_file == solution
    assert resolved.config_file is None


def test_resolve_problem_inputs_allows_explicit_workload_when_problem_dir_default_missing(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "problem"
    problem_dir.mkdir()
    definition = problem_dir / "definition.json"
    solution = tmp_path / "solution.json"
    explicit_workload = tmp_path / "workload.jsonl"
    definition.write_text("{}")
    solution.write_text("{}")
    explicit_workload.write_text("")

    resolved = problem_io.resolve_problem_inputs(
        problem_dir=problem_dir,
        definition_file=None,
        workload_file=explicit_workload,
        solution_file=solution,
        config_file=None,
    )

    assert resolved.definition_file == definition
    assert resolved.workload_file == explicit_workload
    assert resolved.solution_file == solution
    assert resolved.config_file is None


def test_resolve_problem_inputs_rejects_missing_solution() -> None:
    try:
        problem_io.resolve_problem_inputs(
            problem_dir=None,
            definition_file=Path("definition.json"),
            workload_file=Path("workload.jsonl"),
            solution_file=None,
            config_file=None,
        )
    except click.ClickException as exc:
        assert "Provide PROBLEM_DIR with solution.json or --solution" in str(exc)
    else:
        raise AssertionError("expected ClickException")
