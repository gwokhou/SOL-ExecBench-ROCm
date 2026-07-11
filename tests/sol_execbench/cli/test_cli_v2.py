from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from sol_execbench.cli.main import cli
from sol_execbench.cli.protocol import CliResult, artifact


def test_root_help_and_version_are_stable() -> None:
    runner = CliRunner()
    no_args = runner.invoke(cli, [])
    version = runner.invoke(cli, ["--version"])

    assert no_args.exit_code == 0
    assert {
        "evaluate",
        "environment",
        "contract",
        "toolchain",
        "dataset",
        "baseline",
        "hardware",
        "score",
    }.issubset(set(no_args.output.split()))
    assert version.output == "sol-execbench, version 2.0.0\n"


def test_unknown_command_has_suggestion_and_json_error() -> None:
    result = CliRunner().invoke(cli, ["--format", "json", "evalute"])

    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["schema_version"] == "sol_execbench.cli_response.v1"
    assert payload["error"]["code"] == "usage_error"
    assert "Did you mean 'evaluate'" in payload["error"]["message"]
    assert result.stderr == ""


def test_evaluate_json_requires_separate_trace_artifact(tmp_path: Path) -> None:
    definition = tmp_path / "definition.json"
    workload = tmp_path / "workload.jsonl"
    solution = tmp_path / "solution.json"
    for path in (definition, workload, solution):
        path.write_text("{}\n")

    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "evaluate",
            "--definition",
            str(definition),
            "--workload",
            str(workload),
            "--solution",
            str(solution),
        ],
    )

    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "missing_trace_output"


def test_evaluate_json_returns_summary_and_artifact(
    monkeypatch, tmp_path: Path
) -> None:
    problem = tmp_path / "problem"
    problem.mkdir()
    solution = problem / "solution.json"
    solution.write_text("{}\n")
    trace = tmp_path / "trace.jsonl"

    def fake_run(**kwargs):
        kwargs["output_file"].write_text('{"canonical":"trace"}\n')
        return CliResult(
            data={"workloads": 1, "passed": 1, "all_passed": True},
            artifacts=(artifact(kwargs["output_file"], "canonical_trace_jsonl"),),
        )

    monkeypatch.setattr(
        "sol_execbench.cli.commands.evaluate.run_evaluation_cli", fake_run
    )
    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "evaluate",
            str(problem),
            "--solution",
            str(solution),
            "--trace-output",
            str(trace),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["command"] == "evaluate"
    assert payload["artifacts"] == [
        {"path": str(trace), "type": "canonical_trace_jsonl"}
    ]
    assert trace.read_text() == '{"canonical":"trace"}\n'


def test_evaluate_input_relationships_fail_before_execution(tmp_path: Path) -> None:
    problem = tmp_path / "problem"
    problem.mkdir()
    definition = tmp_path / "definition.json"
    definition.write_text("{}")
    solution = tmp_path / "solution.json"
    solution.write_text("{}")

    conflict = CliRunner().invoke(
        cli,
        [
            "evaluate",
            str(problem),
            "--definition",
            str(definition),
            "--solution",
            str(solution),
        ],
    )
    incomplete = CliRunner().invoke(
        cli, ["evaluate", "--definition", str(definition), "--solution", str(solution)]
    )
    invalid_timeout = CliRunner().invoke(
        cli, ["evaluate", str(problem), "--solution", str(solution), "--timeout", "0"]
    )

    assert conflict.exit_code == 2
    assert incomplete.exit_code == 2
    assert invalid_timeout.exit_code == 2


def test_legacy_entry_shapes_are_removed() -> None:
    runner = CliRunner()
    assert runner.invoke(cli, ["doctor", "--json"]).exit_code == 2
    assert runner.invoke(cli, ["baseline", "release-build"]).exit_code == 2
    assert runner.invoke(cli, ["official-score"]).exit_code == 2
