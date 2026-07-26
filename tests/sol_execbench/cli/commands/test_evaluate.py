from __future__ import annotations

import json

from click.testing import CliRunner

from sol_execbench.cli.main import cli


def test_evaluate_profile_choice_accepts_wire_value() -> None:
    result = CliRunner().invoke(
        cli,
        ["--format", "json", "evaluate", "--profile", "rocprofv3"],
    )

    assert result.exit_code == 2
    assert json.loads(result.output)["error"]["code"] == "incomplete_input_set"
