from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from sol_execbench.cli.main import cli


@pytest.mark.parametrize("profile", ["rocprofv3", "rocprofv3-counters"])
def test_evaluate_profile_choice_accepts_wire_value(profile: str) -> None:
    result = CliRunner().invoke(
        cli,
        ["--format", "json", "evaluate", "--profile", profile],
    )

    assert result.exit_code == 2
    assert json.loads(result.output)["error"]["code"] == "incomplete_input_set"
