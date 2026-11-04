# SPDX-License-Identifier: Apache-2.0
"""Profiler setup must neither instrument the launcher nor erase controller files."""

import json
import subprocess
from unittest.mock import Mock

from sol_execbench.cli.evaluation import scope_collection
from sol_execbench.core.bench.rocm_profiler.evaluate_scope_models import (
    SCOPE_ENV,
    SCOPE_INTENT_FILE,
    SCOPE_LAUNCH_FILE,
    SCOPE_TOOL_ENV,
    SCOPE_TRACE_DIRECTORY,
    EvaluateScopeIntent,
)
from sol_execbench.core.bench.rocm_profiler.models import Rocprofv3ProfileResult
from sol_execbench.core.bench.rocm_profiler.profile import (
    prepare_profile_output_directory,
)
from sol_execbench.core.process.environment import ENV_PYTORCH_ALLOC_CONF


def test_collection_preserves_intent_and_profiles_only_candidate(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(ENV_PYTORCH_ALLOC_CONF, "expandable_segments:True")
    staging = tmp_path / "staging"
    staging.mkdir()
    control = staging / SCOPE_INTENT_FILE
    control.write_text("controller-owned")
    intent = Mock(
        spec=EvaluateScopeIntent,
        run_id="a" * 32,
        application_command=("python", "eval_driver.py"),
    )
    tool = tmp_path / "tool.so"
    result = Mock(spec=Rocprofv3ProfileResult)
    calls = []

    def collect(request, *, runner):
        assert request.output_directory == tmp_path / SCOPE_TRACE_DIRECTORY
        prepare_profile_output_directory(
            request.output_directory, request.output_file
        )
        assert control.read_text() == "controller-owned"
        runner(["rocprofv3", "--", *request.application_command], staging, 5)
        return result

    def run(command, *, cwd, timeout, env):
        calls.append((command, env))
        assert command[-1] == "evaluation_orchestrator.py"
        assert env[SCOPE_ENV] == str(tmp_path)
        assert env[ENV_PYTORCH_ALLOC_CONF] == "expandable_segments:False"
        assert SCOPE_TOOL_ENV not in env
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(scope_collection, "collect_rocprofv3_profile", collect)
    monkeypatch.setattr(scope_collection, "run_in_process_group_bounded", run)
    assert (
        scope_collection.collect_evaluate_scope(
            staging=staging,
            directory=tmp_path,
            tool=tool,
            intent=intent,
            timeout=5,
        )
        is result
    )
    launch = json.loads((staging / SCOPE_LAUNCH_FILE).read_text())
    assert launch["command"][-1] == "eval_driver.py"
    assert launch["environment"][SCOPE_TOOL_ENV] == str(tool)
    assert len(calls) == 1
