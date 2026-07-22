from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import yaml

from sol_execbench.cli.evaluation import evaluator
from sol_execbench.cli.evaluation import problem_io
from sol_execbench.cli.evaluation.requests import EvaluationRequest
from sol_execbench.core.bench.config import BenchmarkConfig
from sol_execbench.cli.protocol import CliFailure


@pytest.mark.parametrize("keep_staging", [False, True])
def test_staging_scope_handles_failure_before_packager_construction(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    keep_staging: bool,
) -> None:
    staging_dir = tmp_path / "staging"

    def make_staging_dir(*, prefix: str) -> str:
        assert prefix == "sol_execbench_"
        staging_dir.mkdir()
        return str(staging_dir)

    @contextmanager
    def fail_execution_boundary(_request: EvaluationRequest) -> Iterator[None]:
        raise RuntimeError("GPU lock unavailable")
        yield

    loaded = SimpleNamespace(
        definition=SimpleNamespace(name="toy"),
        workloads=[],
        solution=SimpleNamespace(name="candidate"),
        config=BenchmarkConfig(),
    )
    monkeypatch.setattr(evaluator.tempfile, "mkdtemp", make_staging_dir)
    monkeypatch.setattr(
        evaluator.cli_problem_io,
        "resolve_problem_inputs",
        lambda **_kwargs: SimpleNamespace(config_file=None),
    )
    monkeypatch.setattr(
        evaluator.cli_problem_io,
        "load_problem_inputs",
        lambda _inputs: loaded,
    )
    monkeypatch.setattr(
        evaluator.cli_phases,
        "evaluation_execution_boundary",
        fail_execution_boundary,
    )

    request = EvaluationRequest(
        problem_dir=tmp_path,
        definition_file=None,
        workload_file=None,
        solution_file=tmp_path / "solution.json",
        config_file=None,
        compile_timeout=30,
        timeout=60,
        output_file=None,
        json_output=False,
        lock_clocks=False,
        keep_staging=keep_staging,
        profile="none",
        static_evidence="none",
        decision="none",
        feedback_target_id=None,
        feedback_run_id=None,
        feedback_candidate_id=None,
        feedback_source_sha256=None,
        feedback_sol_version=None,
        verbose=False,
        unsafe_local_execution=True,
    )

    with pytest.raises(RuntimeError, match="GPU lock unavailable"):
        evaluator.run_evaluation_cli(request=request)

    assert staging_dir.exists() is keep_staging


def test_materialized_problem_rejects_wrong_exact_device_target(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem_dir = tmp_path / "torch2hip" / "problem"
    problem_dir.mkdir(parents=True)
    (tmp_path / "materialization-manifest.yaml").write_text(
        yaml.safe_dump({"target": {"gfx_target": "gfx942"}})
    )
    monkeypatch.setattr(
        problem_io,
        "detect_rocm_device",
        lambda _device: SimpleNamespace(gfx_target="gfx1200"),
    )

    with pytest.raises(CliFailure, match="targets gfx942") as raised:
        problem_io.require_materialized_target_match(problem_dir, "cuda:0")

    assert raised.value.code == "evaluation_target_mismatch"


def test_materialized_problem_accepts_matching_exact_device_target(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem_dir = tmp_path / "torch2hip" / "problem"
    problem_dir.mkdir(parents=True)
    (tmp_path / "materialization-manifest.yaml").write_text(
        yaml.safe_dump({"target": {"gfx_target": "gfx1150"}})
    )
    monkeypatch.setattr(
        problem_io,
        "detect_rocm_device",
        lambda device: SimpleNamespace(gfx_target="gfx1150", device=device),
    )

    problem_io.require_materialized_target_match(problem_dir, "cuda:1")
