from pathlib import Path

from sol_execbench.core.scoring.hardware_calibration.rocprof_compute import (
    ProfilerDiscovery,
    ensure_profiler_environment,
    parse_roofline_metrics,
    run_rocprof_compute_bench_only,
)


def _discovery(tmp_path: Path) -> ProfilerDiscovery:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("example==1\n")

    def run(command: list[str], **kwargs: object) -> None:
        if command[:2] == ["uv", "venv"]:
            (Path(command[2]) / "bin").mkdir(parents=True)
            (Path(command[2]) / "bin" / "python").touch()

    return ProfilerDiscovery(
        tool_path=Path("/usr/bin/rocprof-compute"),
        tool_version="7.1.1",
        requirements_path=requirements,
        artifact_root=tmp_path / ".artifacts" / "rocprof-compute",
        interpreter_abi="cp312",
        run=run,
    )


def test_managed_invocation_uses_isolated_environment(tmp_path: Path) -> None:
    discovery = _discovery(tmp_path)
    environment = ensure_profiler_environment(
        discovery, offline=False, auto_install=True
    )
    seen: dict[str, object] = {}

    run_rocprof_compute_bench_only(
        environment,
        run=lambda command, *, env: seen.update(command=command, env=env),
    )

    env = seen["env"]
    assert env["VIRTUAL_ENV"] == str(environment.venv_path)
    assert env["PATH"].split(":")[0] == str(environment.venv_path / "bin")
    assert env["PYTHONNOUSERSITE"] == "1"


def test_offline_missing_dependencies_returns_unknown(tmp_path: Path) -> None:
    environment = ensure_profiler_environment(
        _discovery(tmp_path), offline=True, auto_install=True
    )

    assert environment.state == "unknown"
    assert environment.reason_code == "rocprof_compute_dependencies_unavailable_offline"


def test_roofline_parser_marks_missing_and_unrecognised_metrics_unknown() -> None:
    parsed = parse_roofline_metrics(
        "Metric,Value\nUnexpected peak,999\n",
        {"Peak FP32": ("compute.fp32.vector", "TFLOP/s")},
    )

    assert parsed.raw_output_sha256
    assert {candidate.reason_code for candidate in parsed.candidates} == {
        "rocprof_compute_metric_missing",
        "rocprof_compute_metric_unrecognised",
    }
    assert all(
        candidate.state == "unknown" and candidate.value is None
        for candidate in parsed.candidates
    )


def test_reused_environment_rejects_missing_tool(tmp_path: Path) -> None:
    discovery = _discovery(tmp_path)
    environment = ensure_profiler_environment(
        discovery, offline=False, auto_install=True
    )
    unavailable = ProfilerDiscovery(
        **{**discovery.__dict__, "exists": lambda path: path != discovery.tool_path},
    )

    result = ensure_profiler_environment(unavailable, offline=False, auto_install=False)

    assert environment.state == "measured"
    assert result.state == "unknown"
    assert result.reason_code == "rocprof_compute_tool_unavailable"


def test_reused_environment_rejects_nonexecutable_tool(tmp_path: Path) -> None:
    discovery = _discovery(tmp_path)
    ensure_profiler_environment(discovery, offline=False, auto_install=True)
    unavailable = ProfilerDiscovery(
        **{
            **discovery.__dict__,
            "exists": lambda path: (
                True if path == discovery.tool_path else discovery.exists(path)
            ),
            "is_executable": lambda path: False,
        },
    )

    result = ensure_profiler_environment(unavailable, offline=False, auto_install=False)

    assert result.state == "unknown"
    assert result.reason_code == "rocprof_compute_tool_unavailable"
