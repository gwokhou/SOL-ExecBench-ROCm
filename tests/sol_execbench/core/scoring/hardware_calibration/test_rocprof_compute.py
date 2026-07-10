from pathlib import Path

from sol_execbench.core.scoring.hardware_calibration.rocprof_compute import (
    ProfilerDiscovery,
    ensure_profiler_environment,
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
