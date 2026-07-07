from __future__ import annotations

import subprocess
from pathlib import Path

from sol_execbench.cli import compilation


class _Packager:
    def __init__(self, *, is_cpp: bool) -> None:
        self._is_cpp = is_cpp
        self.compile_output_path: Path | None = None

    def _make_compile_cmd(self, output_path: Path) -> list[str]:
        self.compile_output_path = output_path
        return ["python", "build_ext.py"]


def _env_builder(env):
    return dict(env)


def test_run_compile_phase_skips_non_cpp_solution(tmp_path: Path) -> None:
    called = False

    def _runner(*args, **kwargs):  # noqa: ARG001
        nonlocal called
        called = True
        raise AssertionError("runner should not be called")

    packager = _Packager(is_cpp=False)

    result = compilation.run_compile_phase(
        packager,
        staging_dir=tmp_path,
        compile_timeout=7,
        env_builder=_env_builder,
        runner=_runner,
    )

    assert result == compilation.CompilePhaseResult(
        attempted=False,
        succeeded=False,
        artifact_path=None,
        stdout="",
        filtered_stderr="",
        returncode=0,
    )
    assert called is False
    assert packager.compile_output_path is None


def test_run_compile_phase_executes_compile_command_for_cpp_solution(
    tmp_path: Path,
) -> None:
    calls = []

    def _runner(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="compiled\n",
            stderr="",
        )

    packager = _Packager(is_cpp=True)

    result = compilation.run_compile_phase(
        packager,
        staging_dir=tmp_path,
        compile_timeout=11,
        env_builder=_env_builder,
        runner=_runner,
    )

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (["python", "build_ext.py"],)
    assert kwargs["cwd"] == tmp_path
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["timeout"] == 11
    assert kwargs["env"]["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
    assert packager.compile_output_path == tmp_path / "benchmark_kernel.so"
    assert result == compilation.CompilePhaseResult(
        attempted=True,
        succeeded=True,
        artifact_path=tmp_path / "benchmark_kernel.so",
        stdout="compiled\n",
        filtered_stderr="",
        returncode=0,
    )


def test_run_compile_phase_filters_benign_rocm_stderr(tmp_path: Path) -> None:
    def _runner(*args, **kwargs):  # noqa: ARG001
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout="",
            stderr=(
                "/opt/amdgpu/share/libdrm/amdgpu.ids: No such file or directory\n"
                "real compile error\n"
            ),
        )

    result = compilation.run_compile_phase(
        _Packager(is_cpp=True),
        staging_dir=tmp_path,
        compile_timeout=11,
        env_builder=_env_builder,
        runner=_runner,
    )

    assert result.succeeded is False
    assert result.returncode == 1
    assert "amdgpu.ids" not in result.filtered_stderr
    assert result.filtered_stderr == "real compile error\n"
