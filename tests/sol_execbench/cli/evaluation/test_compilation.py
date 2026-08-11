from __future__ import annotations

import subprocess
from pathlib import Path

from sol_execbench.cli.evaluation import compilation
from sol_execbench.core.process.environment import (
    ENV_SOL_EXECBENCH_CONTAINER_IMAGE_ID,
    ENV_SOL_EXECBENCH_NATIVE_COMPILE_CACHE,
    ENV_SOL_EXECBENCH_SOURCE_REVISION,
)


class _Packager:
    def __init__(self, *, is_cpp: bool, artifact_path: Path) -> None:
        self._is_cpp = is_cpp
        self._artifact_path = artifact_path
        self.compile_output_path: Path | None = None

    def compile(self) -> tuple[list[str], str]:
        self.compile_output_path = self._artifact_path
        return ["python", "build_ext.py"], str(self._artifact_path)


class _CachePackager(_Packager):
    def compile(self) -> tuple[list[str], str]:
        self._artifact_path.parent.mkdir(parents=True, exist_ok=True)
        (self._artifact_path.parent / "solution.json").write_text(
            '{"solution":"same"}',
            encoding="utf-8",
        )
        (self._artifact_path.parent / "build_ext.py").write_text(
            "# same build script\n",
            encoding="utf-8",
        )
        return super().compile()


def _env_builder(env):
    return dict(env)


def test_run_compile_phase_skips_non_cpp_solution(tmp_path: Path) -> None:
    called = False

    def _runner(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("runner should not be called")

    packager = _Packager(
        is_cpp=False,
        artifact_path=tmp_path / "benchmark_kernel.so",
    )

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
        command=(),
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

    packager = _Packager(
        is_cpp=True,
        artifact_path=tmp_path / "benchmark_kernel.so",
    )

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
        command=("python", "build_ext.py"),
    )


def test_run_compile_phase_filters_benign_rocm_stderr(tmp_path: Path) -> None:
    def _runner(*args, **kwargs):
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
        _Packager(
            is_cpp=True,
            artifact_path=tmp_path / "benchmark_kernel.so",
        ),
        staging_dir=tmp_path,
        compile_timeout=11,
        env_builder=_env_builder,
        runner=_runner,
    )

    assert result.succeeded is False
    assert result.returncode == 1
    assert "amdgpu.ids" not in result.filtered_stderr
    assert result.filtered_stderr == "real compile error\n"


def test_run_compile_phase_reuses_content_addressed_native_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_root = tmp_path / "compile-cache"
    monkeypatch.setenv(
        ENV_SOL_EXECBENCH_NATIVE_COMPILE_CACHE,
        str(cache_root),
    )
    monkeypatch.setenv(
        ENV_SOL_EXECBENCH_CONTAINER_IMAGE_ID,
        f"sha256:{'a' * 64}",
    )
    monkeypatch.setenv(ENV_SOL_EXECBENCH_SOURCE_REVISION, "b" * 40)
    monkeypatch.setattr(
        compilation,
        "_compiler_provenance",
        lambda: ("/opt/rocm/bin/hipcc", "c" * 64, "HIP 7.2"),
    )
    calls = 0

    def _runner(command, *, cwd, **_kwargs):
        nonlocal calls
        calls += 1
        (cwd / "benchmark_kernel.so").write_bytes(b"compiled artifact")
        (cwd / "kernel.hip.o").write_bytes(b"static object")
        return subprocess.CompletedProcess(command, 0, "compiled\n", "")

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = compilation.run_compile_phase(
        _CachePackager(
            is_cpp=True,
            artifact_path=first_dir / "benchmark_kernel.so",
        ),
        staging_dir=first_dir,
        compile_timeout=11,
        env_builder=_env_builder,
        runner=_runner,
    )
    second = compilation.run_compile_phase(
        _CachePackager(
            is_cpp=True,
            artifact_path=second_dir / "benchmark_kernel.so",
        ),
        staging_dir=second_dir,
        compile_timeout=11,
        env_builder=_env_builder,
        runner=_runner,
    )

    assert first.succeeded is True
    assert second.succeeded is True
    assert calls == 1
    assert second.stdout == (
        "restored native artifact from content-addressed cache\n"
    )
    assert second.artifact_path is not None
    assert second.artifact_path.read_bytes() == b"compiled artifact"
    assert (second_dir / "kernel.hip.o").read_bytes() == b"static object"
