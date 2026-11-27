# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Tests for candidate staging isolation, -P safe-path execution, and symbol guards."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
import torch
from sol_execbench_type_helpers import (
    make_definition,
    make_solution,
    make_workload,
)

from sol_execbench.core import BenchmarkConfig
from sol_execbench.core.bench.evaluation_requests import (
    WorkloadEvaluationRequest,
)
from sol_execbench.core.bench.reference_protocol import ReferenceTimingCase
from sol_execbench.core.bench.reward_hack import RewardHackError
from sol_execbench.core.data.solution import (
    BuildSpec,
    Solution,
    SourceFile,
    SupportedHardware,
    SupportedLanguages,
)
from sol_execbench.driver import scope_replay
from sol_execbench.driver.candidate_loader import (
    load_staged_user_function,
    verify_native_extension_symbols,
)
from sol_execbench.driver.problem_packager import ProblemPackager
from sol_execbench.driver.staging import (
    CANDIDATE_SOURCE_SUBDIR,
    resolve_stageable_safetensors,
    stage_solution_sources,
)


def _minimal_problem():
    definition = make_definition(
        name="vecadd_iso",
        op_type="elementwise",
        axes={"n": {"type": "const", "value": 16}},
        inputs={
            "x": {"shape": ["n"], "dtype": "float32"},
            "y": {"shape": ["n"], "dtype": "float32"},
        },
        outputs={"z": {"shape": ["n"], "dtype": "float32"}},
        reference="def run(x, y):\n    return x + y\n",
    )
    workloads = [
        make_workload(
            uuid="iso-wkl-1",
            axes={},
            inputs={"x": {"type": "random"}, "y": {"type": "random"}},
            checks=[{"type": "numeric", "output": "z"}],
        ),
    ]
    solution = make_solution(
        name="vecadd_sol",
        definition="vecadd_iso",
        author="test",
        spec={
            "languages": ["pytorch"],
            "target_hardware": ["LOCAL"],
            "entry_point": "pkg/kernel.py::run",
            "destination_passing_style": False,
        },
        sources=[
            {"path": "pkg/helper.py", "content": "SCALE = 1.0\n"},
            {
                "path": "pkg/kernel.py",
                "content": (
                    "from pkg.helper import SCALE\n"
                    "def run(x, y):\n"
                    "    return (x + y) * SCALE\n"
                ),
            },
        ],
    )
    return definition, workloads, solution


def test_packager_isolates_candidate_sources_under_candidate_subdir(
    tmp_path: Path,
) -> None:
    definition, workloads, solution = _minimal_problem()
    staging_dir = tmp_path / "staging"
    pkg = ProblemPackager(
        definition=definition,
        workloads=workloads,
        solution=solution,
        config=BenchmarkConfig(
            lock_clocks=False,
            benchmark_reference=False,
            warmup_runs=1,
            iterations=1,
            trials=1,
        ),
        output_dir=staging_dir,
        keep_output_dir=True,
    )
    assert not (staging_dir / "pkg").exists()
    assert (
        staging_dir / CANDIDATE_SOURCE_SUBDIR / "pkg" / "kernel.py"
    ).is_file()

    cmd = pkg.execute()
    assert cmd == [sys.executable, "-P", "evaluation_orchestrator.py"]

    # Even if a rogue module named json.py sits in staging_dir, -P prevents
    # evaluation_orchestrator.py, reference_worker.py, and eval_driver.py from
    # importing from CWD.
    (staging_dir / "secrets.py").write_text(
        "raise RuntimeError('CWD hijacked')\n"
    )
    proc = subprocess.run(
        cmd,
        cwd=staging_dir,
        env={
            **os.environ,
            "SOL_EXECBENCH_CLOCKS_LOCKED": "0",
            "SOL_EXECBENCH_ALLOW_CPU_TIMING": "1",
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    traces = pkg.convert_stdout_to_traces(proc.stdout)
    assert len(traces) == 1
    assert traces[0].is_successful()


@pytest.mark.parametrize(
    "bad_path",
    [
        "os.py",
        "io.py",
        "json.py",
        "torch.py",
        "triton/hijack.py",
        "sol_execbench/hijack.py",
        "eval_driver.py",
        "evaluation_orchestrator.py",
        "reference_worker.py",
        "build_ext.py",
    ],
)
def test_stage_solution_sources_rejects_shadowing_and_reserved_names(
    tmp_path: Path,
    bad_path: str,
) -> None:
    solution = Solution.model_construct(
        name="bad",
        definition="def",
        author="test",
        spec=BuildSpec.model_construct(
            languages=[SupportedLanguages.PYTORCH],
            target_hardware=[SupportedHardware.LOCAL],
            entry_point="kernel.py::run",
            destination_passing_style=False,
        ),
        sources=[
            SourceFile.model_construct(path=bad_path, content="x = 1\n"),
        ],
    )
    with pytest.raises(ValueError, match="Refusing to stage"):
        stage_solution_sources(solution, tmp_path)


def test_load_staged_user_function_appends_candidate_dir_at_end_of_sys_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, solution = _minimal_problem()
    stage_solution_sources(solution, tmp_path)
    cand_str = str((tmp_path / CANDIDATE_SOURCE_SUBDIR).resolve())
    cleaned = [entry for entry in sys.path if entry != cand_str]
    monkeypatch.setattr(sys, "path", cleaned)

    saved_modules = {
        name: sys.modules[name]
        for name in list(sys.modules)
        if name == "pkg" or name.startswith("pkg.")
    }
    for name in list(saved_modules):
        sys.modules.pop(name, None)
    try:
        fn = load_staged_user_function(solution, tmp_path)
        assert sys.path[-1] == cand_str
        assert sys.path[0] != str(tmp_path)
        x = torch.tensor([1.0, 2.0])
        y = torch.tensor([3.0, 4.0])
        assert torch.equal(
            cast(torch.Tensor, fn(x, y)), torch.tensor([4.0, 6.0])
        )
    finally:
        for name in list(sys.modules):
            if name == "pkg" or name.startswith("pkg."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


def test_resolve_stageable_safetensors_rejects_symlinks_escaping_root(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    outside = tmp_path / "outside_secret.safetensors"
    outside.write_bytes(b"secret")
    escaped_link = repo_root / "escaped.safetensors"
    escaped_link.symlink_to(outside)

    source, rel = resolve_stageable_safetensors(
        "escaped.safetensors",
        repo_root=repo_root,
        flashinfer_trace_dir=None,
    )
    assert source is None
    assert rel is None

    source_t, rel_t = resolve_stageable_safetensors(
        "../outside_secret.safetensors",
        repo_root=repo_root,
        flashinfer_trace_dir=None,
    )
    assert source_t is None
    assert rel_t is None


@pytest.mark.parametrize(
    "symbol",
    [
        "pthread_create@GLIBC_2.34",
        "dlsym@GLIBC_2.34",
        "dlopen@@GLIBC_2.34",
        "PyRun_SimpleString",
        "PyImport_AddModule",
        "hipStreamCreateWithFlags",
        "hipGraphLaunch",
    ],
)
def test_verify_native_extension_symbols_rejects_forbidden_undefined_symbols(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    symbol: str,
) -> None:
    fake_so = tmp_path / "benchmark_kernel.so"
    fake_so.write_bytes(b"\x7fELF" + b"\x00" * 60)

    def fake_run(cmd, *, timeout, **_kwargs):
        assert cmd[0] == "/usr/bin/nm"
        assert timeout == 10
        return subprocess.CompletedProcess(
            cmd,
            0,
            f"                 U {symbol}\n",
            "",
        )

    monkeypatch.setattr(
        "sol_execbench.driver.candidate_loader.run_in_process_group_bounded",
        fake_run,
    )
    with pytest.raises(RewardHackError, match="forbidden symbol"):
        verify_native_extension_symbols(
            fake_so,
            tool_resolver=lambda name: "/usr/bin/nm" if name == "nm" else None,
        )


def test_scope_replay_invoke_rejects_tensor_subclass_before_boundary_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _SubTensor(torch.Tensor):
        pass

    boundary_events: list[int] = []
    tool = Mock()
    tool.sol_scope_boundary.side_effect = boundary_events.append

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args: None)
    workload = make_workload(
        uuid="scope-lazy-wkl",
        axes={},
        inputs={"x": {"type": "random"}},
        checks=[{"type": "numeric", "output": "z"}],
    )
    definition = make_definition(
        name="scope_lazy",
        op_type="elementwise",
        axes={"n": {"type": "const", "value": 4}},
        inputs={"x": {"shape": ["n"], "dtype": "float32"}},
        outputs={"z": {"shape": ["n"], "dtype": "float32"}},
        reference="def run(x):\n    return x\n",
    )
    request = cast(
        WorkloadEvaluationRequest,
        SimpleNamespace(
            definition=definition,
            workloads=[workload],
            device="cpu",
            output_names=["z"],
            output_dtypes_torch={"z": torch.float32},
            destination_passing_style=False,
            dependencies=SimpleNamespace(
                reference_client=SimpleNamespace(
                    timing_iteration_case=lambda **_kw: ReferenceTimingCase(
                        inputs=[torch.ones(4)],
                        outputs=[],
                        reference_latency_ms=1.0,
                        input_sha256="f" * 64,
                    ),
                    validate_timing_outputs=lambda _out: None,
                ),
                user_fn=lambda x: x.as_subclass(_SubTensor),
                check_integrity=lambda *_args: None,
                integrity_snapshot={},
                driver_globals={},
            ),
        ),
    )

    with pytest.raises(RewardHackError, match="_SubTensor"):
        scope_replay._invoke(request, tool, scoped=True)
    # Boundary opened (1) and exception raised BEFORE boundary close (0)!
    assert boundary_events == [1]
