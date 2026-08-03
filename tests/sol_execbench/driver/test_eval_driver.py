# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Subprocess integration tests for the SOL ExecBench eval driver.

Runs eval_driver.py directly in a temp directory — no GPU server required.
Verifies that:
  - A correct solution emits PASSED traces.
  - Clock-lock unavailability warnings propagate into trace logs.
  - Reward-hack defenses (monkey-patch, thread injection, lazy outputs) fire and
    emit REWARD_HACK traces with descriptive log messages.
  - @torch.compile solutions do NOT trigger a thread injection false positive.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from sol_execbench import driver
from sol_execbench.core.bench.reference_protocol import TRUSTED_DEFINITION_FILE
from sol_execbench.core.integrity.schema_versions import (
    BENCHMARK_CONFIG_SCHEMA_VERSION,
    DEFINITION_SCHEMA_VERSION,
    SOLUTION_SCHEMA_VERSION,
    WORKLOAD_SCHEMA_VERSION,
)

_TEMPLATES_DIR = Path(driver.__file__).parent / "templates"


def _stage_evaluation_templates(tmp_path: Path) -> None:
    for name in (
        "eval_driver.py",
        "reference_worker.py",
        "evaluation_orchestrator.py",
    ):
        (tmp_path / name).write_text((_TEMPLATES_DIR / name).read_text())


def parse_eval_result(stdout: str, stderr: str) -> list[dict]:
    """Parse JSONL Trace dicts from eval_driver stdout."""
    traces = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("{"):
            traces.append(json.loads(line))
    return traces


# ---------------------------------------------------------------------------
# Minimal problem used by all subprocess tests
# ---------------------------------------------------------------------------

_MINIMAL_DEFINITION = {
    "name": "test_vecadd",
    "op_type": "elementwise",
    "axes": {"n": {"type": "const", "value": 64}},
    "inputs": {
        "x": {"shape": ["n"], "dtype": "float32"},
        "y": {"shape": ["n"], "dtype": "float32"},
    },
    "outputs": {"z": {"shape": ["n"], "dtype": "float32"}},
    "reference": "import torch\ndef run(x, y):\n    return x + y",
}

_MINIMAL_WORKLOAD = {
    "axes": {},
    "inputs": {
        "x": {"type": "random"},
        "y": {"type": "random"},
    },
    "checks": [{"type": "numeric", "output": "z"}],
    "uuid": "sub-test-0001",
}

_SOLUTION_SPEC = {
    "name": "test_vecadd_solution",
    "definition": "test_vecadd",
    "author": "test",
    "spec": {
        "languages": ["pytorch"],
        "target_hardware": ["LOCAL"],
        "entry_point": "kernel.py::run",
        # Must be explicit: driver defaults to True (DPS) if omitted.
        "destination_passing_style": False,
    },
    "sources": [{"path": "kernel.py", "content": "# loaded by test"}],
}


def _stage_definitions(tmp_path: Path, definition: dict) -> None:
    """Stage a worker-only definition and a candidate-visible redacted copy."""
    definition = {
        "schema_version": DEFINITION_SCHEMA_VERSION,
        **definition,
    }
    (tmp_path / TRUSTED_DEFINITION_FILE).write_text(json.dumps(definition))
    parameters = ", ".join(definition["inputs"])
    reference_stub = (
        f"def run({parameters}):\n"
        "    raise RuntimeError('trusted reference unavailable')\n"
    )
    custom_inputs_entrypoint = definition.get("custom_inputs_entrypoint")
    if custom_inputs_entrypoint:
        reference_stub += (
            f"\ndef {custom_inputs_entrypoint}(axes, device):\n"
            "    raise RuntimeError('trusted input generator unavailable')\n"
        )
    candidate_definition = {
        **definition,
        "reference": reference_stub,
    }
    (tmp_path / "definition.json").write_text(json.dumps(candidate_definition))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run_eval_driver(
    tmp_path: Path,
    kernel_code: str,
    bench_config: dict | None = None,
    extra_env: dict | None = None,
    definition: dict | None = None,
    workload: dict | None = None,
) -> list[dict]:
    """Write all staging files and run eval_driver.py in a subprocess.

    Args:
        tmp_path: Temporary directory to use as the staging directory.
        kernel_code: Python source to write to kernel.py.
        bench_config: BenchmarkConfig overrides to write to config.json.
            Defaults to ``{"lock_clocks": False}`` to suppress ROCm clock-lock
            warnings on machines where clock locking is not available.
        extra_env: Additional environment variables for the subprocess.
        definition: Problem definition dict. Defaults to ``_MINIMAL_DEFINITION``.
        workload: Workload dict. Defaults to ``_MINIMAL_WORKLOAD``.

    Returns:
        List of Trace dicts parsed from the driver's stdout.

    """
    result = _run_eval_driver_process(
        tmp_path,
        kernel_code,
        bench_config=bench_config,
        extra_env=extra_env,
        definition=definition,
        workload=workload,
    )
    return parse_eval_result(result.stdout, result.stderr)


def _run_eval_driver_process(
    tmp_path: Path,
    kernel_code: str,
    bench_config: dict | None = None,
    extra_env: dict | None = None,
    definition: dict | None = None,
    workload: dict | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the isolated reference/candidate workflow and return its result."""
    _stage_evaluation_templates(tmp_path)
    _stage_definitions(
        tmp_path,
        definition if definition is not None else _MINIMAL_DEFINITION,
    )
    (tmp_path / "workload.jsonl").write_text(
        json.dumps(
            {
                "schema_version": WORKLOAD_SCHEMA_VERSION,
                **(workload if workload is not None else _MINIMAL_WORKLOAD),
            },
        ),
    )
    solution = {
        "schema_version": SOLUTION_SCHEMA_VERSION,
        **_SOLUTION_SPEC,
        "sources": [{"path": "kernel.py", "content": kernel_code}],
    }
    (tmp_path / "solution.json").write_text(json.dumps(solution))
    (tmp_path / "kernel.py").write_text(kernel_code)

    # Disable clock locking by default where ROCm clock access is absent.
    cfg = (
        bench_config
        if bench_config is not None
        else {
            "lock_clocks": False,
        }
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "schema_version": BENCHMARK_CONFIG_SCHEMA_VERSION,
                **cfg,
            },
        ),
    )

    env = {
        **os.environ,
        "SOL_EXECBENCH_CLOCKS_LOCKED": "0",
        "SOL_EXECBENCH_ALLOW_CPU_TIMING": "1",
    }
    if extra_env:
        env.update(extra_env)

    return subprocess.run(
        [sys.executable, "evaluation_orchestrator.py"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_reference_source_is_removed_before_candidate_execution(tmp_path):
    result = _run_eval_driver_process(
        tmp_path,
        "def run(x, y):\n    return x + y\n",
    )

    assert result.returncode == 0
    assert not (tmp_path / TRUSTED_DEFINITION_FILE).exists()
    assert not (tmp_path / "_reference.py").exists()
    candidate_definition = json.loads(
        (tmp_path / "definition.json").read_text(),
    )
    assert candidate_definition["reference"] != _MINIMAL_DEFINITION["reference"]


@pytest.mark.xdist_group("serial")
def test_passing_solution(tmp_path):
    """A correct vector-add kernel produces a PASSED trace."""
    kernel = "import torch\ndef run(x, y):\n    return x + y\n"
    traces = _run_eval_driver(tmp_path, kernel)

    assert len(traces) == 1
    ev = traces[0]["evaluation"]
    assert ev["status"] == "PASSED", (
        f"Expected PASSED, got {ev['status']}; log={ev.get('log')}"
    )


def _custom_inputs_definition(generate_body: str) -> dict:
    return {
        "name": "test_custom_matmul",
        "op_type": "matmul",
        "axes": {
            "m": {"type": "const", "value": 4},
            "k": {"type": "const", "value": 3},
            "n": {"type": "const", "value": 2},
        },
        "custom_inputs_entrypoint": "generate_inputs",
        "inputs": {
            "a": {"shape": ["m", "k"], "dtype": "float32"},
            "b": {"shape": ["k", "n"], "dtype": "float32"},
        },
        "outputs": {"c": {"shape": ["m", "n"], "dtype": "float32"}},
        "reference": (
            "import torch\n"
            f"{generate_body}\n\n"
            "@torch.no_grad()\n"
            "def run(a, b):\n"
            "    return a @ b\n"
        ),
    }


def _custom_inputs_workload() -> dict:
    return {
        "axes": {},
        "inputs": {"a": {"type": "custom"}, "b": {"type": "custom"}},
        "checks": [{"type": "numeric", "output": "c"}],
        "uuid": "custom-driver-0001",
    }


def test_custom_inputs_success_path_is_cpu_safe(tmp_path):
    """The driver resolves and uses benchmark-defined custom inputs."""
    definition = _custom_inputs_definition(
        "def generate_inputs(axes, device):\n"
        "    return {\n"
        "        'a': torch.rand((axes['m'], axes['k']), device=device),\n"
        "        'b': torch.rand((axes['k'], axes['n']), device=device),\n"
        "    }\n",
    )
    kernel = "import torch\ndef run(a, b):\n    return a @ b\n"

    traces = _run_eval_driver(
        tmp_path,
        kernel,
        definition=definition,
        workload=_custom_inputs_workload(),
    )

    assert len(traces) == 1
    assert traces[0]["evaluation"]["status"] == "PASSED"


def test_fixed_custom_inputs_fail_closed_during_timing(tmp_path):
    """A generator that ignores iteration seeds cannot claim v4 timing."""
    definition = _custom_inputs_definition(
        "def generate_inputs(axes, device):\n"
        "    return {\n"
        "        'a': torch.ones((axes['m'], axes['k']), device=device),\n"
        "        'b': torch.ones((axes['k'], axes['n']), device=device),\n"
        "    }\n",
    )
    traces = _run_eval_driver(
        tmp_path,
        "def run(a, b):\n    return a @ b\n",
        definition=definition,
        workload=_custom_inputs_workload(),
    )

    assert traces[0]["evaluation"]["status"] == "REWARD_HACK"
    assert "unique input values" in traces[0]["evaluation"].get("log", "")


def test_custom_inputs_schema_mismatch_preempts_reference_run(tmp_path):
    """Bad generated tensors are classified before reference run()."""
    definition = _custom_inputs_definition(
        "def generate_inputs(axes, device):\n"
        "    return {\n"
        "        'a': torch.ones((axes['m'], axes['k'] + 1), device=device),\n"
        "        'b': torch.ones((axes['k'], axes['n']), device=device),\n"
        "    }\n",
    )
    kernel = "import torch\ndef run(a, b):\n    return a @ b\n"

    traces = _run_eval_driver(
        tmp_path,
        kernel,
        definition=definition,
        workload=_custom_inputs_workload(),
    )

    ev = traces[0]["evaluation"]
    assert ev["status"] == "RUNTIME_ERROR"
    assert "gen_inputs_schema_mismatch" in ev.get("log", "")
    assert "Reference run() failed" not in ev.get("log", "")
    assert "seed=" in ev.get("log", "")
    assert "generated_keys=['a', 'b']" in ev.get("log", "")


def test_custom_inputs_device_mismatch_preempts_reference_run(tmp_path):
    """Wrong-device generated tensors are classified before reference run()."""
    if torch.cuda.is_available():
        pytest.skip(
            "CPU mismatch path only applies when CUDA/HIP is unavailable",
        )
    definition = _custom_inputs_definition(
        "def generate_inputs(axes, device):\n"
        "    return {\n"
        "        'a': torch.ones((axes['m'], axes['k']), device='meta'),\n"
        "        'b': torch.ones((axes['k'], axes['n']), device=device),\n"
        "    }\n",
    )
    kernel = "import torch\ndef run(a, b):\n    return a @ b\n"

    traces = _run_eval_driver(
        tmp_path,
        kernel,
        definition=definition,
        workload=_custom_inputs_workload(),
    )

    ev = traces[0]["evaluation"]
    assert ev["status"] == "RUNTIME_ERROR"
    assert "gen_inputs_device_mismatch" in ev.get("log", "")
    assert "Reference run() failed" not in ev.get("log", "")


def test_custom_inputs_non_oom_generation_error_class(tmp_path):
    """Non-OOM generation failures stay separate from reference failures."""
    definition = _custom_inputs_definition(
        "def generate_inputs(axes, device):\n"
        "    raise RuntimeError('synthetic generation failure')\n",
    )
    kernel = "import torch\ndef run(a, b):\n    return a @ b\n"

    traces = _run_eval_driver(
        tmp_path,
        kernel,
        definition=definition,
        workload=_custom_inputs_workload(),
    )

    ev = traces[0]["evaluation"]
    assert ev["status"] == "RUNTIME_ERROR"
    assert "gen_inputs_error" in ev.get("log", "")
    assert "synthetic generation failure" in ev.get("log", "")
    assert "Reference run() failed" not in ev.get("log", "")


def test_reference_outputs_aliasing_inputs_are_stabilized(tmp_path):
    """Reference outputs that alias inputs stay stable across user mutation."""
    definition = {
        **_MINIMAL_DEFINITION,
        "name": "alias_reference",
        "reference": "def run(x, y):\n    return x\n",
    }
    kernel = "def run(x, y):\n    x.add_(1)\n    return x\n"
    traces = _run_eval_driver(tmp_path, kernel, definition=definition)

    assert len(traces) == 1
    ev = traces[0]["evaluation"]
    assert ev["status"] == "INCORRECT_NUMERICAL", (
        f"Expected INCORRECT_NUMERICAL, got {ev['status']}; log={ev.get('log')}"
    )


def test_reference_timing_failure_is_explicit_when_requested(tmp_path):
    """Reference timing failures should not silently look like a zero baseline."""
    reference = (
        "import torch\n"
        "_calls = 0\n"
        "def run(x, y):\n"
        "    global _calls\n"
        "    _calls += 1\n"
        "    if _calls > 11:\n"
        "        raise RuntimeError('reference timing boom')\n"
        "    return x + y\n"
    )
    definition = {**_MINIMAL_DEFINITION, "reference": reference}
    kernel = "import torch\ndef run(x, y):\n    return x + y\n"
    traces = _run_eval_driver(
        tmp_path,
        kernel,
        bench_config={
            "lock_clocks": False,
            "benchmark_reference": True,
            "warmup_runs": 1,
            "iterations": 1,
        },
        definition=definition,
    )

    assert len(traces) == 1
    ev = traces[0]["evaluation"]
    assert ev["status"] == "RUNTIME_ERROR"
    assert "reference timing boom" in ev.get("log", "")


def test_phase_counting_cannot_skip_timed_computation(tmp_path):
    """Every timed result is validated, so correctness-call counting cannot pass."""
    kernel = (
        "import torch\n"
        "_calls = 0\n"
        "def run(x, y):\n"
        "    global _calls\n"
        "    _calls += 1\n"
        "    if _calls <= 10:\n"
        "        return x + y\n"
        "    return torch.empty_like(x)\n"
    )

    traces = _run_eval_driver(
        tmp_path,
        kernel,
        bench_config={
            "lock_clocks": False,
            "benchmark_reference": False,
            "warmup_runs": 1,
            "iterations": 1,
            "trials": 1,
        },
    )

    assert traces[0]["evaluation"]["status"] == "REWARD_HACK"
    assert "timed invocation" in traces[0]["evaluation"]["log"]


def test_custom_inputs_use_a_distinct_seed_for_each_correctness_round(tmp_path):
    """A candidate cannot reuse the first result across ten custom-input rounds."""
    definition = _custom_inputs_definition(
        "def generate_inputs(axes, device):\n"
        "    value = float(torch.initial_seed() % 997 + 1)\n"
        "    return {\n"
        "        'a': torch.full((axes['m'], axes['k']), value, device=device),\n"
        "        'b': torch.ones((axes['k'], axes['n']), device=device),\n"
        "    }\n",
    )
    kernel = (
        "_saved = None\n"
        "def run(a, b):\n"
        "    global _saved\n"
        "    if _saved is None:\n"
        "        _saved = a @ b\n"
        "    return _saved\n"
    )

    traces = _run_eval_driver(
        tmp_path,
        kernel,
        definition=definition,
        workload=_custom_inputs_workload(),
    )

    assert traces[0]["evaluation"]["status"] == "INCORRECT_NUMERICAL"


def test_reference_outputs_are_frozen_before_user_call(tmp_path):
    """A user in-place input update must not mutate the saved reference output."""
    definition = {
        "name": "test_alias_freeze",
        "op_type": "elementwise",
        "axes": {"n": {"type": "const", "value": 64}},
        "inputs": {"x": {"shape": ["n"], "dtype": "float32"}},
        "outputs": {"z": {"shape": ["n"], "dtype": "float32"}},
        "reference": "def run(x):\n    return x\n",
    }
    workload = {
        "axes": {},
        "inputs": {"x": {"type": "random"}},
        "checks": [{"type": "numeric", "output": "z"}],
        "uuid": "alias-freeze-0001",
    }
    _stage_evaluation_templates(tmp_path)
    _stage_definitions(tmp_path, definition)
    (tmp_path / "workload.jsonl").write_text(
        json.dumps(
            {"schema_version": WORKLOAD_SCHEMA_VERSION, **workload},
        ),
    )
    kernel = "def run(x):\n    original = x.clone()\n    x.add_(1)\n    return original\n"
    solution = {
        "schema_version": SOLUTION_SCHEMA_VERSION,
        **_SOLUTION_SPEC,
        "definition": "test_alias_freeze",
        "sources": [{"path": "kernel.py", "content": kernel}],
    }
    (tmp_path / "solution.json").write_text(json.dumps(solution))
    (tmp_path / "kernel.py").write_text(kernel)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "schema_version": BENCHMARK_CONFIG_SCHEMA_VERSION,
                "lock_clocks": False,
                "benchmark_reference": False,
            },
        ),
    )

    result = subprocess.run(
        [sys.executable, "evaluation_orchestrator.py"],
        cwd=tmp_path,
        env={
            **os.environ,
            "SOL_EXECBENCH_CLOCKS_LOCKED": "0",
            "SOL_EXECBENCH_ALLOW_CPU_TIMING": "1",
        },
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    traces = parse_eval_result(result.stdout, result.stderr)

    assert len(traces) == 1
    assert traces[0]["evaluation"]["status"] == "PASSED"


def test_noisy_user_stdout_stays_out_of_trace_jsonl(tmp_path):
    """Import-time and run-time user prints must not corrupt stdout JSONL."""
    kernel = (
        "import torch\n"
        "print('import noise from solution')\n"
        "def run(x, y):\n"
        "    print('run noise from solution')\n"
        "    return x + y\n"
    )

    result = _run_eval_driver_process(tmp_path, kernel)
    traces = parse_eval_result(result.stdout, result.stderr)

    assert len(traces) == 1
    assert traces[0]["evaluation"]["status"] == "PASSED"
    assert all(
        json.loads(line) for line in result.stdout.splitlines() if line.strip()
    )
    assert "import noise from solution" not in result.stdout
    assert "run noise from solution" not in result.stdout
    assert "import noise from solution" in result.stderr
    assert "run noise from solution" in result.stderr


def test_monkey_patch_detected(tmp_path):
    """Kernel that patches torch.cuda.Event.elapsed_time at import → REWARD_HACK."""
    kernel = (
        "import torch\n"
        "torch.cuda.Event.elapsed_time = lambda self, end: 0.0\n"
        "\n"
        "def run(x, y):\n"
        "    return x + y\n"
    )
    traces = _run_eval_driver(tmp_path, kernel)

    assert len(traces) == 1
    ev = traces[0]["evaluation"]
    assert ev["status"] == "REWARD_HACK", (
        f"Expected REWARD_HACK, got {ev['status']}; log={ev.get('log')}"
    )
    assert "monkey-patched" in ev.get("log", ""), (
        f"Expected 'monkey-patched' in log; got: {ev.get('log')}"
    )


@pytest.mark.xdist_group("serial")
def test_thread_injection_detected(tmp_path):
    """Kernel that spawns a daemon thread inside run() → REWARD_HACK."""
    kernel = (
        "import threading\n"
        "import time\n"
        "\n"
        "def run(x, y):\n"
        "    threading.Thread(target=lambda: time.sleep(10), daemon=True).start()\n"
        "    return x + y\n"
    )
    traces = _run_eval_driver(tmp_path, kernel)

    assert len(traces) == 1
    ev = traces[0]["evaluation"]
    assert ev["status"] == "REWARD_HACK", (
        f"Expected REWARD_HACK, got {ev['status']}; log={ev.get('log')}"
    )
    assert "thread" in ev.get("log", "").lower(), (
        f"Expected 'thread' in log; got: {ev.get('log')}"
    )


def test_torch_compile_no_reward_hack(tmp_path):
    """@torch.compile must not trigger thread injection false positive.

    Candidate compilation is synchronous, while the thread monitor remains
    strict for every candidate-created worker during timing.
    """
    kernel = (
        "import torch\n"
        "import torch.nn.functional as F\n"
        "\n"
        "@torch.compile\n"
        "@torch.no_grad()\n"
        "def run(x, y):\n"
        "    return x + y\n"
    )
    traces = _run_eval_driver(tmp_path, kernel)

    assert len(traces) == 1
    ev = traces[0]["evaluation"]
    assert ev["status"] == "PASSED", (
        f"Expected PASSED (torch.compile false positive fixed), "
        f"got {ev['status']}; log={ev.get('log')}"
    )


def test_candidate_forces_synchronous_inductor_compilation(tmp_path):
    """The candidate process overrides inherited async-compile settings."""
    kernel = (
        "import os\n"
        "\n"
        "def run(x, y):\n"
        "    assert os.environ['TORCHINDUCTOR_COMPILE_THREADS'] == '1'\n"
        "    return x + y\n"
    )
    traces = _run_eval_driver(
        tmp_path,
        kernel,
        extra_env={"TORCHINDUCTOR_COMPILE_THREADS": "8"},
    )

    assert len(traces) == 1
    ev = traces[0]["evaluation"]
    assert ev["status"] == "PASSED", (
        f"Expected synchronous compilation policy; "
        f"got {ev['status']}; log={ev.get('log')}"
    )


def test_lazy_output_detected(tmp_path):
    """Kernel that returns a torch.Tensor subclass → REWARD_HACK."""
    kernel = (
        "import torch\n"
        "\n"
        "class _FakeTensor(torch.Tensor):\n"
        "    pass\n"
        "\n"
        "def run(x, y):\n"
        "    return (x + y).as_subclass(_FakeTensor)\n"
    )
    traces = _run_eval_driver(tmp_path, kernel)

    assert len(traces) == 1
    ev = traces[0]["evaluation"]
    assert ev["status"] == "REWARD_HACK", (
        f"Expected REWARD_HACK, got {ev['status']}; log={ev.get('log')}"
    )
    assert "_FakeTensor" in ev.get("log", ""), (
        f"Expected '_FakeTensor' in log; got: {ev.get('log')}"
    )


# Check if triton is available for the regression test
_triton_available = importlib.util.find_spec("triton") is not None


@pytest.mark.requires_rocm_gpu
@pytest.mark.skipif(
    not _triton_available or not torch.cuda.is_available(),
    reason="triton or cuda device not available",
)
def test_triton_jit_reference(tmp_path):
    """Reference with @triton.jit must not crash the eval driver.

    Regression test: exec(compile(code, '<reference>', 'exec')) causes
    triton.JITFunction.__init__ to call inspect.getsourcelines() which
    fails because '<reference>' is not a real file.  The fix writes the
    reference to _reference.py and imports it via importlib.
    """
    # Definition whose reference uses @triton.jit for a vector-add.
    triton_reference = (
        "import torch\n"
        "import triton\n"
        "import triton.language as tl\n"
        "\n"
        "@triton.jit\n"
        "def _vecadd_kernel(x_ptr, y_ptr, z_ptr, n, BLOCK: tl.constexpr):\n"
        "    pid = tl.program_id(0)\n"
        "    offs = pid * BLOCK + tl.arange(0, BLOCK)\n"
        "    mask = offs < n\n"
        "    x = tl.load(x_ptr + offs, mask=mask)\n"
        "    y = tl.load(y_ptr + offs, mask=mask)\n"
        "    tl.store(z_ptr + offs, x + y, mask=mask)\n"
        "\n"
        "def run(x, y):\n"
        "    z = torch.empty_like(x)\n"
        "    n = x.numel()\n"
        "    _vecadd_kernel[(n + 255) // 256,](x, y, z, n, BLOCK=256)\n"
        "    return z\n"
    )
    definition = {**_MINIMAL_DEFINITION, "reference": triton_reference}

    kernel = "import torch\ndef run(x, y):\n    return x + y\n"
    traces = _run_eval_driver(tmp_path, kernel, definition=definition)

    assert len(traces) == 1
    ev = traces[0]["evaluation"]
    assert ev["status"] == "PASSED", (
        f"Expected PASSED (triton @jit reference), got {ev['status']}; "
        f"log={ev.get('log')}"
    )


# ---------------------------------------------------------------------------
# Evil tests: load_inline and stream keyword defenses
# ---------------------------------------------------------------------------


@pytest.mark.xdist_group("serial")
def test_load_inline_blocked_in_run(tmp_path):
    """Kernel that calls load_inline inside run() → static REWARD_HACK."""
    kernel = (
        "from torch.utils import cpp_extension\n"
        "\n"
        "def run(x, y):\n"
        "    cpp_extension.load_inline(name='evil', cpp_sources='void f(){}')\n"
        "    return x + y\n"
    )
    traces = _run_eval_driver(tmp_path, kernel)

    assert len(traces) == 1
    ev = traces[0]["evaluation"]
    assert ev["status"] == "REWARD_HACK", (
        f"Expected REWARD_HACK, got {ev['status']}; log={ev.get('log')}"
    )
    assert "unauthorized_file_or_loader" in ev.get("log", "")


def test_load_inline_blocked_at_import(tmp_path):
    """Kernel that calls load_inline at import time → static REWARD_HACK."""
    kernel = (
        "from torch.utils.cpp_extension import load_inline\n"
        "load_inline(name='evil', cpp_sources='void f(){}')\n"
        "\n"
        "def run(x, y):\n"
        "    return x + y\n"
    )
    traces = _run_eval_driver(tmp_path, kernel)

    assert len(traces) == 1
    ev = traces[0]["evaluation"]
    assert ev["status"] == "REWARD_HACK", (
        f"Expected REWARD_HACK, got {ev['status']}; log={ev.get('log')}"
    )
    assert "unauthorized_file_or_loader" in ev.get("log", "")


def test_static_source_review_blocks_non_default_stream(tmp_path):
    """Kernel that creates a non-default stream is rejected before import."""
    kernel = (
        "import torch\n"
        "stream = torch.cuda.Stream()\n"
        "\n"
        "def run(x, y):\n"
        "    with torch.cuda.stream(stream):\n"
        "        return x + y\n"
    )
    traces = _run_eval_driver(tmp_path, kernel)

    assert len(traces) == 1
    ev = traces[0]["evaluation"]
    assert ev["status"] == "REWARD_HACK"
    assert "hidden_async_stream" in ev.get("log", "")


def test_static_source_review_blocks_semantic_cache(tmp_path):
    """Kernel that keys cached output by input data pointer is rejected."""
    kernel = (
        "_cache = {}\n"
        "\n"
        "def run(x, y):\n"
        "    key = (x.data_ptr(), y.data_ptr())\n"
        "    if key not in _cache:\n"
        "        _cache[key] = x + y\n"
        "    return _cache[key]\n"
    )
    traces = _run_eval_driver(tmp_path, kernel)

    assert len(traces) == 1
    ev = traces[0]["evaluation"]
    assert ev["status"] == "REWARD_HACK"
    assert "semantic_output_cache" in ev.get("log", "")


def test_static_source_review_flags_precision_downgrade_without_blocking(
    tmp_path,
):
    """Internal downcasts are diagnostic; runtime correctness remains authoritative."""
    kernel = "def run(x, y):\n    return (x.half() + y.half()).float()\n"
    traces = _run_eval_driver(tmp_path, kernel)

    assert len(traces) == 1
    ev = traces[0]["evaluation"]
    assert ev["status"] in {"PASSED", "INCORRECT_NUMERICAL"}
    assert "precision_downgrade" not in ev.get("log", "")


# ---------------------------------------------------------------------------
# NaN / Inf JSON compliance tests
# ---------------------------------------------------------------------------


@pytest.mark.xdist_group("serial")
def test_nan_kernel_produces_valid_json(tmp_path):
    """Kernel that outputs NaN must produce strictly valid JSON traces.

    Regression test: Python's json.dumps(allow_nan=True) serializes NaN as
    the JavaScript literal ``NaN`` which is not valid JSON per RFC 8259.
    Strict parsers (e.g., Rust serde_json) reject the entire line, silently
    dropping the trace.  The fix sanitizes NaN/Inf to null before dumping.
    """
    kernel = "import torch\n\ndef run(x, y):\n    return torch.full_like(x, float('nan'))\n"
    traces = _run_eval_driver(tmp_path, kernel)

    assert len(traces) == 1
    ev = traces[0]["evaluation"]
    assert ev["status"] == "INCORRECT_NUMERICAL", (
        f"Expected INCORRECT_NUMERICAL, got {ev['status']}; log={ev.get('log')}"
    )

    # The critical check: re-serialize to JSON with allow_nan=False.
    # This mimics what a strict JSON parser (Rust, Go, etc.) would do.
    # It must NOT raise ValueError.
    import json as json_mod

    strict_json = json_mod.dumps(traces[0], allow_nan=False)
    reparsed = json_mod.loads(strict_json)
    assert reparsed["evaluation"]["status"] == "INCORRECT_NUMERICAL"


def test_inf_kernel_produces_valid_json(tmp_path):
    """Kernel that outputs Inf must produce strictly valid JSON traces."""
    kernel = "import torch\n\ndef run(x, y):\n    return torch.full_like(x, float('inf'))\n"
    traces = _run_eval_driver(tmp_path, kernel)

    assert len(traces) == 1
    ev = traces[0]["evaluation"]
    assert ev["status"] == "INCORRECT_NUMERICAL", (
        f"Expected INCORRECT_NUMERICAL, got {ev['status']}; log={ev.get('log')}"
    )

    import json as json_mod

    strict_json = json_mod.dumps(traces[0], allow_nan=False)
    reparsed = json_mod.loads(strict_json)
    assert reparsed["evaluation"]["status"] == "INCORRECT_NUMERICAL"


def test_nan_correctness_fields_are_finite_with_flags(tmp_path):
    """NaN outputs must produce finite correctness values with has_nan=True."""
    kernel = "import torch\n\ndef run(x, y):\n    return torch.full_like(x, float('nan'))\n"
    traces = _run_eval_driver(tmp_path, kernel)

    assert len(traces) == 1
    correctness = traces[0]["evaluation"].get("correctness", {})

    # Error values must be finite (not NaN/Inf)
    max_abs = correctness.get("max_absolute_error", 0.0)
    max_rel = correctness.get("max_relative_error", 0.0)
    assert isinstance(max_abs, (int, float)) and math.isfinite(max_abs), (
        f"max_absolute_error should be finite, got {max_abs!r}"
    )
    assert isinstance(max_rel, (int, float)) and math.isfinite(max_rel), (
        f"max_relative_error should be finite, got {max_rel!r}"
    )

    # Boolean flag must indicate NaN was detected
    assert correctness.get("has_nan") is True, (
        f"has_nan should be True for NaN output, got {correctness}"
    )


def test_passing_trace_correctness_unchanged(tmp_path):
    """PASSED traces must preserve finite correctness values (not clobber to null)."""
    kernel = "import torch\ndef run(x, y):\n    return x + y\n"
    traces = _run_eval_driver(tmp_path, kernel)

    assert len(traces) == 1
    ev = traces[0]["evaluation"]
    assert ev["status"] == "PASSED"

    correctness = ev.get("correctness", {})
    max_abs = correctness.get("max_absolute_error")
    max_rel = correctness.get("max_relative_error")

    # For a correct kernel, error values must be finite numbers (not null)
    assert isinstance(max_abs, (int, float)) and math.isfinite(max_abs), (
        f"Expected finite max_absolute_error for PASSED trace, got {max_abs!r}"
    )
    assert isinstance(max_rel, (int, float)) and math.isfinite(max_rel), (
        f"Expected finite max_relative_error for PASSED trace, got {max_rel!r}"
    )
