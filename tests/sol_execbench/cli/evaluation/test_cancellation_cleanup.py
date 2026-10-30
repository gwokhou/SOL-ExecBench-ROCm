"""CPU subprocess regressions for cancellation of the public execution boundary."""

from __future__ import annotations

import fcntl
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

import pytest

_CHILD = """
import os, signal, time
from pathlib import Path
root = Path(os.environ['CANCEL_TEST_ROOT'])
def stopping(*_):
    (root / 'stopping').touch()
signal.signal(signal.SIGTERM, stopping)
(root / 'child.pid').write_text(str(os.getpid()))
while True:
    time.sleep(.02)
"""

_PARENT = """
import os, sys
from types import SimpleNamespace
from sol_execbench.core.bench import gpu_execution
from sol_execbench.core.bench.clock_ownership import ClockDevice
gpu_execution.resolve_gpu_device = lambda: ClockDevice(bdf='0000:05:00.0', uuid=os.environ['GPU_LOCK_UUID'])
gpu_execution.observe_clock_levels = lambda _: ('AMDSMI_DEV_PERF_LEVEL_AUTO',)
from sol_execbench.cli.evaluation.phases import evaluation_execution_boundary
from sol_execbench.cli.evaluation.profile_mode import ProfileMode
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded
request = SimpleNamespace(timeout=15, device='cuda:0', unsafe_local_execution=False,
                          profile=ProfileMode.NONE, lock_clocks=False)
with evaluation_execution_boundary(request):
    run_in_process_group_bounded([sys.executable, '-c', os.environ['CANCEL_TEST_CHILD']], timeout=12)
"""

_PROFILER_PARENT = """
import os, sys
from sol_execbench.core.bench import gpu_execution
from sol_execbench.core.bench.clock_ownership import ClockDevice
gpu_execution.resolve_gpu_device = lambda: ClockDevice(bdf="0000:05:00.0", uuid=os.environ["GPU_LOCK_UUID"])
gpu_execution.observe_clock_levels = lambda _: ("AMDSMI_DEV_PERF_LEVEL_AUTO",)
from sol_execbench.core.bench.rocm_profiler.commands import default_runner
with gpu_execution.native_gpu_execution():
    default_runner([sys.executable, '-c', os.environ['CANCEL_TEST_CHILD']], timeout_seconds=12)
"""


def _wait_file(path: Path, timeout: float = 8) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            pytest.fail(f"child did not publish {path.name}")
        time.sleep(0.01)


@pytest.mark.parametrize("first_signal", [signal.SIGTERM, signal.SIGINT])
@pytest.mark.parametrize(
    "entry", [_PARENT, _PROFILER_PARENT], ids=["evaluate", "profiler"]
)
def test_cancel_drains_child_group_before_unlock(tmp_path, first_signal, entry):
    environment = dict(os.environ)
    for key in tuple(environment):
        if key.startswith("SOL_EXECBENCH_"):
            environment.pop(key)
    environment.update(
        GPU_LOCK_DIR=str(tmp_path),
        SOL_EXECBENCH_CLOCK_STATE_DIR=str(tmp_path / "private"),
        CANCEL_TEST_ROOT=str(tmp_path),
        CANCEL_TEST_CHILD=_CHILD,
    )
    process = subprocess.Popen(
        [sys.executable, "-c", entry],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    child_pid = None
    try:
        _wait_file(tmp_path / "child.pid")
        child_pid = int((tmp_path / "child.pid").read_text())
        process.send_signal(first_signal)
        _wait_file(tmp_path / "stopping", timeout=2)
        process.send_signal(signal.SIGTERM)
        with (
            (
                tmp_path
                / "gpu-11111111-2222-3333-4444-555555555555.L1-ownership.lock"
            ).open("a+") as lock,
            pytest.raises(BlockingIOError),
        ):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 128 + first_signal, (stdout, stderr)
        with (
            tmp_path
            / "gpu-11111111-2222-3333-4444-555555555555.L1-ownership.lock"
        ).open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not Path(f"/proc/{child_pid}").exists()
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)
        if child_pid is not None:
            with suppress(ProcessLookupError):
                os.killpg(child_pid, signal.SIGKILL)
