"""CPU compilation concurrency, atomic output and process cleanup regressions."""

import os
import sys
import threading
import time
from pathlib import Path

import pytest

from sol_execbench.core.process.locks import exclusive_file_lock

PRODUCER = "scripts/internal/rdna4/run_rdna4_resource_peak_calibration.py"
QUALIFIED = (
    "scripts/internal/rdna4/run_qualified_rdna4_resource_peak_calibration.py"
)


def test_compile_parallel_is_bounded_and_ordered(
    load_script, monkeypatch, tmp_path
):
    producer = load_script(PRODUCER)
    producer.PROBES = tuple({"source": f"{i}.hip"} for i in range(5))
    active = peak = 0
    lock = threading.Lock()
    barrier = threading.Barrier(2)
    monkeypatch.setattr(
        producer, "compile_job_limit", lambda requested: min(requested, 2)
    )

    def compile_probe(source, workdir, *_, **__):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        if source.stem in {"0", "1"}:
            barrier.wait(timeout=2)
        time.sleep(0.01)
        with lock:
            active -= 1
        return workdir / source.name

    monkeypatch.setattr(producer, "_compile_probe", compile_probe)
    result = producer._compile_calibration_binaries(
        "gfx1200", tmp_path, "mock", compile_jobs=8
    )
    assert peak == 2
    assert active == 0
    assert list(result) == [f"{i}.hip" for i in range(5)]


def test_failed_compile_preserves_previous_binary(
    load_script, monkeypatch, tmp_path
):
    producer = load_script(PRODUCER)
    source = tmp_path / "probe.hip"
    binary = producer._probe_binary_path(source, tmp_path)
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"previous")

    def fail(cmd, **_):
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"partial")
        raise RuntimeError("compiler failed")

    monkeypatch.setattr(producer, "_run", fail)
    with pytest.raises(RuntimeError, match="compiler failed"):
        producer._compile_probe(source, tmp_path, "fake", "gfx1200")
    assert binary.read_bytes() == b"previous"
    assert not list(binary.parent.glob(".compile-*"))


def test_failed_worker_cancels_live_compiler_descendants(
    load_script, monkeypatch, tmp_path
):
    producer = load_script(PRODUCER)
    producer.PROBES = ({"source": "slow.hip"}, {"source": "fail.hip"})
    producer.PROBE_DIR = tmp_path
    monkeypatch.setattr(producer, "compile_job_limit", lambda _: 2)
    child_pid = tmp_path / "child.pid"
    compiler = tmp_path / "compiler"
    compiler.write_text(
        f"#!{sys.executable}\n"
        "import pathlib,subprocess,sys,time\n"
        f"p=pathlib.Path({str(child_pid)!r})\n"
        "if sys.argv[-1].endswith('slow.hip'):\n"
        " c=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n"
        " p.write_text(str(c.pid))\n"
        " time.sleep(60)\n"
        "else:\n"
        " end=time.monotonic()+3\n"
        " while not p.exists() and time.monotonic()<end: time.sleep(.01)\n"
        " sys.exit(7)\n"
    )
    compiler.chmod(0o700)
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="exit 7"):
        producer._compile_calibration_binaries(
            "gfx1200", tmp_path, str(compiler), compile_jobs=2
        )
    assert time.monotonic() - start < 5
    pid = int(child_pid.read_text())
    status = Path(f"/proc/{pid}/stat")
    assert not status.exists() or status.read_text().split()[2] == "Z"
    assert not list(tmp_path.rglob("*.bin"))
    assert not list(tmp_path.rglob(".compile-*"))


def test_static_qualification_competing_writer_cannot_enter(
    load_script, monkeypatch, tmp_path
):
    qualified = load_script(QUALIFIED)
    args = qualified._parse_args(
        [
            "qualify-static",
            "--qualification-root",
            str(tmp_path / "gate"),
            "--out",
            str(tmp_path / "audit.json"),
        ]
    )

    def forbidden(*_):
        raise AssertionError("concurrent writer reached compilation")

    monkeypatch.setattr(qualified, "_run_qualification_locked", forbidden)
    with (
        exclusive_file_lock(tmp_path / "gate" / ".static-compile.lock"),
        pytest.raises(RuntimeError, match="already held"),
    ):
        qualified._run_qualification(
            args, qualified.BatchGPUQualificationStage.STATIC
        )


def test_toolchain_identity_detects_driver_and_environment_change(
    load_script, monkeypatch, tmp_path
):
    from types import SimpleNamespace

    qualified = load_script(QUALIFIED)
    compiler = tmp_path / "clang++"
    compiler.write_bytes(b"first")
    monkeypatch.delenv("HIP_CLANG_PATH", raising=False)
    monkeypatch.setattr(
        qualified,
        "_legacy",
        lambda: {
            "_required_rocm_tool": lambda _: str(compiler),
            "_run": lambda _: SimpleNamespace(stdout="version 1"),
        },
    )
    first = qualified._toolchain_identity("hipcc")
    compiler.write_bytes(b"second")
    assert first != qualified._toolchain_identity("hipcc")
    second = qualified._toolchain_identity("hipcc")
    monkeypatch.setenv("HIPCC_COMPILE_FLAGS_APPEND", "-ffast-math")
    assert second != qualified._toolchain_identity("hipcc")


def test_compile_termination_cancels_workers_before_return(
    load_script, monkeypatch, tmp_path
):
    import signal
    from concurrent.futures import CancelledError

    producer = load_script(PRODUCER)
    producer.PROBES = ({"source": "one.hip"}, {"source": "two.hip"})
    monkeypatch.setattr(producer, "compile_job_limit", lambda _: 2)
    cancelled = threading.Event()

    def terminate(*_, cancel_event, **__):
        os.kill(os.getpid(), signal.SIGTERM)
        assert cancel_event.wait(2)
        cancelled.set()
        raise CancelledError

    monkeypatch.setattr(producer, "_compile_probe", terminate)
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(SystemExit) as stopped:
        producer._compile_calibration_binaries(
            "gfx1200", tmp_path, "fake", compile_jobs=2
        )
    assert stopped.value.code == 128 + signal.SIGTERM
    assert cancelled.is_set()
    assert signal.getsignal(signal.SIGTERM) == previous
