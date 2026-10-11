# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Interrupt an owned cpp_extension build and retry its exact build directory."""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.process.logs import redacted_text_tail
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded

INTERRUPT_SECONDS = 10
RETRY_SECONDS = 120
MAX_OUTPUT_BYTES = 8192
LOAD_PROGRAM = """
import ctypes, sys
from torch.utils.cpp_extension import load
path = load(name='upgrade_lock_probe', sources=[sys.argv[1]],
            build_directory=sys.argv[2], is_python_module=False, verbose=True)
assert ctypes.CDLL(path).upgrade_probe_answer() == 42
"""
COMPILER_WRAPPER = """#!/bin/sh
case " $* " in
  *" -c "*) touch "$UPGRADE_COMPILE_MARKER"; sleep 90 ;;
esac
exec /usr/bin/c++ "$@"
"""


def probe(directory: Path) -> dict[str, object]:
    """Only signal children launched by the bounded process-group runner."""
    wrapper = directory / "compiler"
    wrapper.write_text(COMPILER_WRAPPER)
    wrapper.chmod(0o700)
    marker = directory / "compile-entered"
    source = Path(__file__).with_name("extension_probe.cpp").resolve()
    command = [sys.executable, "-c", LOAD_PROGRAM, str(source), str(directory)]
    environment = {
        **os.environ,
        "CXX": str(wrapper),
        "UPGRADE_COMPILE_MARKER": str(marker),
    }
    interrupted = False
    try:
        run_in_process_group_bounded(
            command,
            env=environment,
            timeout=INTERRUPT_SECONDS,
            max_capture_bytes=MAX_OUTPUT_BYTES,
        )
    except subprocess.TimeoutExpired:
        interrupted = marker.is_file()
    if not interrupted:
        return {
            "status": "incomplete",
            "reason": "compiler_interruption_not_observed",
        }
    result = run_in_process_group_bounded(
        command,
        env={**os.environ, "CXX": "/usr/bin/c++"},
        timeout=RETRY_SECONDS,
        max_capture_bytes=MAX_OUTPUT_BYTES,
    )
    return {
        "status": "passed" if result.returncode == 0 else "failed",
        "interrupted_inside_compile": True,
        "same_build_directory": True,
        "retry_returncode": result.returncode,
        "retry_stderr": redacted_text_tail(result.stderr, MAX_OUTPUT_BYTES),
        "scope": "CPU cpp_extension lock recovery, not HIP compiler correctness",
    }


def main() -> None:
    """Save an engineering receipt; temporary files belong solely to this probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="upgrade-extension-") as temporary:
        try:
            result = probe(Path(temporary))
        except subprocess.TimeoutExpired:
            result = {"status": "failed", "reason": "retry_timeout"}
    atomic_write_json_value(args.output, result)


if __name__ == "__main__":
    main()
