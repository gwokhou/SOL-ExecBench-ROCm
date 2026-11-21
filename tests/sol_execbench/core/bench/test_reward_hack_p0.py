# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""P0 coverage for sealed runtime guards, native thread detection, and static review."""

from __future__ import annotations

import ctypes
import os
import time
from types import SimpleNamespace

import pytest
import torch

from sol_execbench.core.bench.reference_protocol import ReferenceClient
from sol_execbench.core.bench.reward_hack import (
    RewardHackError,
    ThreadInjectionMonitor,
    check_monkey_patch,
    check_runtime_integrity,
    check_thread_injection_from_monitor,
    review_solution_sources,
    snapshot_runtime_integrity,
)


def _solution(content: str, path: str = "kernel.py"):
    return SimpleNamespace(
        sources=[SimpleNamespace(path=path, content=content)]
    )


class TestSealedRuntimeTargets:
    @pytest.mark.parametrize(
        ("owner_fn", "attr"),
        [
            pytest.param(lambda: torch.cuda, "Event", id="cuda_event_cls"),
            pytest.param(lambda: torch.cuda.Event, "record", id="event_record"),
            pytest.param(
                lambda: torch.cuda.Event,
                "synchronize",
                id="event_synchronize",
            ),
            pytest.param(
                lambda: torch.cuda.Event,
                "elapsed_time",
                id="event_elapsed_time",
            ),
            pytest.param(
                lambda: torch.cuda, "synchronize", id="cuda_synchronize"
            ),
            pytest.param(
                lambda: torch.cuda, "current_device", id="cuda_current_device"
            ),
            pytest.param(
                lambda: torch.cuda, "set_device", id="cuda_set_device"
            ),
            pytest.param(lambda: torch.Tensor, "zero_", id="tensor_zero"),
            pytest.param(
                lambda: ReferenceClient,
                "validate_timing_outputs",
                id="ref_client_validate",
            ),
            pytest.param(
                lambda: ReferenceClient,
                "timing_iteration_case",
                id="ref_client_timing_iter",
            ),
            pytest.param(
                lambda: ReferenceClient,
                "correctness_case",
                id="ref_client_correctness",
            ),
        ],
    )
    def test_check_monkey_patch_and_runtime_integrity_detect_replacements(
        self,
        monkeypatch: pytest.MonkeyPatch,
        owner_fn,
        attr: str,
    ) -> None:
        owner = owner_fn()
        if getattr(owner, attr, None) is None:
            pytest.skip(f"{attr} unavailable on target")
        snapshot = snapshot_runtime_integrity({})
        monkeypatch.setattr(owner, attr, lambda *args, **kwargs: None)

        with pytest.raises(RewardHackError, match="monkey-patched"):
            check_monkey_patch()
        with pytest.raises(RewardHackError, match="monkey-patched"):
            check_runtime_integrity(snapshot, {})

    def test_device_event_base_methods_are_sealed_when_present(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        base = getattr(torch._C, "_CudaEventBase", None)
        if base is None:
            pytest.skip("_CudaEventBase unavailable")
        try:
            monkeypatch.setattr(base, "record", lambda self, *a: None)
        except TypeError:
            # Immutable C-level tp_dict is inherently immune to Python setattr.
            return
        with pytest.raises(RewardHackError, match="monkey-patched"):
            check_monkey_patch()


class TestNativeThreadMonitor:
    @pytest.mark.skipif(
        not os.path.isdir("/proc/self/task"),
        reason="Linux /proc/self/task required",
    )
    def test_monitor_catches_native_pthread_invisible_to_python_threading(
        self,
    ) -> None:
        libc = ctypes.CDLL("libc.so.6")
        pthread_t = ctypes.c_ulong()
        release_flag = ctypes.c_int(0)

        @ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)
        def native_worker(_arg: int) -> int:
            # Spin briefly with nanosleep/usleep via libc so GIL is unnecessary
            while release_flag.value == 0:
                libc.usleep(1000)
            return 0

        monitor = ThreadInjectionMonitor(interval_s=0.002)
        with monitor:
            rc = libc.pthread_create(
                ctypes.byref(pthread_t),
                None,
                native_worker,
                None,
            )
            assert rc == 0
            time.sleep(0.03)
            release_flag.value = 1
            libc.pthread_join(pthread_t, None)

        assert monitor.starts_observed == 0  # Python hooks were bypassed
        assert monitor.peak > monitor.baseline
        with pytest.raises(RewardHackError, match="Thread injection detected"):
            check_thread_injection_from_monitor(monitor)


class TestStaticReviewP0Closures:
    @pytest.mark.parametrize(
        ("code", "expected_rule"),
        [
            (
                "import torch\ntorch.cuda.synchronize = lambda *a: None\ndef run(x): return x\n",
                "runtime_monkey_patch",
            ),
            (
                "import torch\nsetattr(torch.cuda.Event, 'record', lambda s: None)\ndef run(x): return x\n",
                "runtime_monkey_patch",
            ),
            (
                "import torch\nclass Sneaky(torch.Tensor):\n    pass\ndef run(x): return x\n",
                "lazy_output_interception",
            ),
            (
                "class Proxy:\n    @classmethod\n    def __torch_function__(cls, f, t, a=(), k=None):\n        pass\n",
                "lazy_output_interception",
            ),
            (
                "class Proxy:\n    @classmethod\n    def __torch_dispatch__(cls, f, t, a=(), k=None):\n        pass\n",
                "lazy_output_interception",
            ),
            (
                "import torch\ndef run(x):\n    return torch.Tensor._make_subclass(torch.Tensor, x)\n",
                "lazy_output_interception",
            ),
            (
                "import io\ndef run(x):\n    io.open(3, 'w').write('{}')\n    return x\n",
                "unauthorized_file_or_loader",
            ),
            (
                "def run(x):\n    raise SystemExit(0)\n",
                "unauthorized_file_or_loader",
            ),
            (
                "def run(x):\n    exit(0)\n",
                "unauthorized_file_or_loader",
            ),
            (
                "def run(x):\n    quit(0)\n",
                "unauthorized_file_or_loader",
            ),
        ],
    )
    def test_python_exploit_patterns_are_blocked(
        self,
        code: str,
        expected_rule: str,
    ) -> None:
        review = review_solution_sources(_solution(code))
        assert review.blocked is True
        assert expected_rule in {issue.rule for issue in review.issues}

    @pytest.mark.parametrize(
        "path",
        ["os.py", "io.py", "torch.py", "sol_execbench/evil.py"],
    )
    def test_protected_module_names_are_blocked(self, path: str) -> None:
        review = review_solution_sources(
            _solution("def run(x):\n    return x\n", path=path),
        )
        assert review.blocked is True
        assert "unauthorized_file_or_loader" in {
            issue.rule for issue in review.issues
        }

    @pytest.mark.parametrize(
        "cpp_source",
        [
            'void f() { PyRun_SimpleString("import os"); }\n',
            'void f() { PyImport_ImportModule("torch"); }\n',
            'void f() { PyObject_SetAttrString(0, "x", 0); }\n',
            "void f() { PyEval_GetBuiltins(); }\n",
            'void* f() { return dlsym(0, "hipStreamCreate"); }\n',
            'void f() { syscall(1, 3, "{}", 2); }\n',
            "void f() { _exit(0); }\n",
            "void f() { quick_exit(0); }\n",
            'void f() { dprintf(3, "{}\\n"); }\n',
            'void f() { write(3, "{}", 2); }\n',
            'void f() { fopen("/tmp/x", "w"); }\n',
            'void f() { std::ofstream out("/tmp/x"); }\n',
            "#define CALL(a, b) a##b\nvoid f() {}\n",
            'void* f() { return dl\\\nsym(0, "x"); }\n',
        ],
    )
    def test_cpp_trace_forgery_and_preprocessor_obfuscation_blocked(
        self,
        cpp_source: str,
    ) -> None:
        review = review_solution_sources(
            _solution(cpp_source, path="kernel.hip"),
        )
        assert review.blocked is True
        assert "unauthorized_file_or_loader" in {
            issue.rule for issue in review.issues
        }

    def test_cpp_block_and_line_comments_do_not_false_positive(self) -> None:
        cpp_source = (
            "/* dlsym PyRun_SimpleString hipStreamCreate */\n"
            "// write(3, buf, 1) fopen _exit(0)\n"
            "void run() {}\n"
        )
        review = review_solution_sources(
            _solution(cpp_source, path="kernel.hip"),
        )
        assert review.blocked is False
