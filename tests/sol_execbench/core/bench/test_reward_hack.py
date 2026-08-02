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


"""Tests for sol_execbench.core.bench.reward_hack."""

import json
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from sol_execbench.core.bench.reward_hack import (
    RewardHackError,
    check_eval_integrity,
    check_lazy_outputs,
    check_monkey_patch,
    check_thread_injection,
    review_solution_sources,
    snapshot_critical_functions,
)

# ── check_monkey_patch ────────────────────────────────────────────────────────


class TestCheckMonkeyPatch:
    def test_passes_when_not_patched(self):
        """No exception when elapsed_time identity is unchanged."""
        check_monkey_patch()  # must not raise

    def test_raises_when_elapsed_time_replaced(self, monkeypatch):
        """Raise RewardHackError when the captured elapsed-time identity changes."""
        from sol_execbench.core.bench import reward_hack

        if reward_hack._ELAPSED_TIME_ADDR is None:
            pytest.skip(
                "ROCm GPU unavailable; _ELAPSED_TIME_ADDR was not captured at import",
            )

        monkeypatch.setattr(
            torch.cuda.Event,
            "elapsed_time",
            lambda self, other: 0.0,
        )
        with pytest.raises(RewardHackError, match="monkey-patched"):
            check_monkey_patch()


# ── check_thread_injection ────────────────────────────────────────────────────


class TestCheckThreadInjection:
    def test_passes_when_count_unchanged(self):
        check_thread_injection(5, 5)

    def test_passes_when_count_decreases(self):
        check_thread_injection(5, 3)

    def test_raises_when_count_increases(self):
        with pytest.raises(RewardHackError, match="Thread injection"):
            check_thread_injection(3, 5)

    def test_error_message_includes_both_counts(self):
        with pytest.raises(RewardHackError) as exc:
            check_thread_injection(2, 7)
        msg = str(exc.value)
        assert "7" in msg and "2" in msg


# ── ThreadInjectionMonitor (concurrent thread-count sampling) ──────────────


class TestThreadInjectionMonitor:
    def test_rejects_nonpositive_sampling_interval(self):
        from sol_execbench.core.bench.reward_hack import ThreadInjectionMonitor

        with pytest.raises(ValueError, match="interval must be positive"):
            ThreadInjectionMonitor(interval_s=0)

    def test_no_flag_when_peak_equals_baseline(self):
        from sol_execbench.core.bench.reward_hack import (
            ThreadInjectionMonitor,
            check_thread_injection_from_monitor,
        )

        monitor = ThreadInjectionMonitor()
        monitor._baseline = 4
        monitor._peak = 4
        check_thread_injection_from_monitor(monitor)  # must not raise

    def test_flags_when_peak_exceeds_baseline(self):
        from sol_execbench.core.bench.reward_hack import (
            ThreadInjectionMonitor,
            check_thread_injection_from_monitor,
        )

        monitor = ThreadInjectionMonitor()
        monitor._baseline = 4
        monitor._peak = 7
        with pytest.raises(RewardHackError, match="peak 7"):
            check_thread_injection_from_monitor(monitor)

    def test_detects_a_real_transient_worker(self):
        """Detect a worker that exists only during the timed region.

        The concurrent sampler catches workers that a before/after delta misses.
        """
        import threading
        import time

        from sol_execbench.core.bench.reward_hack import ThreadInjectionMonitor

        monitor = ThreadInjectionMonitor(interval_s=0.002)
        with monitor:
            stop = threading.Event()

            def worker() -> None:
                stop.wait(0.2)

            workers = [
                threading.Thread(target=worker, daemon=True) for _ in range(2)
            ]
            for w in workers:
                w.start()
            time.sleep(0.05)
            stop.set()

        assert monitor.peak > monitor.baseline

    def test_event_guard_catches_worker_shorter_than_sampling_interval(self):
        """Thread starts are recorded synchronously, without a sampling gap."""
        import threading

        from sol_execbench.core.bench.reward_hack import (
            ThreadInjectionMonitor,
            check_thread_injection_from_monitor,
        )

        monitor = ThreadInjectionMonitor(interval_s=1.0)
        with monitor:
            worker = threading.Thread(target=lambda: None)
            worker.start()
            worker.join()

        assert monitor.peak == monitor.baseline
        assert monitor.starts_observed > 0
        with pytest.raises(RewardHackError, match="thread start event"):
            check_thread_injection_from_monitor(monitor)

    def test_event_guard_catches_low_level_thread_start(self):
        import _thread
        import threading

        from sol_execbench.core.bench.reward_hack import ThreadInjectionMonitor

        finished = threading.Event()
        monitor = ThreadInjectionMonitor(interval_s=1.0)
        with monitor:
            _thread.start_new_thread(finished.set, ())
            assert finished.wait(timeout=1.0)

        assert monitor.starts_observed > 0

    def test_event_guards_are_restored_after_timed_exception(self):
        import _thread
        import threading

        from sol_execbench.core.bench.reward_hack import ThreadInjectionMonitor

        original_thread_start = threading.Thread.start
        original_raw_start = _thread.start_new_thread
        with (
            pytest.raises(RuntimeError, match="timing failed"),
            ThreadInjectionMonitor(),
        ):
            raise RuntimeError("timing failed")

        assert threading.Thread.start is original_thread_start
        assert _thread.start_new_thread is original_raw_start


# ── check_lazy_outputs ────────────────────────────────────────────────────────


class TestCheckLazyOutputs:
    def test_passes_for_real_tensors(self):
        check_lazy_outputs([torch.zeros(2), torch.ones(3)])

    def test_passes_for_empty_list(self):
        check_lazy_outputs([])

    def test_raises_for_non_tensor(self):
        with pytest.raises(RewardHackError, match="Lazy evaluation"):
            check_lazy_outputs([torch.zeros(2), 42])

    def test_raises_for_none(self):
        with pytest.raises(RewardHackError, match="NoneType"):
            check_lazy_outputs([None])

    def test_raises_for_tensor_subclass(self):
        """Tensor subclasses (e.g. FakeTensor views) are rejected by strict type() check."""

        class SubTensor(torch.Tensor):
            pass

        base = torch.zeros(2)
        sub = base.as_subclass(SubTensor)
        with pytest.raises(RewardHackError):
            check_lazy_outputs([sub])

    def test_error_message_includes_actual_type_name(self):
        with pytest.raises(RewardHackError) as exc:
            check_lazy_outputs(["a string"])
        assert "str" in str(exc.value)


# ── snapshot_critical_functions / check_eval_integrity ───────────────────────


class TestEvalIntegrity:
    def test_snapshot_captures_present_names(self):
        ns = {"foo": lambda: None, "bar": 42}
        snap = snapshot_critical_functions(ns, ["foo", "bar", "missing"])
        assert "foo" in snap and "bar" in snap
        assert "missing" not in snap

    def test_passes_when_namespace_unchanged(self):
        def fn_a():
            return None

        def fn_b():
            return None

        ns = {"fn_a": fn_a, "fn_b": fn_b}
        snap = snapshot_critical_functions(ns, ["fn_a", "fn_b"])
        check_eval_integrity(snap, ns)  # must not raise

    def test_raises_when_function_replaced(self):
        def original():
            return None

        ns: dict[str, Any] = {"time_runnable": original}
        snap = snapshot_critical_functions(ns, ["time_runnable"])
        ns["time_runnable"] = lambda *a, **kw: 0.001  # attacker replaces it
        with pytest.raises(
            RewardHackError,
            match="time_runnable.*monkey-patched",
        ):
            check_eval_integrity(snap, ns)

    def test_raises_when_function_deleted(self):
        ns = {"compute_error_stats": lambda: None}
        snap = snapshot_critical_functions(ns, ["compute_error_stats"])
        del ns["compute_error_stats"]
        with pytest.raises(RewardHackError, match="compute_error_stats"):
            check_eval_integrity(snap, ns)

    def test_reports_first_violated_name(self):
        def fn():
            return None

        ns = {"a": fn, "b": fn}
        snap = snapshot_critical_functions(ns, ["a", "b"])
        ns["b"] = lambda: None  # replace only b
        with pytest.raises(RewardHackError, match="'b'"):
            check_eval_integrity(snap, ns)

    def test_empty_snapshot_always_passes(self):
        check_eval_integrity({}, {"anything": 123})

    def test_simulates_time_runnable_attack(self):
        """End-to-end: mimics the kernel.py attack via sys.modules['__main__']."""

        def real_time_runnable(fn, args, device, warmup=10, rep=50):
            return 1.0

        ns: dict[str, Any] = {"time_runnable": real_time_runnable}
        snap = snapshot_critical_functions(ns, ["time_runnable"])

        # Simulate the attack: user code wraps time_runnable
        orig = ns["time_runnable"]

        def patched(fn, args, device, warmup=None, **kwargs):
            return orig(fn, args, device)

        ns["time_runnable"] = patched

        with pytest.raises(RewardHackError, match="time_runnable"):
            check_eval_integrity(snap, ns)


# ── static source review ─────────────────────────────────────────────────────


def _solution_with_source(content: str, path: str = "kernel.py"):
    source = SimpleNamespace(path=path, content=content)
    return SimpleNamespace(sources=[source])


class TestStaticSourceReview:
    def test_reports_non_default_stream_patterns(self):
        review = review_solution_sources(
            _solution_with_source(
                "import torch\n"
                "s = torch.cuda.Stream()\n"
                "def run(x):\n"
                "    with torch.cuda.stream(s):\n"
                "        return x + 1\n",
            ),
        )

        assert review.blocked is True
        assert {issue.rule for issue in review.issues} == {
            "hidden_async_stream",
        }

    @pytest.mark.parametrize(
        "content",
        [
            "import torch\ngraph = torch.cuda.CUDAGraph()\n",
            "import torch\nwith torch.cuda.graph(torch.cuda.CUDAGraph()):\n    pass\n",
            "from torch.cuda import make_graphed_callables as graph_it\ngraph_it(lambda: None, ())\n",
            "def run(graph):\n    graph.capture_begin()\n    graph.capture_end()\n    graph.replay()\n",
            "import torch\nmake_graph = getattr(torch.cuda, 'CUDAGraph')\ngraph = make_graph()\n",
        ],
    )
    def test_reports_cuda_graph_capture_patterns(self, content):
        review = review_solution_sources(_solution_with_source(content))

        assert review.blocked is True
        assert "hidden_async_stream" in {issue.rule for issue in review.issues}

    @pytest.mark.parametrize(
        "content",
        [
            "import torch\ndef run(x):\n    torch.jit.fork(torch.relu, x)\n    return x\n",
            "import torch as t\ndef run(x):\n    t.jit.fork(torch.relu, x)\n    return x\n",
            "import torch.jit as jit\ndef run(x):\n    jit.fork(torch.relu, x)\n    return x\n",
            "from torch.jit import fork\ndef run(x):\n    fork(torch.relu, x)\n    return x\n",
        ],
    )
    def test_reports_jit_forking_concurrency_exploit(self, content):
        """Block TorchScript JIT forking as a concurrency exploit.

        ``torch.jit.fork`` runs work on a C++ thread invisible to the runtime
        thread-count guard. This covers direct calls and import aliases.
        """
        review = review_solution_sources(_solution_with_source(content))

        assert review.blocked is True
        assert "parallel_execution" in {issue.rule for issue in review.issues}

    @pytest.mark.parametrize(
        "content",
        [
            "import threading\nthreading.Thread(target=lambda: None).start()\n",
            "from _thread import start_new_thread\nstart_new_thread(lambda: None, ())\n",
            "from concurrent.futures import ThreadPoolExecutor\nThreadPoolExecutor()\n",
            "import multiprocessing\nmultiprocessing.Process(target=lambda: None)\n",
        ],
    )
    def test_blocks_python_thread_creation_sources(self, content):
        review = review_solution_sources(_solution_with_source(content))

        assert review.blocked is True
        assert "parallel_execution" in {issue.rule for issue in review.issues}

    def test_reports_data_pointer_cache_patterns(self):
        review = review_solution_sources(
            _solution_with_source(
                "_cache = {}\n"
                "def run(x):\n"
                "    key = x.data_ptr()\n"
                "    return _cache.setdefault(key, x + 1)\n",
            ),
        )

        assert review.blocked is True
        assert "semantic_output_cache" in {
            issue.rule for issue in review.issues
        }

    def test_reports_unauthorized_file_and_loader_patterns(self):
        review = review_solution_sources(
            _solution_with_source(
                "import base64, ctypes\n"
                "payload = base64.b64decode('AA==')\n"
                "lib = ctypes.CDLL('/tmp/libx.so')\n"
                "def run(x):\n"
                "    return x\n",
            ),
        )

        assert review.blocked is True
        assert "unauthorized_file_or_loader" in {
            issue.rule for issue in review.issues
        }

    def test_reports_os_process_execution_patterns(self):
        review = review_solution_sources(
            _solution_with_source(
                "import os\ndef run(x):\n    os.system('true')\n    return x\n",
            ),
        )

        assert review.blocked is True
        assert "unauthorized_file_or_loader" in {
            issue.rule for issue in review.issues
        }

    def test_reports_dynamic_import_process_execution_patterns(self):
        review = review_solution_sources(
            _solution_with_source(
                "def run(x):\n    __import__('os').system('true')\n    return x\n",
            ),
        )

        assert review.blocked is True
        assert "unauthorized_file_or_loader" in {
            issue.rule for issue in review.issues
        }

    def test_reports_getattr_os_process_execution_patterns(self):
        review = review_solution_sources(
            _solution_with_source(
                "import os\n"
                "def run(x):\n"
                "    getattr(os, 'system')('true')\n"
                "    return x\n",
            ),
        )

        assert review.blocked is True
        assert "unauthorized_file_or_loader" in {
            issue.rule for issue in review.issues
        }

    def test_reports_getattr_dynamic_import_process_execution_patterns(self):
        review = review_solution_sources(
            _solution_with_source(
                "def run(x):\n"
                "    getattr(__import__('os'), 'system')('true')\n"
                "    return x\n",
            ),
        )

        assert review.blocked is True
        assert "unauthorized_file_or_loader" in {
            issue.rule for issue in review.issues
        }

    def test_reports_getattr_indirect_stream_creation(self):
        """getattr(torch.cuda, 'Stream') bypasses the direct-name stream check."""
        review = review_solution_sources(
            _solution_with_source(
                "import torch\n"
                "def run(x):\n"
                "    make_stream = getattr(torch.cuda, 'Stream')\n"
                "    with make_stream():\n"
                "        return x + 1\n",
            ),
        )

        assert review.blocked is True
        assert "hidden_async_stream" in {issue.rule for issue in review.issues}

    @pytest.mark.parametrize(
        "content",
        [
            "import pickle\npayload = pickle.loads(b'0')\ndef run(x):\n    return x\n",
            "import importlib\nos = importlib.import_module('os')\ndef run(x):\n    return x\n",
            "import socket\ndef run(x):\n    return x\n",
            "from pathlib import Path\ndef run(x):\n    return Path('/tmp/x').read_text()\n",
            "import torch\ndef run(x):\n    torch.ops.load_library('/tmp/libx.so')\n    return x\n",
        ],
    )
    def test_reports_additional_file_network_loader_bypass_families(
        self,
        content,
    ):
        review = review_solution_sources(_solution_with_source(content))

        assert review.blocked is True
        assert "unauthorized_file_or_loader" in {
            issue.rule for issue in review.issues
        }

    @pytest.mark.parametrize(
        "content",
        [
            "import subprocess\ndef run(x):\n    return x\n",
            "def run(x):\n    return eval('1')\n",
            "def run(x):\n    return compile('1', '<x>', 'eval')\n",
            "import ctypes\nlib = ctypes.cdll.LoadLibrary('/tmp/libx.so')\n",
            "import urllib.request\ndef run(x):\n    return urllib.request.urlopen('http://localhost')\n",
            "import requests\ndef run(x):\n    return requests.request('GET', 'http://localhost')\n",
            "import torch\ns = torch.cuda.ExternalStream(1)\ndef run(x):\n    return x\n",
            "def run(x):\n    cache = {}\n    return cache.setdefault(x.data_ptr(), x)\n",
            "import os as runtime_os\ndef run(x):\n    runtime_os.system('true')\n    return x\n",
            "from os import system as run_process\ndef run(x):\n    run_process('true')\n    return x\n",
            "import torch as t\ns = t.cuda.Stream()\ndef run(x):\n    return x\n",
            "from torch.cuda import Stream as HiddenStream\ns = HiddenStream()\n",
        ],
    )
    def test_reports_ast_detected_bypass_families(self, content):
        review = review_solution_sources(_solution_with_source(content))

        assert review.blocked is True

    def test_ignores_blocked_words_inside_python_strings_and_comments(self):
        review = review_solution_sources(
            _solution_with_source(
                "# subprocess open('/tmp/x') torch.cuda.Stream()\n"
                "def run(x):\n"
                "    note = \"ctypes.CDLL('/tmp/x') and x.data_ptr()\"\n"
                "    return x + 1\n",
            ),
            output_dtypes={"out": torch.float32},
        )

        assert review.blocked is False

    def test_allows_triton_language_load_kernel_memory_reads(self):
        review = review_solution_sources(
            _solution_with_source(
                "import triton\n"
                "import triton.language as tl\n"
                "@triton.jit\n"
                "def kernel(x, offs):\n"
                "    return tl.load(x + offs)\n",
            ),
        )

        assert review.blocked is False

    def test_still_blocks_non_triton_load_methods(self):
        review = review_solution_sources(
            _solution_with_source(
                "class Loader:\n"
                "    def load(self):\n"
                "        return 1\n"
                "def run(x):\n"
                "    return Loader().load()\n",
            ),
        )

        assert review.blocked is True
        assert "unauthorized_file_or_loader" in {
            issue.rule for issue in review.issues
        }

    def test_blocking_message_includes_structured_review_evidence(self):
        review = review_solution_sources(
            _solution_with_source(
                "def run(x):\n    return open('/tmp/x').read()\n",
            ),
        )

        message = review.format_blocking_message()
        payload = message.split("structured_evidence=", maxsplit=1)[1]

        assert review.blocked is True
        assert json.loads(payload) == review.to_dict()

    def test_allows_plain_os_import_without_process_execution(self):
        review = review_solution_sources(
            _solution_with_source(
                "import os\n"
                "def run(x):\n"
                "    _ = os.environ.get('UNUSED', '')\n"
                "    return x\n",
            ),
        )

        assert review.blocked is False

    def test_flags_precision_downgrade_for_float32_outputs_without_blocking(
        self,
    ):
        review = review_solution_sources(
            _solution_with_source("def run(x):\n    return x.half().float()\n"),
            output_dtypes={"out": torch.float32},
        )

        assert review.blocked is False
        assert [issue.rule for issue in review.issues] == [
            "precision_downgrade",
        ]
        assert {issue.severity for issue in review.issues} == {"flag"}

    def test_flags_precision_downgrade_dtype_keyword_for_float32_outputs(self):
        review = review_solution_sources(
            _solution_with_source(
                "import torch\ndef run(x):\n    return x.to(dtype=torch.float16)\n",
            ),
            output_dtypes={"out": torch.float32},
        )

        assert review.blocked is False
        assert "precision_downgrade" in {issue.rule for issue in review.issues}

    def test_allows_precision_casts_when_outputs_are_mixed_precision(self):
        review = review_solution_sources(
            _solution_with_source(
                "import torch\n"
                "def run(grad_output, weight):\n"
                "    grad_x = grad_output.float() * weight\n"
                "    grad_x_bf16 = grad_x.to(torch.bfloat16)\n"
                "    grad_weight = grad_output.float().sum(dim=0)\n"
                "    return grad_x_bf16, grad_weight\n",
            ),
            output_dtypes={
                "grad_x": torch.bfloat16,
                "grad_weight": torch.float32,
            },
        )

        assert review.blocked is False
        assert review.to_dict() == {"blocked": False, "issues": []}

    def test_allows_model_kv_cache_tensor_assignments(self):
        review = review_solution_sources(
            _solution_with_source(
                "import torch\n"
                "def get_inputs(axes, device):\n"
                "    cache_len = axes['cache_len']\n"
                "    grad_key_cache = torch.randn(1, 1, 16, 64, device=device)\n"
                "    return {'grad_key_cache': grad_key_cache, 'cache_len': cache_len}\n"
                "def run(key_cache, key_states):\n"
                "    updated_key_cache = torch.cat([key_cache, key_states], dim=2)\n"
                "    return updated_key_cache\n",
            ),
            output_dtypes={"updated_key_cache": torch.bfloat16},
        )

        assert review.blocked is False
        assert review.to_dict() == {"blocked": False, "issues": []}

    def test_allows_legitimate_torch_compile_solution(self):
        review = review_solution_sources(
            _solution_with_source(
                "import torch\n@torch.compile\ndef run(x, y):\n    return x + y\n",
            ),
            output_dtypes={"z": torch.float32},
        )

        assert review.blocked is False
        assert review.to_dict() == {"blocked": False, "issues": []}

    def test_allows_hip_current_stream_text(self):
        review = review_solution_sources(
            _solution_with_source(
                "auto stream = at::cuda::getCurrentCUDAStream();\n"
                "kernel<<<grid, block, 0, stream>>>(args);\n",
                path="kernel.hip",
            ),
        )

        assert review.blocked is False

    def test_allows_native_data_ptr_for_library_calls(self):
        review = review_solution_sources(
            _solution_with_source(
                "hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,\n"
                "             n, m, k, &alpha, b.data_ptr<float>(), n,\n"
                "             a.data_ptr<float>(), k, &beta,\n"
                "             out.data_ptr<float>(), n);\n",
                path="main.cpp",
            ),
        )

        assert review.blocked is False


# ── sealed timing / integrity guards (dlopen-b1, static-b2) ──────────────────


class TestSealedIntegrityGuards:
    """Regression coverage for the immutable strong-reference guards.

    These replace the former ``id()``-in-a-mutable-global pattern that a native
    dlopen constructor or a ``sys.modules['__main__']`` write could defeat
    (audit findings ``runtime.py:40`` / dlopen-b1 and ``runtime.py:322`` /
    static-b2).
    """

    def test_sealed_reference_rejects_attribute_mutation(self):
        from sol_execbench.core.bench.reward_hack.runtime import (
            _SealedReference,
        )

        guard = _SealedReference(len)
        with pytest.raises(AttributeError):
            guard._reference = print  # type: ignore[misc]
        with pytest.raises(AttributeError):
            del guard._reference
        assert guard.is_intact(len)
        assert not guard.is_intact(print)

    def test_snapshot_is_immutable_to_attribute_or_item_write(self):
        from sol_execbench.core.bench.reward_hack.runtime import (
            _IntegritySnapshot,
        )

        def original() -> None:
            return None

        snapshot = snapshot_critical_functions({"fn": original}, ["fn"])
        assert isinstance(snapshot, _IntegritySnapshot)
        # A candidate reaching the snapshot via __main__ cannot rewrite the
        # stored reference to match a patched live function.
        with pytest.raises(AttributeError):
            snapshot._references = {"fn": print}  # type: ignore[misc]
        with pytest.raises(AttributeError):
            del snapshot._references
        with pytest.raises(TypeError):
            snapshot["fn"] = print  # ty: ignore[invalid-assignment]  # read-only

    def test_snapshot_holds_strong_reference_not_id(self):
        """Strong ref survives deletion + gc; an id()-based guard would not."""
        import gc

        def original() -> str:
            return "real"

        ns = {"fn": original}
        snapshot = snapshot_critical_functions(ns, ["fn"])
        # Drop every other reference to the function and collect garbage so its
        # id() address may be reused by a later allocation.
        del ns["fn"]
        del original
        gc.collect()
        # The captured reference is still alive and identity still resolves.
        ns_after = {"fn": snapshot.reference_of("fn")}
        check_eval_integrity(snapshot, ns_after)  # must not raise

    def test_check_uses_identity_not_id_value(self):
        """Replacement is detected even if a new object reuses the address."""

        def real() -> None:
            return None

        ns: dict[str, Any] = {"fn": real}
        snapshot = snapshot_critical_functions(ns, ["fn"])
        ns["fn"] = lambda: None  # attacker swaps in a fresh callable
        with pytest.raises(RewardHackError, match="fn"):
            check_eval_integrity(snapshot, ns)

    def test_verify_timing_function_intact_is_callable(self):
        from sol_execbench.core.bench.reward_hack import (
            verify_timing_function_intact,
        )

        # CPU-only target: guard no-ops. GPU target: pristine fn is intact.
        verify_timing_function_intact()  # must not raise


# ── static-review bypass closures (static-b1/b6/b7/b8) ───────────────────────


class TestStaticBypassClosures:
    """Block the five static-obfuscation families the follow-up audit confirmed.

    Each parametrized snippet is a representative exploit primitive that must be
    rejected with ``unauthorized_file_or_loader`` (audit static-b1/b6/b7/b8).
    """

    @pytest.mark.parametrize(
        "content",
        [
            # static-b8: trace-forgery fd primitives + native fd/mapping modules
            "import os\ndef run(x):\n    os.write(3, b'forged')\n    return x\n",
            "import os\ndef run(x):\n    os.dup(1)\n    return x\n",
            "import os\ndef run(x):\n    os.fdopen(3, 'w')\n    return x\n",
            "import mmap\ndef run(x):\n    mmap.mmap(-1, 4096)\n    return x\n",
            "import fcntl\ndef run(x):\n    fcntl.fcntl(1, 0)\n    return x\n",
            # static-b1: __builtins__ re-exposes __import__/eval after blocking
            "def run(x):\n    return __builtins__.__import__('os')\n",
            "def run(x):\n    return __builtins__['__import__']('os')\n",
            # static-b6: types constructs callables from raw bytecode
            "import types\ndef run(x):\n    return types.FunctionType(x, {})\n",
            "import types\ndef run(x):\n    return types.CodeType(*x)\n",
            # static-b7: os.__dict__ / os.__getattribute__ restore os.system
            "import os\ndef run(x):\n    return os.__dict__['system']('true')\n",
            "import os\ndef run(x):\n    return os.__getattribute__(os, 'system')('true')\n",
            "import os\ndef run(x):\n    return getattr(os, '__dict__')['system']('true')\n",
        ],
    )
    def test_blocks_obfuscation_primitives(self, content):
        review = review_solution_sources(_solution_with_source(content))
        assert review.blocked is True
        assert "unauthorized_file_or_loader" in {
            issue.rule for issue in review.issues
        }

    @pytest.mark.parametrize(
        "path",
        ["kernel.hip", "main.cpp", "kernel.cu"],
    )
    def test_blocks_native_process_and_loader_primitives(self, path):
        """HIP/C++ sources must be reviewed for loader/process primitives too."""
        review = review_solution_sources(
            _solution_with_source(
                'extern "C" void run() {\n'
                '    system("rm -rf /");\n'
                '    popen("ls", "r");\n'
                '    dlopen("/tmp/x.so", 1);\n'
                "}\n",
                path=path,
            ),
        )
        assert review.blocked is True
        assert "unauthorized_file_or_loader" in {
            issue.rule for issue in review.issues
        }

    def test_native_data_ptr_still_allowed_in_cpp(self):
        """Tightened rules must not block legitimate C++ library data_ptr calls."""
        review = review_solution_sources(
            _solution_with_source(
                'extern "C" void run(float* a) {\n'
                "    hipblasSgemm(handle, m, n, k, a);\n"
                "}\n",
                path="main.cpp",
            ),
        )
        assert review.blocked is False


# ── value-keyed output-cache closure (seed-seed2) ────────────────────────────


class TestValueKeyedCacheClosure:
    """Block value-keyed output caches (unbounded decorators + population writes).

    Audit seed-seed2: the memory pool already varies ``data_ptr`` per iteration
    but re-presents byte-identical VALUES, so a value-keyed cache collapses
    in-trial latency. Static detection closes the construction surface (the
    within-trial identical-values residual is documented as boundary-constrained).
    """

    @pytest.mark.parametrize(
        "content",
        [
            # cache POPULATION: the write that makes a value/data-keyed cache work
            (
                "def run(x):\n"
                "    key = x.sum().item()\n"
                "    if key not in _cache:\n"
                "        _cache[key] = x + 1\n"
                "    return _cache[key]\n"
                "_cache = {}\n"
            ),
            (
                "class K:\n"
                "    def run(self, x):\n"
                "        self.memo[x.int().sum().item()] = x + 1\n"
                "        return self.memo[x.int().sum().item()]\n"
            ),
            # unbounded value-keyed decorators (sibling of lru_cache)
            (
                "import functools\n"
                "@functools.cache\n"
                "def run(x):\n"
                "    return x + 1\n"
            ),
            (
                "from functools import cache\n"
                "@cache\n"
                "def run(x):\n"
                "    return x + 1\n"
            ),
            ("import functools\nrun = functools.cache(lambda x: x + 1)\n"),
        ],
    )
    def test_blocks_value_keyed_cache_construction(self, content):
        review = review_solution_sources(_solution_with_source(content))
        assert review.blocked is True
        assert "semantic_output_cache" in {
            issue.rule for issue in review.issues
        }

    def test_legitimate_tensor_subscript_write_is_allowed(self):
        """A normal output-element write must not be mistaken for a cache write."""
        review = review_solution_sources(
            _solution_with_source(
                "import torch\n"
                "def run(x):\n"
                "    out = torch.empty_like(x)\n"
                "    out[0] = x[0] + 1\n"
                "    out[1:] = x[1:]\n"
                "    return out\n",
            ),
        )
        assert "semantic_output_cache" not in {
            issue.rule for issue in review.issues
        }
