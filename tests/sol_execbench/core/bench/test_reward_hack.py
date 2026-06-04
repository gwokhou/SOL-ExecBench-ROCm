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
    RewardHackDetected,
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
        """RewardHackDetected when elapsed_time identity differs from the captured address."""
        from sol_execbench.core.bench import reward_hack

        if reward_hack._ELAPSED_TIME_ADDR is None:
            pytest.skip(
                "ROCm GPU unavailable; _ELAPSED_TIME_ADDR was not captured at import"
            )

        monkeypatch.setattr(torch.cuda.Event, "elapsed_time", lambda self, other: 0.0)
        with pytest.raises(RewardHackDetected, match="monkey-patched"):
            check_monkey_patch()


# ── check_thread_injection ────────────────────────────────────────────────────


class TestCheckThreadInjection:
    def test_passes_when_count_unchanged(self):
        check_thread_injection(5, 5)

    def test_passes_when_count_decreases(self):
        check_thread_injection(5, 3)

    def test_raises_when_count_increases(self):
        with pytest.raises(RewardHackDetected, match="Thread injection"):
            check_thread_injection(3, 5)

    def test_error_message_includes_both_counts(self):
        with pytest.raises(RewardHackDetected) as exc:
            check_thread_injection(2, 7)
        msg = str(exc.value)
        assert "7" in msg and "2" in msg


# ── check_lazy_outputs ────────────────────────────────────────────────────────


class TestCheckLazyOutputs:
    def test_passes_for_real_tensors(self):
        check_lazy_outputs([torch.zeros(2), torch.ones(3)])

    def test_passes_for_empty_list(self):
        check_lazy_outputs([])

    def test_raises_for_non_tensor(self):
        with pytest.raises(RewardHackDetected, match="Lazy evaluation"):
            check_lazy_outputs([torch.zeros(2), 42])

    def test_raises_for_none(self):
        with pytest.raises(RewardHackDetected, match="NoneType"):
            check_lazy_outputs([None])

    def test_raises_for_tensor_subclass(self):
        """Tensor subclasses (e.g. FakeTensor views) are rejected by strict type() check."""

        class SubTensor(torch.Tensor):
            pass

        base = torch.zeros(2)
        sub = base.as_subclass(SubTensor)
        with pytest.raises(RewardHackDetected):
            check_lazy_outputs([sub])

    def test_error_message_includes_actual_type_name(self):
        with pytest.raises(RewardHackDetected) as exc:
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
        fn_a = lambda: None  # noqa: E731
        fn_b = lambda: None  # noqa: E731
        ns = {"fn_a": fn_a, "fn_b": fn_b}
        snap = snapshot_critical_functions(ns, ["fn_a", "fn_b"])
        check_eval_integrity(snap, ns)  # must not raise

    def test_raises_when_function_replaced(self):
        original = lambda: None  # noqa: E731
        ns: dict[str, Any] = {"time_runnable": original}
        snap = snapshot_critical_functions(ns, ["time_runnable"])
        ns["time_runnable"] = lambda *a, **kw: 0.001  # attacker replaces it
        with pytest.raises(RewardHackDetected, match="time_runnable.*monkey-patched"):
            check_eval_integrity(snap, ns)

    def test_raises_when_function_deleted(self):
        ns = {"compute_error_stats": lambda: None}
        snap = snapshot_critical_functions(ns, ["compute_error_stats"])
        del ns["compute_error_stats"]
        with pytest.raises(RewardHackDetected, match="compute_error_stats"):
            check_eval_integrity(snap, ns)

    def test_reports_first_violated_name(self):
        fn = lambda: None  # noqa: E731
        ns = {"a": fn, "b": fn}
        snap = snapshot_critical_functions(ns, ["a", "b"])
        ns["b"] = lambda: None  # replace only b
        with pytest.raises(RewardHackDetected, match="'b'"):
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

        with pytest.raises(RewardHackDetected, match="time_runnable"):
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
                "        return x + 1\n"
            )
        )

        assert review.blocked is True
        assert {issue.rule for issue in review.issues} == {"hidden_async_stream"}

    def test_reports_data_pointer_cache_patterns(self):
        review = review_solution_sources(
            _solution_with_source(
                "_cache = {}\n"
                "def run(x):\n"
                "    key = x.data_ptr()\n"
                "    return _cache.setdefault(key, x + 1)\n"
            )
        )

        assert review.blocked is True
        assert "semantic_output_cache" in {issue.rule for issue in review.issues}

    def test_reports_unauthorized_file_and_loader_patterns(self):
        review = review_solution_sources(
            _solution_with_source(
                "import base64, ctypes\n"
                "payload = base64.b64decode('AA==')\n"
                "lib = ctypes.CDLL('/tmp/libx.so')\n"
                "def run(x):\n"
                "    return x\n"
            )
        )

        assert review.blocked is True
        assert "unauthorized_file_or_loader" in {issue.rule for issue in review.issues}

    def test_reports_os_process_execution_patterns(self):
        review = review_solution_sources(
            _solution_with_source(
                "import os\n"
                "def run(x):\n"
                "    os.system('true')\n"
                "    return x\n"
            )
        )

        assert review.blocked is True
        assert "unauthorized_file_or_loader" in {issue.rule for issue in review.issues}

    def test_reports_dynamic_import_process_execution_patterns(self):
        review = review_solution_sources(
            _solution_with_source(
                "def run(x):\n"
                "    __import__('os').system('true')\n"
                "    return x\n"
            )
        )

        assert review.blocked is True
        assert "unauthorized_file_or_loader" in {issue.rule for issue in review.issues}

    def test_reports_getattr_os_process_execution_patterns(self):
        review = review_solution_sources(
            _solution_with_source(
                "import os\n"
                "def run(x):\n"
                "    getattr(os, 'system')('true')\n"
                "    return x\n"
            )
        )

        assert review.blocked is True
        assert "unauthorized_file_or_loader" in {issue.rule for issue in review.issues}

    def test_reports_getattr_dynamic_import_process_execution_patterns(self):
        review = review_solution_sources(
            _solution_with_source(
                "def run(x):\n"
                "    getattr(__import__('os'), 'system')('true')\n"
                "    return x\n"
            )
        )

        assert review.blocked is True
        assert "unauthorized_file_or_loader" in {issue.rule for issue in review.issues}

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
    def test_reports_additional_file_network_loader_bypass_families(self, content):
        review = review_solution_sources(_solution_with_source(content))

        assert review.blocked is True
        assert "unauthorized_file_or_loader" in {issue.rule for issue in review.issues}

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
                "    return x + 1\n"
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
                "    return tl.load(x + offs)\n"
            )
        )

        assert review.blocked is False

    def test_still_blocks_non_triton_load_methods(self):
        review = review_solution_sources(
            _solution_with_source(
                "class Loader:\n"
                "    def load(self):\n"
                "        return 1\n"
                "def run(x):\n"
                "    return Loader().load()\n"
            )
        )

        assert review.blocked is True
        assert "unauthorized_file_or_loader" in {issue.rule for issue in review.issues}

    def test_blocking_message_includes_structured_review_evidence(self):
        review = review_solution_sources(
            _solution_with_source("def run(x):\n    return open('/tmp/x').read()\n")
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
                "    return x\n"
            )
        )

        assert review.blocked is False

    def test_reports_precision_downgrade_for_float32_outputs(self):
        review = review_solution_sources(
            _solution_with_source("def run(x):\n    return x.half().float()\n"),
            output_dtypes={"out": torch.float32},
        )

        assert review.blocked is True
        assert "precision_downgrade" in {issue.rule for issue in review.issues}

    def test_reports_precision_downgrade_dtype_keyword_for_float32_outputs(self):
        review = review_solution_sources(
            _solution_with_source(
                "import torch\n"
                "def run(x):\n"
                "    return x.to(dtype=torch.float16)\n"
            ),
            output_dtypes={"out": torch.float32},
        )

        assert review.blocked is True
        assert "precision_downgrade" in {issue.rule for issue in review.issues}

    def test_allows_precision_casts_when_outputs_are_mixed_precision(self):
        review = review_solution_sources(
            _solution_with_source(
                "import torch\n"
                "def run(grad_output, weight):\n"
                "    grad_x = grad_output.float() * weight\n"
                "    grad_x_bf16 = grad_x.to(torch.bfloat16)\n"
                "    grad_weight = grad_output.float().sum(dim=0)\n"
                "    return grad_x_bf16, grad_weight\n"
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
                "    grad_key_cache = torch.randn(1, 1, 16, 64, device=device)\n"
                "    return {'grad_key_cache': grad_key_cache}\n"
                "def run(key_cache, key_states):\n"
                "    updated_key_cache = torch.cat([key_cache, key_states], dim=2)\n"
                "    return updated_key_cache\n"
            ),
            output_dtypes={"updated_key_cache": torch.bfloat16},
        )

        assert review.blocked is False
        assert review.to_dict() == {"blocked": False, "issues": []}

    def test_allows_legitimate_torch_compile_solution(self):
        review = review_solution_sources(
            _solution_with_source(
                "import torch\n@torch.compile\ndef run(x, y):\n    return x + y\n"
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
            )
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
            )
        )

        assert review.blocked is False
