# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""P0 coverage for FP8 (including fnuz) and FP4 (float4_e2m1fn_x2) pipelines."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing.connection import Connection

import pytest
import torch
from sol_execbench_type_helpers import make_definition

from sol_execbench.core.bench.correctness import (
    check_tensor_sanity,
    compute_error_stats,
)
from sol_execbench.core.bench.input_generation import _rand_tensor
from sol_execbench.core.bench.input_heuristics import (
    _generate_heuristic_tensor,
)
from sol_execbench.core.bench.memory_pool import ShiftingMemoryPoolAllocator
from sol_execbench.core.bench.output_allocation import (
    allocate_outputs,
    normalize_outputs,
)
from sol_execbench.core.bench.reference_protocol import (
    ReferenceCase,
    receive_case,
    send_case,
)
from sol_execbench.core.data.workload import ToleranceSpec

_ALL_FP8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
)


@contextmanager
def _pipe_pair() -> Iterator[tuple[Connection, Connection]]:
    read_fd, write_fd = os.pipe()
    sender = Connection(write_fd, readable=False, writable=True)
    receiver = Connection(read_fd, readable=True, writable=False)
    try:
        yield sender, receiver
    finally:
        sender.close()
        receiver.close()


@pytest.mark.parametrize("dtype", _ALL_FP8_DTYPES)
def test_fp8_sanity_and_error_stats_do_not_crash_on_cpu_or_gpu(
    dtype: torch.dtype,
) -> None:
    devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
    for device in devices:
        ref = torch.tensor([0.0, 1.0, -1.5, 0.5], device=device).to(dtype)
        sol = ref.clone()
        assert check_tensor_sanity(sol, ref, allow_negative_inf=True) is None

        metrics, exceeds = compute_error_stats(
            sol,
            ref,
            ToleranceSpec(max_atol=1e-2, max_rtol=1e-2),
        )
        assert not exceeds
        assert metrics.max_absolute_error == 0.0

        all_zero = torch.zeros_like(ref)
        zero_fail = check_tensor_sanity(all_zero, ref)
        assert zero_fail is not None
        assert zero_fail.max_absolute_error > 0.0


@pytest.mark.parametrize(
    "dtype",
    [torch.float8_e4m3fnuz, torch.float8_e5m2fnuz],
)
def test_fp8_fnuz_input_heuristics_and_rand_tensor(
    dtype: torch.dtype,
) -> None:
    for name in ("norm_weight", "rstd", "weight", "attn_weights"):
        assert (
            _generate_heuristic_tensor(
                name,
                (16, 16),
                dtype,
                torch.device("cpu"),
            )
            is None
        )
    generated = _rand_tensor((16, 16), dtype, torch.device("cpu"))
    assert generated.dtype == dtype
    assert generated.shape == (16, 16)


def test_fp4_ipc_roundtrip_allocation_and_memory_pool() -> None:
    devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
    for device in devices:
        definition = make_definition(
            name="fp4_quant",
            op_type="quantize",
            axes={
                "M": {"type": "const", "value": 4},
                "K": {"type": "const", "value": 8},
                "KP": {"type": "const", "value": 4},
            },
            inputs={"x": {"shape": ["M", "K"], "dtype": "float32"}},
            outputs={"y": {"shape": ["M", "KP"], "dtype": "float4_e2m1fn_x2"}},
            reference="def run(x): pass\n",
        )
        outputs = allocate_outputs(
            definition, {"M": 4, "K": 8, "KP": 4}, device
        )
        assert len(outputs) == 1
        assert outputs[0].dtype == torch.float4_e2m1fn_x2
        assert outputs[0].shape == (4, 4)
        assert outputs[0].view(torch.uint8).sum().item() == 0

        src_fp4 = torch.tensor(
            [[0x12, 0x34, 0x56, 0x78]] * 4,
            dtype=torch.uint8,
            device=device,
        ).view(torch.float4_e2m1fn_x2)

        with _pipe_pair() as (sender, receiver):
            send_case(
                sender,
                ReferenceCase(inputs=[src_fp4], outputs=[outputs[0]]),
            )
            received = receive_case(receiver, device=device)

        assert received.inputs[0].dtype == torch.float4_e2m1fn_x2
        assert received.inputs[0].shape == (4, 4)
        assert torch.equal(
            received.inputs[0].view(torch.uint8),
            src_fp4.view(torch.uint8),
        )

        pool = ShiftingMemoryPoolAllocator(
            [src_fp4],
            [outputs[0]],
            total_iterations=3,
        )
        ptrs = []
        for _ in range(3):
            in_arg, out_arg = pool.get_unique_args()
            assert in_arg.dtype == torch.float4_e2m1fn_x2
            assert out_arg.dtype == torch.float4_e2m1fn_x2
            assert torch.equal(
                in_arg.view(torch.uint8),
                src_fp4.view(torch.uint8),
            )
            assert int(out_arg.view(torch.uint8).sum().item()) == 0
            out_arg.view(torch.uint8).fill_(0xFF)
            ptrs.append((in_arg.data_ptr(), out_arg.data_ptr()))
        assert len(set(ptrs)) == 3

        norm = normalize_outputs(
            src_fp4.view(torch.uint8).cpu().view(torch.float4_e2m1fn_x2),
            device=torch.device(device),
            output_names=["y"],
            output_dtypes={"y": torch.float4_e2m1fn_x2},
        )
        assert norm["y"].dtype == torch.float4_e2m1fn_x2
        assert norm["y"].device == torch.device(device)
        assert check_tensor_sanity(src_fp4, src_fp4) is None
