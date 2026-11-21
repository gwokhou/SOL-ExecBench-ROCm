"""Pure-CPU tests for Stage-2 ROCm inference primitives and warning-free CLI."""

from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from scripts.mvp_v1 import (
    MVPOperator,
    PrecisionMode,
    bound_fused_moe,
    bound_fused_rmsnorm_rope,
    bound_mla_paged_decode,
    bound_swiglu_fp8_quant,
    load_hardware_ceiling,
)
from scripts.mvp_v1.anti_distortion import (
    RotatingBufferPool,
    validate_materialized_outputs,
)
from scripts.mvp_v1.kernels import (
    KernelBackend,
    ref_fused_moe,
    ref_fused_rmsnorm_rope,
    ref_mla_paged_decode,
    ref_swiglu_fp8_quant,
    run_fused_moe_grouped,
    run_swiglu_fp8_quant_pytorch,
)
from scripts.mvp_v1.triton_ops_ext import supports_splitk_mla
from scripts.mvp_v1.vendor_adapters import (
    PreparedRunner,
    materialize_comparable_output,
)
from scripts.mvp_v1.vendor_adapters_ext import (
    try_make_aiter_fp8_moe_runner,
    try_make_aiter_mla_runner,
    try_make_aiter_rmsnorm_rope_runner,
    try_make_aiter_swiglu_quant_runner,
    try_make_vllm_fp8_moe_runner,
    try_make_vllm_mla_runner,
    try_make_vllm_rmsnorm_rope_runner,
    try_make_vllm_swiglu_quant_runner,
)
from scripts.mvp_v1.workloads import (
    SweepMode,
    build_runtime_case,
    build_workload_specs,
    materialize_trace_corpus,
)
from sol_execbench.core.bench.correctness import compute_error_stats
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.data.workload_validation import (
    validate_problem_contract,
)


def _assert_pool_ptr(
    runner: PreparedRunner,
    arg_idx: int,
    recorded: list[int],
) -> None:
    pool = RotatingBufferPool(
        runner.candidate_args, min_ring_bytes=1, max_replicas=2
    )
    args = pool.next_args()
    recorded.clear()
    assert runner.fn is not None
    runner.fn(*args)
    assert recorded == [args[arg_idx].data_ptr()]


def test_stage2_analytical_roofline_bounds() -> None:
    mi300x = load_hardware_ceiling("MI300X")

    mla = bound_mla_paged_decode(
        8, 16, 512, 64, 32768, 16, PrecisionMode.BF16, mi300x
    )
    assert mla.operator is MVPOperator.MLA_PAGED_DECODE
    assert mla.bottleneck_spec == "memory"
    expected_mla_flops = float(8 * 16 * 32768 * (4.0 * 512 + 2.0 * 64 + 5.0))
    assert mla.flops == pytest.approx(expected_mla_flops)

    norm_rope = bound_fused_rmsnorm_rope(
        128, 32, 128, PrecisionMode.BF16, mi300x
    )
    assert norm_rope.operator is MVPOperator.FUSED_RMSNORM_ROPE
    assert norm_rope.bottleneck_spec == "memory"
    assert norm_rope.compulsory_bytes == pytest.approx(
        float(3 * 128 * 32 * 128 * 2 + 128 * 2 + 128 * 128 * 4)
    )

    sq = bound_swiglu_fp8_quant(1024, 4096, mi300x)
    assert sq.operator is MVPOperator.SWIGLU_FP8_QUANT
    assert sq.bottleneck_spec == "memory"
    assert sq.compulsory_bytes == pytest.approx(
        float(4 * 1024 * 4096 + 1024 * 4096 + 4 * 1024)
    )

    moe_fp8 = bound_fused_moe(
        8, 2048, 1024, 64, 6, PrecisionMode.FP8, mi300x, active_experts=30
    )
    moe_bf16 = bound_fused_moe(
        8, 2048, 1024, 64, 6, PrecisionMode.BF16, mi300x, active_experts=30
    )
    assert moe_fp8.compulsory_bytes < moe_bf16.compulsory_bytes


def test_full_stage2_corpus_validates_all_34_workloads(tmp_path: Path) -> None:
    specs = build_workload_specs(SweepMode.FULL)
    assert len(specs) == 34
    written = materialize_trace_corpus(tmp_path, SweepMode.FULL)
    assert len(written) == 8
    total_loaded = 0
    for def_name, wl_path in written.items():
        defn = Definition.model_validate_json(
            (tmp_path / "workloads" / def_name / "definition.json").read_text(
                "utf-8"
            )
        )
        workloads = [
            Workload.model_validate_json(line)
            for line in wl_path.read_text("utf-8").splitlines()
            if line.strip()
        ]
        validate_problem_contract(defn, workloads)
        total_loaded += len(workloads)
    assert total_loaded == 34


def test_stage2_cpu_reference_and_pytorch_runners_pass_accuracy() -> None:
    mi300x = load_hardware_ceiling("MI300X")
    for spec in build_workload_specs(SweepMode.SMOKE):
        if spec.operator is MVPOperator.SKINNY_GEMM:
            continue
        case = build_runtime_case(
            spec, device="cpu", ceiling=mi300x, backend=KernelBackend.PYTORCH
        )
        assert case.candidate_fn is not None
        out = case.candidate_fn(*case.candidate_args)
        validate_materialized_outputs(out, "cpu")
        if spec.operator is MVPOperator.SWIGLU_FP8_QUANT:
            assert isinstance(out, tuple)
            assert len(out) == 2
            assert out[0].dtype is torch.float8_e4m3fnuz
            assert out[1].dtype is torch.float32
        cand_tensor = materialize_comparable_output(out)
        assert cand_tensor is not None
        ref_tensor = case.reference_fn(*case.prototype_args)
        _, exceeds = compute_error_stats(
            cand_tensor, ref_tensor, spec.tolerance
        )
        assert not exceeds

    q = torch.ones((2, 16, 80), dtype=torch.bfloat16)
    kv = torch.ones((8, 16, 80), dtype=torch.bfloat16)
    assert supports_splitk_mla(q, kv, 64) is True
    assert supports_splitk_mla(q, kv, 48) is False


def test_stage2_vendor_adapters_prepack_zero_copy_and_reject_corrupt(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "scripts.mvp_v1.vendor_adapters._is_rocm_torch", lambda: True
    )
    ptrs: list[int] = []
    corrupt = {"flag": False}

    def _mock_mla(q_in, kv_in, out, *_rest, sm_scale=1.0):
        del sm_scale
        ptrs.append(kv_in.data_ptr())
        res = ref_mla_paged_decode(
            q_in, kv_in.squeeze(2), _rest[1], _rest[2], _rest[3], out.shape[-1]
        )
        out.copy_(res + (5.0 if corrupt["flag"] else 0.0))

    def _mock_qk_norm_rope(pos, x_in, res_in, w_in, cs_in, *_opt):
        del pos
        ptrs.append(cs_in.data_ptr())
        out = ref_fused_rmsnorm_rope(x_in, res_in, w_in, cs_in)
        return out + (5.0 if corrupt["flag"] else 0.0)

    def _mock_swiglu_quant(y_fp8, y_scale, gu_in):
        ptrs.append(gu_in.data_ptr())
        q_t, s_t = run_swiglu_fp8_quant_pytorch(
            gu_in, fp8_dtype=torch.float8_e4m3fnuz
        )
        y_fp8.copy_(q_t)
        y_scale.copy_(s_t + (50.0 if corrupt["flag"] else 0.0))

    def _mock_fp8_moe(t_in, w1, w2, lg, k, **_kw):
        ptrs.append(w1.data_ptr())
        out = torch.zeros((t_in.shape[0], t_in.shape[1]), dtype=torch.bfloat16)
        return out + (25.0 if corrupt["flag"] else 0.0)

    def _mock_aiter_fp8_moe(t_in, w1, w2, wn, ids, **_kw):
        ptrs.append(w1.data_ptr())
        return torch.zeros((t_in.shape[0], t_in.shape[1]), dtype=torch.bfloat16)

    fake_ops = types.SimpleNamespace(
        mla_decode_fwd=_mock_mla,
        fused_qk_norm_rope=_mock_qk_norm_rope,
        silu_and_mul_quant=_mock_swiglu_quant,
    )
    fake_vllm = types.SimpleNamespace(
        _custom_ops=fake_ops,
        model_executor=types.SimpleNamespace(
            layers=types.SimpleNamespace(
                fused_moe=types.SimpleNamespace(fused_moe=_mock_fp8_moe)
            )
        ),
    )
    fake_aiter = types.SimpleNamespace(
        fused_moe=_mock_aiter_fp8_moe,
        mla=types.SimpleNamespace(mla_decode_fwd=_mock_mla),
        fused_qk_norm_rope=_mock_qk_norm_rope,
        silu_and_mul_scaled_quant=_mock_swiglu_quant,
        ops=types.SimpleNamespace(
            shuffle=types.SimpleNamespace(
                shuffle_weight=lambda t, layout=(16, 16): t.clone()
            )
        ),
    )
    for name, mod in {
        "vllm": fake_vllm,
        "vllm._custom_ops": fake_ops,
        "vllm.model_executor.layers.fused_moe": (
            fake_vllm.model_executor.layers.fused_moe
        ),
        "aiter": fake_aiter,
        "aiter.mla": fake_aiter.mla,
        "aiter.ops": fake_aiter.ops,
        "aiter.ops.shuffle": fake_aiter.ops.shuffle,
    }.items():
        monkeypatch.setitem(sys.modules, name, cast(Any, mod))

    tok = torch.zeros((2, 8), dtype=torch.float8_e4m3fnuz)
    gw = torch.zeros((4, 8, 4), dtype=torch.float8_e4m3fnuz)
    dw = torch.zeros((4, 4, 8), dtype=torch.float8_e4m3fnuz)
    lg = torch.ones((2, 4), dtype=torch.bfloat16)
    moe_args: list[Any] = [tok, gw, gw, dw, lg, 2]
    assert ref_fused_moe(*moe_args).shape == (2, 8)
    assert run_fused_moe_grouped(*moe_args).shape == (2, 8)
    moe_v = try_make_vllm_fp8_moe_runner(
        moe_args, ref_fn=ref_fused_moe, max_atol=1e-1
    )
    moe_a = try_make_aiter_fp8_moe_runner(
        moe_args, ref_fn=ref_fused_moe, max_atol=1e-1
    )
    assert moe_v.implementation_name == "vllm_fp8_fused_moe"
    assert moe_a.implementation_name == "aiter_fp8_fused_moe"
    _assert_pool_ptr(moe_v, 1, ptrs)
    _assert_pool_ptr(moe_a, 1, ptrs)

    q = torch.ones((1, 16, 80), dtype=torch.bfloat16)
    kv = torch.ones((2, 16, 80), dtype=torch.bfloat16)
    mla_args: list[Any] = [
        q,
        kv,
        torch.tensor([0, 2], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([16], dtype=torch.int32),
        64,
    ]
    mla_v = try_make_vllm_mla_runner(
        mla_args, ref_fn=ref_mla_paged_decode, max_atol=1e-2
    )
    mla_a = try_make_aiter_mla_runner(
        mla_args, ref_fn=ref_mla_paged_decode, max_atol=1e-2
    )
    assert mla_v.implementation_name == "vllm_mla_decode_fwd"
    assert mla_a.implementation_name == "aiter_mla_decode_fwd"
    assert mla_v.candidate_args[1].shape == (2, 16, 1, 80)
    _assert_pool_ptr(mla_v, 1, ptrs)
    _assert_pool_ptr(mla_a, 1, ptrs)

    x = torch.randn((4, 8, 64), dtype=torch.bfloat16)
    w = torch.ones((64,), dtype=torch.bfloat16)
    cs = torch.ones((4, 64), dtype=torch.float32)
    nr_args: list[Any] = [x, x, w, cs]
    nr_v = try_make_vllm_rmsnorm_rope_runner(
        nr_args, ref_fn=ref_fused_rmsnorm_rope, max_atol=1e-2
    )
    nr_a = try_make_aiter_rmsnorm_rope_runner(
        nr_args, ref_fn=ref_fused_rmsnorm_rope, max_atol=1e-2
    )
    assert nr_v.implementation_name == "vllm_fused_qk_norm_rope"
    assert nr_a.implementation_name == "aiter_fused_qk_norm_rope"
    assert nr_v.candidate_args[3].dtype is torch.bfloat16
    _assert_pool_ptr(nr_v, 3, ptrs)
    _assert_pool_ptr(nr_a, 3, ptrs)

    gu = torch.randn((4, 256), dtype=torch.bfloat16) * 0.5
    sq_v = try_make_vllm_swiglu_quant_runner(
        [gu],
        fp8_dtype=torch.float8_e4m3fnuz,
        ref_fn=ref_swiglu_fp8_quant,
        max_atol=2.5e-1,
    )
    sq_a = try_make_aiter_swiglu_quant_runner(
        [gu],
        fp8_dtype=torch.float8_e4m3fnuz,
        ref_fn=ref_swiglu_fp8_quant,
        max_atol=2.5e-1,
    )
    assert sq_v.implementation_name == "vllm_silu_and_mul_fp8_quant"
    assert sq_a.implementation_name == "aiter_silu_and_mul_fp8_quant"
    _assert_pool_ptr(sq_v, 0, ptrs)
    _assert_pool_ptr(sq_a, 0, ptrs)

    corrupt["flag"] = True
    for bad in (
        try_make_vllm_fp8_moe_runner(
            moe_args, ref_fn=ref_fused_moe, max_atol=1e-1
        ),
        try_make_vllm_mla_runner(
            mla_args, ref_fn=ref_mla_paged_decode, max_atol=1e-2
        ),
        try_make_aiter_rmsnorm_rope_runner(
            nr_args, ref_fn=ref_fused_rmsnorm_rope, max_atol=1e-2
        ),
        try_make_vllm_swiglu_quant_runner(
            [gu],
            fp8_dtype=torch.float8_e4m3fnuz,
            ref_fn=ref_swiglu_fp8_quant,
            max_atol=2.5e-1,
        ),
    ):
        assert bad.fn is None
        assert bad.implementation_name.startswith("unavailable:")


def test_py312_compat_module_emits_no_runpy_warning(tmp_path: Path) -> None:
    probe = tmp_path / "ok_probe.py"
    probe.write_text("VALUE = 1 + 2\n", encoding="utf-8")
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{root / 'src'}:{root}"
    proc = subprocess.run(
        [
            sys.executable,
            "-W",
            "error",
            "-m",
            "scripts.mvp_v1.py312_compat",
            str(probe),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "RuntimeWarning" not in proc.stderr
