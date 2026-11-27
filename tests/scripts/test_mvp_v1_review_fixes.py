"""Focused unit tests for Part I (scripts/mvp_v1) P0 and P1 review fixes."""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import types
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import pytest
import torch

from scripts.mvp_v1 import (
    PrecisionMode,
    VendorBackend,
    bound_fused_moe,
    load_hardware_ceiling,
)
from scripts.mvp_v1.anti_distortion import (
    _parse_float,
    _sample_amd_telemetry,
    acquire_dvfs_telemetry_window,
    time_kernel_isolated,
)
from scripts.mvp_v1.cloud_preflight import _validate_bundle_payload
from scripts.mvp_v1.cloud_runner import run_single_op_replay
from scripts.mvp_v1.counter_capture import (
    PROFILE_RANGE_MARKER,
    _execute_rocm_groups,
    _filter_pass_csv_to_marker,
    collect_hardware_counters,
)
from scripts.mvp_v1.kernels import (
    KernelBackend,
    make_skinny_gemm_runner,
    ref_fused_moe,
    ref_fused_rmsnorm_rope,
    ref_skinny_gemm,
)
from scripts.mvp_v1.vendor_adapters import (
    _self_validate,
    try_make_aiter_gemm_runner,
    try_make_aiter_gqa_runner,
    try_make_vllm_gemm_runner,
    try_make_vllm_gqa_runner,
)
from scripts.mvp_v1.vendor_adapters_ext import (
    try_make_aiter_fp8_moe_runner,
    try_make_aiter_rmsnorm_rope_runner,
    try_make_vllm_rmsnorm_rope_runner,
)
from scripts.mvp_v1.workloads import SweepMode
from sol_execbench.core.bench.clock_ownership import ClockDevice


def test_p0_1_rocm_pa_256_partition_scratch_and_fallbacks(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.mvp_v1.vendor_adapters._is_rocm_torch", lambda: True
    )
    q = torch.ones((2, 4, 16), dtype=torch.bfloat16)
    k = torch.ones((128, 16, 2, 16), dtype=torch.bfloat16)
    # 64 pages * 16 tokens/page = 1024 tokens -> 4 partitions at 256, 2 at 512.
    gqa_args: list[Any] = [
        q,
        k,
        k,
        torch.tensor([0, 64, 128], dtype=torch.int32),
        torch.arange(128, dtype=torch.int32),
        torch.tensor([16, 16], dtype=torch.int32),
    ]
    seen_parts: list[int] = []

    def _pa_rocm_256(out, exp_sums, max_logits, tmp_out, q_in, *_rest):
        num_parts = int(exp_sums.shape[2])
        seen_parts.append(num_parts)
        if num_parts < 4 or max_logits.shape[2] < 4 or tmp_out.shape[2] < 4:
            raise RuntimeError("OOB partition buffer for 256-token ROCm kernel")
        exp_sums[:, :, :4].zero_()
        max_logits[:, :, :4].zero_()
        tmp_out[:, :, :4, :].zero_()
        out.copy_(q_in)

    fake_ops_256 = types.SimpleNamespace(paged_attention_rocm=_pa_rocm_256)
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        cast(Any, types.SimpleNamespace(_custom_ops=fake_ops_256)),
    )
    monkeypatch.setitem(
        sys.modules, "vllm._custom_ops", cast(Any, fake_ops_256)
    )
    res_256 = try_make_vllm_gqa_runner(
        gqa_args, ref_fn=lambda q_in, *_r: q_in, max_atol=1e-2
    )
    assert res_256.implementation_name == "vllm_paged_attention_rocm"
    assert seen_parts == [4]

    seen_512: list[int] = []

    def _pa_rocm_512(out, exp_sums, _max_logits, _tmp_out, q_in, *_rest):
        seen_512.append(int(exp_sums.shape[2]))
        if int(exp_sums.shape[2]) != 2:
            raise RuntimeError("512-partition build expects 2 partitions")
        out.copy_(q_in)

    monkeypatch.setitem(
        sys.modules,
        "aiter",
        cast(Any, types.SimpleNamespace(paged_attention_rocm=_pa_rocm_512)),
    )
    res_512 = try_make_aiter_gqa_runner(
        gqa_args, ref_fn=lambda q_in, *_r: q_in, max_atol=1e-2
    )
    assert res_512.implementation_name == "aiter_paged_attention_rocm"
    assert seen_512 == [4, 2]

    def _pa_v1_only(*call_args):
        if len(call_args) != 19:
            raise TypeError("v1 expects 19 args")
        call_args[0].copy_(call_args[1])

    fake_ops_v1 = types.SimpleNamespace(paged_attention_v1=_pa_v1_only)
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        cast(Any, types.SimpleNamespace(_custom_ops=fake_ops_v1)),
    )
    monkeypatch.setitem(sys.modules, "vllm._custom_ops", cast(Any, fake_ops_v1))
    res_v1 = try_make_vllm_gqa_runner(
        gqa_args, ref_fn=lambda q_in, *_r: q_in, max_atol=1e-2
    )
    assert res_v1.implementation_name == "vllm_paged_attention_v1"


def test_p0_2_self_validate_clones_and_rmsnorm_rope_inplace_safety(
    monkeypatch,
) -> None:
    x_orig = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    cand_args: list[Any] = [x_orig.clone()]
    ref_args: list[Any] = [x_orig]
    order: list[str] = []

    def _mutating_throwing_cand(t: torch.Tensor) -> torch.Tensor:
        order.append("cand")
        t.add_(99.0)
        raise RuntimeError("mid-kernel abort after mutation")

    def _ref(t: torch.Tensor) -> torch.Tensor:
        order.append("ref")
        return t * 2.0

    assert (
        _self_validate(_mutating_throwing_cand, cand_args, _ref, ref_args, 1e-2)
        is False
    )
    assert order == ["ref", "cand"]
    assert torch.equal(cand_args[0], x_orig)
    assert torch.equal(ref_args[0], x_orig)

    monkeypatch.setattr(
        "scripts.mvp_v1.vendor_adapters._is_rocm_torch", lambda: True
    )
    x = torch.randn((4, 4, 32), dtype=torch.bfloat16)
    res = torch.randn((4, 4, 32), dtype=torch.bfloat16)
    w = torch.ones((32,), dtype=torch.bfloat16)
    cs = torch.ones((4, 32), dtype=torch.float32)
    proto = [x.clone(), res.clone(), w, cs]
    expected = ref_fused_rmsnorm_rope(*proto)

    def _inplace_none_fused(_pos, x_in, res_in, w_in, cs_in, *_extra) -> None:
        computed = ref_fused_rmsnorm_rope(x_in, res_in, w_in, cs_in)
        res_in.add_(x_in)
        x_in.copy_(computed)

    fake_ops = types.SimpleNamespace(fused_qk_norm_rope=_inplace_none_fused)
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        cast(Any, types.SimpleNamespace(_custom_ops=fake_ops)),
    )
    monkeypatch.setitem(sys.modules, "vllm._custom_ops", cast(Any, fake_ops))
    monkeypatch.setitem(
        sys.modules,
        "aiter",
        cast(
            Any, types.SimpleNamespace(fused_qk_norm_rope=_inplace_none_fused)
        ),
    )

    for runner in (
        try_make_vllm_rmsnorm_rope_runner(
            proto, ref_fn=ref_fused_rmsnorm_rope, max_atol=2e-2
        ),
        try_make_aiter_rmsnorm_rope_runner(
            proto, ref_fn=ref_fused_rmsnorm_rope, max_atol=2e-2
        ),
    ):
        assert runner.fn is not None
        assert torch.equal(proto[0], x)
        assert torch.equal(proto[1], res)
        stats = time_kernel_isolated(
            runner.fn,
            runner.candidate_args,
            device="cpu",
            llc_bytes=64 * 1024,
            warmup=5,
            rep=20,
        )
        assert len(stats.samples_ms) == 20
        out_after = runner.fn(*runner.candidate_args)
        assert torch.allclose(out_after, expected, atol=2e-2)


def test_p0_2_vllm_rope_cos_sin_dtype_and_aiter_fp8_moe_bf16(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "scripts.mvp_v1.vendor_adapters._is_rocm_torch", lambda: True
    )
    x = torch.randn((4, 2, 16), dtype=torch.bfloat16)
    res = torch.randn((4, 2, 16), dtype=torch.bfloat16)
    w = torch.ones((16,), dtype=torch.bfloat16)
    cs_f32 = torch.ones((4, 16), dtype=torch.float32)

    def _norm(h, r, w_in, eps):
        r.add_(h)
        rms = torch.rsqrt(r.float().pow(2).mean(dim=-1, keepdim=True) + eps)
        h.copy_((r.float() * rms * w_in.float()).to(h.dtype))

    def _rope(_pos, q_flat, _k, head_dim, cs_cache, _is_neox):
        if cs_cache.dtype != q_flat.dtype:
            raise RuntimeError("cos_sin_cache dtype must match query dtype")
        q_3d = q_flat.view(-1, q_flat.shape[-1] // head_dim, head_dim)
        d2 = head_dim // 2
        u, v = q_3d[..., :d2].float(), q_3d[..., d2:].float()
        c = cs_cache[:, None, :d2].float()
        s = cs_cache[:, None, d2:].float()
        q_3d.copy_(
            torch.cat([u * c - v * s, v * c + u * s], dim=-1).to(q_3d.dtype)
        )

    fake_ops = types.SimpleNamespace(
        fused_add_rms_norm=_norm, rotary_embedding=_rope
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        cast(Any, types.SimpleNamespace(_custom_ops=fake_ops)),
    )
    monkeypatch.setitem(sys.modules, "vllm._custom_ops", cast(Any, fake_ops))
    rope_runner = try_make_vllm_rmsnorm_rope_runner(
        [x, res, w, cs_f32], ref_fn=ref_fused_rmsnorm_rope, max_atol=2e-2
    )
    assert rope_runner.implementation_name == "vllm_rmsnorm_rope"
    assert rope_runner.candidate_args[3].dtype is torch.bfloat16

    tok = torch.zeros((2, 8), dtype=torch.float8_e4m3fnuz)
    gw = torch.zeros((4, 8, 4), dtype=torch.float8_e4m3fnuz)
    dw = torch.zeros((4, 4, 8), dtype=torch.float8_e4m3fnuz)
    lg = torch.ones((2, 4), dtype=torch.bfloat16)
    moe_args: list[Any] = [tok, gw, gw, dw, lg, 2]

    for required_dt in (torch.bfloat16, torch.float8_e4m3fnuz):

        def _aiter_moe(t_in, _w1, _w2, _wn, _ids, _req=required_dt, **_kw):
            if t_in.dtype is not _req:
                raise TypeError(f"expected {_req}, got {t_in.dtype}")
            return torch.zeros(
                (t_in.shape[0], t_in.shape[1]), dtype=torch.bfloat16
            )

        monkeypatch.setitem(
            sys.modules,
            "aiter",
            cast(Any, types.SimpleNamespace(fused_moe=_aiter_moe)),
        )
        res_moe = try_make_aiter_fp8_moe_runner(
            moe_args, ref_fn=ref_fused_moe, max_atol=1e-1
        )
        assert res_moe.implementation_name == "aiter_fp8_fused_moe"
        assert res_moe.candidate_args[0].dtype is required_dt


def test_p1_1_vllm_honest_fallback_names_and_ty_contracts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "scripts.mvp_v1.vendor_adapters._is_rocm_torch", lambda: True
    )
    x_bf = torch.ones((4, 16), dtype=torch.bfloat16)
    w_bf = torch.full((16, 8), 0.5, dtype=torch.bfloat16)
    x_fp8 = x_bf.to(torch.float8_e4m3fnuz)
    w_fp8 = w_bf.to(torch.float8_e4m3fnuz)
    sa = torch.ones((), dtype=torch.float32)
    sb = torch.ones((), dtype=torch.float32)

    # vllm without _custom_ops must fail closed rather than masquerading PyTorch.
    monkeypatch.setitem(
        sys.modules, "vllm", cast(Any, types.SimpleNamespace(_custom_ops=None))
    )
    monkeypatch.delitem(sys.modules, "vllm._custom_ops", raising=False)
    monkeypatch.setattr(
        "scripts.mvp_v1.vendor_adapters._import_vllm_custom_ops",
        lambda: (types.SimpleNamespace(), None),
    )
    unavail_bf = try_make_vllm_gemm_runner(
        [x_bf, w_bf], fp8=False, ref_fn=ref_skinny_gemm, max_atol=5e-2
    )
    unavail_fp8 = try_make_vllm_gemm_runner(
        [x_fp8, w_fp8],
        fp8=True,
        scale_a=sa,
        scale_b=sb,
        ref_fn=ref_skinny_gemm,
        max_atol=5e-1,
    )
    assert (
        unavail_bf.implementation_name == "unavailable:vllm_ops_not_installed"
    )
    assert (
        unavail_fp8.implementation_name == "unavailable:vllm_ops_not_installed"
    )

    # Loaded _custom_ops without custom GEMM names Torch fallbacks honestly.
    empty_ops = types.SimpleNamespace()
    monkeypatch.setattr(
        "scripts.mvp_v1.vendor_adapters._import_vllm_custom_ops",
        lambda: (types.SimpleNamespace(_custom_ops=empty_ops), empty_ops),
    )
    fb_bf = try_make_vllm_gemm_runner(
        [x_bf, w_bf], fp8=False, ref_fn=ref_skinny_gemm, max_atol=5e-2
    )
    assert fb_bf.implementation_name == "vllm_torch_linear_fallback"

    monkeypatch.setattr(
        torch,
        "_scaled_mm",
        lambda a, b, *, scale_a, scale_b, out_dtype: (
            (a.float() @ b.float()) * (scale_a * scale_b)
        ).to(out_dtype),
    )
    fb_fp8 = try_make_vllm_gemm_runner(
        [x_fp8, w_fp8],
        fp8=True,
        scale_a=sa,
        scale_b=sb,
        ref_fn=ref_skinny_gemm,
        max_atol=5e-1,
    )
    assert fb_fp8.implementation_name == "vllm_torch_scaled_mm_fallback"

    def _align16_mm(a, b, *, scale_a, scale_b, out_dtype):
        if a.shape[0] % 16 != 0:
            raise RuntimeError("align16")
        return ((a.float() @ b.float()) * (scale_a * scale_b)).to(out_dtype)

    monkeypatch.setattr(torch, "_scaled_mm", _align16_mm)
    fb_pad = try_make_vllm_gemm_runner(
        [x_fp8, w_fp8],
        fp8=True,
        scale_a=sa,
        scale_b=sb,
        ref_fn=ref_skinny_gemm,
        max_atol=5e-1,
    )
    assert fb_pad.implementation_name == "vllm_torch_scaled_mm_pad16_fallback"

    assert _parse_float(MappingProxyType({"value": "1850 MHz"})) == 1850.0

    summary_p = tmp_path / "summary.json"
    report_p = tmp_path / "report.md"
    traces_p = tmp_path / "traces.jsonl"
    summary_p.write_text(
        json.dumps({"target": {"target_id": "MI300X"}, "rows": ["bad_row"]}),
        encoding="utf-8",
    )
    report_p.write_text("# Report\n", encoding="utf-8")
    traces_p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="rows must be objects"):
        _validate_bundle_payload(summary_p, report_p, traces_p)


def test_p1_2_rocprof_fail_closed_missing_marker_and_pid_prefix(
    tmp_path: Path,
    monkeypatch,
) -> None:
    missing_dir = tmp_path / "no_marker"
    missing_dir.mkdir(parents=True)
    raw_csv = missing_dir / "10_counter_collection.csv"
    raw_csv.write_text(
        "Dispatch_Id,Kernel_Name,Grid_Size,Workgroup_Size,Counter_Name,Counter_Value,Start_Timestamp\n"
        '1,flush_k,"[1,1,1]","[64,1,1]",FETCH_SIZE,1073741824,500\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=PROFILE_RANGE_MARKER):
        _filter_pass_csv_to_marker(raw_csv, missing_dir)

    monkeypatch.setattr(
        "scripts.mvp_v1.counter_capture._resolve_rocprofv3_binary",
        lambda: Path("/opt/rocm/bin/rocprofv3"),
    )
    monkeypatch.setattr(
        "scripts.mvp_v1.counter_capture._query_rocm_available_counters",
        lambda _bin, _env: frozenset({"SQ_WAVES"}),
    )

    def _pass_without_marker(
        _rocprof: Path,
        _group: list[str],
        _cmd: list[str],
        *,
        pass_dir: Path,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        del env
        pass_dir.mkdir(parents=True, exist_ok=True)
        (pass_dir / "10_counter_collection.csv").write_text(
            raw_csv.read_text("utf-8"), encoding="utf-8"
        )
        return subprocess.CompletedProcess(["rocprofv3"], 0, "", "")

    monkeypatch.setattr(
        "scripts.mvp_v1.counter_capture._run_one_rocm_pass",
        _pass_without_marker,
    )
    failed = collect_hardware_counters(
        ["python"],
        backend=VendorBackend.AMD_ROCM,
        architecture="gfx942",
        workload_id="w_fail_closed",
    )
    assert failed.status == "failed"
    assert PROFILE_RANGE_MARKER in (failed.note or "")

    def _pass_multi_pid(
        _rocprof: Path,
        _group: list[str],
        _cmd: list[str],
        *,
        pass_dir: Path,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        del env
        pass_dir.mkdir(parents=True, exist_ok=True)
        (pass_dir / "10_marker_api_trace.csv").write_text(
            "Domain,Function,Start_Timestamp,End_Timestamp\n"
            f"MARKER_CORE_RANGE_API,{PROFILE_RANGE_MARKER},1000,2000\n",
            encoding="utf-8",
        )
        (pass_dir / "10_counter_collection.csv").write_text(
            "Dispatch_Id,Kernel_Name,Grid_Size,Workgroup_Size,Counter_Name,Counter_Value,Start_Timestamp\n"
            '1,main_k,"[2,1,1]","[64,1,1]",SQ_WAVES,24,1500\n',
            encoding="utf-8",
        )
        (pass_dir / "99_counter_collection.csv").write_text(
            "Dispatch_Id,Kernel_Name,Grid_Size,Workgroup_Size,Counter_Name,Counter_Value,Start_Timestamp\n"
            '1,helper_k,"[8,1,1]","[64,1,1]",SQ_WAVES,9999,1500\n',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(["rocprofv3"], 0, "", "")

    monkeypatch.setattr(
        "scripts.mvp_v1.counter_capture._run_one_rocm_pass",
        _pass_multi_pid,
    )
    csvs, err = _execute_rocm_groups(
        Path("/opt/rocm/bin/rocprofv3"),
        [["SQ_WAVES"]],
        ["python"],
        tmp=tmp_path / "multi_pid",
        env={},
    )
    assert err is None
    assert [p.name for p in csvs] == ["measured_10_counter_collection.csv"]


def test_p1_3_non_unit_fp8_scales_and_moe_fp8_activation_bytes(
    monkeypatch,
) -> None:
    x = torch.ones((4, 16), dtype=torch.bfloat16).to(torch.float8_e4m3fnuz)
    w_t = (
        torch.full((8, 16), 0.5, dtype=torch.bfloat16)
        .to(torch.float8_e4m3fnuz)
        .t()
    )
    sa = torch.tensor(0.25, dtype=torch.float32)
    sb = torch.tensor(3.0, dtype=torch.float32)
    expected = ((x.float() @ w_t.float()) * 0.75).to(torch.bfloat16)

    assert torch.allclose(ref_skinny_gemm(x, w_t, sa, sb), expected)
    cpu_runner = make_skinny_gemm_runner(
        prototype_args=[x, w_t], fp8=True, device="cpu", scale_a=sa, scale_b=sb
    )
    assert cpu_runner.implementation_name == "cpu_fp8_ref"
    assert cpu_runner.fn is not None
    assert torch.allclose(cpu_runner.fn(*cpu_runner.candidate_args), expected)

    monkeypatch.setattr(
        "scripts.mvp_v1.vendor_adapters._is_rocm_torch", lambda: True
    )
    # Unscaled vendor op must FAIL _self_validate when sa * sb == 0.75 != 1.0!
    unscaled_ops = types.SimpleNamespace(
        cutlass_scaled_mm=lambda a, b, **_kw: (a.float() @ b.float()).to(
            torch.bfloat16
        )
    )
    scaled_ops = types.SimpleNamespace(
        cutlass_scaled_mm=lambda a, b, *, scale_a, scale_b, out_dtype: (
            (a.float() @ b.float()) * (scale_a.float() * scale_b.float())
        ).to(out_dtype)
    )
    monkeypatch.setattr(
        torch,
        "_scaled_mm",
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("disabled")),
    )
    monkeypatch.setattr(
        "scripts.mvp_v1.vendor_adapters._import_vllm_custom_ops",
        lambda: (types.SimpleNamespace(_custom_ops=unscaled_ops), unscaled_ops),
    )
    rejected = try_make_vllm_gemm_runner(
        [x, w_t],
        fp8=True,
        scale_a=sa,
        scale_b=sb,
        ref_fn=ref_skinny_gemm,
        max_atol=5e-2,
    )
    assert rejected.fn is None

    monkeypatch.setattr(
        "scripts.mvp_v1.vendor_adapters._import_vllm_custom_ops",
        lambda: (types.SimpleNamespace(_custom_ops=scaled_ops), scaled_ops),
    )
    accepted_v = try_make_vllm_gemm_runner(
        [x, w_t],
        fp8=True,
        scale_a=sa,
        scale_b=sb,
        ref_fn=ref_skinny_gemm,
        max_atol=5e-2,
    )
    assert accepted_v.implementation_name == "vllm_fp8_scaled_mm"
    assert accepted_v.fn is not None
    assert torch.allclose(accepted_v.fn(*accepted_v.candidate_args), expected)

    monkeypatch.setitem(
        sys.modules,
        "aiter",
        cast(
            Any,
            types.SimpleNamespace(
                gemm_a8w8=lambda a, b_nk, s_a, s_b, *_r: (
                    (a.float() @ b_nk.t().float()) * (s_a.float() * s_b.float())
                ).to(torch.bfloat16)
            ),
        ),
    )
    accepted_a = try_make_aiter_gemm_runner(
        [x, w_t],
        fp8=True,
        scale_a=sa,
        scale_b=sb,
        ref_fn=ref_skinny_gemm,
        max_atol=5e-2,
    )
    assert accepted_a.implementation_name == "aiter_gemm_a8w8"
    assert accepted_a.fn is not None
    assert torch.allclose(accepted_a.fn(*accepted_a.candidate_args), expected)

    mi300x = load_hardware_ceiling("MI300X")
    m, h, inter, e, k = 8, 2048, 1024, 64, 6
    b_fp8 = bound_fused_moe(
        m, h, inter, e, k, PrecisionMode.FP8, mi300x, active_experts=10
    )
    expected_fp8_bytes = float(
        10 * 3 * h * inter * 1 + (m * h * 1 + m * h * 2 + m * e * 2)
    )
    assert b_fp8.compulsory_bytes == pytest.approx(expected_fp8_bytes)


def test_p1_4_clock_unlock_on_partial_lock_uuid_telemetry_and_device_ctx(
    monkeypatch,
) -> None:
    test_uuid = "00000000-0000-0000-0000-000000000042"
    lock_calls: list[str | None] = []
    unlock_calls: list[str | None] = []
    monkeypatch.delenv("GPU_LOCK_UUID", raising=False)
    monkeypatch.setattr(
        "scripts.mvp_v1.anti_distortion.observe_clock_devices",
        lambda: (ClockDevice(bdf="0000:c1:00.0", uuid=test_uuid),),
    )
    monkeypatch.setattr(
        "scripts.mvp_v1.anti_distortion.sample_gpu_telemetry",
        lambda _dev, vendor: __import__(
            "scripts.mvp_v1.anti_distortion", fromlist=["TelemetrySample"]
        ).TelemetrySample(vendor=vendor),
    )

    def _failing_lock(uuid: str | None = None) -> None:
        lock_calls.append(uuid)
        raise RuntimeError("STABLE_PEAK readback failed after set")

    def _throwing_unlock(uuid: str | None = None) -> bool:
        unlock_calls.append(uuid)
        raise RuntimeError("unlock error must be suppressed")

    monkeypatch.setattr(
        "scripts.mvp_v1.anti_distortion.set_peak_clocks", _failing_lock
    )
    monkeypatch.setattr(
        "scripts.mvp_v1.anti_distortion.unlock_clocks", _throwing_unlock
    )
    with acquire_dvfs_telemetry_window(
        "cuda:0", VendorBackend.AMD_ROCM, lock_clocks=True
    ) as holder:
        assert holder["receipt"].lock_acquired is False
    assert lock_calls == [test_uuid]
    assert unlock_calls == [test_uuid]

    # UUID-aware _sample_amd_telemetry with fallback & per-GPU process filter.
    monkeypatch.setenv("GPU_LOCK_UUID", test_uuid)
    seen_level_uuids: list[str | None] = []
    monkeypatch.setattr(
        "scripts.mvp_v1.anti_distortion.observe_clock_levels",
        lambda uuid=None: (seen_level_uuids.append(uuid), ("STABLE_PEAK",))[-1],
    )
    monkeypatch.setattr(
        "scripts.mvp_v1.anti_distortion.detect_concurrent_gpu_processes",
        lambda: [
            {"pid": 11, "device": "0", "name": "neighbor"},
            {"pid": 22, "device": "2", "name": "target_proc"},
        ],
    )
    monkeypatch.setattr(
        "scripts.mvp_v1.anti_distortion._amd_smi_candidates",
        lambda: ["amd-smi"],
    )
    seen_selectors: list[str] = []

    def _fake_smi(
        cmd: list[str], **_kw: Any
    ) -> subprocess.CompletedProcess[str]:
        if "list" in cmd:
            return subprocess.CompletedProcess(
                cmd,
                0,
                json.dumps(
                    [
                        {"gpu": 0, "bdf": "0000:01:00.0", "uuid": "other"},
                        {"gpu": 2, "bdf": "0000:c1:00.0", "uuid": test_uuid},
                    ]
                ),
                "",
            )
        gpu_arg = cmd[cmd.index("--gpu") + 1]
        seen_selectors.append(gpu_arg)
        payload = {"gpu_data": [{"clock": {"gfx_0": {"clk": {"value": 2100}}}}]}
        return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")

    monkeypatch.setattr(
        "scripts.mvp_v1.anti_distortion.run_in_process_group_bounded", _fake_smi
    )
    sample = _sample_amd_telemetry("cuda:0")
    assert seen_level_uuids == [test_uuid]
    assert seen_selectors == [test_uuid]
    assert sample.sclk_mhz == 2100.0
    assert sample.concurrent_processes == 1

    # Verify torch.cuda.device(device) stream scoping in time_kernel_isolated
    # and run_single_op_replay.
    entered_devices: list[str] = []

    @contextlib.contextmanager
    def _mock_cuda_device(dev: str):
        entered_devices.append(str(dev))
        yield

    class _FakeEvent:
        def __init__(self, *, enable_timing: bool = False) -> None:
            del enable_timing

        def record(self) -> None:
            pass

        def synchronize(self) -> None:
            pass

        def elapsed_time(self, _other: object) -> float:
            return 0.25

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", _mock_cuda_device)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_a: None)
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(
        "scripts.mvp_v1.anti_distortion._device_free_bytes", lambda _d: None
    )
    monkeypatch.setattr(
        "scripts.mvp_v1.anti_distortion.recommended_llc_flush_bytes",
        lambda *_a, **_kw: 4096,
    )
    monkeypatch.setattr(
        "scripts.mvp_v1.anti_distortion.LLCRotatingFlusher",
        lambda _d, _b: types.SimpleNamespace(flush=lambda: None),
    )
    monkeypatch.setattr(
        "scripts.mvp_v1.anti_distortion.validate_materialized_outputs",
        lambda _o, _d: None,
    )
    t_stats = time_kernel_isolated(
        lambda a: a + 1.0,
        [torch.ones((4,), dtype=torch.float32)],
        device="cuda:1",
        llc_bytes=4096,
        warmup=1,
        rep=2,
    )
    assert t_stats.median_ms == pytest.approx(0.25)
    assert entered_devices == ["cuda:1"]

    entered_devices.clear()
    monkeypatch.setattr(
        "scripts.mvp_v1.cloud_runner.build_runtime_case",
        lambda *_a, **_kw: types.SimpleNamespace(
            candidate_fn=lambda a: a + 1.0,
            candidate_args=[torch.ones((4,), dtype=torch.float32)],
        ),
    )
    monkeypatch.setattr(
        "scripts.mvp_v1.cloud_runner.LLCRotatingFlusher",
        lambda _d, _b: types.SimpleNamespace(flush=lambda: None),
    )
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda _m: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)
    rc = run_single_op_replay(
        "skinny_gemm_bf16_m1_n512_k256",
        mode=SweepMode.SMOKE,
        device="cuda:1",
        ceiling=load_hardware_ceiling("MI300X"),
        backend=KernelBackend.PYTORCH,
    )
    assert rc == 0
    assert entered_devices == ["cuda:1"]
