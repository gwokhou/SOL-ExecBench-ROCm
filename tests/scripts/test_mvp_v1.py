"""Pure-CPU unit and integration tests for the Phase 1 3-operator MVP."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from scripts.mvp_v1 import (
    MVPOperator,
    PrecisionMode,
    VendorBackend,
    bound_fused_moe,
    bound_gqa_paged_decode,
    bound_skinny_gemm,
    load_hardware_ceiling,
)
from scripts.mvp_v1.analytical_roofline import detect_hardware_ceiling
from scripts.mvp_v1.anti_distortion import (
    LLCRotatingFlusher,
    RotatingBufferPool,
    _extract_amd_metrics,
    acquire_dvfs_telemetry_window,
    recommended_llc_flush_bytes,
    validate_materialized_outputs,
)
from scripts.mvp_v1.cloud_runner import (
    main as runner_main,
    run_benchmark_campaign,
)
from scripts.mvp_v1.counter_capture import (
    PROFILE_RANGE_MARKER,
    CounterCaptureSummary,
    _execute_rocm_groups,
    _filter_pass_csv_to_marker,
    collect_hardware_counters,
    parse_rocprof_mvp_passes,
)
from scripts.mvp_v1.kernels import KernelBackend, make_skinny_gemm_runner
from scripts.mvp_v1.report import compare_summaries_markdown
from scripts.mvp_v1.triton_kernels import (
    _build_moe_tile_schedule,
    select_gqa_num_splits,
)
from scripts.mvp_v1.vendor_adapters import (
    PreparedRunner,
    try_make_aiter_gemm_runner,
    try_make_aiter_gqa_runner,
    try_make_aiter_moe_runner,
    try_make_vllm_gemm_runner,
    try_make_vllm_gqa_runner,
    try_make_vllm_moe_runner,
)
from scripts.mvp_v1.workloads import (
    SweepMode,
    build_workload_specs,
    materialize_trace_corpus,
)
from sol_execbench.core.bench.performance_model.counter_metrics import (
    counter_memory_bytes,
)
from sol_execbench.core.bench.reward_hack import RewardHackError
from sol_execbench.core.bench.rocm_profiler.counters import (
    load_counter_manifest,
    parse_available_counters,
    select_counter_groups,
)
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.dtypes import (
    dtype_storage_bits,
    dtype_str_to_torch_dtype,
)
from sol_execbench.core.data.trace import Trace
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.data.workload_validation import (
    validate_problem_contract,
)
from solar.rocm.architecture import ArchitectureProfile


def test_mi300x_fnuz_fp8_profile_and_dtypes() -> None:
    profile = ArchitectureProfile.load("MI300X")
    assert profile.peak_ops_per_second["fp8"] == pytest.approx(2614.9e12)
    assert profile.tensor_precision("float8_e4m3fnuz") == "fp8"
    assert profile.tensor_precision("float8_e5m2fnuz") == "fp8"
    with pytest.raises(ValueError, match="not supported"):
        profile.tensor_precision("float8_e4m3fn")

    assert dtype_str_to_torch_dtype("float8_e4m3fnuz") is torch.float8_e4m3fnuz
    assert dtype_storage_bits("float8_e4m3fnuz") == 8

    ceiling = load_hardware_ceiling("MI300X")
    assert ceiling.vendor is VendorBackend.AMD_ROCM
    assert ceiling.llc_bytes == 256 * 1024 * 1024
    assert ceiling.fp8_torch_dtype() is torch.float8_e4m3fnuz


def test_gfx942_counter_manifest_fallback_and_bytes() -> None:
    manifest_path = files("sol_execbench.data.rocprofv3_counters").joinpath(
        "gfx942_v1.yaml"
    )
    manifest = load_counter_manifest(str(manifest_path))
    available = parse_available_counters(
        "Counter_Name: SQ_WAVES\n"
        "Counter_Name: TCC_EA0_RDREQ_32B_sum\n"
        "Counter_Name: TCC_EA0_WRREQ_64B_sum\n"
        "Counter_Name: TCC_MISS_sum\n"
        "Counter_Name: SQ_INSTS_MFMA_sum\n"
    )
    groups, missing = select_counter_groups(manifest, available)
    assert missing == []
    assert groups[0] == [
        "SQ_WAVES",
        "TCC_EA0_RDREQ_32B_sum",
        "TCC_EA0_WRREQ_64B_sum",
    ]
    assert ["SQ_INSTS_MFMA_sum"] in groups
    total = counter_memory_bytes(
        {"TCC_EA0_RDREQ_32B_SUM": 100.0, "TCC_EA0_WRREQ_64B_SUM": 10.0}
    )
    assert total == pytest.approx(100 * 32.0 + 10 * 64.0)


def test_analytical_roofline_bounds_and_dedup_moe() -> None:
    mi300x = load_hardware_ceiling("MI300X")
    mi308x = load_hardware_ceiling("MI308X")
    assert mi300x.hbm_bandwidth_bytes_per_s == pytest.approx(5.3e12)
    assert mi308x.peak_fp8_flops_per_s == pytest.approx(2.96e14)
    for unsupported in ("B200", "H20", "H100_SXM"):
        with pytest.raises(ValueError, match="Unsupported MVP target"):
            load_hardware_ceiling(unsupported)

    gemm_fp8 = bound_skinny_gemm(16, 4096, 7168, PrecisionMode.FP8, mi300x)
    expected_bytes = float(16 * 7168 + 7168 * 4096 + 16 * 4096 * 2 + 8)
    assert gemm_fp8.flops == pytest.approx(2.0 * 16 * 4096 * 7168)
    assert gemm_fp8.compulsory_bytes == pytest.approx(expected_bytes)
    assert gemm_fp8.bottleneck_spec == "memory"

    moe_m1 = bound_fused_moe(
        1, 2048, 1024, 64, 6, PrecisionMode.BF16, mi300x, active_experts=6
    )
    moe_all = bound_fused_moe(
        1, 2048, 1024, 64, 6, PrecisionMode.BF16, mi300x, active_experts=64
    )
    assert moe_m1.compulsory_bytes < moe_all.compulsory_bytes

    gqa = bound_gqa_paged_decode(
        8, 32, 4, 128, 32768, 16, PrecisionMode.BF16, mi300x
    )
    assert gqa.operator is MVPOperator.GQA_PAGED_DECODE
    assert gqa.bottleneck_spec == "memory"
    assert gqa.t_sol_ms_spec == pytest.approx(
        (gqa.compulsory_bytes / mi300x.hbm_bandwidth_bytes_per_s) * 1e3
    )


def test_materialize_trace_corpus_round_trips_strict_contracts(
    tmp_path: Path,
) -> None:
    written = materialize_trace_corpus(tmp_path, SweepMode.SMOKE)
    assert set(written) == {
        "skinny_gemm_bf16",
        "skinny_gemm_fp8",
        "fused_moe_bf16",
        "fused_moe_fp8",
        "gqa_paged_decode_bf16",
        "mla_paged_decode_bf16",
        "fused_rmsnorm_rope_bf16",
        "swiglu_fp8_quant",
    }
    for def_name, wl_path in written.items():
        def_path = tmp_path / "workloads" / def_name / "definition.json"
        defn = Definition.model_validate_json(def_path.read_text("utf-8"))
        workloads = [
            Workload.model_validate_json(line)
            for line in wl_path.read_text("utf-8").splitlines()
            if line.strip()
        ]
        assert workloads
        validate_problem_contract(defn, workloads)


def test_anti_distortion_flush_ring_anti_dce_and_dvfs(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    mi300x_flush = recommended_llc_flush_bytes(
        256 * 1024 * 1024,
        device="cuda:0",
        free_memory_bytes=190 * 1024**3,
    )
    assert mi300x_flush == 512 * 1024 * 1024

    min_flush = recommended_llc_flush_bytes(
        64 * 1024 * 1024,
        device="cuda:0",
        free_memory_bytes=16 * 1024**3,
    )
    assert min_flush == 256 * 1024 * 1024

    flusher = LLCRotatingFlusher("cpu", 64 * 1024)
    flusher.flush()

    x = torch.ones((64, 64), dtype=torch.float32)
    pool = RotatingBufferPool([x], min_ring_bytes=128 * 1024, max_replicas=4)
    assert pool.ring_replicas == 4
    ptrs = {pool.next_args()[0].data_ptr() for _ in range(4)}
    assert len(ptrs) == 4

    class _ProxyTensor:
        pass

    with pytest.raises(RewardHackError):
        validate_materialized_outputs(_ProxyTensor(), "cpu")

    unlock_calls: list[object] = []
    monkeypatch.setattr(
        "scripts.mvp_v1.anti_distortion.observe_clock_devices",
        lambda: (),
    )
    monkeypatch.setattr(
        "scripts.mvp_v1.anti_distortion.set_peak_clocks",
        lambda *_a: (_ for _ in ()).throw(PermissionError("no sudo")),
    )
    monkeypatch.setattr(
        "scripts.mvp_v1.anti_distortion.unlock_clocks",
        lambda *a: (unlock_calls.append(a), True)[-1],
    )
    monkeypatch.setattr(
        "scripts.mvp_v1.anti_distortion.sample_gpu_telemetry",
        lambda _dev, vendor: __import__(
            "scripts.mvp_v1.anti_distortion", fromlist=["TelemetrySample"]
        ).TelemetrySample(vendor=vendor),
    )
    with acquire_dvfs_telemetry_window(
        "cuda:0", VendorBackend.AMD_ROCM, lock_clocks=True
    ) as holder:
        assert holder["receipt"].lock_requested is True
        assert holder["receipt"].lock_acquired is False
    assert "degraded" in holder["receipt"].lock_note
    assert len(unlock_calls) == 1


def test_counter_parser_rocm_and_cpu_skip(tmp_path: Path) -> None:
    pass_csv = tmp_path / "pmc_1" / "out_counter_collection.csv"
    pass_csv.parent.mkdir(parents=True)
    pass_csv.write_text(
        "Dispatch_Id,Kernel_Name,Grid_Size,Workgroup_Size,Counter_Name,Counter_Value\n"
        '1,gemm_k,"[64,1,1]","[256,1,1]",SQ_WAVES_sum,128\n'
        '1,gemm_k,"[64,1,1]","[256,1,1]",TCC_EA0_RDREQ_32B_sum,1000\n'
        '1,gemm_k,"[64,1,1]","[256,1,1]",TCC_EA0_WRREQ_64B_sum,200\n'
        '1,gemm_k,"[64,1,1]","[256,1,1]",SQ_INSTS_MFMA_sum,512\n',
        encoding="utf-8",
    )
    rocm_summary = parse_rocprof_mvp_passes([pass_csv], workload_uuid="w1")
    assert rocm_summary.status == "collected"
    assert rocm_summary.memory_bytes == pytest.approx(1000 * 32.0 + 200 * 64.0)
    assert rocm_summary.waves == pytest.approx(128.0)
    assert rocm_summary.mfma_instructions == pytest.approx(512.0)

    cpu_skip = collect_hardware_counters(
        ["python"],
        backend=VendorBackend.CPU,
        architecture="cpu",
        workload_id="w1",
    )
    assert cpu_skip.status == "skipped_cpu"


def test_end_to_end_cpu_smoke_runner_and_vertical_compare(
    tmp_path: Path,
    capsys,
) -> None:
    pt_dir = tmp_path / "pytorch"
    port_dir = tmp_path / "portable"
    assert (
        runner_main(
            [
                "--mode",
                "smoke",
                "--device",
                "cpu",
                "--backend",
                "pytorch",
                "--target-override",
                "MI300X",
                "--output-dir",
                str(pt_dir),
            ]
        )
        == 0
    )
    assert (
        runner_main(
            [
                "--mode",
                "smoke",
                "--device",
                "cpu",
                "--backend",
                "portable",
                "--target-override",
                "MI300X",
                "--output-dir",
                str(port_dir),
            ]
        )
        == 0
    )

    summary = json.loads((port_dir / "summary.json").read_text("utf-8"))
    expected_count = len(build_workload_specs(SweepMode.SMOKE))
    assert summary["backend"] == "portable"
    assert len(summary["rows"]) == expected_count
    assert all(row["correctness_passed"] for row in summary["rows"])

    with (port_dir / "summary.csv").open(newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == expected_count

    traces = [
        Trace.model_validate_json(line)
        for line in (port_dir / "traces.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    assert len(traces) == expected_count
    assert all(trace.is_successful() for trace in traces)

    cmp_md = compare_summaries_markdown(
        pt_dir / "summary.json", port_dir / "summary.json"
    )
    assert (
        "ROCm Vertical Stack SOL Diagnosis — MI300X (pytorch vs portable)"
        in (cmp_md)
    )
    assert "gqa_paged_decode_bf16_b2_kv64" in cmp_md

    capsys.readouterr()
    assert (
        runner_main(
            [
                "--compare",
                str(pt_dir / "summary.json"),
                str(port_dir / "summary.json"),
            ]
        )
        == 0
    )
    assert "MI300X (pytorch vs portable)" in capsys.readouterr().out


def test_amd_smi_gpu_data_metric_extraction() -> None:
    payload = {
        "gpu_data": [
            {
                "power": {"socket_power": {"value": 22}},
                "clock": {
                    "gfx_0": {"clk": {"value": 2500}},
                    "mem_0": {"clk": {"value": 1200}},
                },
                "temperature": {"edge": {"value": 45}},
            }
        ]
    }
    assert _extract_amd_metrics(json.dumps(payload)) == (
        2500.0,
        1200.0,
        22.0,
        45.0,
    )


def test_rocprof_marker_window_and_zero_byte_counter_handling(
    tmp_path: Path,
) -> None:
    pass_dir = tmp_path / "pass_1"
    pass_dir.mkdir(parents=True)
    (pass_dir / "10_marker_api_trace.csv").write_text(
        "Domain,Function,Start_Timestamp,End_Timestamp\n"
        f"MARKER_CORE_RANGE_API,{PROFILE_RANGE_MARKER},1000,2000\n",
        encoding="utf-8",
    )
    raw_csv = pass_dir / "10_counter_collection.csv"
    raw_csv.write_text(
        "Dispatch_Id,Kernel_Name,Grid_Size,Workgroup_Size,Counter_Name,Counter_Value,Start_Timestamp\n"
        '1,warmup_k,"[1,1,1]","[64,1,1]",SQ_WAVES_sum,99,500\n'
        '2,flush_k,"[1,1,1]","[64,1,1]",SQ_WAVES_sum,88,900\n'
        '3,target_k,"[4,1,1]","[64,1,1]",SQ_WAVES_sum,32,1500\n'
        '3,target_k,"[4,1,1]","[64,1,1]",FETCH_SIZE,0,1500\n'
        '4,trailing_k,"[1,1,1]","[64,1,1]",SQ_WAVES_sum,77,2500\n',
        encoding="utf-8",
    )
    filtered = _filter_pass_csv_to_marker(raw_csv, pass_dir)
    rows = list(csv.DictReader(filtered.open(encoding="utf-8")))
    assert [(r["Dispatch_Id"], r["Kernel_Name"]) for r in rows] == [
        ("3", "target_k"),
        ("3", "target_k"),
    ]

    summary = parse_rocprof_mvp_passes([filtered], workload_uuid="w_zero")
    assert summary.status == "collected"
    assert summary.memory_bytes is None
    assert summary.waves == pytest.approx(32.0)
    assert "returned 0" in (summary.note or "")


def test_execute_rocm_groups_splits_on_hardware_capability_limit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seen_groups: list[list[str]] = []

    def _fake_pass(
        _rocprof: Path,
        group: list[str],
        _cmd: list[str],
        *,
        pass_dir: Path,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        del env
        seen_groups.append(list(group))
        if len(group) > 1:
            return subprocess.CompletedProcess(
                ["rocprofv3"],
                -6,
                "",
                "Request exceeds the capabilities of the hardware to collect",
            )
        pass_dir.mkdir(parents=True, exist_ok=True)
        (pass_dir / "1_marker_api_trace.csv").write_text(
            "Domain,Function,Start_Timestamp,End_Timestamp\n"
            f"MARKER_CORE_RANGE_API,{PROFILE_RANGE_MARKER},100,200\n",
            encoding="utf-8",
        )
        out = pass_dir / "1_counter_collection.csv"
        out.write_text(
            "Dispatch_Id,Kernel_Name,Grid_Size,Workgroup_Size,Counter_Name,Counter_Value,Start_Timestamp\n"
            f'1,k,"[1,1,1]","[64,1,1]",{group[0]},16,150\n',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(["rocprofv3"], 0, "", "")

    monkeypatch.setattr(
        "scripts.mvp_v1.counter_capture._run_one_rocm_pass",
        _fake_pass,
    )
    csvs, err = _execute_rocm_groups(
        Path("/opt/rocm/bin/rocprofv3"),
        [["SQ_WAVES_sum", "FETCH_SIZE"]],
        ["python"],
        tmp=tmp_path,
        env={},
    )
    assert err is None
    assert seen_groups == [
        ["SQ_WAVES_sum", "FETCH_SIZE"],
        ["SQ_WAVES_sum"],
        ["FETCH_SIZE"],
    ]
    assert len(csvs) == 2


def test_splitk_gqa_split_selection_and_moe_tile_schedule_bijection() -> None:
    assert select_gqa_num_splits(batch=2, kv_heads=2, total_pages=8) == 2
    assert select_gqa_num_splits(batch=1, kv_heads=4, total_pages=256) == 64
    assert select_gqa_num_splits(batch=16, kv_heads=4, total_pages=32768) == 4

    # 20 tokens x top_k=2 = 40 pairs skewed heavily onto expert 2 (>16 rows) and
    # expert 5, leaving experts 0, 1, 3, 4 completely empty.
    logits = torch.full((20, 6), -10.0)
    logits[:15, 2] = 5.0
    logits[:15, 5] = 4.0
    logits[15:, 2] = 5.0
    logits[15:, 5] = 4.0
    weights, tile_expert, tile_pairs = _build_moe_tile_schedule(logits, top_k=2)

    assert weights.shape == (40,)
    assert torch.allclose(weights.view(20, 2).sum(dim=-1), torch.ones(20))
    valid_pairs = tile_pairs[tile_pairs >= 0].tolist()
    assert sorted(valid_pairs) == list(range(40))
    inactive = tile_expert >= 6
    if inactive.any():
        assert torch.all(tile_pairs[inactive] == -1)


def _assert_zero_copy_from_pool(
    runner: PreparedRunner,
    arg_index: int,
    recorded_ptrs: list[int],
) -> None:
    pool = RotatingBufferPool(
        runner.candidate_args, min_ring_bytes=1, max_replicas=2
    )
    pool_args = pool.next_args()
    recorded_ptrs.clear()
    assert runner.fn is not None
    runner.fn(*pool_args)
    assert recorded_ptrs == [pool_args[arg_index].data_ptr()]


def test_rocm_vendor_adapters_prepack_once_and_validate(monkeypatch) -> None:
    import types

    monkeypatch.setattr(
        "scripts.mvp_v1.vendor_adapters._is_rocm_torch", lambda: True
    )
    seen_ptrs: list[int] = []
    corrupt = {"flag": False}

    def _fake_wv_splitk(w_nk, x, cu):
        del cu
        seen_ptrs.append(w_nk.data_ptr())
        out = x @ w_nk.t()
        return out + 9.0 if corrupt["flag"] else out

    fake_ops = types.SimpleNamespace(
        wvSplitK=_fake_wv_splitk,
        paged_attention_rocm=lambda out, *rest: (
            seen_ptrs.append(rest[4].data_ptr()),
            out.copy_(rest[3]),
        ),
    )
    fake_vllm = types.SimpleNamespace(
        _custom_ops=fake_ops,
        model_executor=types.SimpleNamespace(
            layers=types.SimpleNamespace(
                fused_moe=types.SimpleNamespace(
                    fused_moe=lambda t, w1, w2, lg, k, renormalize=True: (
                        seen_ptrs.append(w1.data_ptr()),
                        torch.zeros_like(t)
                        + (99.0 if corrupt["flag"] else 0.0),
                    )[-1]
                )
            )
        ),
    )
    fake_aiter = types.SimpleNamespace(
        gemm_a16w16=lambda x, w_nk, *_r: (
            seen_ptrs.append(w_nk.data_ptr()),
            x @ w_nk.t(),
        )[-1],
        gemm_a8w8_bpreshuffle=lambda x, w, sa, sb, *_r: (
            seen_ptrs.append(w.data_ptr()),
            ((x.float() @ w.t().float()) * (sa * sb)).to(torch.bfloat16),
        )[-1],
        fused_moe=lambda t, w1, w2, wn, ids: (
            seen_ptrs.append(w1.data_ptr()),
            torch.zeros_like(t),
        )[-1],
        pa_fwd_asm=lambda q, k, v, bt, sl, mb: (
            seen_ptrs.append(k.data_ptr()),
            q + (5.0 if corrupt["flag"] else 0.0),
        )[-1],
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
        "aiter.ops": fake_aiter.ops,
        "aiter.ops.shuffle": fake_aiter.ops.shuffle,
    }.items():
        monkeypatch.setitem(sys.modules, name, cast(Any, mod))

    x_bf = torch.ones((2, 16), dtype=torch.bfloat16)
    w_t_bf = torch.full((16, 8), 0.5, dtype=torch.bfloat16).t().contiguous().t()
    gemm_vllm = try_make_vllm_gemm_runner(
        [x_bf, w_t_bf],
        fp8=False,
        ref_fn=lambda a, b: (a.float() @ b.float()).to(torch.bfloat16),
        max_atol=1e-2,
    )
    assert gemm_vllm.implementation_name == "vllm_wvSplitK"
    assert gemm_vllm.candidate_args[1].is_contiguous()
    _assert_zero_copy_from_pool(gemm_vllm, 1, seen_ptrs)

    gemm_a16 = try_make_aiter_gemm_runner(
        [x_bf, w_t_bf],
        fp8=False,
        ref_fn=lambda a, b: (a.float() @ b.float()).to(torch.bfloat16),
        max_atol=1e-2,
    )
    assert gemm_a16.implementation_name == "aiter_gemm_a16w16"
    _assert_zero_copy_from_pool(gemm_a16, 1, seen_ptrs)

    x_fp8 = x_bf.to(torch.float8_e4m3fnuz)
    w_t_fp8 = w_t_bf.to(torch.float8_e4m3fnuz)
    sa = torch.ones((), dtype=torch.float32)
    sb = torch.ones((), dtype=torch.float32)
    gemm_a8 = try_make_aiter_gemm_runner(
        [x_fp8, w_t_fp8],
        fp8=True,
        scale_a=sa,
        scale_b=sb,
        ref_fn=lambda a, b: (a.float() @ b.float()).to(torch.bfloat16),
        max_atol=2e-1,
    )
    assert gemm_a8.implementation_name == "aiter_gemm_a8w8_bpreshuffle"
    _assert_zero_copy_from_pool(gemm_a8, 1, seen_ptrs)

    tokens = torch.zeros((2, 8), dtype=torch.bfloat16)
    gw = torch.zeros((4, 8, 4), dtype=torch.bfloat16)
    dw = torch.zeros((4, 4, 8), dtype=torch.bfloat16)
    logits = torch.ones((2, 4), dtype=torch.bfloat16)
    moe_args: list[Any] = [tokens, gw, gw, dw, logits, 2]
    moe_vllm = try_make_vllm_moe_runner(
        moe_args, ref_fn=lambda *_r: tokens, max_atol=1e-2
    )
    moe_aiter = try_make_aiter_moe_runner(
        moe_args, ref_fn=lambda *_r: tokens, max_atol=1e-2
    )
    assert moe_vllm.implementation_name == "vllm_fused_moe"
    assert moe_aiter.implementation_name == "aiter_fused_moe"
    assert moe_vllm.candidate_args[1].shape == (4, 8, 8)
    _assert_zero_copy_from_pool(moe_vllm, 1, seen_ptrs)
    _assert_zero_copy_from_pool(moe_aiter, 1, seen_ptrs)

    q = torch.ones((1, 4, 128), dtype=torch.bfloat16)
    k = torch.ones((2, 16, 2, 128), dtype=torch.bfloat16)
    gqa_args: list[Any] = [
        q,
        k,
        k,
        torch.tensor([0, 2], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([16], dtype=torch.int32),
    ]
    gqa_vllm = try_make_vllm_gqa_runner(
        gqa_args, ref_fn=lambda q_in, *_r: q_in, max_atol=1e-2
    )
    gqa_aiter = try_make_aiter_gqa_runner(
        gqa_args, ref_fn=lambda q_in, *_r: q_in, max_atol=1e-2
    )
    assert gqa_vllm.implementation_name == "vllm_paged_attention_rocm"
    assert gqa_aiter.implementation_name == "aiter_pa_fwd_asm"
    assert gqa_aiter.candidate_args[1].shape == (2, 2, 16, 16, 8)
    _assert_zero_copy_from_pool(gqa_vllm, 1, seen_ptrs)
    _assert_zero_copy_from_pool(gqa_aiter, 1, seen_ptrs)

    # AITER ASM paged attention C++ kernel aborts unless head_dim == 128.
    q_d64 = torch.ones((1, 4, 64), dtype=torch.bfloat16)
    k_d64 = torch.ones((2, 16, 2, 64), dtype=torch.bfloat16)
    gqa_d64_aiter = try_make_aiter_gqa_runner(
        [q_d64, k_d64, k_d64, *gqa_args[3:]],
        ref_fn=lambda q_in, *_r: q_in,
        max_atol=1e-2,
    )
    assert gqa_d64_aiter.fn is None
    assert (
        gqa_d64_aiter.implementation_name == "unavailable:aiter_gqa_unsupported"
    )

    corrupt["flag"] = True
    rejected = try_make_aiter_gqa_runner(
        gqa_args, ref_fn=lambda q_in, *_r: q_in, max_atol=1e-2
    )
    assert rejected.fn is None
    assert rejected.implementation_name.startswith("unavailable:")


def test_mi308x_ceilings_cu_aware_detection_and_rejects_non_rocm(
    monkeypatch,
) -> None:
    import types

    mi308x = load_hardware_ceiling("MI308X")
    assert mi308x.expected_compute_units == 80
    assert mi308x.fp8_torch_dtype() is torch.float8_e4m3fnuz
    assert mi308x.hbm_bandwidth_bytes_per_s == pytest.approx(5.3e12)
    assert mi308x.peak_bf16_flops_per_s == pytest.approx(148.0e12)
    assert mi308x.peak_fp8_flops_per_s == pytest.approx(296.0e12)

    assert (
        select_gqa_num_splits(batch=1, kv_heads=4, total_pages=256, num_cus=80)
        == 32
    )
    assert (
        select_gqa_num_splits(batch=1, kv_heads=4, total_pages=256, num_cus=304)
        == 64
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    props_map = [
        ("AMD Instinct MI308X", "gfx942", 80, "MI308X"),
        ("AMD Instinct", "gfx942:sramecc+:xnack-", 80, "MI308X"),
        ("AMD Instinct", "gfx942", 304, "MI300X"),
    ]
    for name, gfx, cus, expected_id in props_map:
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda _idx, n=name, g=gfx, c=cus: types.SimpleNamespace(
                name=n, gcnArchName=g, multi_processor_count=c
            ),
        )
        assert detect_hardware_ceiling("cuda:0").target_id == expected_id

    for bad_name in ("NVIDIA H20", "NVIDIA H200"):
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda _idx, n=bad_name: types.SimpleNamespace(
                name=n, gcnArchName="", multi_processor_count=132
            ),
        )
        with pytest.raises(ValueError, match="Unrecognized GPU"):
            detect_hardware_ceiling("cuda:0")


def test_campaign_counter_circuit_breaker_and_topology_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = {"count": 0}

    def _fake_collect(*_args, **_kwargs) -> CounterCaptureSummary:
        calls["count"] += 1
        return CounterCaptureSummary(
            status="unavailable",
            backend=VendorBackend.CPU,
            note="pmc blocked in VM",
        )

    monkeypatch.setattr(
        "scripts.mvp_v1.cloud_runner.collect_hardware_counters",
        _fake_collect,
    )
    paths = run_benchmark_campaign(
        mode=SweepMode.SMOKE,
        device="cpu",
        target_override="MI308X",
        backend=KernelBackend.AUTO,
        output_dir=tmp_path / "vm_cb",
        collect_counters=True,
        lock_clocks=False,
    )
    assert calls["count"] == 1
    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    statuses = [row["counter_status"] for row in summary["rows"]]
    assert statuses[0] == "unavailable"
    assert all(s == "skipped_circuit_breaker" for s in statuses[1:])
    assert summary["environment_probe"]["expected_compute_units"] == 80
    report_md = paths["report_md"].read_text(encoding="utf-8")
    assert "Runtime Topology" in report_md
    assert "MI308X" in report_md


def test_repo_sources_parse_natively_as_python_3_12() -> None:
    import ast

    repo_root = Path(__file__).resolve().parents[2]
    checked_files = 0
    for sub in ("scripts/mvp_v1", "src/sol_execbench", "src/solar"):
        for py_file in sorted((repo_root / sub).rglob("*.py")):
            if "_vendor" in py_file.parts:
                continue
            original = py_file.read_text(encoding="utf-8")
            ast.parse(original, filename=str(py_file), feature_version=(3, 12))
            checked_files += 1
    assert checked_files >= 600
    assert not (repo_root / "scripts" / "mvp_v1" / "py312_compat.py").exists()


def test_skinny_gemm_fp8_small_m_pad16_and_dequant_fallbacks(
    monkeypatch,
) -> None:
    unavail = PreparedRunner(
        fn=None,
        implementation_name="unavailable:mock",
        candidate_args=[],
        unavailable_reason="mock",
    )
    monkeypatch.setattr(
        "scripts.mvp_v1.kernels.try_make_aiter_gemm_runner",
        lambda *_a, **_kw: unavail,
    )
    monkeypatch.setattr(
        "scripts.mvp_v1.kernels.try_make_vllm_gemm_runner",
        lambda *_a, **_kw: unavail,
    )
    x = torch.ones((4, 32), dtype=torch.bfloat16).to(torch.float8_e4m3fnuz)
    w_t = (
        torch.full((64, 32), 0.5, dtype=torch.bfloat16)
        .to(torch.float8_e4m3fnuz)
        .t()
    )
    sa = torch.tensor(2.0, dtype=torch.float32)
    sb = torch.tensor(0.5, dtype=torch.float32)
    expected = x.to(torch.bfloat16) @ w_t.to(torch.bfloat16)

    def _fake_scaled_mm_native(
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        return ((a.float() @ b.float()) * (scale_a * scale_b)).to(out_dtype)

    monkeypatch.setattr(torch, "_scaled_mm", _fake_scaled_mm_native)
    res_native = make_skinny_gemm_runner(
        prototype_args=[x, w_t],
        fp8=True,
        device="cuda:0",
        scale_a=sa,
        scale_b=sb,
    )
    assert res_native.implementation_name == "torch._scaled_mm"
    assert res_native.fn is not None
    assert torch.allclose(res_native.fn(*res_native.candidate_args), expected)

    def _fake_scaled_mm_align16(
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        if a.shape[0] % 16 != 0:
            raise RuntimeError("hipBLASLt requires M divisible by 16")
        return ((a.float() @ b.float()) * (scale_a * scale_b)).to(out_dtype)

    monkeypatch.setattr(torch, "_scaled_mm", _fake_scaled_mm_align16)
    res_pad16 = make_skinny_gemm_runner(
        prototype_args=[x, w_t],
        fp8=True,
        device="cuda:0",
        scale_a=sa,
        scale_b=sb,
    )
    assert res_pad16.implementation_name == "torch._scaled_mm_pad16"
    assert res_pad16.fn is not None
    out_pad16 = res_pad16.fn(*res_pad16.candidate_args)
    assert out_pad16.shape == (4, 64)
    assert out_pad16.is_contiguous()
    assert torch.allclose(out_pad16, expected)

    def _fake_scaled_mm_broken(
        *_args: object, **_kwargs: object
    ) -> torch.Tensor:
        raise RuntimeError("hipBLASLt unsupported")

    monkeypatch.setattr(torch, "_scaled_mm", _fake_scaled_mm_broken)
    res_deq = make_skinny_gemm_runner(
        prototype_args=[x, w_t],
        fp8=True,
        device="cuda:0",
        scale_a=sa,
        scale_b=sb,
    )
    assert res_deq.implementation_name == "bf16_dequant_matmul_fallback"
    assert res_deq.fn is not None
    assert torch.allclose(res_deq.fn(*res_deq.candidate_args), expected)
