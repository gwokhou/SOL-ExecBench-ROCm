"""Pure-CPU unit and integration tests for the Phase 1 3-operator MVP."""

from __future__ import annotations

import csv
import json
import subprocess
from importlib.resources import files
from pathlib import Path

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
from scripts.mvp_v1.anti_distortion import (
    LLCRotatingFlusher,
    RotatingBufferPool,
    acquire_dvfs_telemetry_window,
    recommended_llc_flush_bytes,
    validate_materialized_outputs,
)
from scripts.mvp_v1.cloud_runner import main as runner_main
from scripts.mvp_v1.counter_capture import (
    collect_hardware_counters,
    parse_ncu_csv_metrics,
    parse_rocprof_mvp_passes,
)
from scripts.mvp_v1.report import compare_summaries_markdown
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
    b200 = load_hardware_ceiling("B200")
    assert mi300x.hbm_bandwidth_bytes_per_s == pytest.approx(5.3e12)
    assert b200.peak_fp8_flops_per_s == pytest.approx(4.5e15)

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
        "gqa_paged_decode_bf16",
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

    b200_flush = recommended_llc_flush_bytes(
        128 * 1024 * 1024,
        device="cuda:0",
        free_memory_bytes=180 * 1024**3,
    )
    assert b200_flush == 256 * 1024 * 1024

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

    monkeypatch.setattr(
        "scripts.mvp_v1.anti_distortion.set_peak_clocks",
        lambda: (_ for _ in ()).throw(PermissionError("no sudo")),
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


def test_counter_parsers_and_graceful_ncu_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
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

    ncu_csv = (
        "==PROF== Connected to process\n"
        '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
        '"0","ampere_bf16","dram__bytes.sum","Mbyte","2.5"\n'
        '"0","ampere_bf16","sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed","%","74.2"\n'
        '"0","ampere_bf16","gpu__time_duration.sum","usecond","120.0"\n'
    )
    cuda_summary = parse_ncu_csv_metrics(ncu_csv)
    assert cuda_summary.status == "collected"
    assert cuda_summary.memory_bytes == pytest.approx(2.5e6)
    assert cuda_summary.tensor_core_pct == pytest.approx(74.2)
    assert cuda_summary.kernel_duration_ns == pytest.approx(120_000.0)

    monkeypatch.setattr(
        "scripts.mvp_v1.counter_capture.resolve_tool_path",
        lambda _name: Path("/usr/bin/ncu"),
    )
    monkeypatch.setattr(
        "scripts.mvp_v1.counter_capture.run_in_process_group_bounded",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["ncu"], 1, "", "==ERROR== ERR_NVGPUCTRPERM"
        ),
    )
    denied = collect_hardware_counters(
        ["python"],
        backend=VendorBackend.NVIDIA_CUDA,
        architecture="sm_100",
        workload_id="w1",
    )
    assert denied.status == "unavailable"
    assert "ERR_NVGPUCTRPERM" in (denied.note or "")


def test_end_to_end_cpu_smoke_runner_and_cross_vendor_compare(
    tmp_path: Path,
    capsys,
) -> None:
    mi_dir = tmp_path / "mi300x"
    b2_dir = tmp_path / "b200"
    assert (
        runner_main(
            [
                "--mode",
                "smoke",
                "--device",
                "cpu",
                "--target-override",
                "MI300X",
                "--output-dir",
                str(mi_dir),
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
                "--target-override",
                "B200",
                "--output-dir",
                str(b2_dir),
            ]
        )
        == 0
    )

    summary = json.loads((mi_dir / "summary.json").read_text("utf-8"))
    expected_count = len(build_workload_specs(SweepMode.SMOKE))
    assert len(summary["rows"]) == expected_count
    assert all(row["correctness_passed"] for row in summary["rows"])

    with (mi_dir / "summary.csv").open(newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == expected_count

    traces = [
        Trace.model_validate_json(line)
        for line in (mi_dir / "traces.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    assert len(traces) == expected_count
    assert all(trace.is_successful() for trace in traces)

    cmp_md = compare_summaries_markdown(
        mi_dir / "summary.json", b2_dir / "summary.json"
    )
    assert "MI300X vs B200" in cmp_md
    assert "gqa_paged_decode_bf16_b2_kv64" in cmp_md

    capsys.readouterr()
    assert (
        runner_main(
            [
                "--compare",
                str(mi_dir / "summary.json"),
                str(b2_dir / "summary.json"),
            ]
        )
        == 0
    )
    assert "MI300X vs B200" in capsys.readouterr().out
