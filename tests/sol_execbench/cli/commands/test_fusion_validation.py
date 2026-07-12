# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from sol_execbench.cli.commands.fusion_validation import (
    _built_in_probe_command,
    collect_fusion_validation,
)
from sol_execbench.core.scoring.fusion_validation import (
    FusionSignature,
    FusionValidationArtifact,
    FusionValidationCase,
    KernelResourceEvidence,
    PerformanceEvidence,
    sha256_file,
)
from sol_execbench.core.scoring.hardware_calibration.environment import GpuEnvironment


def _capacity_kernel() -> KernelResourceEvidence:
    return KernelResourceEvidence(
        kernel_name="sol_fusion_rms_mean_epsilon",
        binary_sha256="a" * 64,
        source_sha256="b" * 64,
        compile_command=("hipcc", "probe.hip"),
        architecture="gfx1200",
        vgpr_count=32,
        sgpr_count=16,
        vgpr_spill_count=0,
        sgpr_spill_count=0,
        private_segment_bytes=0,
        static_lds_bytes=1024,
        dynamic_lds_bytes=0,
        lds_limit_bytes=65536,
        active_blocks_per_multiprocessor=1,
        launch_passed=True,
        correctness_passed=True,
    )


def test_collect_uses_built_in_driver_when_manifest_has_no_override(
    tmp_path: Path,
) -> None:
    workload_uuid = "rms-workload"
    manifest_path = tmp_path / "suite.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "sol_execbench.representative_suite.v1",
                "workloads": [
                    {"definition": "025_rmsnorm_h4096", "workload_uuid": workload_uuid}
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "evidence.json"

    def fake_runner(command, **_kwargs):
        assert tuple(command) == _built_in_probe_command()
        evidence = FusionValidationArtifact(
            architecture="gfx1200",
            gpu_uuid="GPU-a",
            rocm_version="7.1",
            hipcc_version="7.1",
            clocks_locked=False,
            suite_manifest_sha256=sha256_file(manifest_path),
            benchmark_root_sha256="c" * 64,
            generated_at="2026-07-12T00:00:00Z",
            cases=(
                FusionValidationCase(
                    evidence_id="rms-case",
                    workload_uuid=workload_uuid,
                    variant_id="rmsnorm_fp32_mean_epsilon",
                    signature=FusionSignature(
                        "reduction_epilogue.v1",
                        1,
                        ("mean", "add"),
                        "fp32",
                        ((1, 4096),),
                        ((1,),),
                        {"workgroup_size": 256},
                    ),
                    fused=_capacity_kernel(),
                    unfused=(_capacity_kernel(),),
                    capacity_status="passed",
                    performance=PerformanceEvidence("not_measured", (), (), None, None),
                ),
            ),
        )
        output.write_text(json.dumps(evidence.to_dict()), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    result = collect_fusion_validation(
        device=0,
        architecture="gfx1200",
        suite_manifest=manifest_path,
        benchmark_root=tmp_path,
        output=output,
        require_clock_lock=False,
        runner=fake_runner,
        discover=lambda _: GpuEnvironment(
            device=0, architecture="gfx1200", uuid="GPU-a", rocm_version="7.1"
        ),
    )

    assert result.cases[0].capacity_status == "passed"
