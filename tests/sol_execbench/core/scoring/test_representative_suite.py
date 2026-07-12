# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from sol_execbench.core.scoring.representative_suite import (
    REPRESENTATIVE_SUITE_SCOPE,
    build_representative_suite_manifest,
)


def test_representative_suite_freezes_exact_87_workload_denominator(tmp_path) -> None:
    root = tmp_path / "benchmark"
    classes = (
        ("FlashInfer-Bench/004_gemm_n128_k2048", 25),
        ("FlashInfer-Bench/025_rmsnorm_h4096", 14),
        ("L1/005_conv_gated_projection_with_causal_conv", 16),
        ("L1/083_attention_score_value_matmul", 16),
        ("L1/061_tanh_gated_residual_add_backward", 16),
    )
    for relative_path, count in classes:
        file_path = root / relative_path / "workload.jsonl"
        file_path.parent.mkdir(parents=True)
        file_path.write_text(
            "\n".join(
                json.dumps({"uuid": f"{count}-{index}"}) for index in range(count)
            )
            + "\n",
            encoding="utf-8",
        )

    manifest = build_representative_suite_manifest(root)

    assert manifest["scope"] == REPRESENTATIVE_SUITE_SCOPE
    assert len(manifest["workloads"]) == 87
    assert manifest["workloads"][0]["definition"] == "004_gemm_n128_k2048"


def test_representative_suite_rejects_changed_class_denominator(tmp_path) -> None:
    file_path = tmp_path / "FlashInfer-Bench/004_gemm_n128_k2048/workload.jsonl"
    file_path.parent.mkdir(parents=True)
    file_path.write_text('{"uuid": "only-one"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="must contain 25 UUID workloads"):
        build_representative_suite_manifest(tmp_path)
