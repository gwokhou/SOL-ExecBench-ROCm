# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Freeze the bounded gfx1200 representative multi-operator suite."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPRESENTATIVE_SUITE_SCHEMA_VERSION = "sol_execbench.representative_suite.v1"
REPRESENTATIVE_SUITE_SCOPE = "gfx1200:representative-multi-op:87-workloads"
_CLASSES = (
    ("FlashInfer-Bench/004_gemm_n128_k2048", "004_gemm_n128_k2048", 25),
    ("FlashInfer-Bench/025_rmsnorm_h4096", "025_rmsnorm_h4096", 14),
    (
        "L1/005_conv_gated_projection_with_causal_conv",
        "005_conv_gated_projection_with_causal_conv",
        16,
    ),
    ("L1/083_attention_score_value_matmul", "083_attention_score_value_matmul", 16),
    (
        "L1/061_tanh_gated_residual_add_backward",
        "061_tanh_gated_residual_add_backward",
        16,
    ),
)


def build_representative_suite_manifest(benchmark_root: Path) -> dict[str, Any]:
    """Return a deterministic, UUID-pinned suite manifest from local benchmark data."""
    root = Path(benchmark_root)
    workloads: list[dict[str, str]] = []
    classes: list[dict[str, Any]] = []
    for relative_path, definition, expected_count in _CLASSES:
        path = root / relative_path / "workload.jsonl"
        try:
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line
            ]
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cannot load representative suite workload file: {path}"
            ) from exc
        uuids = sorted(row.get("uuid") for row in rows if isinstance(row, dict))
        if len(uuids) != expected_count or any(
            not isinstance(value, str) or not value for value in uuids
        ):
            raise ValueError(
                f"representative class {relative_path} must contain {expected_count} UUID workloads"
            )
        classes.append({"definition": definition, "workload_count": expected_count})
        workloads.extend(
            {"definition": definition, "workload_uuid": uuid} for uuid in uuids
        )
    payload = {
        "schema_version": REPRESENTATIVE_SUITE_SCHEMA_VERSION,
        "architecture": "gfx1200",
        "scope": REPRESENTATIVE_SUITE_SCOPE,
        "classes": classes,
        "workloads": workloads,
    }
    payload["payload_sha256"] = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return payload
