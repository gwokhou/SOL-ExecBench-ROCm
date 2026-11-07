# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Curated R16 SOLAR qualification over the frozen R-set denominator.

The official chain gates on every scored workload; this curated audit uses the
same corpus readiness seam with the eight frozen R workloads and declares R08
as the expected boundary sentinel. A slot passes when it is ready or matches
its declared expected rejection; any other failure fails the whole gate. The
official full-denominator chain remains untouched.
"""

import argparse
import json
import sys
from pathlib import Path

from sol_execbench.core.solar_bridge.corpus_readiness import (
    CorpusReadinessStatus,
    audit_corpus_stage_readiness,
)
from solar.ir.contracts import IRPath
from solar.schema_versions import SchemaVersion

R16_ITEMS: dict[str, str] = {
    "R01": "torch2hip/l1n1_square_matmul/aka-l1n1_square_matmul-w0",
    "R02": "torch2hip/l1n47_sum_reduction/aka-l1n47_sum_reduction-w0",
    "R03": "torch2hip/gpumode_mlp/aka-gpumode_mlp-w0",
    "R04": "torch2hip/dot_product_attention/aka-dot_product_attention-w0",
    "R05": "torch2flydsl/per_token_i8_quant/aka-per_token_i8_quant-w0",
    "R06": "torch2flydsl/moe_topk_softmax/aka-moe_topk_softmax-w0",
    "R07": "torch2hip/l1n8_matmul_irregular/aka-l1n8_matmul_irregular-w0",
    "R08": "instruction2triton/rmsnorm_bwd/aka-rmsnorm_bwd-w0",
}

EXPECTED_REJECTIONS: dict[str, frozenset[str]] = {
    "R08": frozenset(
        {"graph_extraction_failed", "gradient_verification_failed"}
    ),
}


def evaluate_curated_gate(
    records: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Pass only on ready slots or declared, matching expected rejections."""
    verdict: dict[str, object] = {
        "schema_version": SchemaVersion.SOLAR_CURATED_GATE,
        "denominator": "R16",
        "slots": {},
        "passed": True,
    }
    for item, path in sorted(R16_ITEMS.items()):
        record = records[path]
        ready = record.get("status") == "ready"
        expected = EXPECTED_REJECTIONS.get(item, frozenset())
        reason = str(record.get("reason_code") or "")
        expected_match = bool(expected) and reason in expected
        verdict["slots"][item] = {
            "path": path,
            "status": record.get("status"),
            "reason_code": record.get("reason_code"),
            "failure_stage": record.get("failure_stage"),
            "ready": ready,
            "declared_expected_rejection": sorted(expected),
            "expected_match": expected_match,
            "passed": ready or expected_match,
        }
        if not (ready or expected_match):
            verdict["passed"] = False
    return verdict


def main() -> int:
    """Audit the frozen R-set and write the curated gate verdict."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        type=Path,
        help="authored AKA corpus manifest (problems/AMD_AKA/manifest.yaml)",
    )
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--backend",
        choices=[path.value for path in IRPath],
        default=IRPath.TORCHVIEW_EXTENDED_EINSUM.value,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args()

    manifest = args.manifest
    backend = IRPath(args.backend)
    backend_root = args.output_root / backend.value
    readiness_root = backend_root / "readiness"
    result = audit_corpus_stage_readiness(
        manifest,
        readiness_root,
        device=args.device,
        timeout_seconds=args.timeout,
        ir_path=backend,
        selected_item_ids=frozenset(R16_ITEMS.values()),
    )
    records = {
        f"{record['problem_path']}/{record['workload_uuid']}": record
        for record in (
            json.loads(line)
            for line in (readiness_root / "matrix.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        )
    }
    if set(records) != set(R16_ITEMS.values()):
        print(
            f"curated denominator mismatch: "
            f"{set(R16_ITEMS.values()) ^ set(records)}",
            file=sys.stderr,
        )
        return 2
    verdict = evaluate_curated_gate(records)
    verdict["audit_status"] = result.status.value
    verdict["audit_ready"] = result.status is CorpusReadinessStatus.READY
    verdict["ir_path"] = backend.value
    gate_path = backend_root / "curated-gate.json"
    gate_path.write_text(json.dumps(verdict, indent=1, sort_keys=True) + "\n")
    print(
        f"curated gate [{backend.value}]: passed={verdict['passed']} "
        f"(audit={result.status.value}) -> {gate_path}"
    )
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
