"""Frozen family rules and pure assembly helpers for the L3-c matrix."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

FAMILY_RULES: tuple[tuple[str, frozenset[str], frozenset[str]], ...] = (
    (
        "explicit_backward",
        frozenset({"bwd", "backward"}),
        frozenset(),
    ),
    (
        "attention",
        frozenset({"attention"}),
        frozenset({"sdpa", "scaled_dot_product", "attention"}),
    ),
    (
        "low_precision",
        frozenset({"quant", "fp8", "i8", "mxfp8"}),
        frozenset({"quantize", "dequantize"}),
    ),
    ("convolution", frozenset({"conv"}), frozenset({"conv"})),
    (
        "matmul",
        frozenset({"matmul", "gemm", "matrix"}),
        frozenset({"matmul", "mm", "bmm", "einsum", "linear"}),
    ),
    (
        "normalization",
        frozenset({"norm", "softmax"}),
        frozenset({"norm", "softmax"}),
    ),
    (
        "reduction",
        frozenset({"sum", "reduction", "pool"}),
        frozenset(
            {"sum", "mean", "max", "min", "arg", "topk", "reduction", "pool"}
        ),
    ),
    (
        "view_alias",
        frozenset({"transpose"}),
        frozenset(
            {
                "view",
                "reshape",
                "permute",
                "transpose",
                "slice",
                "select",
                "expand",
                "repeat",
                "concat",
                "cat",
                "split",
                "gather",
                "index",
                "scatter",
                "embedding",
            }
        ),
    ),
)
ELEMENTWISE_TOKENS = frozenset(
    {
        "mul",
        "add",
        "div",
        "sub",
        "pow",
        "exp",
        "sqrt",
        "rsqrt",
        "gelu",
        "silu",
        "relu",
        "leaky",
        "sigmoid",
        "tanh",
        "abs",
        "neg",
        "clamp",
        "to",
        "copy",
        "where",
        "erf",
        "reciprocal",
        "log",
    }
)
ROADMAP_FAMILIES = (
    "explicit_backward",
    "attention",
    "low_precision",
    "matmul",
    "normalization",
    "reduction",
    "view_alias",
    "mutation",
)


def classify_target(target: str) -> str:
    """Classify one op target into a family, never silently dropping it."""
    lowered = target.lower()
    for family, _, op_tokens in FAMILY_RULES:
        if any(token in lowered for token in op_tokens):
            return family
    if any(token in lowered for token in ELEMENTWISE_TOKENS):
        return "elementwise"
    return "other"


def classify_families(
    problem_name: str,
    op_targets: Sequence[str],
    *,
    has_mutation: bool,
) -> list[str]:
    """Return the ordered multi-tag family list for one matrix row."""
    families: list[str] = []
    lowered_name = problem_name.lower()
    for family, name_tokens, _ in FAMILY_RULES:
        if any(token in lowered_name for token in name_tokens):
            families.append(family)
    for target in op_targets:
        family = classify_target(target)
        if family not in families:
            families.append(family)
    if has_mutation and "mutation" not in families:
        families.append("mutation")
    return families


def matrix_row(
    item_id: str,
    ir_path: str,
    audit: Mapping[str, Any],
    analysis: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Assemble one coverage row from an audit record and analysis record."""
    status = audit.get("status")
    failure_stage = audit.get("failure_stage")
    reason = audit.get("reason_code")
    gradient_failed = reason == "gradient_verification_failed"
    verification_failed = (
        status != "ready" and failure_stage == "conversion_verification"
    )
    row: dict[str, Any] = {
        "item": item_id,
        "ir_path": ir_path,
        "audit_status": status,
        "extraction": "passed"
        if status == "ready" or failure_stage != "graph_extraction"
        else f"failed:{reason}",
        "conversion": "passed"
        if status == "ready" or failure_stage != "ir_conversion"
        else f"failed:{reason}",
        "verification_precision": "passed"
        if status == "ready" or not verification_failed or gradient_failed
        else f"failed:{reason}",
        "verification_gradient": "failed:gradient_verification_failed"
        if gradient_failed
        else ("passed" if status == "ready" else "not_reached"),
        "effects_alias": None,
        "effects_mutation": None,
        "formal_analysis": "not_reached",
        "resource_totals": None,
        "ops": [],
        "families": [],
    }
    if analysis is not None:
        row["ops"] = list(analysis.get("ops", []))
        row["effects_alias"] = analysis.get("has_alias")
        row["effects_mutation"] = analysis.get("has_mutation")
        analysis_status = analysis.get("status")
        if analysis_status == "analyzed":
            row["formal_analysis"] = "analyzed"
            row["resource_totals"] = analysis.get("totals")
        else:
            row["formal_analysis"] = f"failed:{analysis.get('reason_code')}"
    problem = item_id.rsplit("/", 2)[-2]
    row["families"] = classify_families(
        problem,
        row["ops"],
        has_mutation=bool(row["effects_mutation"]),
    )
    return row


def reconcile_canary(
    canary_records: Mapping[str, Mapping[str, Any]],
    fresh_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare fresh torchview outcomes against the L1 canary wall."""
    mismatches = []
    for item, canary in sorted(canary_records.items()):
        fresh = fresh_records.get(item)
        if fresh is None:
            mismatches.append({"item": item, "kind": "missing_in_fresh"})
            continue
        for field in ("status", "failure_stage", "reason_code"):
            if canary.get(field) != fresh.get(field):
                mismatches.append(
                    {
                        "item": item,
                        "kind": field,
                        "canary": canary.get(field),
                        "fresh": fresh.get(field),
                    }
                )
    return {
        "canary_items": len(canary_records),
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def attribute_failure(row: Mapping[str, Any]) -> dict[str, Any]:
    """Attribute one failed row to a family and its first failed dimension."""
    dimensions = (
        "extraction",
        "conversion",
        "verification_precision",
        "verification_gradient",
        "formal_analysis",
    )
    gap = "none"
    for dimension in dimensions:
        value = str(row.get(dimension, ""))
        if value.startswith("failed"):
            gap = value
            break
    return {
        "item": row["item"],
        "ir_path": row["ir_path"],
        "families": row["families"],
        "gap": gap,
    }


KNOWN_FAMILIES = frozenset(
    {family for family, _, _ in FAMILY_RULES}
    | {"elementwise", "other", "mutation"}
)


def evaluate_l3c_gate(
    rows: Sequence[Mapping[str, Any]],
    reconciliation: Mapping[str, Any],
    attribution: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate the pre-fixed L3-c validity gate (spec section 4)."""
    failures: list[str] = []
    if any("worker_error" in str(row.get("audit_status")) for row in rows):
        failures.append("worker_error_rows_present")
    for row in rows:
        if row["formal_analysis"] == "analyzed":
            if not row["ops"]:
                failures.append(f"{row['item']}:{row['ir_path']}:empty_ops")
            totals = row.get("resource_totals") or {}
            if totals.get("flops") is None and totals.get("macs") is None:
                failures.append(
                    f"{row['item']}:{row['ir_path']}:missing_totals"
                )
        for target in row["ops"]:
            if classify_target(target) not in KNOWN_FAMILIES:
                failures.append(f"unclassified:{target}")
    if not reconciliation["passed"]:
        failures.append("canary_reconciliation_mismatch")
    family_presence = {
        family: any(family in row["families"] for row in rows)
        for family in ROADMAP_FAMILIES
    }
    return {
        "passed": not failures,
        "failures": failures[:64],
        "rows": len(rows),
        "attribution_count": len(attribution),
        "family_presence": family_presence,
    }
