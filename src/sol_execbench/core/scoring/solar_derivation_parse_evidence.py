# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Semantic group and tensor parsing for SOLAR derivation sidecars."""

from __future__ import annotations

from typing import Any

from sol_execbench.core.scoring.parsing_utils import (
    ensure_dict as _ensure_dict,
    parse_confidence as _parse_confidence,
    parse_list as _parse_list,
    parse_optional_str as _parse_optional_str,
    parse_str as _parse_str,
)
from sol_execbench.core.scoring.solar_derivation_models import (
    SOLAR_BOUND_LIMITING_RESOURCES,
    SolarBoundEvidence,
    SolarByteEvidence,
    SolarFormulaEvidence,
    SolarSemanticGroupEvidence,
    SolarSubroleEvidence,
    SolarTensorEvidence,
)

from .solar_derivation_parse_sources import _evidence_source_from_dict
from .solar_derivation_parse_utils import (
    _parse_dict,
    _parse_non_negative_float,
    _parse_object_map,
    _parse_shape,
    _parse_str_map,
    _parse_str_tuple,
    _parse_status,
    _require_exact_keys,
)

def _group_from_dict(payload: Any, index: int) -> SolarSemanticGroupEvidence:
    source = f"groups[{index}]"
    raw = _ensure_dict(payload, source=source)
    _require_exact_keys(
        raw,
        {
            "family",
            "group_id",
            "node_ids",
            "subroles",
            "confidence",
            "status",
            "required_evidence",
            "missing_evidence",
            "warning_prefixes",
            "source",
            "rationale",
            "formula_evidence",
            "byte_evidence",
            "bound_evidence",
        },
        source=source,
    )
    status = _parse_status(raw, "status", source=source)
    return SolarSemanticGroupEvidence(
        family=_parse_str(raw, "family", source=source),
        group_id=_parse_str(raw, "group_id", source=source),
        node_ids=_parse_str_tuple(raw, "node_ids", source=source),
        subroles=tuple(
            _subrole_from_dict(item, subrole_index, group_index=index)
            for subrole_index, item in enumerate(
                _parse_list(raw, "subroles", source=source)
            )
        ),
        confidence=_parse_confidence(raw, "confidence", source=source),
        status=status,
        required_evidence=_parse_str_tuple(raw, "required_evidence", source=source),
        missing_evidence=_parse_str_tuple(raw, "missing_evidence", source=source),
        warning_prefixes=_parse_str_tuple(raw, "warning_prefixes", source=source),
        source=_evidence_source_from_dict(
            _parse_dict(raw, "source", source=source), source=f"{source}.source"
        ),
        rationale=_parse_str(raw, "rationale", source=source),
        formula_evidence=tuple(
            _formula_evidence_from_dict(item, evidence_index, group_index=index)
            for evidence_index, item in enumerate(
                _parse_list(raw, "formula_evidence", source=source)
            )
        ),
        byte_evidence=tuple(
            _byte_evidence_from_dict(item, evidence_index, group_index=index)
            for evidence_index, item in enumerate(
                _parse_list(raw, "byte_evidence", source=source)
            )
        ),
        bound_evidence=tuple(
            _bound_evidence_from_dict(item, evidence_index, group_index=index)
            for evidence_index, item in enumerate(
                _parse_list(raw, "bound_evidence", source=source)
            )
        ),
    )


def _formula_evidence_from_dict(
    payload: Any,
    index: int,
    *,
    group_index: int,
) -> SolarFormulaEvidence:
    source = f"groups[{group_index}].formula_evidence[{index}]"
    raw = _ensure_dict(payload, source=source)
    _require_exact_keys(
        raw,
        {
            "node_id",
            "family",
            "formula_kind",
            "formula",
            "formula_inputs",
            "source",
            "confidence",
            "rationale",
        },
        source=source,
    )
    return SolarFormulaEvidence(
        node_id=_parse_str(raw, "node_id", source=source),
        family=_parse_str(raw, "family", source=source),
        formula_kind=_parse_str(raw, "formula_kind", source=source),
        formula=_parse_str(raw, "formula", source=source),
        formula_inputs=_parse_object_map(raw, "formula_inputs", source=source),
        source=_evidence_source_from_dict(
            _parse_dict(raw, "source", source=source), source=f"{source}.source"
        ),
        confidence=_parse_confidence(raw, "confidence", source=source),
        rationale=_parse_str(raw, "rationale", source=source),
    )


def _byte_evidence_from_dict(
    payload: Any,
    index: int,
    *,
    group_index: int,
) -> SolarByteEvidence:
    source = f"groups[{group_index}].byte_evidence[{index}]"
    raw = _ensure_dict(payload, source=source)
    _require_exact_keys(
        raw,
        {
            "node_id",
            "family",
            "read_bytes",
            "write_bytes",
            "intermediate_bytes",
            "movement_bytes",
            "total_bytes",
            "dtype_inputs",
            "tensor_ids",
            "source",
            "confidence",
            "rationale",
        },
        source=source,
    )
    return SolarByteEvidence(
        node_id=_parse_str(raw, "node_id", source=source),
        family=_parse_str(raw, "family", source=source),
        read_bytes=_parse_non_negative_float(raw, "read_bytes", source=source),
        write_bytes=_parse_non_negative_float(raw, "write_bytes", source=source),
        intermediate_bytes=_parse_non_negative_float(
            raw, "intermediate_bytes", source=source
        ),
        movement_bytes=_parse_non_negative_float(raw, "movement_bytes", source=source),
        total_bytes=_parse_non_negative_float(raw, "total_bytes", source=source),
        dtype_inputs=_parse_str_map(raw, "dtype_inputs", source=source),
        tensor_ids=_parse_str_tuple(raw, "tensor_ids", source=source),
        source=_evidence_source_from_dict(
            _parse_dict(raw, "source", source=source), source=f"{source}.source"
        ),
        confidence=_parse_confidence(raw, "confidence", source=source),
        rationale=_parse_str(raw, "rationale", source=source),
    )


def _bound_evidence_from_dict(
    payload: Any,
    index: int,
    *,
    group_index: int,
) -> SolarBoundEvidence:
    source = f"groups[{group_index}].bound_evidence[{index}]"
    raw = _ensure_dict(payload, source=source)
    _require_exact_keys(
        raw,
        {
            "node_id",
            "family",
            "compute_bound_ms",
            "memory_bound_ms",
            "limiting_resource",
            "sol_bound_ms",
            "source",
            "confidence",
            "rationale",
        },
        source=source,
    )
    limiting_resource = _parse_str(raw, "limiting_resource", source=source)
    if limiting_resource not in SOLAR_BOUND_LIMITING_RESOURCES:
        valid = ", ".join(sorted(SOLAR_BOUND_LIMITING_RESOURCES))
        raise ValueError(
            f"{source}.limiting_resource has invalid value "
            f"'{limiting_resource}', expected one of: {valid}"
        )
    return SolarBoundEvidence(
        node_id=_parse_str(raw, "node_id", source=source),
        family=_parse_str(raw, "family", source=source),
        compute_bound_ms=_parse_non_negative_float(
            raw, "compute_bound_ms", source=source
        ),
        memory_bound_ms=_parse_non_negative_float(
            raw, "memory_bound_ms", source=source
        ),
        limiting_resource=limiting_resource,
        sol_bound_ms=_parse_non_negative_float(raw, "sol_bound_ms", source=source),
        source=_evidence_source_from_dict(
            _parse_dict(raw, "source", source=source), source=f"{source}.source"
        ),
        confidence=_parse_confidence(raw, "confidence", source=source),
        rationale=_parse_str(raw, "rationale", source=source),
    )


def _subrole_from_dict(
    payload: Any,
    index: int,
    *,
    group_index: int,
) -> SolarSubroleEvidence:
    source = f"groups[{group_index}].subroles[{index}]"
    raw = _ensure_dict(payload, source=source)
    _require_exact_keys(
        raw,
        {
            "name",
            "node_ids",
            "tensor_ids",
            "source",
            "confidence",
            "rationale",
            "missing_evidence",
        },
        source=source,
    )
    return SolarSubroleEvidence(
        name=_parse_str(raw, "name", source=source),
        node_ids=_parse_str_tuple(raw, "node_ids", source=source),
        tensor_ids=_parse_str_tuple(raw, "tensor_ids", source=source),
        source=_evidence_source_from_dict(
            _parse_dict(raw, "source", source=source), source=f"{source}.source"
        ),
        confidence=_parse_confidence(raw, "confidence", source=source),
        rationale=_parse_str(raw, "rationale", source=source),
        missing_evidence=_parse_str_tuple(raw, "missing_evidence", source=source),
    )


def _tensor_from_dict(payload: Any, index: int) -> SolarTensorEvidence:
    source = f"tensors[{index}]"
    raw = _ensure_dict(payload, source=source)
    _require_exact_keys(
        raw,
        {
            "tensor_id",
            "name",
            "shape",
            "dtype",
            "semantic_axes",
            "source",
            "producer_node_id",
            "missing_evidence",
        },
        source=source,
    )
    return SolarTensorEvidence(
        tensor_id=_parse_str(raw, "tensor_id", source=source),
        name=_parse_str(raw, "name", source=source),
        shape=_parse_shape(raw, "shape", source=source),
        dtype=_parse_str(raw, "dtype", source=source),
        semantic_axes=_parse_str_tuple(raw, "semantic_axes", source=source),
        source=_evidence_source_from_dict(
            _parse_dict(raw, "source", source=source), source=f"{source}.source"
        ),
        producer_node_id=_parse_optional_str(raw, "producer_node_id", source=source),
        missing_evidence=_parse_str_tuple(raw, "missing_evidence", source=source),
    )
