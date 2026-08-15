# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Deterministic common-scale workload generation for LLM Core."""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from sol_execbench.core.bench.reference_protocol import (
    MAX_REFERENCE_TENSOR_STORAGE_BYTES,
)
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.definition_models import DType
from sol_execbench.core.data.dtypes import dtype_storage_bits
from sol_execbench.core.data.schema_versions import BenchmarkArtifactSchema
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.data.workload_validation import (
    validate_problem_contract,
)
from sol_execbench.core.dataset.corpus_models import (
    WORKLOAD_GENERATION_PROTOCOL_MAJOR,
    CorpusOperationFamily,
    GeneratedWorkloadRecord,
    GenerationSlotRule,
    ModelArchitectureFacts,
    QuantizationScheme,
    ResourceEnvelope,
    ServingPhase,
    ShapeBinding,
    StaticCapability,
    StaticRequirements,
    StaticTargetDescriptor,
    WorkloadGenerationRule,
    WorkloadRegime,
)
from sol_execbench.core.integrity import stable_json_checksum

GIB = 1024**3
MAX_COMMON_SCALE = 1 << 20
WORKLOAD_NAMESPACE = uuid.UUID("d8218ae1-b9c5-54bc-9e4b-7d10a7ac0b4d")


@dataclass(frozen=True)
class GeneratedRuleResult:
    """Concrete workloads generated atomically from one distribution rule."""

    distribution_id: str
    common_scale: int
    workloads: tuple[Workload, ...]
    records: tuple[GeneratedWorkloadRecord, ...]


def capacity_class_bytes(
    usable_budget_bytes: int,
    capacity_classes_gib: tuple[int, ...],
) -> int:
    """Floor measured usable bytes to a frozen capacity class."""
    classes = tuple(value * GIB for value in capacity_classes_gib)
    eligible = tuple(value for value in classes if value <= usable_budget_bytes)
    return eligible[-1] if eligible else 0


def distribution_id(
    *,
    semantic_fingerprint: str,
    definition: Definition,
    rule: WorkloadGenerationRule,
    facts: ModelArchitectureFacts,
    protocol_major: int = WORKLOAD_GENERATION_PROTOCOL_MAJOR,
) -> str:
    """Return the full hardware-independent distribution identity."""
    input_digests = {
        regime.value: input_profile_digest(rule, regime)
        for regime in WorkloadRegime
    }
    return stable_json_checksum(
        {
            "semantic_fingerprint": semantic_fingerprint,
            "generation_rule_digest": stable_json_checksum(
                rule.model_dump(mode="json")
            ),
            "model_architecture_facts_digest": stable_json_checksum(
                facts.model_dump(mode="json")
            ),
            "slots": [slot.model_dump(mode="json") for slot in rule.slots],
            "input_profile_digests": input_digests,
            "correctness_profile_digest": correctness_profile_digest(
                rule.operation_family
            ),
            "generation_protocol_major": protocol_major,
        }
    )


def generation_cohort_id(
    *,
    distribution_id: str,
    target: StaticTargetDescriptor,
    capacity_class_bytes: int,
    capacity_numerator: int,
    capacity_denominator: int,
    executor_version: str,
) -> str:
    """Hash only normalized constraints that can alter generated semantics."""
    return stable_json_checksum(
        {
            "distribution_id": distribution_id,
            "gfx_target": target.gfx_target.strip().lower(),
            "capacity_class_bytes": capacity_class_bytes,
            "dtypes": sorted(set(map(str, target.supported_dtypes))),
            "quantization": sorted(
                set(map(str, target.supported_quantization))
            ),
            "capabilities": sorted(set(map(str, target.capabilities))),
            "max_tensor_bytes": target.max_tensor_bytes,
            "reference_ipc_limit_bytes": target.reference_ipc_limit_bytes,
            "reference_protocol_tensor_limit_bytes": (
                MAX_REFERENCE_TENSOR_STORAGE_BYTES
            ),
            "capacity_numerator": capacity_numerator,
            "capacity_denominator": capacity_denominator,
            "maximum_common_scale": MAX_COMMON_SCALE,
            "executor_version": executor_version,
        }
    )


def generate_rule_workloads(
    *,
    definition: Definition,
    rule: WorkloadGenerationRule,
    facts: ModelArchitectureFacts,
    target: StaticTargetDescriptor,
    capacity_bytes: int,
    cohort_id: str,
    semantic_fingerprint: str,
) -> GeneratedRuleResult | None:
    """Generate all nine slots at one shared scale, or return no partial set."""
    maximum_peak = capacity_bytes // 2
    minimum_scale = _minimum_distinct_scale(rule, facts)
    maximum_scale = _maximum_feasible_scale(
        definition,
        rule,
        facts,
        target,
        maximum_peak,
    )
    if minimum_scale is None or maximum_scale is None:
        return None
    if minimum_scale > maximum_scale:
        return None
    scale = maximum_scale
    rule_distribution_id = distribution_id(
        semantic_fingerprint=semantic_fingerprint,
        definition=definition,
        rule=rule,
        facts=facts,
    )
    workloads, records = _materialize_scale(
        definition,
        rule,
        facts,
        target,
        maximum_peak,
        cohort_id,
        rule_distribution_id,
        scale,
    )
    validate_problem_contract(definition, list(workloads))
    _validate_materialized_distribution(rule, facts, scale, workloads, records)
    return GeneratedRuleResult(
        distribution_id=rule_distribution_id,
        common_scale=scale,
        workloads=workloads,
        records=records,
    )


def _minimum_distinct_scale(
    rule: WorkloadGenerationRule,
    facts: ModelArchitectureFacts,
) -> int | None:
    """Find the first scale after which aligned slot axes stay distinct."""
    proof_limit = _distinctness_proof_limit(rule)
    if proof_limit is None or proof_limit > MAX_COMMON_SCALE:
        return None
    last_duplicate = 0
    for scale in range(1, proof_limit + 1):
        rows = tuple(
            _slot_axes(rule, facts, slot, scale) for slot in rule.slots
        )
        if len({_axes_identity(axes) for axes in rows}) != len(rows):
            last_duplicate = scale
    minimum = last_duplicate + 1
    return minimum if minimum <= MAX_COMMON_SCALE else None


def _distinctness_proof_limit(rule: WorkloadGenerationRule) -> int | None:
    """Bound finite enumeration after which slot extents cannot collide."""
    ratios = tuple(
        Fraction(slot.scale_numerator, slot.scale_denominator)
        for slot in rule.slots
    )
    if len(set(ratios)) != len(ratios):
        return None
    limit = 1
    for slot, ratio in zip(rule.slots, ratios, strict=True):
        inactive_minimum = Fraction(_minimum_extent(slot), 1) / ratio
        limit = max(limit, _floor_fraction(inactive_minimum) + 2)
    for left_index, left in enumerate(ratios):
        for right_index in range(left_index + 1, len(ratios)):
            delta = abs(left - ratios[right_index])
            rounding_error = _rounding_error(rule.slots[left_index])
            rounding_error += _rounding_error(rule.slots[right_index])
            separated = Fraction(rounding_error, 1) / delta
            limit = max(limit, _floor_fraction(separated) + 2)
    return limit


def _floor_fraction(value: Fraction) -> int:
    return value.numerator // value.denominator


def _rounding_error(slot: GenerationSlotRule) -> int:
    return 2 if slot.irregular else _slot_alignment(slot) + 1


def _minimum_extent(slot: GenerationSlotRule) -> int:
    return 3 if slot.irregular else _slot_alignment(slot)


def _maximum_feasible_scale(
    definition: Definition,
    rule: WorkloadGenerationRule,
    facts: ModelArchitectureFacts,
    target: StaticTargetDescriptor,
    maximum_peak: int,
) -> int | None:
    if not common_scale_rule_is_monotonic(rule):
        raise ValueError("generation rule does not prove scale monotonicity")
    low = 1
    high = MAX_COMMON_SCALE
    if not _resource_scale_is_feasible(
        definition, rule, facts, target, maximum_peak, low
    ):
        return None
    best = low
    while low <= high:
        candidate = (low + high) // 2
        if _resource_scale_is_feasible(
            definition, rule, facts, target, maximum_peak, candidate
        ):
            best = candidate
            low = candidate + 1
        else:
            high = candidate - 1
    return best


def _resource_scale_is_feasible(
    definition: Definition,
    rule: WorkloadGenerationRule,
    facts: ModelArchitectureFacts,
    target: StaticTargetDescriptor,
    maximum_peak: int,
    scale: int,
) -> bool:
    axes_rows = tuple(
        _slot_axes(rule, facts, slot, scale) for slot in rule.slots
    )
    if not all(_axes_within_limits(axes, facts) for axes in axes_rows):
        return False
    try:
        requirements = tuple(
            workload_requirements(
                definition,
                axes,
                rule.operation_family,
                rule.quantization,
                rule.capabilities,
            )
            for axes in axes_rows
        )
    except (KeyError, ValueError):
        return False
    return all(
        _requirements_fit(item, target, maximum_peak) for item in requirements
    )


def _requirements_fit(
    requirements: StaticRequirements,
    target: StaticTargetDescriptor,
    maximum_peak: int,
) -> bool:
    resources = requirements.resources
    ipc_limit = min(
        target.reference_ipc_limit_bytes,
        MAX_REFERENCE_TENSOR_STORAGE_BYTES,
    )
    return (
        set(requirements.dtypes) <= set(target.supported_dtypes)
        and set(requirements.quantization) <= set(target.supported_quantization)
        and set(requirements.capabilities) <= set(target.capabilities)
        and resources.reference_peak_bytes <= maximum_peak
        and resources.max_tensor_bytes <= target.max_tensor_bytes
        and resources.reference_ipc_bytes <= ipc_limit
    )


def _materialize_scale(
    definition: Definition,
    rule: WorkloadGenerationRule,
    facts: ModelArchitectureFacts,
    target: StaticTargetDescriptor,
    maximum_peak: int,
    cohort_id: str,
    rule_distribution_id: str,
    scale: int,
) -> tuple[tuple[Workload, ...], tuple[GeneratedWorkloadRecord, ...]]:
    workloads: list[Workload] = []
    records: list[GeneratedWorkloadRecord] = []
    for slot in rule.slots:
        axes = _slot_axes(rule, facts, slot, scale)
        workload = _workload(
            definition,
            rule,
            slot,
            axes,
            cohort_id,
            rule_distribution_id,
        )
        requirements = workload_requirements(
            definition,
            axes,
            rule.operation_family,
            rule.quantization,
            rule.capabilities,
        )
        if not _axes_within_limits(axes, facts):
            raise ValueError("generated axes exceed a frozen rule limit")
        if not _requirements_fit(requirements, target, maximum_peak):
            raise ValueError(
                "generated workload exceeds an execution constraint"
            )
        workloads.append(workload)
        records.append(_record(rule, slot, workload, requirements, scale))
    identities = {_axes_identity(workload.axes) for workload in workloads}
    if len(identities) != len(rule.slots):
        raise ValueError(
            "generated distribution contains duplicate workload axes"
        )
    return tuple(workloads), tuple(records)


def _validate_materialized_distribution(
    rule: WorkloadGenerationRule,
    facts: ModelArchitectureFacts,
    scale: int,
    workloads: tuple[Workload, ...],
    records: tuple[GeneratedWorkloadRecord, ...],
) -> None:
    """Revalidate slot completeness, alignment, order, and distinctness."""
    if len(workloads) != 9 or len(records) != 9:
        raise ValueError(
            "generated distribution must contain exactly nine slots"
        )
    for slot, workload, record in zip(
        rule.slots, workloads, records, strict=True
    ):
        expected_axes = _slot_axes(rule, facts, slot, scale)
        if workload.axes != expected_axes or record.axes != expected_axes:
            raise ValueError("generated axes violate slot alignment semantics")
        if record.slot_id != slot.slot_id or record.uuid != workload.uuid:
            raise ValueError(
                "generated slot metadata is incomplete or reordered"
            )
    identities = {_axes_identity(workload.axes) for workload in workloads}
    if len(identities) != 9:
        raise ValueError("generated slots must have nine distinct axes")


def _workload(
    definition: Definition,
    rule: WorkloadGenerationRule,
    slot: GenerationSlotRule,
    axes: dict[str, int],
    cohort_id: str,
    rule_distribution_id: str,
) -> Workload:
    canonical_axes = json.dumps(
        axes,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    identity = (
        rule_distribution_id
        + cohort_id
        + rule.semantic_id
        + slot.slot_id
        + canonical_axes
    )
    return Workload.model_validate(
        {
            "schema_version": BenchmarkArtifactSchema.WORKLOAD,
            "axes": axes,
            "inputs": _workload_inputs(definition, rule, slot.regime),
            "uuid": str(uuid.uuid5(WORKLOAD_NAMESPACE, identity)),
            "checks": _correctness_checks(definition, rule.operation_family),
        }
    )


def _record(
    rule: WorkloadGenerationRule,
    slot: GenerationSlotRule,
    workload: Workload,
    requirements: StaticRequirements,
    scale: int,
) -> GeneratedWorkloadRecord:
    input_id = f"llm_core.input.{slot.regime.value}.v1"
    correctness_id = _correctness_profile_id(rule.operation_family)
    return GeneratedWorkloadRecord(
        semantic_id=rule.semantic_id,
        slot_id=slot.slot_id,
        uuid=workload.uuid,
        axes=workload.axes,
        role=slot.role,
        regime=slot.regime,
        serving_phase=slot.serving_phase,
        binding=slot.binding,
        scale_numerator=slot.scale_numerator,
        scale_denominator=slot.scale_denominator,
        common_scale=scale,
        input_profile_id=input_id,
        input_profile_digest=input_profile_digest(rule, slot.regime),
        correctness_profile_id=correctness_id,
        correctness_profile_digest=correctness_profile_digest(
            rule.operation_family
        ),
        requirements=requirements,
    )


def _slot_axes(
    rule: WorkloadGenerationRule,
    facts: ModelArchitectureFacts,
    slot: GenerationSlotRule,
    scale: int,
) -> dict[str, int]:
    extent = _slot_extent(slot, scale)
    family = rule.operation_family
    if family is CorpusOperationFamily.LINEAR:
        return _linear_axes(rule.variant, facts, slot, extent)
    if family is CorpusOperationFamily.NORM_ACTIVATION:
        return {"M": extent, "N": _hidden(facts, slot, 64)}
    if family is CorpusOperationFamily.POSITION:
        return _position_axes(facts, slot, extent)
    if family in (
        CorpusOperationFamily.ATTENTION,
        CorpusOperationFamily.ADVANCED_ATTENTION,
    ):
        return _attention_axes(facts, slot, extent)
    if family is CorpusOperationFamily.KV_CACHE:
        return _kv_axes(facts, slot, extent)
    if family is CorpusOperationFamily.MOE:
        return _moe_axes(facts, slot, extent)
    if family is CorpusOperationFamily.QUANTIZATION:
        return {"M": extent, "N": _hidden(facts, slot, 64)}
    raise ValueError(f"unsupported workload generation family: {family}")


def _slot_extent(slot: GenerationSlotRule, scale: int) -> int:
    value = max(1, scale * slot.scale_numerator // slot.scale_denominator)
    if slot.irregular:
        return max(3, value | 1)
    alignment = _slot_alignment(slot)
    return max(alignment, value // alignment * alignment)


def _slot_alignment(slot: GenerationSlotRule) -> int:
    if slot.role.value == "smoke" or slot.regime is WorkloadRegime.LATENCY:
        return 1
    return 16 if slot.regime is WorkloadRegime.CAPACITY else 8


def _linear_axes(
    variant: str,
    facts: ModelArchitectureFacts,
    slot: GenerationSlotRule,
    extent: int,
) -> dict[str, int]:
    hidden = _hidden(facts, slot, 64)
    output = hidden
    if (
        slot.binding is ShapeBinding.MODEL
        and facts.intermediate_size
        and slot.serving_phase is not ServingPhase.DECODE
    ):
        output = facts.intermediate_size
    axes = {"M": extent, "K": hidden, "N": output}
    if "batched" in variant or "grouped" in variant:
        axes["B"] = 2 if slot.binding is ShapeBinding.BOUNDARY else 4
    return axes


def _position_axes(
    facts: ModelArchitectureFacts,
    slot: GenerationSlotRule,
    extent: int,
) -> dict[str, int]:
    heads, dimension = _head_facts(facts, slot)
    if slot.serving_phase is ServingPhase.DECODE:
        return {"B": extent, "H": heads, "S": 1, "D": dimension}
    return {"B": 1, "H": heads, "S": extent, "D": dimension}


def _attention_axes(
    facts: ModelArchitectureFacts,
    slot: GenerationSlotRule,
    extent: int,
) -> dict[str, int]:
    heads, dimension = _head_facts(facts, slot)
    kv_heads = (
        facts.kv_heads
        if slot.binding is ShapeBinding.MODEL and facts.kv_heads
        else max(1, heads // 4)
    )
    if slot.serving_phase is ServingPhase.DECODE:
        sequence, context = 1, extent
    elif slot.irregular:
        sequence, context = extent, extent * 2 + 1
    else:
        sequence = context = extent
    return {
        "B": 1,
        "HQ": heads,
        "HK": kv_heads,
        "S": sequence,
        "T": context,
        "D": dimension,
    }


def _kv_axes(
    facts: ModelArchitectureFacts,
    slot: GenerationSlotRule,
    extent: int,
) -> dict[str, int]:
    _, dimension = _head_facts(facts, slot)
    heads = (
        facts.kv_heads
        if slot.binding is ShapeBinding.MODEL and facts.kv_heads
        else 4
    )
    block = facts.page_size or 16
    if slot.binding is ShapeBinding.BOUNDARY:
        block = min(block, 8)
    sequence = 1 if slot.serving_phase is ServingPhase.DECODE else extent
    pages = max(1, math.ceil(extent / block))
    return {
        "P": pages,
        "BS": block,
        "B": 1,
        "S": sequence,
        "H": heads,
        "D": dimension,
    }


def _moe_axes(
    facts: ModelArchitectureFacts,
    slot: GenerationSlotRule,
    extent: int,
) -> dict[str, int]:
    model = slot.binding is ShapeBinding.MODEL
    hidden = facts.hidden_size if model and facts.hidden_size else 64
    experts = facts.expert_count if model and facts.expert_count else 8
    output = (
        facts.intermediate_size if model and facts.intermediate_size else hidden
    )
    return {"M": extent, "D": hidden, "E": experts, "O": output}


def _hidden(
    facts: ModelArchitectureFacts,
    slot: GenerationSlotRule,
    boundary: int,
) -> int:
    if slot.binding is ShapeBinding.MODEL and facts.hidden_size:
        return facts.hidden_size
    return boundary


def _head_facts(
    facts: ModelArchitectureFacts,
    slot: GenerationSlotRule,
) -> tuple[int, int]:
    if slot.binding is ShapeBinding.MODEL:
        return facts.query_heads or 8, facts.head_dimension or 128
    return 4, 32


def _axes_identity(axes: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(axes.items()))


def common_scale_rule_is_monotonic(rule: WorkloadGenerationRule) -> bool:
    """Prove axes and resource estimates are non-decreasing with scale.

    This proof is deliberately closed over the executor's known family
    implementations. Unknown rule semantics never enter integer binary search.
    """
    supported = {
        CorpusOperationFamily.LINEAR,
        CorpusOperationFamily.NORM_ACTIVATION,
        CorpusOperationFamily.POSITION,
        CorpusOperationFamily.ATTENTION,
        CorpusOperationFamily.ADVANCED_ATTENTION,
        CorpusOperationFamily.KV_CACHE,
        CorpusOperationFamily.MOE,
        CorpusOperationFamily.QUANTIZATION,
    }
    return rule.operation_family in supported and all(
        slot.scale_numerator > 0 and slot.scale_denominator > 0
        for slot in rule.slots
    )


def _axes_within_limits(
    axes: dict[str, int],
    facts: ModelArchitectureFacts,
) -> bool:
    if any(value <= 0 for value in axes.values()):
        return False
    maximum_context = facts.maximum_context
    if maximum_context is None:
        return True
    context_values = [axes[name] for name in ("S", "T") if name in axes]
    if "P" in axes and "BS" in axes:
        context_values.append(axes["P"] * axes["BS"])
    return all(value <= maximum_context for value in context_values)


def input_profile_digest(
    rule: WorkloadGenerationRule,
    regime: WorkloadRegime,
) -> str:
    """Hash the complete input-generation profile content."""
    return stable_json_checksum(
        {
            "profile_id": f"llm_core.input.{regime.value}.v1",
            "numeric_generator": _numeric_generator(regime),
            "integer_generator": {"type": "random"},
            "kv_slots": (
                {"type": "integer", "low": 0, "high": "P * BS"}
                if rule.operation_family is CorpusOperationFamily.KV_CACHE
                else None
            ),
        }
    )


def correctness_profile_digest(family: CorpusOperationFamily) -> str:
    """Hash the complete correctness profile content."""
    return stable_json_checksum(
        {
            "profile_id": _correctness_profile_id(family),
            "template": _correctness_template(family),
        }
    )


def _correctness_profile_id(family: CorpusOperationFamily) -> str:
    if family is CorpusOperationFamily.QUANTIZATION:
        return "llm_core.quantized_numeric.v1"
    if family in (
        CorpusOperationFamily.ATTENTION,
        CorpusOperationFamily.ADVANCED_ATTENTION,
        CorpusOperationFamily.MOE,
    ):
        return "llm_core.accumulation_numeric.v1"
    return "llm_core.bf16_numeric.v1"


def _correctness_template(family: CorpusOperationFamily) -> dict[str, Any]:
    if family is CorpusOperationFamily.QUANTIZATION:
        return {
            "type": "numeric",
            "max_atol": 0.125,
            "max_rtol": 0.125,
            "required_matched_ratio": 1.0,
            "max_error_cap": 0.5,
        }
    tolerance = (
        0.02
        if family
        in (
            CorpusOperationFamily.ATTENTION,
            CorpusOperationFamily.ADVANCED_ATTENTION,
            CorpusOperationFamily.MOE,
        )
        else 0.01
    )
    return {
        "type": "numeric",
        "max_atol": tolerance,
        "max_rtol": tolerance,
        "required_matched_ratio": 1.0,
    }


def _workload_inputs(
    definition: Definition,
    rule: WorkloadGenerationRule,
    regime: WorkloadRegime,
) -> dict[str, Any]:
    generator = _numeric_generator(regime)
    inputs: dict[str, Any] = {}
    integer_types = {DType.INT32, DType.INT64, DType.UINT8, DType.BOOL}
    for name, tensor in definition.inputs.items():
        inputs[name] = (
            {"type": "random"}
            if tensor.dtype in integer_types
            else {"type": "generated", "generator": generator}
        )
    if rule.operation_family is CorpusOperationFamily.KV_CACHE:
        inputs["slots"] = {
            "type": "generated",
            "generator": {"type": "integer", "low": 0, "high": "P * BS"},
        }
    return inputs


def _numeric_generator(regime: WorkloadRegime) -> dict[str, Any]:
    if regime is WorkloadRegime.LATENCY:
        return {"type": "normal", "mean": 0.0, "std": 0.02}
    if regime is WorkloadRegime.THROUGHPUT:
        return {"type": "normal", "mean": 0.0, "std": 0.5}
    if regime is WorkloadRegime.IRREGULAR:
        return {"type": "uniform", "low": -8.0, "high": 8.0}
    return {"type": "normal", "mean": 0.0, "std": 1.0}


def _correctness_checks(
    definition: Definition,
    family: CorpusOperationFamily,
) -> list[dict[str, Any]]:
    template = _correctness_template(family)
    return [{**template, "output": output} for output in definition.outputs]


def workload_requirements(
    definition: Definition,
    axes: dict[str, int],
    family: CorpusOperationFamily,
    quantization: QuantizationScheme | None,
    capabilities: tuple[StaticCapability, ...],
) -> StaticRequirements:
    """Recompute the canonical static resource envelope and requirements."""
    input_sizes = _tensor_sizes(
        definition.inputs, definition.get_input_shapes(axes)
    )
    output_sizes = _tensor_sizes(
        definition.outputs, definition.get_output_shapes(axes)
    )
    input_bytes = sum(input_sizes)
    output_bytes = sum(output_sizes)
    temporary = _temporary_bytes(family, axes, output_bytes)
    dtypes = {
        tensor.dtype
        for tensor in (
            *definition.inputs.values(),
            *definition.outputs.values(),
        )
    }
    dtypes.update(_quantization_dtypes(quantization))
    resources = ResourceEnvelope(
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        max_tensor_bytes=max((*input_sizes, *output_sizes), default=0),
        reference_ipc_bytes=input_bytes + output_bytes,
        temporary_bytes=temporary,
        reference_peak_bytes=input_bytes + output_bytes + temporary,
    )
    return StaticRequirements(
        dtypes=tuple(sorted(dtypes, key=str)),
        quantization=() if quantization is None else (quantization,),
        capabilities=tuple(
            dict.fromkeys((StaticCapability.DENSE_TENSOR, *capabilities))
        ),
        resources=resources,
    )


def _tensor_sizes(
    tensors: Any,
    shapes: dict[str, tuple[int, ...] | None],
) -> list[int]:
    sizes: list[int] = []
    for name, tensor in tensors.items():
        shape = shapes[name]
        elements = 1 if shape is None else math.prod(shape)
        sizes.append((elements * dtype_storage_bits(tensor.dtype) + 7) // 8)
    return sizes


def _temporary_bytes(
    family: CorpusOperationFamily,
    axes: dict[str, int],
    output_bytes: int,
) -> int:
    if family is CorpusOperationFamily.ATTENTION:
        return axes["B"] * axes["HQ"] * axes["S"] * axes["T"] * 4
    if family is CorpusOperationFamily.ADVANCED_ATTENTION:
        dense = axes["B"] * axes["HQ"] * axes["S"] * axes["T"] * 4
        return max(output_bytes, dense // 8)
    if family is CorpusOperationFamily.MOE:
        return axes["M"] * axes["E"] * 4 + output_bytes * 2
    return output_bytes * 2


def _quantization_dtypes(
    scheme: QuantizationScheme | None,
) -> set[DType]:
    if scheme in (
        QuantizationScheme.FP8_PER_TENSOR,
        QuantizationScheme.FP8_PER_TOKEN,
    ):
        return {DType.FLOAT8_E4M3FN}
    if scheme is QuantizationScheme.MXFP8_BLOCK:
        return {DType.FLOAT8_E5M2}
    if scheme in (QuantizationScheme.FP4_GROUP, QuantizationScheme.MXFP4_BLOCK):
        return {DType.FLOAT4_E2M1FN_X2}
    if scheme is not None:
        return {DType.INT8}
    return set()


__all__ = [
    "GeneratedRuleResult",
    "capacity_class_bytes",
    "common_scale_rule_is_monotonic",
    "correctness_profile_digest",
    "distribution_id",
    "generate_rule_workloads",
    "generation_cohort_id",
    "input_profile_digest",
    "workload_requirements",
]
