from __future__ import annotations

import pytest
import torch

from sol_execbench.core.dataset import (
    CDNA4_VALIDATION_DEFERRED_CODE,
    LOW_PRECISION_COMPATIBILITY_EVIDENCE_CODE,
    cdna4_low_precision_skip_reason,
    dequantize_e2m1_codes,
    definition_uses_cdna4_low_precision,
    low_precision_unvalidated_evidence,
    normalize_low_precision_format,
    pack_e2m1_codes,
    pack_low_precision_tensor,
    quantize_e2m1_codes,
    should_skip_cdna4_low_precision_on_arch,
    unpack_e2m1_codes,
)


def test_e2m1_pack_unpack_preserves_nibbles_and_shape():
    codes = torch.tensor([[0, 1, 2], [8, 15, 6]], dtype=torch.uint8)

    packed = pack_e2m1_codes(codes)
    unpacked = unpack_e2m1_codes(packed, codes.shape)

    assert packed.shape == (2, 2)
    assert packed.tolist() == [[0x10, 0x02], [0xF8, 0x06]]
    assert unpacked.tolist() == codes.tolist()


def test_e2m1_quantize_dequantize_round_trip_uses_cpu_reference_codebook():
    values = torch.tensor([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

    codes = quantize_e2m1_codes(values, scale=1.0)
    restored = dequantize_e2m1_codes(codes, scale=1.0)

    assert codes.dtype == torch.uint8
    assert restored.dtype == torch.float32
    assert torch.allclose(restored, values)


def test_quantize_e2m1_codes_matches_existing_fp4_thresholds():
    values = torch.tensor([-5.1, -5.0, -0.25, -0.2, 0.0, 0.25, 0.26, 5.0, 5.1])

    codes = quantize_e2m1_codes(values)

    assert codes.tolist() == [15, 14, 8, 8, 0, 0, 1, 6, 7]


def test_pack_low_precision_tensor_preserves_metadata_and_unvalidated_evidence():
    values = torch.tensor([[0.0, 0.5, 1.0], [-0.5, -1.0, 6.0]], dtype=torch.float32)

    payload = pack_low_precision_tensor(
        values,
        format_name="nv-fp4",
        scale=1.0,
        scale_shape=(1,),
        scale_role="per_tensor",
    )

    assert payload.format_name == "nvfp4"
    assert payload.original_shape == (2, 3)
    assert payload.scale_metadata.to_dict() == {
        "scale": 1.0,
        "scale_shape": [1],
        "scale_dtype": "float32",
        "scale_role": "per_tensor",
    }
    assert torch.allclose(payload.dequantize(), values)
    evidence = payload.evidence.to_dict()
    assert evidence["evidence_code"] == LOW_PRECISION_COMPATIBILITY_EVIDENCE_CODE
    assert evidence["blocker_code"] == CDNA4_VALIDATION_DEFERRED_CODE
    assert evidence["hardware_validation"] is False
    assert evidence["performance_authority"] is False
    assert evidence["blackwell_equivalence"] is False
    assert evidence["score_authority"] is False


def test_low_precision_format_aliases_and_validation_errors():
    assert normalize_low_precision_format("NV-FP4") == "nvfp4"
    assert normalize_low_precision_format("fp4") == "float4_e2m1"
    assert low_precision_unvalidated_evidence("mx-fp4").format_name == "mxfp4"

    with pytest.raises(ValueError, match="Unsupported low-precision"):
        normalize_low_precision_format("int4")
    with pytest.raises(ValueError, match="range"):
        pack_e2m1_codes(torch.tensor([16], dtype=torch.uint8))
    with pytest.raises(ValueError, match="positive"):
        quantize_e2m1_codes(torch.tensor([1.0]), scale=0.0)


def test_scalar_low_precision_tensor_round_trips_shape():
    payload = pack_low_precision_tensor(torch.tensor(1.0), format_name="float4_e2m1fn_x2")

    assert payload.original_shape == ()
    assert payload.packed.shape == (1,)
    assert payload.unpack_codes().shape == ()
    assert payload.dequantize().shape == ()
    assert payload.dequantize().item() == 1.0


def test_cdna3_low_precision_quant_skip_policy():
    definition = {
        "name": "033_nvfp4_moe_routing_with_topk_selection",
        "inputs": {
            "gate_weight_fp4": {
                "dtype": "float4_e2m1fn_x2",
                "description": "NVFP4 gate weight",
            },
            "hidden_states": {"dtype": "bfloat16"},
        },
        "outputs": {"router_logits": {"dtype": "float32"}},
        "reference": "",
    }

    assert definition_uses_cdna4_low_precision(definition) is True
    assert should_skip_cdna4_low_precision_on_arch(definition, "gfx942") is True
    assert (
        should_skip_cdna4_low_precision_on_arch(
            definition, "gfx941:sramecc+:xnack-"
        )
        is True
    )
    assert should_skip_cdna4_low_precision_on_arch(definition, "gfx950") is False
    assert should_skip_cdna4_low_precision_on_arch(definition, "unknown") is False
    assert "cdna3_low_precision_hardware_unsupported" in cdna4_low_precision_skip_reason(
        "gfx942"
    )


def test_cdna3_low_precision_skip_policy_ignores_non_fp4_quant():
    definition = {
        "name": "011_fp8_moe_gate_routing",
        "inputs": {"hidden_states": {"dtype": "bfloat16"}},
        "outputs": {"topk_weight": {"dtype": "float32"}},
        "reference": "",
    }

    assert definition_uses_cdna4_low_precision(definition) is False
    assert should_skip_cdna4_low_precision_on_arch(definition, "gfx942") is False


def test_cdna3_low_precision_skip_policy_detects_scaled_gemm_reference():
    definition = {
        "name": "020_nvfp4_linear_layer",
        "inputs": {
            "x": {"dtype": "bfloat16"},
            "weight": {"dtype": "float8_e4m3fn"},
        },
        "outputs": {"out": {"dtype": "float32"}},
        "reference": "return torch._scaled_mm(x, weight, scale_a, scale_b)",
    }

    assert definition_uses_cdna4_low_precision(definition) is True
    assert should_skip_cdna4_low_precision_on_arch(definition, "gfx942") is True
