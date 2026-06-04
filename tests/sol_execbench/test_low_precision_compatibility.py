from __future__ import annotations

import pytest
import torch

from sol_execbench.core.dataset import (
    CDNA4_VALIDATION_DEFERRED_CODE,
    LOW_PRECISION_COMPATIBILITY_EVIDENCE_CODE,
    dequantize_e2m1_codes,
    low_precision_unvalidated_evidence,
    normalize_low_precision_format,
    pack_e2m1_codes,
    pack_low_precision_tensor,
    quantize_e2m1_codes,
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
