# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for SOLAR P0/P1 soundness, accounting, and verification fixes."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from solar.analysis.contraction_proofs import build_orojenesis_proof_layer
from solar.analysis.external_reads import FULL_OPERAND_READS
from solar.analysis.graph_analyzer import IRGraphAnalyzer
from solar.analysis.graph_context import PreparedAnalysis
from solar.analysis.graph_models import FusionPlan
from solar.analysis.orojenesis.errors import (
    OrojenesisError,
    OrojenesisTimeoutError,
)
from solar.analysis.orojenesis.problem import problem_for_layer
from solar.analysis.orojenesis.process import mapper_subprocess_environment
from solar.analysis.orojenesis.runner import OrojenesisRunner
from solar.analysis.orojenesis_evidence import _layer_compulsory_bytes
from solar.analysis.resources import (
    classify_layer_resources,
    mandatory_mfma_macs,
)
from solar.artifacts import sha256_file, stable_json_checksum
from solar.artifacts.names import (
    ANALYSIS_FILENAME,
    CONVERSION_ATTESTATION_FILENAME,
    MANIFEST_FILENAME,
)
from solar.artifacts.yaml import write_yaml_atomic
from solar.contracts import (
    AnalysisRequest,
    AnalysisResult,
    ConversionRequest,
)
from solar.errors import GradientVerificationError
from solar.graph.extraction import extract_operator_graph
from solar.graph.grad_capture import GradientBoundaryCapture
from solar.graph.torchview.attribute_parsing import SemanticArgumentEncoder
from solar.ir.access import AccessQuality, AccessReason
from solar.ir.contracts import IRPath
from solar.ir.conversion import convert_operator_graph
from solar.ir.extended_einsum.operations.analyzer import EinsumAnalyzer
from solar.ir.extended_einsum.operations.handlers.matmul_ops import (
    MatmulHandler,
)
from solar.ir.extended_einsum.torchview.converter import PyTorchToEinsum
from solar.pipeline.verified_program import load_verified_program
from solar.precision import dtype_bytes, normalize_dtype
from solar.reuse_contracts import VerifiedProgramRequest
from solar.rocm.architecture import ArchitectureProfile
from solar.shapes import (
    ShapeReasonCode,
    certify_structure,
    structural_projection,
)
from solar.types import TensorShapes
from solar.verification.contracts import VerificationPolicy
from solar.verification.errors import IRExecutionError
from solar.verification.executor import IRGraphExecutor
from solar.verification.gradients import verify_gradients
from solar.verification.registry import verification_backend
from solar.verification.verify import verify_callable_conversion


def _compile_and_analyze(
    tmp_path: Path,
    name: str,
    reference,
    inputs: tuple[torch.Tensor, ...],
    *,
    ir_path: IRPath = IRPath.MAKE_FX_ATEN,
    precision: str = "fp32",
    architecture: str | ArchitectureProfile | None = None,
) -> tuple[dict, IRGraphExecutor, dict]:
    out = tmp_path / name
    op_art = extract_operator_graph(
        reference,
        inputs,
        device="cpu",
        output_dir=out,
        name=name,
        extraction_kind=ir_path.extraction_kind,
    )
    ir_art = convert_operator_graph(
        op_art, output_dir=out, ir_kind=ir_path.ir_kind
    )
    graph = yaml.safe_load(ir_art.path.read_text())
    executor = IRGraphExecutor(
        graph, verification_backend(ir_path.ir_kind), check_shapes=True
    )
    analysis = IRGraphAnalyzer().analyze_graph(
        ir_art.path,
        out / "analysis",
        precision=precision,
        strict=True,
        architecture=architecture,
    )
    assert analysis is not None
    return graph, executor, analysis


def test_p0_1_transparent_and_sliced_declared_outputs_preserve_write_bytes(
    tmp_path: Path,
) -> None:
    x = torch.randn(4, 6, dtype=torch.float32)
    _, _, view_analysis = _compile_and_analyze(
        tmp_path,
        "view_out",
        lambda t: (t + 1.0).transpose(0, 1).reshape(-1),
        (x,),
    )
    # 24 float32 input + 24 float32 output = 192 bytes (not 96 bytes).
    assert view_analysis["total"]["fused_elements"] == 48
    assert view_analysis["total"]["fused_bytes"] == 192.0

    _, _, slice_analysis = _compile_and_analyze(
        tmp_path,
        "slice_out",
        lambda t: (t + 1.0)[:2, :],
        (x,),
    )
    # 24 float32 input + 12 float32 sliced output = 36 elements / 144 bytes.
    assert slice_analysis["total"]["fused_elements"] == 36
    assert slice_analysis["total"]["fused_bytes"] == 144.0

    def overlapping_slices(t: torch.Tensor):
        y = t + 1.0
        return y[:3, :], y[1:4, :]

    _, _, overlap_analysis = _compile_and_analyze(
        tmp_path,
        "overlap_out",
        overlapping_slices,
        (x,),
    )
    # Union of rows [0:3] and [1:4] is [0:4] = 24 output elements (no double count).
    assert overlap_analysis["total"]["fused_elements"] == 48
    assert overlap_analysis["total"]["fused_bytes"] == 192.0


def test_p0_2_region_boundary_does_not_bypass_internal_operand_provenance(
    tmp_path: Path,
) -> None:
    matmul = {
        "type": "matmul",
        "semantic_op": {
            "kind": "einsum",
            "target": "einsum",
            "equation": "MK,KN->MN",
            "operands": [{"tensor": 0}, {"tensor": 1}],
            "attributes": {},
            "effects": {},
        },
        "einsum_equation": "MK,KN->MN",
        "is_real_einsum": True,
        "tensor_names": {"inputs": ["internal_act", "w"], "outputs": ["out"]},
        "tensor_shapes": {"inputs": [[2, 4], [4, 4]], "outputs": [[2, 4]]},
        "tensor_dtypes": {
            "inputs": ["torch.float16", "torch.float16"],
            "outputs": ["torch.float16"],
        },
    }
    raw = tmp_path / "orojenesis" / "mm" / "raw.csv"
    raw.parent.mkdir(parents=True)
    raw.write_text("64,160\n")
    plan = FusionPlan(
        fusion={
            "regions": [
                {
                    "id": "r0",
                    "layers": ["mm"],
                    "external_inputs": ["internal_act", "w"],
                    "external_outputs": ["out"],
                }
            ]
        },
        chains=[],
        regions=[],
        proof_layers={"mm": matmul},
    )
    evidence = {
        "schema_version": 1,
        "status": "complete",
        "toolchain": {"verification_mode": "test"},
        "layers": {
            "mm": {
                "word_bits": 16,
                "selected_capacity": {
                    "level": "l2",
                    "capacity_bytes": 64,
                    "point": {"buffer_bytes": 64, "dram_bytes": 160.0},
                },
                "evidence_files": {
                    "raw": {
                        "path": "orojenesis/mm/raw.csv",
                        "sha256": sha256_file(raw),
                    }
                },
            }
        },
        "chains": {},
        "regions": {},
    }
    prepared = SimpleNamespace(
        all_layers={
            "upstream": {
                "type": "conv2d",
                "semantic_op": {
                    "kind": "aten",
                    "target": "conv2d",
                    "effects": {},
                },
                "tensor_names": {"inputs": ["x"], "outputs": ["internal_act"]},
            },
            "w_start": {
                "type": "start",
                "tensor_names": {"inputs": [], "outputs": ["w"]},
                "tensor_dtypes": {"inputs": [], "outputs": ["torch.float16"]},
            },
            "mm": matmul,
        },
        output_dir=tmp_path,
    )
    _, formal = IRGraphAnalyzer()._audit_orojenesis_evidence(
        plan,
        evidence,
        cast(PreparedAnalysis, prepared),
        audited_fused_bytes=64.0,
    )
    assert formal is False
    layers_ev = cast(dict[str, dict[str, object]], evidence["layers"])
    app = cast(dict[str, object], layers_ev["mm"]["formal_applicability"])
    assert app["applicable"] is False
    assert app["graph_input_operands"] is False
    assert app["region_boundary_operands"] is True
    assert app["operand_provenance"] == "unproven_internal_operand"


def test_p0_3_torchview_dtypes_executor_check_and_dequant_payload(
    tmp_path: Path,
) -> None:
    enc = SemanticArgumentEncoder()
    assert enc.encode("__torch_float8_e4m3fnuz__") == {
        "dtype": "float8_e4m3fnuz"
    }
    assert enc.encode("__torch_complex64__") == {"dtype": "complex64"}
    assert PyTorchToEinsum._DTYPE_BITS["float8_e4m3fnuz"] == 8
    assert PyTorchToEinsum._DTYPE_BITS["float8_e5m2"] == 8
    assert PyTorchToEinsum._OUTPUT_DTYPE_INPUT_INDEX["type_as"] == 1

    layers = {
        "x": {
            "type": "auxiliary-tensor",
            "output_dtypes": ["torch.float16"],
            "connections": {
                "inputs": [],
                "outputs": ["sum_op", "mix_op", "arg_op"],
            },
        },
        "y": {
            "type": "auxiliary-tensor",
            "output_dtypes": ["torch.bfloat16"],
            "connections": {"inputs": [], "outputs": ["mix_op"]},
        },
        "sum_op": {
            "type": "sum",
            "input_dtypes": ["torch.float16"],
            "output_dtypes": ["torch.float16"],
            "module_args": {"call_kwargs": {"dtype": {"dtype": "float32"}}},
            "connections": {"inputs": ["x"], "outputs": ["sum_out"]},
        },
        "sum_out": {
            "type": "hidden-tensor",
            "output_dtypes": ["torch.float16"],
            "connections": {"inputs": ["sum_op"], "outputs": []},
        },
        "mix_op": {
            "type": "add",
            "input_dtypes": ["torch.float16", "torch.bfloat16"],
            "output_dtypes": ["torch.float16"],
            "connections": {"inputs": ["x", "y"], "outputs": ["mix_out"]},
        },
        "mix_out": {
            "type": "hidden-tensor",
            "output_dtypes": ["torch.float16"],
            "connections": {"inputs": ["mix_op"], "outputs": []},
        },
        "arg_op": {
            "type": "argmax",
            "input_dtypes": ["torch.float16"],
            "output_dtypes": ["torch.float16"],
            "connections": {"inputs": ["x"], "outputs": ["arg_out"]},
        },
        "arg_out": {
            "type": "hidden-tensor",
            "output_dtypes": ["torch.float16"],
            "connections": {"inputs": ["arg_op"], "outputs": []},
        },
    }
    PyTorchToEinsum(strict=True)._repair_torchview_quirks(
        layers,
        ["sum_op", "mix_op", "arg_op"],
        ["x", "y", "sum_out", "mix_out", "arg_out"],
    )
    assert layers["sum_op"]["output_dtypes"] == ["torch.float32"]
    assert layers["mix_op"]["output_dtypes"] == ["torch.float32"]
    assert layers["arg_op"]["output_dtypes"] == ["torch.int64"]

    # IRGraphExecutor rejects runtime dtype mismatch when check_shapes=True.
    graph, executor, _ = _compile_and_analyze(
        tmp_path,
        "dtype_check",
        lambda a: a + 1.0,
        (torch.ones(4, dtype=torch.float32),),
    )
    with pytest.raises(IRExecutionError, match=r"expected torch\.float32"):
        executor(torch.ones(4, dtype=torch.float16))
    bad_graph = yaml.safe_load(yaml.safe_dump(graph))
    for layer in bad_graph["layers"].values():
        if layer["type"] != "start":
            layer["tensor_dtypes"]["outputs"] = ["torch.float16"]
    bad_exec = IRGraphExecutor(
        bad_graph, verification_backend(IRPath.MAKE_FX_ATEN.ir_kind)
    )
    with pytest.raises(IRExecutionError, match=r"expected torch\.float16"):
        bad_exec(torch.ones(4, dtype=torch.float32))

    # 2-step FP8 dequantization + unquantized bias on make_fx_aten linear.
    profile = ArchitectureProfile.load("RX_9060_XT")

    def fp8_two_step_linear(x_fp8, w_fp8, sx, sw, bias_bf16):
        x_dq = (x_fp8.to(torch.float32) * sx).to(torch.bfloat16)
        w_dq = (w_fp8.to(torch.float32) * sw).to(torch.bfloat16)
        return F.linear(x_dq, w_dq, bias_bf16)

    inputs = (
        torch.ones((4, 8), dtype=torch.float8_e4m3fn),
        torch.ones((6, 8), dtype=torch.float8_e4m3fn),
        torch.tensor(0.5, dtype=torch.float32),
        torch.tensor(0.25, dtype=torch.float32),
        torch.zeros(6, dtype=torch.bfloat16),
    )
    _, _, fp8_analysis = _compile_and_analyze(
        tmp_path,
        "fp8_linear",
        fp8_two_step_linear,
        inputs,
        precision="bf16",
        architecture=profile,
    )
    assert fp8_analysis["total"]["macs_by_precision"] == {"fp8": 192}
    assert fp8_analysis["total"]["resource_work"]["mfma"] == {"fp8->fp32": 384}


def test_p0_4_repeated_operands_metadata_reads_masked_fill_and_resource_rules(
    tmp_path: Path,
) -> None:
    x = torch.randn(4, 4)
    w = torch.randn(4, 4)
    mm_graph, _, self_mm = _compile_and_analyze(
        tmp_path, "self_mm", lambda a: torch.mm(a, a), (x,)
    )
    assert self_mm["total"]["macs"] == 64

    _, _, self_addmm = _compile_and_analyze(
        tmp_path, "self_addmm", lambda a, b: torch.addmm(a, a, b), (x, w)
    )
    assert self_addmm["total"]["macs"] == 64

    for source_layers in (mm_graph["layers"], self_mm["layers"]):
        mm_layer = next(
            layer for layer in source_layers.values() if layer["type"] == "mm"
        )
        proof = build_orojenesis_proof_layer(
            mm_layer, analyzer=EinsumAnalyzer()
        )
        assert proof is not None
        assert len(proof["tensor_shapes"]["inputs"]) == 2
        assert _layer_compulsory_bytes(proof, word_bits=32) == float(3 * 16 * 4)

    # type_as operand 1 and zeros_like/empty_like/full_like operand 0 are metadata-only.
    big = torch.randn(10, 10, dtype=torch.float16)
    small = torch.randn(2, 2, dtype=torch.float32)
    _, _, meta_analysis = _compile_and_analyze(
        tmp_path,
        "meta_ops",
        lambda s, b: s.type_as(b) + torch.zeros_like(b)[:2, :2].type_as(b),
        (small, big),
    )
    reads = meta_analysis["metadata"]["external_reads"]
    assert reads["quality"] == AccessQuality.EXACT.value
    assert reads["elements"] == 4

    # masked_fill: external self blocks, internal self + external mask passes.
    assert "masked_fill" not in FULL_OPERAND_READS
    mask = torch.tensor([[True, False], [False, True]])
    _, _, blocked_mf = _compile_and_analyze(
        tmp_path,
        "mf_ext",
        lambda a, m: a.masked_fill(m, 0.0),
        (small, mask),
    )
    assert blocked_mf["metadata"]["external_reads"]["reasons"] == [
        AccessReason.UNKNOWN_READS.value
    ]
    _, _, proved_mf = _compile_and_analyze(
        tmp_path,
        "mf_int",
        lambda a, m: (a + 1.0).masked_fill(m, 0.0),
        (small, mask),
    )
    assert (
        proved_mf["metadata"]["external_reads"]["quality"]
        == AccessQuality.EXACT.value
    )

    # Resource rules: bool keepdim in sum, norms, reduced kl_div, causal/v_width SDPA.
    sum_bool = classify_layer_resources(
        {
            "type": "sum",
            "semantic_op": {
                "kind": "operation",
                "target": "sum",
                "arguments": [{"tensor": 0}, {"value": True}],
                "kwargs": {},
            },
            "tensor_shapes": {"inputs": [[3, 4]], "outputs": [[1, 1]]},
            "tensor_dtypes": {"inputs": ["fp32"], "outputs": ["fp32"]},
        },
        macs=0,
        fallback_precision="fp32",
        strict=True,
    )
    assert sum_bool["work"] == {"reduction": {"fp32": 11}}

    gn = classify_layer_resources(
        {
            "type": "group_norm",
            "semantic_op": {
                "kind": "operation",
                "target": "group_norm",
                "arguments": [{"tensor": 0}, {"value": 4}],
                "kwargs": {},
            },
            "tensor_shapes": {"inputs": [[2, 8, 3]], "outputs": [[2, 8, 3]]},
            "tensor_dtypes": {"inputs": ["fp32"], "outputs": ["fp32"]},
        },
        macs=0,
        fallback_precision="fp32",
        strict=True,
    )
    assert gn["work"] == {
        "reduction": {"fp32": 80},
        "sfu": {"fp32": 8},
        "valu": {"fp32": 240},
    }

    bn_eval = classify_layer_resources(
        {
            "type": "batch_norm",
            "semantic_op": {
                "kind": "operation",
                "target": "batch_norm",
                "kwargs": {"training": False},
            },
            "tensor_shapes": {"inputs": [[2, 4, 3]], "outputs": [[2, 4, 3]]},
            "tensor_dtypes": {"inputs": ["fp32"], "outputs": ["fp32"]},
        },
        macs=0,
        fallback_precision="fp32",
        strict=True,
    )
    assert bn_eval["work"] == {"sfu": {"fp32": 4}, "valu": {"fp32": 48}}

    kl_reduced = classify_layer_resources(
        {
            "type": "kl_div",
            "semantic_op": {"kind": "operation", "target": "kl_div"},
            "tensor_shapes": {"inputs": [[4, 5], [4, 5]], "outputs": [[]]},
            "tensor_dtypes": {"inputs": ["fp32", "fp32"], "outputs": ["fp32"]},
        },
        macs=0,
        fallback_precision="fp32",
        strict=True,
    )
    assert kl_reduced["work"] == {
        "reduction": {"fp32": 19},
        "sfu": {"fp32": 20},
        "valu": {"fp32": 60},
    }

    sdpa = classify_layer_resources(
        {
            "type": "scaled_dot_product_attention",
            "semantic_op": {
                "kind": "operation",
                "target": "scaled_dot_product_attention",
                "kwargs": {"is_causal": True},
            },
            "tensor_shapes": {
                "inputs": [[2, 4, 8], [2, 4, 8], [2, 4, 16]],
                "outputs": [[2, 4, 16]],
            },
            "tensor_dtypes": {
                "inputs": ["fp16", "fp16", "fp16"],
                "outputs": ["fp16"],
            },
        },
        macs=0,
        fallback_precision="fp16",
        strict=True,
    )
    # 2 batches * (4*5//2 = 10) = 20 causal scores; mfma = 2 * 20 * (8 + 16) = 960.
    assert sdpa["work"] == {
        "mfma": {"fp16->fp32": 960},
        "reduction": {"fp16": 24},
        "sfu": {"fp16": 20},
        "valu": {"fp16": 40},
    }


def test_p0_5_grad_capture_views_cotangents_and_error_matching() -> None:
    t = torch.ones(3, requires_grad=True)
    with (
        pytest.raises(RuntimeError, match="Mutable operation"),
        GradientBoundaryCapture(),
        torch.no_grad(),
    ):
        t.add_(1.0)

    # Non-uniform cotangents catch transpose/softmax VJP errors that value.sum() missed.
    with pytest.raises(GradientVerificationError):
        verify_gradients(
            lambda x: torch.softmax(x, dim=-1),
            cast(
                IRGraphExecutor,
                lambda x: torch.softmax(x, dim=-1).detach() + 0.0 * x,
            ),
            {"source_input_indices": [0]},
            (torch.tensor([0.2, 0.7, -0.4]),),
            VerificationPolicy(atol=1e-6, rtol=1e-6),
        )
    with pytest.raises(GradientVerificationError):
        verify_gradients(
            lambda x: x.transpose(0, 1),
            cast(IRGraphExecutor, lambda x: x),
            {"source_input_indices": [0]},
            (torch.tensor([[1.0, 2.0], [3.0, 4.0]]),),
            VerificationPolicy(atol=1e-6, rtol=1e-6),
        )

    # Executor shape RuntimeError must NOT match reference leaf-inplace autograd error.
    def leaf_inplace_ref(x):
        return x.add_(1.0)

    def shape_bug_exec(x):
        return torch.mm(x, torch.ones(99, 99))

    with pytest.raises(
        GradientVerificationError, match="error semantics differ"
    ):
        verify_gradients(
            leaf_inplace_ref,
            cast(IRGraphExecutor, shape_bug_exec),
            {"source_input_indices": [0]},
            (torch.ones(2, 2),),
            VerificationPolicy(atol=1e-5, rtol=1e-5),
        )


def test_p1_cloning_isolation_and_torchview_module_state(
    tmp_path: Path,
) -> None:
    def inplace_ref(x: torch.Tensor) -> torch.Tensor:
        x.add_(2.0)
        return x * 3.0

    caller_tensor = torch.tensor([1.0, 4.0])
    op_art = extract_operator_graph(
        inplace_ref,
        (caller_tensor,),
        device="cpu",
        output_dir=tmp_path / "inplace_fx",
        name="inplace_fx",
        extraction_kind=IRPath.MAKE_FX_ATEN.extraction_kind,
    )
    assert torch.equal(caller_tensor, torch.tensor([1.0, 4.0]))
    ir_art = convert_operator_graph(
        op_art,
        output_dir=tmp_path / "inplace_fx",
        ir_kind=IRPath.MAKE_FX_ATEN.ir_kind,
    )
    # _verify_case_gradients receives unmutated pre-forward inputs and verifies cleanly.
    receipt = verify_callable_conversion(
        reference=inplace_ref,
        input_factory=lambda seed: (torch.tensor([float(seed), 2.0]),),
        reference_name="inplace_ref",
        reference_sha256="a" * 64,
        graph_path=ir_art.path,
        output_path=tmp_path / "inplace_fx" / "attestation.yaml",
        policy=VerificationPolicy(atol=1e-5, rtol=1e-5, seeds=(3, 4, 5)),
        backend=verification_backend(IRPath.MAKE_FX_ATEN.ir_kind),
    )
    assert (
        receipt["predicate"]["results"][0]["gradient_error_semantics_verified"]
        == 1.0
    )

    # Torchview registers nn.Module on ReferenceModule and preserves duplicate Parameter & buffer edges.
    class SharedParamAndBufferModule(nn.Module):
        buf: torch.Tensor

        def __init__(self) -> None:
            super().__init__()
            self.p = nn.Parameter(torch.ones(4, 4))
            self.register_buffer("buf", torch.full((4, 4), 0.5))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.addmm(self.p, x, self.p) + self.buf

    mod = SharedParamAndBufferModule()
    tv_art = extract_operator_graph(
        mod,
        (torch.ones(4, 4),),
        device="cpu",
        output_dir=tmp_path / "tv_mod",
        name="tv_mod",
        extraction_kind=IRPath.TORCHVIEW_EXTENDED_EINSUM.extraction_kind,
    )
    tv_graph = yaml.safe_load(tv_art.path.read_text())
    addmm_node = next(
        layer
        for layer in tv_graph["layers"].values()
        if layer["type"] == "addmm"
    )
    assert len(addmm_node["input_shapes"]) == 3
    assert len(addmm_node["connections"]["inputs"]) == 3


def test_p1_structure_certification_and_broadcast_matmul_problem(
    tmp_path: Path,
) -> None:
    # Rewired arguments are caught by structural_projection.
    def add_xy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a - b

    def add_yx(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return b - a

    art_ab = extract_operator_graph(
        add_xy,
        (torch.ones(4), torch.ones(4)),
        device="cpu",
        output_dir=tmp_path / "ab",
        name="ab",
        extraction_kind=IRPath.MAKE_FX_ATEN.extraction_kind,
    )
    art_ba = extract_operator_graph(
        add_yx,
        (torch.ones(4), torch.ones(4)),
        device="cpu",
        output_dir=tmp_path / "ba",
        name="ba",
        extraction_kind=IRPath.MAKE_FX_ATEN.extraction_kind,
    )
    proj_ab = structural_projection(yaml.safe_load(art_ab.path.read_text()))
    proj_ba = structural_projection(yaml.safe_load(art_ba.path.read_text()))
    assert (
        certify_structure(
            {"ab": proj_ab, "ba": proj_ba},
            {"ab": {"N": 4}, "ba": {"N": 4}},
        ).status
        is ShapeReasonCode.SHAPE_BRANCH_REJECTED
    )

    # Pairwise-consistent max(M, N) branch is rejected by global symbol check.
    def const_entry(val: int) -> list[dict]:
        return [
            {
                "type": "topk",
                "target": "topk",
                "exact_target": "topk",
                "overload": "default",
                "effects": {},
                "source_input_index": None,
                "connections": {"inputs": [], "outputs": []},
                "tensor_inputs": [],
                "tensor_outputs": [],
                "tensor_arguments": [],
                "tensor_kwargs": {},
                "constant_arguments": [{"value": val}],
                "constant_kwargs": {},
            }
        ]

    verdict = certify_structure(
        {"i0": const_entry(4), "i1": const_entry(8), "i2": const_entry(8)},
        {
            "i0": {"M": 4, "N": 4},
            "i1": {"M": 8, "N": 4},
            "i2": {"M": 4, "N": 8},
        },
    )
    assert verdict.status is ShapeReasonCode.SHAPE_BRANCH_REJECTED

    # Broadcast batched matmul builds a consistent Orojenesis problem.
    op = MatmulHandler().generate_einsum(
        "matmul",
        TensorShapes(
            inputs=[[1, 8, 4, 16], [3, 1, 16, 32]],
            outputs=[[3, 8, 4, 32]],
        ),
    )
    layer = {
        "semantic_op": {"kind": "einsum", "equation": op.equation},
        "tensor_shapes": {
            "inputs": [[1, 8, 4, 16], [3, 1, 16, 32]],
            "outputs": [[3, 8, 4, 32]],
        },
    }
    problem = problem_for_layer(layer)
    assert len(problem["problem"]["shape"]["data-spaces"]) == 3
    assert (
        mandatory_mfma_macs(
            "matmul",
            [[4, 8]],
            [[4, 4]],
            {"arguments": [{"tensor": 0}, {"tensor": 0}]},
        )
        == 128
    )


def test_p1_precision_contracts_orojenesis_safety_and_atomic_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert normalize_dtype("torch.float4_e2m1fn_x2") == "float4_e2m1fn_x2"
    assert dtype_bytes("torch.float4_e2m1fn_x2") == 1.0
    assert dtype_bytes("nvfp4") == 0.5

    rx = ArchitectureProfile.load("RX_9060_XT")
    with pytest.raises(ValueError, match="not supported"):
        rx.tensor_precision("torch.float64", "fp32")
    with pytest.raises(ValueError, match="not supported"):
        rx.tensor_precision("fp64", "fp32")

    with pytest.raises(ValueError, match="trace_seed"):
        ConversionRequest(
            analysis_id="id",
            reference=lambda x: x,
            input_factory=lambda s: (torch.ones(2),),
            reference_name="ref",
            reference_sha256="b" * 64,
            trace_seed=200,
            verification=VerificationPolicy(atol=1e-3, rtol=1e-3, seeds=(200,)),
        )

    env = mapper_subprocess_environment(
        {"PATH": "/usr/bin", "LD_LIBRARY_PATH": "/bad/lib"},
        cwd=tmp_path / "env",
    )
    assert "LD_LIBRARY_PATH" not in env

    # int4 curve byte math: 100 words * 4 bits / 8 = 50.0 bytes.
    csv_path = tmp_path / "int4.csv"
    csv_path.write_text("64,4.0,100\n")
    curve = OrojenesisRunner.parse_curve(csv_path, word_bytes=4 / 8.0)
    assert curve[0]["dram_bytes"] == 50.0

    # Divisor cap and monotonic deadline on multi-einsum runner.
    fake_runner = object.__new__(OrojenesisRunner)
    fake_runner.home = tmp_path
    fake_runner.mapper = tmp_path / "timeloop-mapper"
    fake_runner.timeout_seconds = 1
    fake_runner.toolchain_identity = {"verification_mode": "fake"}

    def mm_layer(inp: str, w: str, out: str, m: int, k: int, n: int) -> dict:
        return {
            "type": "matmul",
            "semantic_op": {"kind": "einsum", "equation": "MK,KN->MN"},
            "einsum_equation": "MK,KN->MN",
            "is_real_einsum": True,
            "tensor_names": {"inputs": [inp, w], "outputs": [out]},
            "tensor_shapes": {"inputs": [[m, k], [k, n]], "outputs": [[m, n]]},
            "tensor_dtypes": {
                "inputs": ["torch.float16", "torch.float16"],
                "outputs": ["torch.float16"],
            },
            "connections": {"inputs": [], "outputs": []},
        }

    # 720720 has 240 divisors > MAX_MULTI_EINSUM_DIVISORS (128).
    with pytest.raises(OrojenesisError, match="divisor count"):
        fake_runner.run_multi_chain(
            [
                ("m0", mm_layer("x", "w0", "h", 720720, 4, 4)),
                ("m1", mm_layer("h", "w1", "y", 720720, 4, 4)),
            ],
            tmp_path / "div_cap",
            word_bits=16,
        )

    clock = [100.0]
    monkeypatch.setattr(
        "solar.analysis.orojenesis.runner.time.monotonic",
        lambda: clock[0],
    )

    def advancing_runner(*args, **kwargs):
        clock[0] += 0.7
        cwd = kwargs["cwd"]
        row = ["64", "1", "24", "m", "0", "16", "8", "0", "0", "0", "8", "8"]
        row += ["0"] * 9 + ["8", "8", "8"]
        (cwd / "timeloop-mapper.oaves.csv").write_text(",".join(row) + "\n")
        return subprocess.CompletedProcess(args[0], 0, "", "")

    monkeypatch.setattr(
        fake_runner, "_process_runner", advancing_runner, raising=False
    )
    with pytest.raises(OrojenesisTimeoutError, match="exceeded"):
        fake_runner.run_multi_chain(
            [
                ("m0", mm_layer("x", "w0", "h", 4, 2, 2)),
                ("m1", mm_layer("h", "w1", "y", 4, 2, 2)),
            ],
            tmp_path / "deadline_chain",
            word_bits=16,
        )

    # Atomic YAML write leaves no temporary sibling files behind.
    yaml_target = tmp_path / "atomic" / "out.yaml"
    write_yaml_atomic(yaml_target, {"ok": [1, 2, 3]})
    assert yaml.safe_load(yaml_target.read_text()) == {"ok": [1, 2, 3]}
    assert list(yaml_target.parent.iterdir()) == [yaml_target]


def test_p1_load_verified_program_rejects_untrusted_orojenesis_mapper(
    tmp_path: Path,
) -> None:
    from solar.api import analyze

    root = tmp_path / "bundle"
    conv = ConversionRequest(
        analysis_id="vp-trust",
        reference=lambda x: x + 1.0,
        input_factory=lambda s: (torch.ones(4),),
        reference_name="ref",
        reference_sha256="c" * 64,
        ir_path=IRPath.MAKE_FX_ATEN,
    )
    res = analyze(
        AnalysisRequest(
            conversion=conv,
            architecture="RX_9060_XT",
            output_dir=root,
            precision="fp32",
            require_verified_audit=False,
        )
    )
    assert isinstance(res, AnalysisResult)
    analysis_path = root / ANALYSIS_FILENAME
    analysis = yaml.safe_load(analysis_path.read_text())
    analysis["metadata"]["orojenesis"]["status"] = "complete"
    analysis["metadata"]["orojenesis"]["toolchain"] = {
        "artifact": {"sha256": "0" * 64}
    }
    write_yaml_atomic(analysis_path, analysis, no_aliases=False)
    manifest_path = root / MANIFEST_FILENAME
    manifest = yaml.safe_load(manifest_path.read_text())
    for item in manifest["artifacts"]:
        if item["path"] == ANALYSIS_FILENAME:
            item["sha256"] = sha256_file(analysis_path)
    write_yaml_atomic(manifest_path, manifest, no_aliases=False)
    receipt = yaml.safe_load(
        (root / CONVERSION_ATTESTATION_FILENAME).read_text()
    )
    req = VerifiedProgramRequest(
        directory=root,
        manifest_sha256=sha256_file(manifest_path),
        reference_sha256="c" * 64,
        ir_path=IRPath.MAKE_FX_ATEN,
        verification=conv.verification,
        execution_sha256=stable_json_checksum(
            receipt["predicate"]["execution"]
        ),
    )
    with pytest.raises(ValueError, match="not trusted"):
        load_verified_program(req)
