"""Fresh clone storage does not imply an unknown full operand read."""

from pathlib import Path

import pytest
import torch
import yaml

from scripts.internal.solar.base01_records import PROFILE
from solar.analysis.graph_analyzer import IRGraphAnalyzer
from solar.artifacts import sha256_file
from solar.contracts import ConversionRequest
from solar.ir.access import AccessReason
from solar.ir.access_artifact import load_access_document
from solar.ir.contracts import IRPath, layer_operation
from solar.ir.registry import ir_backend
from solar.pipeline.readiness import (
    ConversionReadinessRequest,
    audit_conversion,
)


def _reference(case):
    if case == "clone":
        return lambda x: x.clone()
    if case == "slice-clone":
        return lambda x: x[:2].clone()
    if case == "clone-sum":
        return lambda x: x.clone().sum()
    if case == "clone-slice":
        return lambda x: x.clone()[:2].sum()
    if case == "conditional-contiguous":
        return lambda x: x.contiguous().sum()
    if case == "clone-mutation":
        return lambda x: x.clone().add_(1)
    if case == "copy-mutation":
        return lambda x: x.copy_(x + 1)
    raise AssertionError(case)


@pytest.mark.parametrize("path", list(IRPath))
@pytest.mark.parametrize(
    ("case", "elements", "reason"),
    [
        ("clone", 32, None),
        ("slice-clone", 16, None),
        ("clone-sum", 32, None),
        ("clone-slice", None, "copy_boundary"),
        ("conditional-contiguous", None, "copy_boundary"),
        ("clone-mutation", None, "effect_boundary"),
        ("copy-mutation", None, "effect_boundary"),
    ],
)
def test_copy_read_footprints_preserve_storage_and_effect_boundaries(
    tmp_path, path, case, elements, reason
):
    # make_fx removes the no-op contiguous call on this dense source.
    # TorchView retains a conditional alias which must remain unproved.
    if case == "conditional-contiguous" and path is IRPath.MAKE_FX_ATEN:
        elements, reason = 32, None
    root = tmp_path / "conversion"
    conversion = ConversionRequest(
        analysis_id=case,
        reference=_reference(case),
        input_factory=lambda _seed: (torch.arange(32.0).reshape(4, 8),),
        reference_name=case,
        reference_sha256=sha256_file(Path(__file__)),
        ir_path=path,
    )
    ready = audit_conversion(
        ConversionReadinessRequest(
            conversion=conversion,
            architecture=PROFILE,
            output_dir=root,
            require_verified_audit=False,
        )
    )
    if case == "copy-mutation" and path is IRPath.TORCHVIEW_EXTENDED_EINSUM:
        assert not ready.ready
        assert ready.reason_code == "strict_conversion_failed"
        assert "Unsupported operation: copy" in (ready.message or "")
        return
    assert ready.ready, ready.to_dict()
    graph_path = root / path.graph_filename
    graph = yaml.safe_load(graph_path.read_text())
    access = load_access_document(graph_path)
    for layer_id, layer in graph["layers"].items():
        if layer_operation(layer).get("target") == "clone":
            assert layer_id not in access.transparent_layers
            assert all(
                name not in access.tensors
                and access.reasons[name] is AccessReason.COPY
                for name in layer["tensor_names"]["outputs"]
            )
    result = IRGraphAnalyzer(
        validator=ir_backend(path.ir_kind).validate
    ).analyze_graph(
        graph_path,
        tmp_path / "analysis",
        strict=True,
        architecture=PROFILE,
        copy_graph=False,
    )
    assert result is not None
    reads = result["metadata"]["external_reads"]
    assert reads["elements"] == elements, reads
    assert reads["bytes"] == (None if elements is None else elements * 4)
    assert reads["quality"] == ("unavailable" if elements is None else "exact")
    if reason:
        assert reason in reads["reasons"]
    for name in ("unfused", "ideal_fused"):
        assert result["metadata"]["views"][name]["status"] == (
            "unavailable" if elements is None else "available"
        )
