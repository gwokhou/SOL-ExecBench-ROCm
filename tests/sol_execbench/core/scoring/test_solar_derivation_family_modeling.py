from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring.amd_bound_estimate.estimates import estimate_bound_work
from sol_execbench.core.scoring.amd_bound_graph import OpFamily, build_bound_graph
from sol_execbench.core.scoring.solar_derivation import (
    build_solar_derivation_evidence,
    solar_derivation_from_dict,
)
from sol_execbench_type_helpers import JsonDict, make_definition, make_workload


FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "solar_derivation"


def _load_fixture(name: str) -> JsonDict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _expectation(name: str) -> JsonDict:
    expectation = _load_fixture(name)["expectation"]
    assert isinstance(expectation, dict)
    return expectation


def _assert_group_matches_fixture(
    group,
    expectation: JsonDict,
) -> None:
    assert group.status == expectation["expected_status"]
    assert group.confidence == expectation["expected_confidence"]
    assert [subrole.name for subrole in group.subroles] == expectation[
        "expected_subroles"
    ]
    if expectation["expected_status"] == "scored":
        assert set(expectation["required_evidence"]) <= set(group.required_evidence)
    assert set(expectation["missing_evidence"]) <= set(group.missing_evidence)
    assert set(expectation["warning_prefixes"]) <= set(group.warning_prefixes)


def _attention_definition(*, mask: bool = False) -> Definition:
    inputs: dict[str, Any] = {
        "q": {
            "shape": ["batch", "heads", "sequence_q", "head_dim"],
            "dtype": "float32",
        },
        "k": {
            "shape": ["batch", "heads", "sequence_k", "head_dim"],
            "dtype": "float32",
        },
        "v": {
            "shape": ["batch", "heads", "sequence_k", "head_dim"],
            "dtype": "float32",
        },
        "w_o": {"shape": ["head_dim", "head_dim"], "dtype": "float32"},
    }
    if mask:
        inputs["mask"] = {
            "shape": ["batch", "heads", "sequence_q", "sequence_k"],
            "dtype": "float32",
        }
    reference = (
        "import torch\n\n"
        "def run(q, k, v, w_o):\n"
        "    scores = q @ k.transpose(-2, -1)\n"
        "    probs = torch.softmax(scores, dim=-1)\n"
        "    return (probs @ v) @ w_o\n"
    )
    if mask:
        reference = (
            "import torch\n\n"
            "def run(q, k, v, w_o, mask):\n"
            "    scores = q @ k.transpose(-2, -1)\n"
            "    scores = scores + mask\n"
            "    probs = torch.softmax(scores, dim=-1)\n"
            "    return (probs @ v) @ w_o\n"
        )
    return make_definition(
        name="attention_demo",
        axes={
            "batch": {"type": "var"},
            "heads": {"type": "const", "value": 4},
            "sequence_q": {"type": "const", "value": 16},
            "sequence_k": {"type": "const", "value": 16},
            "head_dim": {"type": "const", "value": 32},
        },
        inputs=inputs,
        outputs={
            "out": {
                "shape": ["batch", "heads", "sequence_q", "head_dim"],
                "dtype": "float32",
            }
        },
        reference=reference,
    )


def _attention_workload(*, mask: bool = False) -> Workload:
    inputs: dict[str, Any] = {
        "q": {"type": "random"},
        "k": {"type": "random"},
        "v": {"type": "random"},
        "w_o": {"type": "random"},
    }
    if mask:
        inputs["mask"] = {"type": "random"}
    return make_workload(axes={"batch": 2}, inputs=inputs, uuid="attention-workload")


def _moe_definition(
    *, dynamic: bool = False, taxonomy_only: bool = False
) -> Definition:
    if taxonomy_only:
        return make_definition(
            name="moe_taxonomy_only",
            axes={
                "tokens": {"type": "const", "value": 128},
                "hidden": {"type": "const", "value": 256},
            },
            inputs={
                "x": {"shape": ["tokens", "hidden"], "dtype": "float16"},
                "opaque_moe": {"shape": ["hidden", "hidden"], "dtype": "float16"},
            },
            outputs={"out": {"shape": ["tokens", "hidden"], "dtype": "float16"}},
            reference="def run(x, opaque_moe):\n    return opaque_moe(x)\n",
        )
    inputs: dict[str, Any] = {
        "x": {"shape": ["tokens", "hidden"], "dtype": "float16"},
        "router": {"shape": ["hidden", "experts"], "dtype": "float16"},
        "expert_weights": {
            "shape": ["experts", "hidden", "hidden"],
            "dtype": "float16",
        },
    }
    if dynamic:
        inputs["threshold"] = {"shape": None, "dtype": "float16"}
        reference = (
            "def run(x, router, expert_weights, threshold):\n"
            "    scores = router(x)\n"
            "    chosen = scores > threshold\n"
            "    return dispatch_dynamic(x, expert_weights, chosen)\n"
        )
    else:
        reference = (
            "import torch\n\n"
            "def run(x, router, expert_weights):\n"
            "    scores = router(x)\n"
            "    gates = torch.topk(scores, k=2, dim=-1)\n"
            "    return dispatch_and_combine(x, expert_weights, gates)\n"
        )
    return make_definition(
        name="moe_dynamic_route" if dynamic else "moe_static_route",
        axes={
            "tokens": {"type": "const", "value": 128},
            "hidden": {"type": "const", "value": 256},
            "experts": {"type": "const", "value": 8},
        },
        inputs=inputs,
        outputs={"out": {"shape": ["tokens", "hidden"], "dtype": "float16"}},
        reference=reference,
    )


def _moe_workload(*, dynamic: bool = False, taxonomy_only: bool = False) -> Workload:
    if taxonomy_only:
        return make_workload(
            axes={},
            inputs={"x": {"type": "random"}, "opaque_moe": {"type": "random"}},
            uuid="moe-taxonomy-workload",
        )
    inputs = {
        "x": {"type": "random"},
        "router": {"type": "random"},
        "expert_weights": {"type": "random"},
    }
    if dynamic:
        inputs["threshold"] = {"type": "random"}
    return make_workload(
        axes={},
        inputs=inputs,
        uuid="moe-dynamic-workload" if dynamic else "moe-static-workload",
    )


def _ssm_mamba_definition(
    *, missing_recurrence: bool = False, custom_scan: bool = False
) -> Definition:
    if custom_scan:
        return make_definition(
            name="ssm_mamba_custom_scan",
            axes={
                "batch": {"type": "const", "value": 2},
                "sequence": {"type": "const", "value": 64},
                "hidden": {"type": "const", "value": 128},
            },
            inputs={
                "x": {"shape": ["batch", "sequence", "hidden"], "dtype": "float16"},
                "opaque_scan": {"shape": ["hidden", "hidden"], "dtype": "float16"},
            },
            outputs={
                "out": {"shape": ["batch", "sequence", "hidden"], "dtype": "float16"}
            },
            reference="def run(x, opaque_scan):\n    return opaque_scan(x)\n",
        )
    inputs = {
        "x": {"shape": ["batch", "sequence", "hidden"], "dtype": "float16"},
        "w_in": {"shape": ["hidden", "hidden"], "dtype": "float16"},
        "conv_weight": {"shape": ["hidden", "one", "kernel"], "dtype": "float16"},
    }
    if missing_recurrence:
        inputs["params"] = {"shape": ["hidden"], "dtype": "float16"}
        inputs["w_out"] = {"shape": ["hidden", "hidden"], "dtype": "float16"}
        reference = (
            "def run(x, w_in, conv_weight, params, w_out):\n"
            "    z = in_proj(x, w_in)\n"
            "    z = depthwise_conv(z, conv_weight)\n"
            "    y = selective_scan(z, params)\n"
            "    return out_proj(y, w_out)\n"
        )
    else:
        inputs.update(
            {
                "a": {"shape": ["hidden", "state"], "dtype": "float16"},
                "b": {"shape": ["hidden", "state"], "dtype": "float16"},
                "c": {"shape": ["hidden", "state"], "dtype": "float16"},
                "w_out": {"shape": ["hidden", "hidden"], "dtype": "float16"},
            }
        )
        reference = (
            "def run(x, w_in, conv_weight, a, b, c, w_out):\n"
            "    z = in_proj(x, w_in)\n"
            "    z = depthwise_conv(z, conv_weight)\n"
            "    y = selective_scan(z, a, b, c)\n"
            "    y = gate(y)\n"
            "    return out_proj(y, w_out)\n"
        )
    return make_definition(
        name="ssm_mamba_missing_recurrence"
        if missing_recurrence
        else "ssm_mamba_static",
        axes={
            "batch": {"type": "const", "value": 2},
            "sequence": {"type": "const", "value": 64},
            "hidden": {"type": "const", "value": 128},
            "state": {"type": "const", "value": 16},
            "one": {"type": "const", "value": 1},
            "kernel": {"type": "const", "value": 3},
        },
        inputs=inputs,
        outputs={"out": {"shape": ["batch", "sequence", "hidden"], "dtype": "float16"}},
        reference=reference,
    )


def _ssm_mamba_workload(
    *, missing_recurrence: bool = False, custom_scan: bool = False
) -> Workload:
    if custom_scan:
        return make_workload(
            axes={},
            inputs={"x": {"type": "random"}, "opaque_scan": {"type": "random"}},
            uuid="ssm-custom-workload",
        )
    inputs = {
        "x": {"type": "random"},
        "w_in": {"type": "random"},
        "conv_weight": {"type": "random"},
    }
    if missing_recurrence:
        inputs.update({"params": {"type": "random"}, "w_out": {"type": "random"}})
    else:
        inputs.update(
            {
                "a": {"type": "random"},
                "b": {"type": "random"},
                "c": {"type": "random"},
                "w_out": {"type": "random"},
            }
        )
    return make_workload(axes={}, inputs=inputs, uuid="ssm-mamba-workload")


def test_ssm_mamba_static_sidecar_matches_positive_fixture_contract():
    expectation = _expectation("ssm_mamba_positive.json")
    evidence = build_solar_derivation_evidence(
        _ssm_mamba_definition(), _ssm_mamba_workload()
    )
    group = next(group for group in evidence.groups if group.family == "ssm_mamba")

    _assert_group_matches_fixture(group, expectation)
    assert group.missing_evidence == ()
    assert group.warning_prefixes == ()
    assert [item.formula_kind for item in group.formula_evidence] == [
        "ssm_mamba_visible_subrole_bytes",
        "ssm_mamba_visible_subrole_bytes",
        "ssm_mamba_static_scan_flops",
        "ssm_mamba_visible_subrole_bytes",
        "ssm_mamba_visible_subrole_bytes",
        "ssm_mamba_visible_subrole_bytes",
    ]
    assert any(item.total_bytes > 0.0 for item in group.byte_evidence)
    assert group.bound_evidence


def test_ssm_mamba_missing_recurrence_sidecar_matches_degraded_fixture_contract():
    expectation = _expectation("ssm_mamba_degraded_missing_recurrence.json")
    evidence = build_solar_derivation_evidence(
        _ssm_mamba_definition(missing_recurrence=True),
        _ssm_mamba_workload(missing_recurrence=True),
    )
    group = next(group for group in evidence.groups if group.family == "ssm_mamba")

    _assert_group_matches_fixture(group, expectation)
    assert any(
        item.formula_kind == "ssm_mamba_degraded_scan_bytes"
        for item in group.formula_evidence
    )
    assert not any(subrole.name == "state_update" for subrole in group.subroles)


def test_ssm_mamba_custom_scan_sidecar_matches_unsupported_fixture_contract():
    expectation = _expectation("ssm_mamba_unsupported_custom_scan.json")
    evidence = build_solar_derivation_evidence(
        _ssm_mamba_definition(custom_scan=True),
        _ssm_mamba_workload(custom_scan=True),
    )
    group = next(group for group in evidence.groups if group.family == "ssm_mamba")

    _assert_group_matches_fixture(group, expectation)
    assert not any(subrole.name == "state_update" for subrole in group.subroles)


def test_moe_static_route_sidecar_matches_positive_fixture_contract():
    expectation = _expectation("moe_positive.json")
    evidence = build_solar_derivation_evidence(_moe_definition(), _moe_workload())
    group = next(group for group in evidence.groups if group.family == "moe")

    _assert_group_matches_fixture(group, expectation)
    assert group.missing_evidence == ()
    assert group.warning_prefixes == ()
    assert {item.formula_kind for item in group.formula_evidence} == {
        "moe_static_route_flops",
        "moe_dynamic_route_bytes",
    }
    assert any(item.total_bytes > 0.0 for item in group.byte_evidence)
    assert group.bound_evidence


def test_moe_dynamic_route_sidecar_matches_degraded_fixture_contract():
    expectation = _expectation("moe_degraded_dynamic_routing.json")
    evidence = build_solar_derivation_evidence(
        _moe_definition(dynamic=True),
        _moe_workload(dynamic=True),
    )
    group = next(group for group in evidence.groups if group.family == "moe")

    _assert_group_matches_fixture(group, expectation)
    assert any(
        item.formula_kind == "moe_dynamic_route_bytes"
        for item in group.formula_evidence
    )


def test_moe_taxonomy_only_sidecar_matches_unsupported_fixture_contract():
    expectation = _expectation("moe_unsupported_taxonomy_only.json")
    evidence = build_solar_derivation_evidence(
        _moe_definition(taxonomy_only=True),
        _moe_workload(taxonomy_only=True),
    )
    group = next(group for group in evidence.groups if group.family == "moe")

    _assert_group_matches_fixture(group, expectation)
    assert group.subroles == ()


def test_attention_sidecar_records_subroles_formula_bytes_and_bounds():
    evidence = build_solar_derivation_evidence(
        _attention_definition(),
        _attention_workload(),
    )
    payload = solar_derivation_from_dict(evidence.to_dict()).to_dict()
    group = next(group for group in payload["groups"] if group["family"] == "attention")

    assert group["status"] == "scored"
    assert group["confidence"] == "supported"
    assert [subrole["name"] for subrole in group["subroles"]] == [
        "k_projection",
        "output_projection",
        "pv_aggregation",
        "q_projection",
        "qk_scores",
        "softmax",
        "v_projection",
    ]
    formula_by_role = {
        subrole["name"]: next(
            item
            for item in group["formula_evidence"]
            if item["node_id"] == subrole["node_ids"][0]
        )
        for subrole in group["subroles"]
        if subrole["name"]
        in {"qk_scores", "softmax", "pv_aggregation", "output_projection"}
    }
    assert formula_by_role["qk_scores"]["formula_kind"] == "attention_scores_flops"
    assert formula_by_role["qk_scores"]["formula_inputs"] == {
        "B": 2,
        "H": 4,
        "D": 32,
        "S_k": 16,
        "S_q": 16,
    }
    assert formula_by_role["softmax"]["formula_kind"] == "attention_softmax_flops"
    assert formula_by_role["pv_aggregation"]["formula_kind"] == "attention_pv_flops"
    assert formula_by_role["output_projection"]["formula_kind"] == "gemm_flops"
    assert len(group["byte_evidence"]) == len(group["formula_evidence"])
    assert len(group["bound_evidence"]) == len(group["formula_evidence"])
    assert group["missing_evidence"] == []


def test_attention_partial_mask_degrades_with_mask_semantics_missing():
    evidence = build_solar_derivation_evidence(
        _attention_definition(mask=True),
        _attention_workload(mask=True),
    )
    group = next(group for group in evidence.groups if group.family == "attention")

    assert group.status == "degraded"
    assert group.confidence == "inexact"
    assert "mask:semantics" in group.missing_evidence
    assert "mask:sparsity" in group.missing_evidence
    assert any(
        warning.startswith("inexact_operator:attention_mask")
        for warning in group.warning_prefixes
    )
    assert any(subrole.name == "scale_or_mask" for subrole in group.subroles)


def test_attention_dynamic_sequence_axes_are_unscored_without_fabricated_subroles():
    definition = make_definition(
        name="dynamic_attention",
        axes={
            "batch": {"type": "var"},
            "heads": {"type": "const", "value": 4},
            "head_dim": {"type": "const", "value": 32},
        },
        inputs={
            "q": {"shape": ["batch", "heads", "head_dim"], "dtype": "float32"},
            "k": {"shape": ["batch", "heads", "head_dim"], "dtype": "float32"},
            "v": {"shape": ["batch", "heads", "head_dim"], "dtype": "float32"},
            "lengths": {"shape": ["batch"], "dtype": "int64"},
        },
        outputs={"out": {"shape": ["batch", "heads", "head_dim"], "dtype": "float32"}},
        reference=(
            "import torch\n\n"
            "def run(q, k, v, lengths):\n"
            "    n = int(lengths.max().item())\n"
            "    return torch.softmax(q[:, :n] @ k[:, :n].transpose(-2, -1), dim=-1) @ v[:, :n]\n"
        ),
    )
    workload = make_workload(
        axes={"batch": 2},
        inputs={
            "q": {"type": "random"},
            "k": {"type": "random"},
            "v": {"type": "random"},
            "lengths": {"type": "random"},
        },
        uuid="dynamic-attention-workload",
    )

    evidence = build_solar_derivation_evidence(definition, workload)
    group = next(group for group in evidence.groups if group.family == "attention")

    assert group.status == "unscored"
    assert group.confidence == "unsupported"
    assert "axis:static_sequence" in group.missing_evidence
    assert "shape:sequence_q" in group.missing_evidence
    assert "shape:sequence_k" in group.missing_evidence
    assert not any(subrole.name == "q_projection" for subrole in group.subroles)


def test_attention_estimates_keep_projection_family_inside_attention_group():
    graph = build_bound_graph(_attention_definition(), _attention_workload())
    estimates = estimate_bound_work(graph)

    attention_nodes = [
        node for node in graph.nodes if node.op_family == OpFamily.ATTENTION
    ]
    attention_estimates = [
        estimate for estimate in estimates if estimate.op_family == OpFamily.ATTENTION
    ]

    assert {node.attributes.get("subrole") for node in attention_nodes} >= {
        "qk_scores",
        "softmax",
        "pv_aggregation",
        "output_projection",
    }
    assert {estimate.formula_kind for estimate in attention_estimates} >= {
        "attention_scores_flops",
        "attention_softmax_flops",
        "attention_pv_flops",
        "gemm_flops",
    }


def test_convolution_sidecar_records_subroles_formula_bytes_and_bounds():
    definition = make_definition(
        name="conv3d_sidecar_demo",
        axes={
            "B": {"type": "const", "value": 1},
            "C": {"type": "const", "value": 2},
            "O": {"type": "const", "value": 4},
            "D": {"type": "const", "value": 5},
            "H": {"type": "const", "value": 6},
            "W": {"type": "const", "value": 6},
            "KD": {"type": "const", "value": 3},
            "OH": {"type": "const", "value": 4},
            "OW": {"type": "const", "value": 4},
        },
        inputs={
            "x": {"shape": ["B", "C", "D", "H", "W"], "dtype": "float32"},
            "weight": {"shape": ["O", "C", "KD", "KD", "KD"], "dtype": "float32"},
            "bias": {"shape": ["O"], "dtype": "float32"},
        },
        outputs={"out": {"shape": ["B", "O", "KD", "OH", "OW"], "dtype": "float32"}},
        reference=(
            "import torch.nn.functional as F\n\n"
            "def run(x, weight, bias):\n"
            "    return F.conv3d(x, weight, bias, stride=(1, 1, 1), padding=(0, 0, 0), "
            "dilation=(1, 1, 1), groups=1)\n"
        ),
    )
    workload = make_workload(
        axes={},
        inputs={
            "x": {"type": "random"},
            "weight": {"type": "random"},
            "bias": {"type": "random"},
        },
        uuid="conv3d-sidecar-workload",
    )

    evidence = build_solar_derivation_evidence(definition, workload)
    group = next(group for group in evidence.groups if group.family == "convolution")

    assert group.status == "scored"
    assert group.confidence == "supported"
    assert [subrole.name for subrole in group.subroles] == [
        "bias",
        "convolution_metadata",
        "input",
        "output",
        "weight",
    ]
    assert group.formula_evidence[0].formula_kind == "convolution_flops"
    assert group.byte_evidence[0].total_bytes > 0.0
    assert group.bound_evidence[0].limiting_resource in {"compute", "memory"}
    assert group.missing_evidence == ()


def test_convolution_omitted_padding_uses_framework_default():
    definition = make_definition(
        name="conv_missing_padding",
        axes={
            "B": {"type": "const", "value": 1},
            "C": {"type": "const", "value": 2},
            "O": {"type": "const", "value": 4},
            "L": {"type": "const", "value": 8},
            "K": {"type": "const", "value": 3},
            "OL": {"type": "const", "value": 6},
        },
        inputs={
            "x": {"shape": ["B", "C", "L"], "dtype": "float32"},
            "weight": {"shape": ["O", "C", "K"], "dtype": "float32"},
        },
        outputs={"out": {"shape": ["B", "O", "OL"], "dtype": "float32"}},
        reference=(
            "import torch.nn.functional as F\n\n"
            "def run(x, weight):\n"
            "    return F.conv1d(x, weight, stride=1, dilation=1, groups=1)\n"
        ),
    )
    workload = make_workload(
        axes={},
        inputs={"x": {"type": "random"}, "weight": {"type": "random"}},
        uuid="conv-missing-padding-workload",
    )

    evidence = build_solar_derivation_evidence(definition, workload)
    group = next(group for group in evidence.groups if group.family == "convolution")

    assert group.status == "scored"
    assert group.confidence == "supported"
    assert "convolution:padding" not in group.missing_evidence


def test_embedding_positional_sidecar_records_memory_bound_evidence():
    definition = make_definition(
        name="embedding_sidecar_demo",
        axes={
            "T": {"type": "const", "value": 32},
            "N": {"type": "const", "value": 4},
            "D": {"type": "const", "value": 8},
        },
        inputs={
            "table": {"shape": ["T", "D"], "dtype": "float16"},
            "indices": {"shape": ["N"], "dtype": "int64"},
            "x": {"shape": ["N", "D"], "dtype": "float16"},
            "pos": {"shape": ["N", "D"], "dtype": "float16"},
            "sin": {"shape": ["N", "D"], "dtype": "float16"},
            "cos": {"shape": ["N", "D"], "dtype": "float16"},
        },
        outputs={"out": {"shape": ["N", "D"], "dtype": "float16"}},
        reference=(
            "import torch\n"
            "import torch.nn.functional as F\n\n"
            "def run(table, indices, x, pos, sin, cos):\n"
            "    token = F.embedding(indices, table)\n"
            "    rotated = (x * cos) + (x * sin)\n"
            "    return x + pos + token + rotated\n"
        ),
    )
    workload = make_workload(
        axes={},
        inputs={
            "table": {"type": "random"},
            "indices": {"type": "random"},
            "x": {"type": "random"},
            "pos": {"type": "random"},
            "sin": {"type": "random"},
            "cos": {"type": "random"},
        },
        uuid="embedding-sidecar-workload",
    )

    evidence = build_solar_derivation_evidence(definition, workload)
    group = next(
        group for group in evidence.groups if group.family == "embedding_positional"
    )

    assert group.status == "scored"
    assert group.confidence == "supported"
    assert {"embedding_lookup", "positional_add", "rotary_like"} <= {
        subrole.name for subrole in group.subroles
    }
    assert {item.formula_kind for item in group.formula_evidence} == {
        "embedding_positional_bytes"
    }
    assert all(item.total_bytes > 0.0 for item in group.byte_evidence)
    assert all(item.limiting_resource == "memory" for item in group.bound_evidence)


def test_embedding_dynamic_indices_degrade_without_selected_byte_fabrication():
    definition = make_definition(
        name="embedding_dynamic_indices",
        axes={
            "T": {"type": "const", "value": 32},
            "D": {"type": "const", "value": 8},
        },
        inputs={
            "table": {"shape": ["T", "D"], "dtype": "float16"},
            "indices": {"shape": None, "dtype": "int64"},
        },
        outputs={"out": {"shape": None, "dtype": "float16"}},
        reference=(
            "import torch.nn.functional as F\n\n"
            "def run(table, indices):\n"
            "    return F.embedding(indices, table)\n"
        ),
    )
    workload = make_workload(
        axes={},
        inputs={"table": {"type": "random"}, "indices": {"type": "random"}},
        uuid="embedding-dynamic-workload",
    )

    evidence = build_solar_derivation_evidence(definition, workload)
    group = next(
        group for group in evidence.groups if group.family == "embedding_positional"
    )

    assert group.status == "degraded"
    assert group.confidence == "inexact"
    assert "embedding_positional:output_shape" in group.missing_evidence
    assert "embedding_positional:selected_elements" in group.missing_evidence
