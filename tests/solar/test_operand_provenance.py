from __future__ import annotations

from solar.analysis.operand_provenance import (
    contraction_external_source_dtypes,
    contraction_operands_are_graph_external,
)


def _start(name: str, *, dtype: str | None = "torch.float16") -> dict:
    dtypes = [] if dtype is None else [dtype]
    return {
        "type": "start",
        "tensor_names": {"inputs": [], "outputs": [name]},
        "tensor_dtypes": {"inputs": [], "outputs": dtypes},
    }


def test_alias_cycle_missing_dtype_and_effects_fail_closed():
    layers = {
        "start": _start("input"),
        "missing_dtype": _start("untyped", dtype=None),
        "bad_alias": {
            "type": "view",
            "semantic_op": {
                "kind": "aten",
                "target": "view",
                "effects": {
                    "aliases": [{"input": 9, "output": 0, "conditional": False}]
                },
            },
            "tensor_names": {"inputs": ["input"], "outputs": ["bad"]},
        },
        "mutating": {
            "type": "mul",
            "semantic_op": {
                "kind": "aten",
                "target": "mul",
                "effects": {"mutates": True},
            },
            "tensor_names": {"inputs": ["input"], "outputs": ["mutated"]},
        },
        "cycle": {
            "type": "view",
            "semantic_op": {
                "kind": "aten",
                "target": "view",
                "effects": {
                    "aliases": [{"input": 0, "output": 0, "conditional": False}]
                },
            },
            "tensor_names": {"inputs": ["loop"], "outputs": ["loop"]},
        },
    }
    for tensor_name in ("untyped", "bad", "mutated", "loop"):
        contraction = {"tensor_names": {"inputs": [tensor_name]}}
        assert not contraction_operands_are_graph_external(contraction, layers)
        assert contraction_external_source_dtypes(contraction, layers) == set()


def test_conditional_alias_is_not_treated_as_zero_copy():
    layers = {
        "start": _start("input"),
        "conditional": {
            "type": "view",
            "semantic_op": {
                "kind": "aten",
                "target": "view",
                "effects": {
                    "aliases": [{"input": 0, "output": 0, "conditional": True}]
                },
            },
            "tensor_names": {"inputs": ["input"], "outputs": ["output"]},
        },
    }
    assert not contraction_operands_are_graph_external(
        {"tensor_names": {"inputs": ["output"]}}, layers
    )
