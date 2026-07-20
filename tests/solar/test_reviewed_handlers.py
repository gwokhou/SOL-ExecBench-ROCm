from __future__ import annotations

import hashlib
import json
from pathlib import Path

import networkx as nx
import pytest

from solar.einsum import pytorch_to_einsum
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.einsum.reviewed_handlers import (
    ReviewedHandlerError,
    expand_reviewed_handlers,
)


_HANDLER_SOURCE = """\
def create_mystery_subgraph(node_id: str, node_data: Dict[str, Any]):
    del node_data
    return {
        node_id + ".first": {"type": "add"},
        node_id + ".last": {"type": "relu"},
    }
"""


def _write_handler(root: Path, *, approved: bool = True) -> None:
    digest = hashlib.sha256(_HANDLER_SOURCE.encode()).hexdigest()
    (root / "mystery.py").write_text(_HANDLER_SOURCE)
    (root / "mystery_handler.json").write_text(
        json.dumps(
            {
                "node_type": "mystery",
                "is_generated": True,
                "code_file": "mystery.py",
                "source_sha256": digest,
                "metadata": {
                    "verification": "passed",
                    "formal_review": "approved" if approved else "pending",
                    "source_sha256": digest,
                },
            }
        )
    )


def test_formal_expander_uses_only_approved_content_addressed_handlers(
    tmp_path: Path,
) -> None:
    _write_handler(tmp_path)
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [
            ("source", {"type": "input-tensor"}),
            ("operation", {"type": "mystery"}),
            ("sink", {"type": "output-tensor"}),
        ]
    )
    graph.add_edges_from([("source", "operation"), ("operation", "sink")])

    expanded = expand_reviewed_handlers(graph, handler_directory=tmp_path)

    assert "operation" not in expanded
    assert list(nx.topological_sort(expanded)) == [
        "source",
        "operation.first",
        "operation.last",
        "sink",
    ]


def test_formal_expander_rejects_unapproved_or_tampered_handlers(
    tmp_path: Path,
) -> None:
    _write_handler(tmp_path, approved=False)
    graph = nx.DiGraph()
    graph.add_node("operation", type="mystery")
    with pytest.raises(ReviewedHandlerError, match="formal_review must be approved"):
        expand_reviewed_handlers(graph, handler_directory=tmp_path)

    _write_handler(tmp_path)
    (tmp_path / "mystery.py").write_text(_HANDLER_SOURCE + "\n# changed\n")
    with pytest.raises(ReviewedHandlerError, match="source hash mismatch"):
        expand_reviewed_handlers(graph, handler_directory=tmp_path)


def test_reviewed_handler_stage_is_read_only_and_fail_closed(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    graph = nx.DiGraph()
    graph.add_node("operation", type="mystery")

    assert expand_reviewed_handlers(graph, handler_directory=missing).nodes == (
        graph.nodes
    )
    assert not missing.exists()

    missing.mkdir()
    _write_handler(missing, approved=False)
    with pytest.raises(ReviewedHandlerError, match="formal_review must be approved"):
        expand_reviewed_handlers(graph, handler_directory=missing)


def test_strict_converter_routes_through_reviewed_handler_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    converter = PyTorchToEinsum(strict=True, cache_dir=str(tmp_path))
    graph = nx.DiGraph()
    graph.add_node("operation", type="add")
    calls: list[nx.DiGraph] = []
    monkeypatch.setattr(converter, "_build_op_graph", lambda value: (graph, [], []))
    monkeypatch.setattr(
        converter,
        "_expand_reviewed_ops",
        lambda value: calls.append(value) or value,
    )
    monkeypatch.setattr(
        converter,
        "_build_einsum_graph",
        lambda *args: {"layers": {}},
    )
    monkeypatch.setattr(
        pytorch_to_einsum, "annotate_semantics", lambda value, **kw: value
    )
    monkeypatch.setattr(
        pytorch_to_einsum, "validate_semantic_graph", lambda value: None
    )
    monkeypatch.setattr(pytorch_to_einsum, "add_taco_expressions", lambda value: value)
    monkeypatch.setattr(converter, "_validate_exact_graph", lambda value: None)
    monkeypatch.setattr(
        converter, "_validate_tensor_shape_consistency", lambda value: None
    )

    converter._convert_loaded_graph({}, expand_complex_ops=True)

    assert calls == [graph]
