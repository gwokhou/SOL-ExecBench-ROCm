from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import cast

import networkx as nx
import pytest

from solar.einsum import graph_expander
from solar.einsum.graph_expander import GraphExpander
from solar.einsum.llm_agent import NodeTypeConversionAgent
from solar.einsum.node_type_registry import NodeTypeHandlerFactory


def _subgraph(node_id, node_data):
    del node_data
    return {
        f"{node_id}.first": {"type": "add"},
        f"{node_id}.last": {"type": "relu"},
    }


def test_custom_handler_expands_and_reconnects_graph(tmp_path) -> None:
    expander = GraphExpander(debug=True, cache_dir=str(tmp_path))
    expander.registry.register(
        "composite",
        NodeTypeHandlerFactory.create_handler_from_methods(
            "composite", create_subgraph_method=_subgraph
        ),
    )
    graph = nx.DiGraph()
    graph.add_node("source", type="input-tensor")
    graph.add_node(
        "complex", type="composite", input_shapes=[[1], [1], [1]], node_id="complex"
    )
    graph.add_node("sink", type="output-tensor")
    graph.add_edges_from([("source", "complex"), ("complex", "sink")])

    expanded = expander.expand(graph)

    assert "complex" not in expanded
    assert list(nx.topological_sort(expanded)) == [
        "source",
        "complex.first",
        "complex.last",
        "sink",
    ]
    assert expander.debug is True
    assert expander.registry.has_handler("composite")


def test_expander_leaves_empty_simple_and_unhandled_graphs_unchanged(tmp_path) -> None:
    expander = GraphExpander(cache_dir=str(tmp_path))
    assert expander.expand(nx.DiGraph()).number_of_nodes() == 0

    graph = nx.DiGraph()
    graph.add_node("add", type="add")
    graph.add_node("gelu", type="gelu")
    graph.add_node("reduction", type="reduction_custom")
    assert set(expander.expand(graph)) == {"add", "gelu", "reduction"}
    assert expander._expand_node("missing", {"type": "missing"}) is None


def test_connect_subgraph_handles_empty_and_absent_neighbors(tmp_path) -> None:
    expander = GraphExpander(cache_dir=str(tmp_path))
    graph = nx.DiGraph()
    expander._connect_subgraph(graph, {}, ["missing"], ["missing"])
    graph.add_nodes_from(["first", "second"])
    expander._connect_subgraph(
        graph,
        {"first": {}, "second": {}},
        ["missing"],
        ["also_missing"],
    )
    assert list(graph.edges) == [("first", "second")]


def test_unknown_handler_is_generated_cached_and_used(tmp_path) -> None:
    code = (
        "def create_subgraph(node_id, node_data):\n"
        "    return {node_id + '_generated': {'type': 'relu'}}\n"
    )
    metadata = {
        "verification": "passed",
        "source_sha256": hashlib.sha256(code.encode()).hexdigest(),
    }

    class Agent:
        def generate_conversion_code(self, node_type, node_data):
            assert node_type == "mystery"
            assert node_data["type"] == "mystery"
            return code, metadata

    expander = GraphExpander(cache_dir=str(tmp_path))
    expander._agent = cast(NodeTypeConversionAgent, Agent())

    result = expander._handle_unknown_node_type(
        "mystery", {"type": "mystery", "node_id": "node"}
    )

    assert result == {"node_generated": {"type": "relu"}}
    assert expander.registry.has_handler("mystery")
    assert (tmp_path / "mystery_handler.json").is_file()
    assert expander._handle_unknown_node_type("other", {"type": "other"}) is None


def test_agent_failures_are_best_effort_or_fail_closed(tmp_path) -> None:
    class FailingAgent:
        def generate_conversion_code(self, node_type, node_data):
            raise RuntimeError("generation failed")

    best_effort = GraphExpander(cache_dir=str(tmp_path / "best"))
    best_effort._agent = cast(NodeTypeConversionAgent, FailingAgent())
    assert best_effort._handle_unknown_node_type("x", {}) is None

    strict = GraphExpander(cache_dir=str(tmp_path / "strict"), fail_closed=True)
    strict._agent = cast(NodeTypeConversionAgent, FailingAgent())
    with pytest.raises(RuntimeError, match="generation failed"):
        strict._handle_unknown_node_type("x", {})


def test_agent_initialization_uses_environment_and_interactive_fallback(
    tmp_path, monkeypatch
) -> None:
    class Config:
        def __init__(self, **kwargs):
            self.model = "fake"
            self.kwargs = kwargs

    created: list[Config] = []

    monkeypatch.setattr(graph_expander, "AgentConfig", Config, raising=False)
    monkeypatch.setattr("solar.einsum.llm_agent.AgentConfig", Config)
    monkeypatch.setattr(
        "solar.einsum.llm_agent.NodeTypeConversionAgent",
        lambda config: created.append(config) or SimpleNamespace(config=config),
    )
    monkeypatch.setattr(
        "solar.einsum.llm_agent.get_api_key_interactive", lambda: "interactive"
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    expander = GraphExpander(
        debug=True, enable_agent=True, cache_dir=str(tmp_path), fail_closed=True
    )

    assert expander._agent is not None
    assert created[0].kwargs["api_key"] == "interactive"


def test_agent_initialization_can_disable_or_propagate_failure(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("solar.einsum.llm_agent.get_api_key_interactive", lambda: None)
    disabled = GraphExpander(enable_agent=True, cache_dir=str(tmp_path / "disabled"))
    assert disabled._agent is None

    class BrokenConfig:
        def __init__(self, **kwargs):
            raise RuntimeError("bad config")

    monkeypatch.setattr("solar.einsum.llm_agent.AgentConfig", BrokenConfig)
    with pytest.raises(RuntimeError, match="bad config"):
        GraphExpander(
            enable_agent=True,
            api_key="key",
            fail_closed=True,
            cache_dir=str(tmp_path / "failed"),
        )
