"""Composition smoke checks for SOLAR workflow façades."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from solar.analysis.graph_analyzer import IRGraphAnalyzer
from solar.graph.torchview.processor import TorchviewProcessor
from solar.ir.extended_einsum.torchview.converter import PyTorchToEinsum
from solar.ir.extended_einsum.torchview.converter_pipeline import GraphEmitter


def test_workflow_facades_construct_composed_components() -> None:
    assert isinstance(IRGraphAnalyzer(), IRGraphAnalyzer)
    assert isinstance(TorchviewProcessor(), TorchviewProcessor)
    assert isinstance(PyTorchToEinsum(), PyTorchToEinsum)
    assert len(IRGraphAnalyzer()._components) == 4
    assert len(TorchviewProcessor()._components) == 4
    assert len(PyTorchToEinsum()._components) == 8


def test_converter_reuse_creates_isolated_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converter = PyTorchToEinsum()
    states: list[object] = []

    def fake_convert(
        component: GraphEmitter, *_args: object, **_kwargs: object
    ) -> dict[str, object]:
        states.append(component._host._state)
        return {}

    monkeypatch.setattr(GraphEmitter, "convert", fake_convert)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda index: converter.convert(
                    tmp_path / f"input-{index}.yaml",
                    tmp_path / f"output-{index}",
                ),
                range(2),
            )
        )
    assert results == [{}, {}]
    assert len({id(state) for state in states}) == 2


def test_processor_reuse_creates_isolated_state(
    tmp_path: Path,
) -> None:
    processor = TorchviewProcessor()
    states: list[object] = []

    def fake_extract(*_args: object) -> list[object]:
        states.append(processor._state)
        return []

    processor._extract_layer_nodes = fake_extract
    processor._save_pytorch_graph_yaml = lambda *_args, **_kwargs: None
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda index: processor.process_graph(
                    object(),
                    str(tmp_path / f"output-{index}"),
                    f"kernel-{index}",
                ),
                range(2),
            )
        )
    assert results == [[], []]
    assert len({id(state) for state in states}) == 2
