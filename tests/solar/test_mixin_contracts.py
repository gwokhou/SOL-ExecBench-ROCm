"""Runtime completeness checks for SOLAR's cooperative mixin systems."""

from __future__ import annotations

from inspect import isabstract

from solar.analysis.graph_analyzer import IRGraphAnalyzer
from solar.graph.torchview.processor import TorchviewProcessor
from solar.ir.extended_einsum.torchview.converter import PyTorchToEinsum


def test_final_mixin_compositions_satisfy_runtime_contracts() -> None:
    assert not isabstract(IRGraphAnalyzer)
    assert not isabstract(TorchviewProcessor)
    assert not isabstract(PyTorchToEinsum)

    assert isinstance(IRGraphAnalyzer(), IRGraphAnalyzer)
    assert isinstance(TorchviewProcessor(), TorchviewProcessor)
    assert isinstance(PyTorchToEinsum(), PyTorchToEinsum)
