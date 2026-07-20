from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
import yaml

from solar.common.types import ProcessingConfig
from solar.graph import pytorch_processor
from solar.graph.pytorch_processor import PyTorchProcessor


META_MODEL = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, value, mask):
        return torch.where(mask, value * self.weight, value)

def get_init_inputs():
    return [3]

def get_inputs():
    return [torch.ones(2, 3), torch.tensor([[True, False, True], [False, True, False]])]

def launch_reference_implementation(model, *inputs):
    return model(*inputs)
"""


CPU_FALLBACK_MODEL = """
import torch
import torch.nn as nn

HIDDEN_SIZE = 3

class ReferenceModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, value):
        return torch.tanh(value)

def get_inputs():
    value = torch.ones(2, HIDDEN_SIZE)
    value.numpy()
    return [value]
"""


def _processor(monkeypatch, **config: Any) -> PyTorchProcessor:
    monkeypatch.setattr(
        pytorch_processor, "_check_torchview_parameter_support", lambda: None
    )
    return PyTorchProcessor(ProcessingConfig(**config))


@pytest.mark.parametrize(
    ("filename", "source", "expected_dtype"),
    [
        ("model.py", META_MODEL, "torch.float32"),
        ("source_reference.py", CPU_FALLBACK_MODEL, "torch.float32"),
    ],
)
def test_process_model_file_supports_meta_and_cpu_fallback(
    tmp_path: Path,
    monkeypatch,
    filename: str,
    source: str,
    expected_dtype: str,
) -> None:
    model_path = tmp_path / filename
    model_path.write_text(source)
    output = tmp_path / "output"
    processor = _processor(monkeypatch, debug=True, force_rerun=True)

    assert processor.process_model_file(str(model_path), str(output)) is True

    graph = yaml.safe_load((output / "pytorch_graph.yaml").read_text())
    inputs = [
        layer
        for layer in graph["layers"].values()
        if layer["type"] == "auxiliary-tensor"
    ]
    assert inputs
    assert inputs[0]["output_dtypes"] == [expected_dtype]
    expected_copy = filename if filename.startswith("source_") else f"source_{filename}"
    assert (output / expected_copy).is_file()

    processor.config.force_rerun = False
    assert processor.process_model_file(str(model_path), str(output)) is True


def test_process_model_file_reports_load_and_graph_failures(
    tmp_path: Path, monkeypatch
) -> None:
    processor = _processor(monkeypatch, debug=True)
    missing = tmp_path / "missing.py"
    assert (
        processor.process_model_file(str(missing), str(tmp_path / "missing")) is False
    )

    model = tmp_path / "model.py"
    model.write_text(META_MODEL)
    monkeypatch.setattr(
        processor, "_generate_torchview_graph", lambda *args, **kwargs: None
    )
    assert processor.process_model_file(str(model), str(tmp_path / "no-graph")) is False


def test_load_model_rejects_missing_contracts(tmp_path: Path, monkeypatch) -> None:
    processor = _processor(monkeypatch, debug=True)
    no_model = tmp_path / "no_model.py"
    no_model.write_text("def get_inputs(): return []\n")
    assert processor._load_model(str(no_model)) == (None, None, None)

    no_inputs = tmp_path / "no_inputs.py"
    no_inputs.write_text("import torch.nn as nn\nclass Model(nn.Module): pass\n")
    assert processor._load_model(str(no_inputs)) == (None, None, None)

    bad_model = tmp_path / "bad_model.py"
    bad_model.write_text(
        "import torch.nn as nn\n"
        "class Model(nn.Module):\n"
        "    def __init__(self, required): super().__init__()\n"
        "def get_inputs(): return []\n"
    )
    assert processor._load_model(str(bad_model)) == (None, None, None)


def test_model_class_and_construction_strategies(monkeypatch) -> None:
    processor = _processor(monkeypatch, debug=True)

    class NoArg(nn.Module):
        pass

    class NeedsWidth(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.hidden_size = hidden_size

    assert processor._get_model_class(SimpleNamespace(Model=NoArg)) == (NoArg, "Model")
    assert processor._get_model_class(SimpleNamespace(ReferenceModel=NoArg)) == (
        NoArg,
        "ReferenceModel",
    )
    assert processor._get_model_class(SimpleNamespace()) == (None, "")
    assert isinstance(
        processor._create_model_instance(SimpleNamespace(Model=NoArg)), NoArg
    )
    inferred = processor._create_model_instance(
        SimpleNamespace(Model=NeedsWidth, HIDDEN_SIZE=7)
    )
    assert isinstance(inferred, NeedsWidth)
    assert inferred.hidden_size == 7

    failing = SimpleNamespace(
        Model=NeedsWidth,
        get_init_inputs=lambda: (_ for _ in ()).throw(RuntimeError("bad")),
    )
    assert processor._create_model_instance(failing) is None


def test_infer_init_args_handles_defaults_missing_and_bad_signature(
    monkeypatch,
) -> None:
    processor = _processor(monkeypatch, debug=True)

    class Configured(nn.Module):
        def __init__(self, num_heads, dropout=0.0):
            super().__init__()

    assert processor._infer_init_args_from_module(
        SimpleNamespace(NUM_HEADS=8), Configured
    ) == ((8,), {})
    assert processor._infer_init_args_from_module(SimpleNamespace(), Configured) == (
        (),
        {},
    )
    monkeypatch.setattr(pytorch_processor.inspect, "signature", lambda value: 1 / 0)
    assert processor._infer_init_args_from_module(SimpleNamespace(), Configured) == (
        (),
        {},
    )


def test_model_call_inference_covers_launch_and_fallback(monkeypatch) -> None:
    processor = _processor(monkeypatch)
    model = nn.Identity()
    inputs = [torch.ones(1)]

    default = processor._infer_model_call_from_launch(SimpleNamespace(), model, inputs)
    assert torch.equal(default(model, inputs), inputs[0])

    def launch_reference_implementation(model, *inputs):
        return model(*inputs)

    unpack = processor._infer_model_call_from_launch(
        SimpleNamespace(
            launch_reference_implementation=launch_reference_implementation
        ),
        model,
        inputs,
    )
    assert torch.equal(unpack(model, inputs), inputs[0])
    monkeypatch.setattr(pytorch_processor.inspect, "getsource", lambda value: 1 / 0)
    fallback = processor._infer_model_call_from_launch(
        SimpleNamespace(launch_reference_implementation=lambda *args: None),
        model,
        inputs[0],
    )
    assert torch.equal(fallback(model, inputs[0]), inputs[0])


def test_move_inputs_and_rnn_detection(monkeypatch) -> None:
    processor = _processor(monkeypatch)
    nested = (
        torch.ones(2),
        [torch.zeros(1)],
        {"flag": torch.tensor([True]), "value": 3},
    )
    meta = processor._move_inputs_to_device(nested, "meta")
    assert meta[0].device.type == "meta"
    assert meta[1][0].device.type == "meta"
    assert meta[2]["flag"].device.type == "meta"
    cpu = processor._move_inputs_to_device(meta, "cpu")
    assert cpu[0].device.type == "cpu"
    assert cpu[2]["value"] == 3

    hidden = nn.Identity()
    setattr(hidden, "hidden", object())
    assert processor._is_rnn_model(hidden) is True
    named = nn.Module()
    named.add_module("gru_block", nn.Identity())
    assert processor._is_rnn_model(named) is True
    assert processor._is_rnn_model(nn.Linear(1, 1)) is False


def test_save_visual_graph_handles_large_missing_and_failed_render(
    tmp_path: Path, monkeypatch
) -> None:
    processor = _processor(monkeypatch, debug=True, save_graph=True)

    class Visual:
        def __init__(self) -> None:
            self.saved: str | None = None
            self.rendered: dict[str, object] | None = None

        def save(self, filename: str) -> None:
            self.saved = filename

        def render(self, **kwargs: object) -> None:
            self.rendered = kwargs

    visual = Visual()
    large = SimpleNamespace(
        visual_graph=visual,
        edge_list=[None] * (processor._MAX_RENDER_EDGES + 1),
    )
    processor._save_torchview_graph(large, str(tmp_path))
    assert visual.saved is not None
    assert visual.saved.endswith("torchview_graph.gv")

    small = SimpleNamespace(visual_graph=visual, edge_list=[])
    processor._save_torchview_graph(small, str(tmp_path))
    assert visual.rendered is not None
    assert visual.rendered["format"] == "pdf"
    processor._save_torchview_graph(SimpleNamespace(), str(tmp_path))

    class FailingVisual(Visual):
        def render(self, **kwargs: object) -> None:
            raise RuntimeError("render")

    processor._save_torchview_graph(
        SimpleNamespace(visual_graph=FailingVisual(), edge_list=[]), str(tmp_path)
    )


def test_patch_input_dtypes_handles_all_boundary_shapes(
    tmp_path: Path, monkeypatch
) -> None:
    processor = _processor(monkeypatch)
    processor._patch_input_dtypes(tmp_path, [torch.ones(1)])
    graph_path = tmp_path / "pytorch_graph.yaml"
    graph_path.write_text("not: [valid")
    processor._patch_input_dtypes(tmp_path, [torch.ones(1)])

    graph = {
        "layers": {
            "input": {
                "type": "input-tensor",
                "input_dtypes": ["torch.float32"],
                "output_dtypes": ["torch.float32"],
            }
        }
    }
    graph_path.write_text(yaml.safe_dump(graph))
    processor._patch_input_dtypes(tmp_path, {"mask": torch.tensor([True])})
    patched = yaml.safe_load(graph_path.read_text())
    assert patched["layers"]["input"]["output_dtypes"] == ["torch.bool"]
    processor._patch_input_dtypes(tmp_path, ["not-a-tensor"])


def test_parameter_support_probe_reports_draw_and_contract_failures(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        pytorch_processor.torchview,
        "draw_graph",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("draw")),
    )
    with pytest.raises(RuntimeError, match="torchview probe failed"):
        pytorch_processor._check_torchview_parameter_support()

    monkeypatch.setattr(
        pytorch_processor.torchview,
        "draw_graph",
        lambda *args, **kwargs: SimpleNamespace(
            edge_list=[(SimpleNamespace(name="x"),)]
        ),
    )
    with pytest.raises(RuntimeError, match="does not generate parameter-tensor"):
        pytorch_processor._check_torchview_parameter_support()


def test_safe_mode_setup_uses_shared_environment_helper(monkeypatch) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(
        pytorch_processor, "setup_safe_environment", lambda: calls.append(True)
    )
    monkeypatch.setattr(
        pytorch_processor, "_check_torchview_parameter_support", lambda: None
    )

    PyTorchProcessor(ProcessingConfig(safe_mode=True))

    assert calls == [True]
